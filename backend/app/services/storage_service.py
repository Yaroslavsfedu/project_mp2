"""Слой объектного хранилища: локальный диск (dev) или MinIO (prod).

Интерфейс намеренно повторяет S3-семантику (ключи, presigned URL),
чтобы переключение `STORAGE_BACKEND=minio` не затрагивало код пайплайна.
"""
from __future__ import annotations

import asyncio
import re
import shutil
from datetime import timedelta
from pathlib import Path

from app.core.config import settings

_EXTRA_SAFE = set("._- ()[]")

# UUID в каноническом виде: только так мы строим путь из ID (защита от Path Traversal)
_UUID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")

# Ключи прежней плоской схемы (originals/{id}/..., pages/{id}/...) остаются читаемыми:
# путь хранится в БД (documents.s3_key_original, ocr_pages.image_key), поэтому миграция файлов не нужна.
LEGACY_ORIGINALS_PREFIX = "originals/"
LEGACY_PAGES_PREFIX = "pages/"
FILE_PREFIXES = ("uploads/", LEGACY_ORIGINALS_PREFIX, "pages/")


def shard_dirs(doc_id: str) -> str:
    """Двухуровневый шард по ID документа: ``ab/cd`` (по аналогии с Git objects).

    65 536 листовых каталогов: при 2 млн документов ~30 файлов на каталог, а не 2 млн в одном.
    """
    if not _UUID_RE.match(doc_id or ""):
        raise ValueError("Некорректный идентификатор документа")
    return f"{doc_id[:2]}/{doc_id[2:4]}"


def original_key(doc_id: str, filename: str) -> str:
    """Ключ оригинала: ``uploads/ab/cd/{doc_id}_{filename}``. filename уже санирован."""
    return f"uploads/{shard_dirs(doc_id)}/{doc_id}_{filename}"


def page_image_key(doc_id: str, page_number: int) -> str:
    """Ключ растра страницы: ``pages/ab/cd/{doc_id}/page-N.png``."""
    return f"pages/{shard_dirs(doc_id)}/{doc_id}/page-{int(page_number)}.png"


def document_prefixes(doc_id: str) -> list[str]:
    """Все префиксы, где могут лежать файлы документа: новые (шард) и прежние (плоские)."""
    shard = shard_dirs(doc_id)
    return [
        f"pages/{shard}/{doc_id}",
        f"{LEGACY_PAGES_PREFIX}{doc_id}",
        f"{LEGACY_ORIGINALS_PREFIX}{doc_id}",
    ]


def safe_filename(name: str) -> str:
    """Защита от Path Traversal и управляющих символов в имени файла.

    Буквы и цифры любых алфавитов (в т.ч. кириллица) сохраняются, опасные
    символы (слэши, кавычки, управляющие коды) заменяются на «_».
    """
    name = Path(name or "document").name.replace("\\", "_").replace("/", "_")
    cleaned = "".join(ch if (ch.isalnum() or ch in _EXTRA_SAFE) else "_" for ch in name)
    cleaned = cleaned.strip(" .")
    return (cleaned or "document")[:200]


class StorageService:
    def __init__(self) -> None:
        self.backend = settings.STORAGE_BACKEND
        self.root = settings.storage_dir
        self.root.mkdir(parents=True, exist_ok=True)
        self._client = None

    # ---------------- MinIO ----------------
    def _minio(self):
        if self._client is None:
            from minio import Minio  # ленивый импорт

            client = Minio(
                settings.MINIO_ENDPOINT,
                access_key=settings.MINIO_ACCESS_KEY,
                secret_key=settings.MINIO_SECRET_KEY,
                secure=settings.MINIO_SECURE,
            )
            if not client.bucket_exists(settings.MINIO_BUCKET):
                client.make_bucket(settings.MINIO_BUCKET)
            self._client = client
        return self._client

    # ---------------- API ----------------
    def _local_path(self, key: str) -> Path:
        path = (self.root / key).resolve()
        if not str(path).startswith(str(self.root.resolve())):
            raise ValueError("Недопустимый ключ хранилища")
        return path

    async def put(self, key: str, data: bytes) -> str:
        if self.backend == "minio":
            import io

            def _put() -> None:
                self._minio().put_object(
                    settings.MINIO_BUCKET, key, io.BytesIO(data), length=len(data)
                )

            await asyncio.to_thread(_put)
            return key
        path = self._local_path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return key

    async def delete_key(self, key: str) -> None:
        """Удаляет один объект по точному ключу (оригинал, отдельная страница)."""
        if self.backend == "minio":
            def _remove() -> None:
                self._minio().remove_object(settings.MINIO_BUCKET, key)

            try:
                await asyncio.to_thread(_remove)
            except Exception:  # noqa: BLE001 - объекта могло не быть
                pass
            return
        self._local_path(key).unlink(missing_ok=True)

    async def get(self, key: str) -> bytes:
        if self.backend == "minio":
            def _get() -> bytes:
                response = self._minio().get_object(settings.MINIO_BUCKET, key)
                try:
                    return response.read()
                finally:
                    response.close()
                    response.release_conn()

            return await asyncio.to_thread(_get)
        return self._local_path(key).read_bytes()

    async def local_path(self, key: str) -> str:
        """Путь к файлу на диске (для PyMuPDF/OpenCV). Для MinIO — временная копия."""
        if self.backend != "minio":
            return str(self._local_path(key))
        tmp_dir = settings.DATA_DIR / "tmp"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        target = tmp_dir / safe_filename(Path(key).name)
        target.write_bytes(await self.get(key))
        return str(target)

    async def presigned_url(self, key: str) -> str:
        if self.backend == "minio":
            def _sign() -> str:
                return self._minio().presigned_get_object(
                    settings.MINIO_BUCKET,
                    key,
                    expires=timedelta(minutes=settings.PRESIGNED_URL_TTL_MINUTES),
                )

            return await asyncio.to_thread(_sign)
        # локальный режим: файл отдаётся через защищённый эндпоинт API
        return f"{settings.API_V1_PREFIX}/files/{key}"

    async def delete_document_files(self, doc_id: str, original_key_value: str | None) -> None:
        """Удаляет оригинал и страницы документа: и по новым шард-путям, и по прежним плоским."""
        if original_key_value:
            await self.delete_key(original_key_value)
        for prefix in document_prefixes(doc_id):
            await self.delete_prefix(prefix)

    async def delete_prefix(self, prefix: str) -> None:
        if self.backend == "minio":
            def _delete() -> None:
                client = self._minio()
                for obj in client.list_objects(settings.MINIO_BUCKET, prefix=prefix, recursive=True):
                    client.remove_object(settings.MINIO_BUCKET, obj.object_name)

            await asyncio.to_thread(_delete)
            return
        target = self.root / prefix
        if target.is_dir():
            shutil.rmtree(target, ignore_errors=True)
        elif target.exists():
            target.unlink(missing_ok=True)


storage = StorageService()
