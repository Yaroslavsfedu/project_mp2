"""Вспомогательные функции HTTP-слоя."""
from __future__ import annotations

from urllib.parse import quote


def content_disposition(filename: str, disposition: str = "attachment") -> str:
    """Заголовок Content-Disposition с поддержкой кириллицы (RFC 6266 / RFC 5987).

    Starlette кодирует заголовки в latin-1, поэтому имя с русскими буквами
    в обычном виде вызвало бы ошибку 500. Здесь дублируем ASCII-вариант
    и даём UTF-8 вариант через filename*.
    """
    ascii_name = filename.encode("ascii", "ignore").decode() or "document"
    ascii_name = ascii_name.replace('"', "_").replace("\\", "_")
    return f"{disposition}; filename=\"{ascii_name}\"; filename*=UTF-8''{quote(filename)}"
