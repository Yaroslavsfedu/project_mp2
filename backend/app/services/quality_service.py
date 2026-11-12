"""Метрики качества распознавания и извлечения сущностей (CER / WER / F1).

Метрики считаются по эталонному датасету (`samples/ground_truth.json`):
* CER — Character Error Rate (расстояние Левенштейна / длина эталона);
* WER — Word Error Rate по токенам;
* Precision / Recall / F1 — по извлечённым сущностям, микро-усреднение по полям;
* Accuracy — точность классификации типа документа.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

ENTITY_FIELDS = (
    "persons",
    "organizations",
    "dates",
    "amounts",
    "inn_ogrn",
    "addresses",
)

# Для длинных строковых сущностей допускаем частичное совпадение:
# адрес или название организации в документе может быть обрезано переносом строки.
FUZZY_FIELDS = {"organizations", "addresses"}
FUZZY_MIN_LENGTH = 10

_PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)
_WS_RE = re.compile(r"\s+", re.UNICODE)


# ----------------------------- базовые метрики -----------------------------
def normalize_for_cer(text: str) -> str:
    text = text.lower().replace("ё", "е")
    text = _WS_RE.sub(" ", text)
    return text.strip()


def normalize_tokens(text: str) -> list[str]:
    text = normalize_for_cer(text)
    return text.split() if text else []


def levenshtein(reference: str, hypothesis: str) -> int:
    """Расстояние Левенштейна (итеративно, O(min(n,m)) памяти)."""
    if reference == hypothesis:
        return 0
    if not reference:
        return len(hypothesis)
    if not hypothesis:
        return len(reference)
    if len(reference) < len(hypothesis):
        reference, hypothesis = hypothesis, reference

    previous = list(range(len(hypothesis) + 1))
    for i, ref_char in enumerate(reference, start=1):
        current = [i]
        for j, hyp_char in enumerate(hypothesis, start=1):
            insert = current[j - 1] + 1
            delete = previous[j] + 1
            replace = previous[j - 1] + (ref_char != hyp_char)
            current.append(min(insert, delete, replace))
        previous = current
    return previous[-1]


def levenshtein_tokens(reference: list[str], hypothesis: list[str]) -> int:
    """Расстояние Левенштейна на уровне токенов (для WER)."""
    if reference == hypothesis:
        return 0
    if not reference:
        return len(hypothesis)
    if not hypothesis:
        return len(reference)
    previous = list(range(len(hypothesis) + 1))
    for i, ref_token in enumerate(reference, start=1):
        current = [i]
        for j, hyp_token in enumerate(hypothesis, start=1):
            current.append(
                min(
                    current[j - 1] + 1,
                    previous[j] + 1,
                    previous[j - 1] + (ref_token != hyp_token),
                )
            )
        previous = current
    return previous[-1]


def cer(reference: str, hypothesis: str) -> float:
    ref = normalize_for_cer(reference)
    hyp = normalize_for_cer(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0
    return levenshtein(ref, hyp) / len(ref)


def wer(reference: str, hypothesis: str) -> float:
    ref_tokens = normalize_tokens(reference)
    hyp_tokens = normalize_tokens(hypothesis)
    if not ref_tokens:
        return 0.0 if not hyp_tokens else 1.0
    return levenshtein_tokens(ref_tokens, hyp_tokens) / len(ref_tokens)


def _normalize_entity(value: str) -> str:
    value = value.lower().replace("ё", "е")
    value = _PUNCT_RE.sub(" ", value)
    return _WS_RE.sub(" ", value).strip()


def _entity_set(values: Iterable[str] | str | None) -> set[str]:
    if values is None:
        return set()
    if isinstance(values, str):
        values = [values]
    return {_normalize_entity(v) for v in values if v and _normalize_entity(v)}


@dataclass
class EntityScore:
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    per_field: dict[str, dict[str, float]] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives,
            "per_field": self.per_field,
        }


def _match_sets(ref_set: set[str], pred_set: set[str], fuzzy: bool) -> tuple[int, int, int]:
    """Возвращает (TP, FP, FN) с опциональным частичным сопоставлением строк."""
    matched_ref: set[str] = set()
    matched_pred: set[str] = set()

    for value in ref_set & pred_set:
        matched_ref.add(value)
        matched_pred.add(value)

    if fuzzy:
        for ref_value in sorted(ref_set - matched_ref, key=len, reverse=True):
            for pred_value in sorted(pred_set - matched_pred, key=len, reverse=True):
                if len(ref_value) < FUZZY_MIN_LENGTH or len(pred_value) < FUZZY_MIN_LENGTH:
                    continue
                if ref_value in pred_value or pred_value in ref_value:
                    matched_ref.add(ref_value)
                    matched_pred.add(pred_value)
                    break

    tp = len(matched_ref)
    fp = len(pred_set) - len(matched_pred)
    fn = len(ref_set) - len(matched_ref)
    return tp, fp, fn


def score_entities(
    reference: dict[str, Any],
    predicted: dict[str, Any],
) -> EntityScore:
    """Микро-усреднённые P/R/F1 по всем документам и полям."""
    score = EntityScore()
    fields = list(ENTITY_FIELDS) + ["document_number"]
    for field_name in fields:
        ref_set = _entity_set(reference.get(field_name))
        pred_set = _entity_set(predicted.get(field_name))
        tp, fp, fn = _match_sets(ref_set, pred_set, field_name in FUZZY_FIELDS)
        score.true_positives += tp
        score.false_positives += fp
        score.false_negatives += fn
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        score.per_field[field_name] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "tp": tp,
            "fp": fp,
            "fn": fn,
        }

    tp = score.true_positives
    fp = score.false_positives
    fn = score.false_negatives
    score.precision = tp / (tp + fp) if tp + fp else 0.0
    score.recall = tp / (tp + fn) if tp + fn else 0.0
    score.f1 = (
        2 * score.precision * score.recall / (score.precision + score.recall)
        if score.precision + score.recall
        else 0.0
    )
    return score


# ----------------------------- отчёт -----------------------------
def build_report(
    per_document: list[dict[str, Any]],
    extra: dict[str, Any] | None = None,
    skipped: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Собирает сводный отчёт: средние метрики, худшие документы, сведения о среде."""
    if not per_document:
        return {
            "generated_at": datetime.utcnow().isoformat(timespec="seconds"),
            "documents": 0,
            "per_document": [],
            "skipped": skipped or [],
            "metrics": {},
        }

    count = len(per_document)
    avg_cer = sum(item["cer"] for item in per_document) / count
    avg_wer = sum(item["wer"] for item in per_document) / count
    avg_time = sum(item.get("duration_sec", 0.0) for item in per_document) / count
    type_correct = sum(1 for item in per_document if item.get("doc_type_correct"))

    reference_all: dict[str, Any] = {}
    predicted_all: dict[str, Any] = {}
    for item in per_document:
        for field_name in list(ENTITY_FIELDS) + ["document_number"]:
            ref_value = item.get("reference_entities", {}).get(field_name)
            pred_value = item.get("predicted_entities", {}).get(field_name)
            if ref_value is None and pred_value is None:
                continue
            ref_list = ref_value if isinstance(ref_value, list) else ([ref_value] if ref_value else [])
            pred_list = pred_value if isinstance(pred_value, list) else ([pred_value] if pred_value else [])
            reference_all.setdefault(field_name, []).extend(ref_list)
            predicted_all.setdefault(field_name, []).extend(pred_list)

    entities = score_entities(reference_all, predicted_all)

    report = {
        "generated_at": datetime.utcnow().isoformat(timespec="seconds"),
        "documents": count,
        "metrics": {
            "cer": round(avg_cer, 4),
            "wer": round(avg_wer, 4),
            "ner_precision": round(entities.precision, 4),
            "ner_recall": round(entities.recall, 4),
            "ner_f1": round(entities.f1, 4),
            "doc_type_accuracy": round(type_correct / count, 4),
            "avg_duration_sec": round(avg_time, 3),
        },
        "entities": entities.as_dict(),
        "per_document": per_document,
        "skipped": skipped or [],
        "environment": extra or {},
        "worst_documents": sorted(per_document, key=lambda item: item["cer"], reverse=True)[:3],
    }
    return report


def report_to_markdown(report: dict[str, Any]) -> str:
    metrics = report.get("metrics", {})
    env = report.get("environment", {})
    lines = [
        "# Отчёт о качестве распознавания и извлечения данных",
        "",
        f"Дата формирования: **{report.get('generated_at', '—')}**",
        f"Документов в выборке: **{report.get('documents', 0)}**",
        "",
        "## Ключевые метрики",
        "",
        "| Метрика | Значение | Цель (ТЗ) |",
        "|---|---|---|",
        f"| CER (Character Error Rate) | {metrics.get('cer', 0) * 100:.2f} % | ≤ 3 % |",
        f"| WER (Word Error Rate) | {metrics.get('wer', 0) * 100:.2f} % | ≤ 8 % |",
        f"| NER F1 (микро) | {metrics.get('ner_f1', 0) * 100:.2f} % | ≥ 85 % |",
        f"| NER Precision | {metrics.get('ner_precision', 0) * 100:.2f} % | — |",
        f"| NER Recall | {metrics.get('ner_recall', 0) * 100:.2f} % | — |",
        f"| Точность классификации | {metrics.get('doc_type_accuracy', 0) * 100:.2f} % | ≥ 90 % |",
        f"| Среднее время обработки документа | {metrics.get('avg_duration_sec', 0):.2f} с | — |",
        "",
        "## Среда выполнения",
        "",
        f"- OCR-движок: `{env.get('ocr_engine', '—')}`",
        f"- LLM-движок: `{env.get('llm_engine', '—')}` (модель: `{env.get('llm_model', '—')}`)",
        f"- Эмбеддинги: `{env.get('embedding_engine', '—')}`",
        "",
        "## Метрики по полям сущностей",
        "",
        "| Поле | Precision | Recall | F1 | TP | FP | FN |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, data in report.get("entities", {}).get("per_field", {}).items():
        lines.append(
            f"| {name} | {data['precision'] * 100:.1f} % | {data['recall'] * 100:.1f} % | "
            f"{data['f1'] * 100:.1f} % | {data['tp']} | {data['fp']} | {data['fn']} |"
        )

    if report.get("skipped"):
        lines += [
            "",
            "## Пропущенные документы",
            "",
            "| Документ | Причина |",
            "|---|---|",
        ]
        for item in report["skipped"]:
            lines.append(f"| {item.get('filename')} | {item.get('reason')} |")

    lines += [
        "",
        "## Документы",
        "",
        "| Документ | Страниц | CER | WER | Тип распознан | Время, с |",
        "|---|---|---|---|---|---|",
    ]
    for item in report.get("per_document", []):
        lines.append(
            f"| {item['filename']} | {item.get('pages', 0)} | {item['cer'] * 100:.2f} % | "
            f"{item['wer'] * 100:.2f} % | {'да' if item.get('doc_type_correct') else 'нет'} | "
            f"{item.get('duration_sec', 0):.2f} |"
        )
    lines.append("")
    return "\n".join(lines)


def save_report(report: dict[str, Any], out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "quality.json"
    md_path = out_dir / "quality.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(report_to_markdown(report), encoding="utf-8")
    return json_path, md_path


def load_report(report_path: Path) -> dict[str, Any] | None:
    if not report_path.exists():
        return None
    try:
        return json.loads(report_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
