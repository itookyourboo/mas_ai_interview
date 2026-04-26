"""
Калибровка single-model (без RAG, без multi-agent) по шкале 1..5.

Сценарий:
1) Использует тот же калибровочный датасет, что scripts/calibrate_agents.py.
2) Для каждого кейса вызывает одну "сырую" модель-оценщик.
3) Считает метрики MAE / exact match / confusion matrix и скорость.

Пример:
    uv run python scripts/calibrate_single_model.py \
      --models "qwen2.5:14b-instruct,deepseek-r1:latest" \
      --max-concurrency 1 \
      --case-timeout-sec 300
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any

from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import settings as _settings  # noqa: E402
from src import debug_utils as _debug_utils  # noqa: E402
from src import llm_provider as _llm_provider  # noqa: E402
from src import parse_hh as _parse_hh  # noqa: E402
from src.rag import retriever as _rag_retriever  # noqa: E402

# Совместимость с импортами вида "import settings" внутри src-модулей.
sys.modules.setdefault("settings", _settings)
sys.modules.setdefault("debug_utils", _debug_utils)
sys.modules.setdefault("llm_provider", _llm_provider)
sys.modules.setdefault("parse_hh", _parse_hh)
sys.modules.setdefault("rag.retriever", _rag_retriever)

from src import settings  # noqa: E402
from src.agents import AGENT_SCORING_POLICY, SCORE_RUBRIC_1_TO_5  # noqa: E402
from scripts.calibrate_agents import (  # noqa: E402
    _build_confusion_matrix,
    _build_dataset,
    _clamp_1_5,
    _parse_scores_filter,
)


class SingleJudgeOutput(BaseModel):
    score: int = Field(ge=1, le=5)
    feedback: str
    follow_ups_answered: int = Field(ge=0)


JUDGE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """Вы — эксперт по техническим собеседованиям.
Оценивайте ответ по целочисленной шкале 1..5 и верните строго JSON.

{rubric}

{policy}

Учитывайте:
1) корректность ответа на основной вопрос;
2) полноту раскрытия;
3) корректность ответов на follow-up вопросы;
4) глубину и примеры.

ЖЕСТКОЕ ПРАВИЛО:
- 5 ставьте только если корректно покрыты основной вопрос и ВСЕ follow-up вопросы.
- Если follow-up не покрыты полностью, максимум 4.
""",
        ),
        (
            "human",
            """Вопрос:
{question}

Тип вопроса: {question_type}
Уровень кандидата: {candidate_level}
Стек: {tech_stack}
Теги: {tags}

Follow-up вопросы:
{follow_ups}

Ответ кандидата:
{answer}

Верните только JSON:
{{
  "score": <1..5>,
  "feedback": "<кратко, 1-3 предложения>",
  "follow_ups_answered": <сколько follow-up покрыто корректно>
}}
""",
        ),
    ]
)


def _format_followups(items: list[str]) -> str:
    if not items:
        return "Нет follow-up вопросов."
    return "\n".join(f"- {x}" for x in items)


def _parse_models(raw: str) -> list[str]:
    return [m.strip() for m in raw.split(",") if m.strip()]


async def run_single_model_calibration(
    *,
    model_name: str,
    case_timeout_sec: float,
    max_concurrency: int,
    scores_filter: list[int],
) -> dict[str, Any]:
    dataset = _build_dataset()
    selected_scores = sorted(set(scores_filter or [1, 2, 3, 4, 5]))
    total_cases = len(dataset) * len(selected_scores)

    llm = ChatOllama(
        model=model_name,
        base_url=settings.OLLAMA_BASE_URL,
        temperature=0.0,
    )
    judge_chain = JUDGE_PROMPT | llm.with_structured_output(SingleJudgeOutput)
    sem = asyncio.Semaphore(max(1, int(max_concurrency)))

    rows: list[dict[str, Any]] = []
    latencies: list[float] = []
    case_entries: list[tuple[int, int, Any, int, str]] = []
    cid = 0
    for q_idx, case in enumerate(dataset, start=1):
        for expected in selected_scores:
            cid += 1
            case_entries.append((cid, q_idx, case, expected, case.answers_by_score[expected]))

    async def run_case(case_id: int, q_idx: int, case: Any, expected: int, answer: str) -> dict[str, Any]:
        async with sem:
            started = time.perf_counter()
            error_text = ""
            predicted = 1.0
            feedback = ""
            followups_answered = 0
            try:
                result = await asyncio.wait_for(
                    judge_chain.ainvoke(
                        {
                            "rubric": SCORE_RUBRIC_1_TO_5,
                            "policy": AGENT_SCORING_POLICY,
                            "question": case.question,
                            "question_type": case.question_type,
                            "candidate_level": case.candidate_level,
                            "tech_stack": case.tech_stack,
                            "tags": ", ".join(case.tags),
                            "follow_ups": _format_followups(case.follow_ups),
                            "answer": answer,
                        }
                    ),
                    timeout=case_timeout_sec,
                )
                payload = result.model_dump() if isinstance(result, BaseModel) else dict(result)
                predicted = float(payload.get("score", 1.0))
                feedback = str(payload.get("feedback", ""))
                followups_answered = int(payload.get("follow_ups_answered", 0) or 0)
            except asyncio.TimeoutError:
                error_text = f"timeout>{case_timeout_sec}s"
            except Exception as e:  # noqa: BLE001
                error_text = str(e)

            elapsed = time.perf_counter() - started
            latencies.append(elapsed)
            print(
                f"[{case_id}/{total_cases}] Q{q_idx} exp={expected} pred={predicted:.2f} "
                f"rounded={_clamp_1_5(predicted)} t={elapsed:.2f}s"
                + (f" error={error_text}" if error_text else ""),
                flush=True,
            )
            return {
                "case_id": case_id,
                "question_index": q_idx,
                "question": case.question,
                "question_type": case.question_type,
                "candidate_level": case.candidate_level,
                "tech_stack": case.tech_stack,
                "tags": list(case.tags),
                "follow_ups": list(case.follow_ups),
                "expected_score": expected,
                "answer": answer,
                "predicted_score": round(predicted, 3),
                "predicted_rounded": _clamp_1_5(predicted),
                "follow_ups_answered": followups_answered,
                "feedback": feedback,
                "run_error": error_text,
                "latency_sec": round(elapsed, 3),
            }

    tasks = [asyncio.create_task(run_case(*entry)) for entry in case_entries]
    rows = await asyncio.gather(*tasks)
    rows.sort(key=lambda r: int(r["case_id"]))

    mae = mean(abs(float(r["expected_score"]) - float(r["predicted_score"])) for r in rows)
    exact = sum(1 for r in rows if int(r["expected_score"]) == int(r["predicted_rounded"])) / len(rows)
    within_1 = (
        sum(1 for r in rows if abs(int(r["expected_score"]) - int(r["predicted_rounded"])) <= 1) / len(rows)
    )

    per_expected: dict[int, dict[str, float | None]] = {}
    for score in selected_scores:
        subset = [r for r in rows if int(r["expected_score"]) == score]
        if not subset:
            per_expected[score] = {
                "count": 0.0,
                "avg_predicted": None,
                "avg_rounded": None,
                "exact_match_rate": None,
            }
            continue
        per_expected[score] = {
            "count": float(len(subset)),
            "avg_predicted": round(mean(float(r["predicted_score"]) for r in subset), 4),
            "avg_rounded": round(mean(float(r["predicted_rounded"]) for r in subset), 4),
            "exact_match_rate": round(
                sum(1 for r in subset if int(r["predicted_rounded"]) == score) / len(subset),
                4,
            ),
        }

    lat_sorted = sorted(latencies)
    p50 = lat_sorted[len(lat_sorted) // 2] if lat_sorted else 0.0
    p95 = lat_sorted[int(0.95 * (len(lat_sorted) - 1))] if lat_sorted else 0.0

    failed = [r for r in rows if int(r["expected_score"]) != int(r["predicted_rounded"])]
    return {
        "model": model_name,
        "created_at": datetime.now().isoformat(),
        "dataset_summary": {
            "questions": len(dataset),
            "answers_per_question": len(selected_scores),
            "selected_scores": selected_scores,
            "total_cases": len(rows),
            "max_concurrency": max(1, int(max_concurrency)),
            "case_timeout_sec": case_timeout_sec,
        },
        "overall_metrics": {
            "mae_total_score": round(mae, 4),
            "exact_match_rate_rounded": round(exact, 4),
            "within_1_rate_rounded": round(within_1, 4),
        },
        "speed_metrics": {
            "avg_latency_sec": round(mean(latencies), 4) if latencies else None,
            "p50_latency_sec": round(p50, 4),
            "p95_latency_sec": round(p95, 4),
        },
        "per_expected_score_metrics": per_expected,
        "confusion_matrix_rounded": _build_confusion_matrix(rows),
        "failed_cases_count": len(failed),
        "rows": rows,
    }


async def main() -> None:
    parser = argparse.ArgumentParser(description="Single-model calibration (no RAG/no multi-agent)")
    parser.add_argument(
        "--models",
        default="qwen2.5:14b-instruct,deepseek-r1:latest",
        help="Comma-separated Ollama model names",
    )
    parser.add_argument("--case-timeout-sec", type=float, default=300.0)
    parser.add_argument("--max-concurrency", type=int, default=1)
    parser.add_argument("--scores", default="1,2,3,4,5")
    args = parser.parse_args()

    models = _parse_models(args.models)
    selected_scores = _parse_scores_filter(args.scores)
    out_dir = PROJECT_ROOT / "data" / "calibration"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    all_reports: dict[str, Any] = {"created_at": datetime.now().isoformat(), "reports": []}
    for model_name in models:
        print(
            f"\n=== Running single-model calibration for {model_name} "
            f"(scores={selected_scores}, timeout={args.case_timeout_sec}s) ===",
            flush=True,
        )
        report = await run_single_model_calibration(
            model_name=model_name,
            case_timeout_sec=args.case_timeout_sec,
            max_concurrency=args.max_concurrency,
            scores_filter=selected_scores,
        )
        all_reports["reports"].append(report)

        model_slug = model_name.replace(":", "_").replace("/", "_")
        model_path = out_dir / f"single_model_calibration_{model_slug}_{ts}.json"
        model_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(
            f"Model={model_name}: MAE={report['overall_metrics']['mae_total_score']:.3f}, "
            f"Exact={report['overall_metrics']['exact_match_rate_rounded']:.1%}, "
            f"Within1={report['overall_metrics']['within_1_rate_rounded']:.1%}, "
            f"p50={report['speed_metrics']['p50_latency_sec']:.2f}s, "
            f"p95={report['speed_metrics']['p95_latency_sec']:.2f}s",
            flush=True,
        )
        print(f"Saved report: {model_path}", flush=True)

    combined_path = out_dir / f"single_model_calibration_compare_{ts}.json"
    combined_path.write_text(json.dumps(all_reports, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved combined report: {combined_path}")


if __name__ == "__main__":
    # Позволяет запуск через `uv run python ...` с PYTHONUNBUFFERED
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    asyncio.run(main())

