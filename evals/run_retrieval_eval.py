"""Run a small golden-set retrieval eval against the indexed Chroma collection."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any

from app.schemas.documents import RetrievedChunk
from app.services.retrieval.service import get_retrieval_service

DEFAULT_DATASET = Path(__file__).with_name("retrieval_golden.jsonl")


@dataclass(frozen=True)
class ExpectedSource:
    document: str
    pages: set[int]


@dataclass(frozen=True)
class GoldenCase:
    id: str
    question: str
    expected_sources: list[ExpectedSource]
    rationale: str


@dataclass(frozen=True)
class CaseResult:
    id: str
    question: str
    document_recall: float
    page_recall: float | None
    reciprocal_rank: float
    passed: bool
    hits: list[str]


@dataclass(frozen=True)
class EvalSummary:
    cases: int
    document_recall: float
    page_recall: float | None
    mrr: float
    exact_source_pass_rate: float


def load_cases(path: Path = DEFAULT_DATASET) -> list[GoldenCase]:
    cases: list[GoldenCase] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        raw = json.loads(line)
        cases.append(_parse_case(raw, line_number))
    return cases


def evaluate_case(case: GoldenCase, chunks: list[RetrievedChunk]) -> CaseResult:
    doc_hits = [
        expected
        for expected in case.expected_sources
        if any(chunk.document == expected.document for chunk in chunks)
    ]
    page_expected = [expected for expected in case.expected_sources if expected.pages]
    page_hits = [
        expected
        for expected in page_expected
        if any(
            chunk.document == expected.document and chunk.page in expected.pages for chunk in chunks
        )
    ]

    first_rank = _first_expected_rank(case, chunks)
    page_recall = len(page_hits) / len(page_expected) if page_expected else None
    passed = len(doc_hits) == len(case.expected_sources) and (
        page_recall is None or page_recall == 1.0
    )
    return CaseResult(
        id=case.id,
        question=case.question,
        document_recall=len(doc_hits) / len(case.expected_sources),
        page_recall=page_recall,
        reciprocal_rank=1 / first_rank if first_rank else 0.0,
        passed=passed,
        hits=[f"{chunk.document}:p{chunk.page}" for chunk in chunks],
    )


def summarize(results: list[CaseResult]) -> EvalSummary:
    page_results = [result.page_recall for result in results if result.page_recall is not None]
    return EvalSummary(
        cases=len(results),
        document_recall=mean(result.document_recall for result in results),
        page_recall=mean(page_results) if page_results else None,
        mrr=mean(result.reciprocal_rank for result in results),
        exact_source_pass_rate=mean(1.0 if result.passed else 0.0 for result in results),
    )


def run_eval(cases: list[GoldenCase], top_k: int) -> list[CaseResult]:
    retrieval = get_retrieval_service()
    return [evaluate_case(case, retrieval.retrieve(case.question, top_k=top_k)) for case in cases]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run golden retrieval evals.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    cases = load_cases(args.dataset)
    results = run_eval(cases, args.top_k)
    summary = summarize(results)

    for result in results:
        status = "PASS" if result.passed else "FAIL"
        page_recall = "n/a" if result.page_recall is None else f"{result.page_recall:.2f}"
        print(
            f"{status} {result.id}: "
            f"doc_recall={result.document_recall:.2f} "
            f"page_recall={page_recall} "
            f"rr={result.reciprocal_rank:.2f} "
            f"hits={', '.join(result.hits)}"
        )

    summary_page_recall = "n/a" if summary.page_recall is None else f"{summary.page_recall:.2f}"
    print("\nSummary")
    print(f"cases={summary.cases}")
    print(f"document_recall@{args.top_k}={summary.document_recall:.2f}")
    print(f"page_recall@{args.top_k}={summary_page_recall}")
    print(f"mrr@{args.top_k}={summary.mrr:.2f}")
    print(f"exact_source_pass_rate@{args.top_k}={summary.exact_source_pass_rate:.2f}")


def _parse_case(raw: dict[str, Any], line_number: int) -> GoldenCase:
    expected_sources = [
        ExpectedSource(
            document=str(source["document"]),
            pages={int(page) for page in source.get("pages", [])},
        )
        for source in raw.get("expected_sources", [])
    ]
    if not expected_sources:
        raise ValueError(f"Eval case on line {line_number} has no expected_sources")
    return GoldenCase(
        id=str(raw["id"]),
        question=str(raw["question"]),
        expected_sources=expected_sources,
        rationale=str(raw.get("rationale", "")),
    )


def _first_expected_rank(case: GoldenCase, chunks: list[RetrievedChunk]) -> int | None:
    for rank, chunk in enumerate(chunks, start=1):
        if any(chunk.document == expected.document for expected in case.expected_sources):
            return rank
    return None


if __name__ == "__main__":
    main()
