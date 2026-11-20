#!/usr/bin/env python3
"""Score ranked VQA answer-generation runs against single and multiple references.

The scorer evaluates every eligible ground-truth query, so missing queries/ranks
cannot improve a run by reducing the denominator.  Fifteen known broken-video
query/video pairs are excluded before all effectiveness and timing calculations.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import statistics
import sys
from collections import Counter
from dataclasses import dataclass, field
from importlib import metadata
from pathlib import Path
from typing import Any, Iterable, Sequence


SCORER_VERSION = "1.1.0"
CUTOFFS = (1, 5, 10)
METRICS = ("sts", "meteor", "bertscore_f1")
METRIC_LABELS = {"sts": "STS", "meteor": "METEOR", "bertscore_f1": "BERTScore F1"}
REFERENCE_MODES = ("correct_only", "all_references")

EXCLUDED_QUERY_VIDEO_PAIRS = frozenset(
    {
        (19, "9EyY7TK0mCA"),
        (265, "_3mWTYMXSok"),
        (470, "2ZefngWxXoE"),
        (608, "I4BgRvGNpjM"),
        (950, "UZDZb9Cijdo"),
        (959, "APlvy8fawLM"),
        (1242, "6f7tV70a_y4"),
        (1381, "7cIqwelU2o4"),
        (1407, "6T30Pikgiw4"),
        (1505, "Axv5y-i_ieA"),
        (1632, "bujAtvNcGiw"),
        (2044, "BHvt4VcAIBQ"),
        (2099, "8rzskun9Qyc"),
        (2156, "LjGrmx69MQU"),
        (2273, "8EX5QO6NDq0"),
    }
)


class InputError(ValueError):
    """Raised when an input file cannot be interpreted safely."""


@dataclass(frozen=True)
class GroundTruthItem:
    qid: int
    video_id: str
    question: str
    correct_answer: str
    alternatives: tuple[str, ...]

    @property
    def all_references(self) -> tuple[str, ...]:
        # Deduplicate exact repeated references without changing their order.
        return tuple(dict.fromkeys((self.correct_answer, *self.alternatives)))


@dataclass(frozen=True)
class SubmittedAnswer:
    rank: int
    answer: str
    time: float | None


@dataclass
class RunData:
    run_name: str
    run_file: Path
    answers_by_qid: dict[int, dict[int, SubmittedAnswer]] = field(default_factory=dict)
    video_by_qid: dict[int, str] = field(default_factory=dict)
    issues: list[dict[str, Any]] = field(default_factory=list)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score ranked VQA answer-generation runs with STS, METEOR, BERTScore, and NDCG."
    )
    parser.add_argument("--ground-truth", required=True, type=Path, help="Ground-truth JSON file")
    parser.add_argument("--runs-dir", required=True, type=Path, help="Folder containing one subfolder per run")
    parser.add_argument("--output-dir", required=True, type=Path, help="Directory for CSV/JSON reports")
    parser.add_argument(
        "--run-file-name",
        default=None,
        help="Exact filename inside each run directory; otherwise exactly one non-hidden file is required",
    )
    parser.add_argument("--sts-model", default="sentence-transformers/all-mpnet-base-v2")
    parser.add_argument("--bertscore-model", default="roberta-large")
    parser.add_argument("--device", default=None, help="cpu, cuda, cuda:0, mps, etc.; default is library-selected")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-rank", type=int, default=10)
    parser.add_argument(
        "--allow-incomplete-runs",
        action="store_true",
        help="Score incomplete/malformed runs with missing entries treated as zero (default: also score, but exit nonzero)",
    )
    parser.add_argument(
        "--download-nltk-data",
        action="store_true",
        help="Download the NLTK wordnet resource if it is unavailable",
    )
    return parser.parse_args(argv)


def _load_json_array(path: Path) -> list[Any]:
    try:
        with path.open("r", encoding="utf-8-sig") as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise InputError(f"Cannot read valid JSON from {path}: {exc}") from exc
    if not isinstance(data, list):
        raise InputError(f"Top-level JSON value in {path} must be an array")
    return data


def _required_nonempty_string(value: Any, field_name: str, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise InputError(f"{context}: {field_name} must be a nonempty string")
    return value.strip()


def _parse_qid(value: Any, context: str) -> int:
    if isinstance(value, bool):
        raise InputError(f"{context}: Q_ID must be an integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise InputError(f"{context}: Q_ID must be an integer") from exc
    if isinstance(value, float) and not value.is_integer():
        raise InputError(f"{context}: Q_ID must be an integer")
    return parsed


def load_ground_truth(path: Path) -> tuple[list[GroundTruthItem], list[GroundTruthItem]]:
    raw = _load_json_array(path)
    seen: set[int] = set()
    eligible: list[GroundTruthItem] = []
    excluded: list[GroundTruthItem] = []
    for index, obj in enumerate(raw):
        context = f"{path}, item {index}"
        if not isinstance(obj, dict):
            raise InputError(f"{context}: item must be an object")
        qid = _parse_qid(obj.get("Q_ID"), context)
        if qid in seen:
            raise InputError(f"{context}: duplicate Q_ID {qid} in ground truth")
        seen.add(qid)
        video_id = _required_nonempty_string(obj.get("Video_ID"), "Video_ID", context)
        correct = _required_nonempty_string(obj.get("Correct_Answer"), "Correct_Answer", context)
        question_value = obj.get("Question", "")
        if question_value is None:
            question_value = ""
        if not isinstance(question_value, str):
            raise InputError(f"{context}: Question must be a string")
        alternatives_value = obj.get("Alternative_Correct_Answers", [])
        if alternatives_value is None:
            alternatives_value = []
        if not isinstance(alternatives_value, list):
            raise InputError(f"{context}: Alternative_Correct_Answers must be an array")
        alternatives: list[str] = []
        for alt_index, alternative in enumerate(alternatives_value):
            alternatives.append(
                _required_nonempty_string(
                    alternative, f"Alternative_Correct_Answers[{alt_index}]", context
                )
            )
        item = GroundTruthItem(qid, video_id, question_value.strip(), correct, tuple(alternatives))
        (excluded if (qid, video_id) in EXCLUDED_QUERY_VIDEO_PAIRS else eligible).append(item)
    if not eligible:
        raise InputError("Ground truth contains no eligible queries after exclusions")
    return eligible, excluded


def discover_run_files(runs_dir: Path, run_file_name: str | None) -> list[tuple[str, Path]]:
    if not runs_dir.is_dir():
        raise InputError(f"Runs directory does not exist: {runs_dir}")
    discovered: list[tuple[str, Path]] = []
    for child in sorted(runs_dir.iterdir(), key=lambda p: p.name):
        if not child.is_dir():
            continue
        if run_file_name:
            candidate = child / run_file_name
            if candidate.is_file():
                discovered.append((child.name, candidate))
            else:
                raise InputError(f"Expected run file does not exist: {candidate}")
            continue
        # Submission filenames need not have a .json extension. Ignore hidden
        # administrative files, but require an unambiguous single regular file.
        candidates = sorted(
            candidate
            for candidate in child.iterdir()
            if candidate.is_file() and not candidate.name.startswith(".")
        )
        if len(candidates) == 1:
            discovered.append((child.name, candidates[0]))
        elif len(candidates) > 1:
            raise InputError(
                f"Run directory {child} has multiple candidate files; specify --run-file-name"
            )
        else:
            raise InputError(f"Run directory contains no candidate file: {child}")
    if not discovered:
        raise InputError(f"No run files found beneath {runs_dir}")
    return discovered


def _issue(run: RunData, severity: str, code: str, qid: int | None, detail: str) -> None:
    run.issues.append(
        {"run_name": run.run_name, "severity": severity, "code": code, "Q_ID": qid, "detail": detail}
    )


def load_run(run_name: str, path: Path, gt_by_qid: dict[int, GroundTruthItem], max_rank: int) -> RunData:
    run = RunData(run_name=run_name, run_file=path)
    try:
        raw = _load_json_array(path)
    except InputError as exc:
        _issue(run, "error", "invalid_json", None, str(exc))
        return run

    seen_qids: set[int] = set()
    for index, obj in enumerate(raw):
        context = f"item {index}"
        if not isinstance(obj, dict):
            _issue(run, "error", "invalid_query_object", None, f"{context} is not an object")
            continue
        try:
            qid = _parse_qid(obj.get("Q_ID"), context)
        except InputError as exc:
            _issue(run, "error", "invalid_qid", None, str(exc))
            continue
        if qid in seen_qids:
            _issue(run, "error", "duplicate_query", qid, "Duplicate query entry; later entry ignored")
            continue
        seen_qids.add(qid)
        video = obj.get("Video_ID")
        if not isinstance(video, str) or not video.strip():
            _issue(run, "error", "invalid_video_id", qid, "Video_ID must be a nonempty string")
            continue
        video = video.strip()

        # Known broken queries are ignored completely, including their times.
        if (qid, video) in EXCLUDED_QUERY_VIDEO_PAIRS:
            continue
        gt_item = gt_by_qid.get(qid)
        if gt_item is None:
            _issue(run, "warning", "extra_query", qid, "Q_ID is not an eligible ground-truth query; ignored")
            continue
        if video != gt_item.video_id:
            _issue(
                run,
                "error",
                "video_id_mismatch",
                qid,
                f"Submitted {video!r}; expected {gt_item.video_id!r}. Query treated as missing",
            )
            continue

        answers_value = obj.get("Answers")
        if not isinstance(answers_value, list):
            _issue(run, "error", "invalid_answers", qid, "Answers must be an array")
            continue
        rank_map: dict[int, SubmittedAnswer] = {}
        for answer_index, answer_obj in enumerate(answers_value):
            if not isinstance(answer_obj, dict):
                _issue(run, "error", "invalid_answer_object", qid, f"Answer item {answer_index} is not an object")
                continue
            rank_value = answer_obj.get("Rank")
            if isinstance(rank_value, bool):
                rank = -1
            else:
                try:
                    rank = int(rank_value)
                except (TypeError, ValueError):
                    rank = -1
            if rank < 1 or rank > max_rank or (isinstance(rank_value, float) and not rank_value.is_integer()):
                _issue(run, "error", "invalid_rank", qid, f"Invalid Rank {rank_value!r}; expected 1..{max_rank}")
                continue
            if rank in rank_map:
                _issue(run, "error", "duplicate_rank", qid, f"Duplicate Rank {rank}; later answer ignored")
                continue
            text_value = answer_obj.get("Answer")
            if not isinstance(text_value, str) or not text_value.strip():
                _issue(run, "error", "invalid_answer_text", qid, f"Rank {rank} Answer must be a nonempty string")
                continue
            time_value = answer_obj.get("Time")
            parsed_time: float | None
            if time_value is None:
                parsed_time = None
                _issue(run, "warning", "missing_time", qid, f"Rank {rank} has no Time")
            elif isinstance(time_value, bool):
                parsed_time = None
                _issue(run, "error", "invalid_time", qid, f"Rank {rank} Time must be a finite nonnegative number")
            else:
                try:
                    parsed_time = float(time_value)
                except (TypeError, ValueError):
                    parsed_time = None
                if parsed_time is None or not math.isfinite(parsed_time) or parsed_time < 0:
                    parsed_time = None
                    _issue(run, "error", "invalid_time", qid, f"Rank {rank} Time must be a finite nonnegative number")
            rank_map[rank] = SubmittedAnswer(rank=rank, answer=text_value.strip(), time=parsed_time)

        missing_ranks = sorted(set(range(1, max_rank + 1)) - set(rank_map))
        if missing_ranks:
            _issue(run, "error", "missing_ranks", qid, f"Missing ranks: {missing_ranks}; scored as zero")
        run.answers_by_qid[qid] = rank_map
        run.video_by_qid[qid] = video

    for qid in gt_by_qid:
        if qid not in run.answers_by_qid:
            _issue(run, "error", "missing_query", qid, "Eligible query is absent or invalid; scored as zero")
    return run


class QualityScorer:
    """Batched STS/BERTScore plus deterministic NLTK METEOR."""

    def __init__(
        self,
        sts_model: str,
        bertscore_model: str,
        device: str | None,
        batch_size: int,
        download_nltk_data: bool,
    ) -> None:
        try:
            import nltk
            from bert_score import BERTScorer
            from nltk.corpus import wordnet
            from nltk.translate.meteor_score import meteor_score
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "Missing scoring dependency. Install packages from requirements_vqa_answer_scorer.txt"
            ) from exc

        try:
            wordnet.ensure_loaded()
        except LookupError:
            if not download_nltk_data:
                raise RuntimeError(
                    "NLTK WordNet data is missing. Run with --download-nltk-data once, "
                    "or run: python -m nltk.downloader wordnet omw-1.4"
                )
            if not nltk.download("wordnet", quiet=False) or not nltk.download("omw-1.4", quiet=False):
                raise RuntimeError("NLTK could not download wordnet/omw-1.4")
            wordnet.ensure_loaded()

        self.sts_model_name = sts_model
        self.bertscore_model_name = bertscore_model
        self.batch_size = batch_size
        self._meteor_score = meteor_score
        self._wordnet = wordnet
        self._sts = SentenceTransformer(sts_model, device=device)
        bert_kwargs: dict[str, Any] = {
            "model_type": bertscore_model,
            "lang": "en",
            "rescale_with_baseline": False,
            "batch_size": batch_size,
        }
        if device:
            bert_kwargs["device"] = device
        self._bert = BERTScorer(**bert_kwargs)

    @property
    def bertscore_hash(self) -> str | None:
        """Return BERTScore's reproducibility hash when exposed by the package."""
        value = getattr(self._bert, "hash", None)
        return str(value) if value is not None else None

    @staticmethod
    def _tokens(text: str) -> list[str]:
        # wordpunct-style tokenization avoids a hidden dependency on Punkt data.
        import re

        return re.findall(r"\w+|[^\w\s]", text.lower(), flags=re.UNICODE)

    def score_pairs(self, candidates: Sequence[str], references: Sequence[str]) -> dict[str, list[float]]:
        if len(candidates) != len(references):
            raise ValueError("Candidate/reference pair counts differ")
        if not candidates:
            return {metric: [] for metric in METRICS}

        import numpy as np

        unique_texts = list(dict.fromkeys((*candidates, *references)))
        embeddings = self._sts.encode(
            unique_texts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True,
        )
        embedding_by_text = dict(zip(unique_texts, embeddings))
        sts_values = [
            float(np.dot(embedding_by_text[candidate], embedding_by_text[reference]))
            for candidate, reference in zip(candidates, references)
        ]
        meteor_values = [
            float(
                self._meteor_score(
                    [self._tokens(reference)], self._tokens(candidate), wordnet=self._wordnet
                )
            )
            for candidate, reference in zip(candidates, references)
        ]
        _precision, _recall, f1 = self._bert.score(list(candidates), list(references), batch_size=self.batch_size)
        bert_values = [float(value) for value in f1.detach().cpu().tolist()]
        return {"sts": sts_values, "meteor": meteor_values, "bertscore_f1": bert_values}


def dcg(relevances: Sequence[float], cutoff: int) -> float:
    """Linear-gain discounted cumulative gain, with missing ranks padded by zero."""
    return sum(max(0.0, float(rel)) / math.log2(rank + 1) for rank, rel in enumerate(relevances[:cutoff], 1))


def ndcg(relevances: Sequence[float], cutoff: int) -> float:
    """NDCG@k; IDCG sorts all submitted relevance values before truncating to k."""
    actual = list(relevances)
    ideal = sorted((max(0.0, float(value)) for value in actual), reverse=True)
    ideal_dcg = dcg(ideal, cutoff)
    return 0.0 if ideal_dcg == 0.0 else dcg(actual, cutoff) / ideal_dcg


def _mean(values: Iterable[float]) -> float:
    vals = list(values)
    return statistics.fmean(vals) if vals else 0.0


def percentile(values: Sequence[float], percentile_value: float) -> float | None:
    """NumPy-compatible linear percentile without requiring NumPy for reports."""
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * percentile_value / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def build_pair_work(
    runs: Sequence[RunData], gt_items: Sequence[GroundTruthItem]
) -> tuple[list[str], list[str], list[tuple[str, int, int, int]]]:
    """Create each candidate/reference pair once for reuse by both reference modes."""
    candidates: list[str] = []
    references: list[str] = []
    keys: list[tuple[str, int, int, int]] = []
    for run in runs:
        for gt in gt_items:
            for rank, submitted in sorted(run.answers_by_qid.get(gt.qid, {}).items()):
                for reference_index, reference in enumerate(gt.all_references):
                    candidates.append(submitted.answer)
                    references.append(reference)
                    keys.append((run.run_name, gt.qid, rank, reference_index))
    return candidates, references, keys


def aggregate_pair_scores(
    keys: Sequence[tuple[str, int, int, int]], pair_scores: dict[str, list[float]]
) -> dict[tuple[str, int, int], dict[str, dict[str, float]]]:
    result: dict[tuple[str, int, int], dict[str, dict[str, float]]] = {}
    for pair_index, (run_name, qid, rank, reference_index) in enumerate(keys):
        answer_key = (run_name, qid, rank)
        modes = result.setdefault(answer_key, {mode: {} for mode in REFERENCE_MODES})
        for metric in METRICS:
            value = pair_scores[metric][pair_index]
            if reference_index == 0:
                modes["correct_only"][metric] = value
            current = modes["all_references"].get(metric)
            if current is None or value > current:
                modes["all_references"][metric] = value
    return result


def calculate_reports(
    runs: Sequence[RunData],
    gt_items: Sequence[GroundTruthItem],
    answer_scores: dict[tuple[str, int, int], dict[str, dict[str, float]]],
    max_rank: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    per_query: list[dict[str, Any]] = []
    summary: list[dict[str, Any]] = []
    for run in runs:
        for mode in REFERENCE_MODES:
            query_rows: list[dict[str, Any]] = []
            for gt in gt_items:
                rank_map = run.answers_by_qid.get(gt.qid, {})
                row: dict[str, Any] = {
                    "run_name": run.run_name,
                    "reference_mode": mode,
                    "Q_ID": gt.qid,
                    "Video_ID": gt.video_id,
                    "query_submitted": int(gt.qid in run.answers_by_qid),
                    "valid_answer_count": len(rank_map),
                }
                for rank in range(1, max_rank + 1):
                    submitted = rank_map.get(rank)
                    row[f"answer_r{rank}"] = submitted.answer if submitted else ""
                    for metric in METRICS:
                        row[f"{metric}_r{rank}"] = (
                            answer_scores.get((run.run_name, gt.qid, rank), {})
                            .get(mode, {})
                            .get(metric, 0.0)
                        )
                for metric in METRICS:
                    relevances = [float(row[f"{metric}_r{rank}"]) for rank in range(1, max_rank + 1)]
                    for cutoff in CUTOFFS:
                        effective_cutoff = min(cutoff, max_rank)
                        prefix = relevances[:effective_cutoff]
                        row[f"{metric}@{cutoff}"] = relevances[0] if cutoff == 1 else max(prefix, default=0.0)
                        row[f"mean_{metric}@{cutoff}"] = _mean(prefix)
                        row[f"{metric}_ndcg@{cutoff}"] = ndcg(relevances, effective_cutoff)
                query_rows.append(row)
                per_query.append(row)

            summary_row: dict[str, Any] = {
                "run_name": run.run_name,
                "reference_mode": mode,
                "eligible_query_count": len(gt_items),
                "submitted_query_count": sum(int(row["query_submitted"]) for row in query_rows),
                "missing_query_count": sum(1 - int(row["query_submitted"]) for row in query_rows),
                "complete_query_count": sum(int(row["valid_answer_count"] == max_rank) for row in query_rows),
            }
            for metric in METRICS:
                for cutoff in CUTOFFS:
                    summary_row[f"{metric}@{cutoff}"] = _mean(row[f"{metric}@{cutoff}"] for row in query_rows)
                    summary_row[f"mean_{metric}@{cutoff}"] = _mean(
                        row[f"mean_{metric}@{cutoff}"] for row in query_rows
                    )
                    summary_row[f"{metric}_ndcg@{cutoff}"] = _mean(
                        row[f"{metric}_ndcg@{cutoff}"] for row in query_rows
                    )
            summary.append(summary_row)
    return per_query, summary


def calculate_time_reports(
    runs: Sequence[RunData], gt_items: Sequence[GroundTruthItem], max_rank: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    per_query: list[dict[str, Any]] = []
    summary: list[dict[str, Any]] = []
    eligible_qids = {item.qid for item in gt_items}
    for run in runs:
        for gt in gt_items:
            rank_map = run.answers_by_qid.get(gt.qid, {})
            values = [answer.time for answer in rank_map.values() if answer.time is not None]
            top1 = rank_map.get(1)
            per_query.append(
                {
                    "run_name": run.run_name,
                    "Q_ID": gt.qid,
                    "Video_ID": gt.video_id,
                    "valid_time_count": len(values),
                    "missing_time_count": max_rank - len(values),
                    "top1_time": top1.time if top1 and top1.time is not None else "",
                    "sum_time": sum(values) if values else "",
                    "mean_time": _mean(values) if values else "",
                    "median_time": statistics.median(values) if values else "",
                    "min_time": min(values) if values else "",
                    "max_time": max(values) if values else "",
                }
            )
        all_times = [
            answer.time
            for qid, ranks in run.answers_by_qid.items()
            if qid in eligible_qids
            for answer in ranks.values()
            if answer.time is not None
        ]
        top1_times = [
            ranks[1].time
            for qid, ranks in run.answers_by_qid.items()
            if qid in eligible_qids and 1 in ranks and ranks[1].time is not None
        ]
        summary.append(
            {
                "run_name": run.run_name,
                "eligible_query_count": len(gt_items),
                "expected_answer_time_count": len(gt_items) * max_rank,
                "reported_answer_time_count": len(all_times),
                "missing_answer_time_count": len(gt_items) * max_rank - len(all_times),
                "zero_answer_time_count": sum(value == 0 for value in all_times),
                "sum_answer_time": sum(all_times) if all_times else "",
                "mean_answer_time": _mean(all_times) if all_times else "",
                "median_answer_time": statistics.median(all_times) if all_times else "",
                "min_answer_time": min(all_times) if all_times else "",
                "max_answer_time": max(all_times) if all_times else "",
                "p90_answer_time": percentile(all_times, 90) if all_times else "",
                "p95_answer_time": percentile(all_times, 95) if all_times else "",
                "reported_top1_time_count": len(top1_times),
                "mean_top1_time": _mean(top1_times) if top1_times else "",
                "median_top1_time": statistics.median(top1_times) if top1_times else "",
                "p95_top1_time": percentile(top1_times, 95) if top1_times else "",
            }
        )
    return per_query, summary


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty report: {path}")
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")


def _rounded(value: Any, digits: int = 6) -> Any:
    return round(value, digits) if isinstance(value, float) else value


def build_human_summary(
    summary_rows: Sequence[dict[str, Any]],
    time_rows: Sequence[dict[str, Any]],
    validation_rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Build a compact nested report intended for direct human inspection."""
    times_by_run = {row["run_name"]: row for row in time_rows}
    validation_counts: dict[str, Counter[str]] = {}
    for issue in validation_rows:
        run_name = str(issue["run_name"])
        validation_counts.setdefault(run_name, Counter())[str(issue["severity"])] += 1

    run_order = list(dict.fromkeys(str(row["run_name"]) for row in summary_rows))
    runs_payload: list[dict[str, Any]] = []
    for run_name in run_order:
        mode_rows = {
            str(row["reference_mode"]): row
            for row in summary_rows
            if row["run_name"] == run_name
        }
        representative = next(iter(mode_rows.values()))
        run_payload: dict[str, Any] = {
            "run_name": run_name,
            "coverage": {
                "eligible_queries": representative["eligible_query_count"],
                "submitted_queries": representative["submitted_query_count"],
                "missing_queries": representative["missing_query_count"],
                "queries_with_all_ranks": representative["complete_query_count"],
            },
            "effectiveness": {},
            "timing": {
                key: _rounded(value)
                for key, value in times_by_run.get(run_name, {}).items()
                if key != "run_name"
            },
            "validation": dict(validation_counts.get(run_name, Counter())),
        }
        for mode in REFERENCE_MODES:
            row = mode_rows[mode]
            cutoffs_payload: dict[str, Any] = {}
            for cutoff in CUTOFFS:
                cutoffs_payload[str(cutoff)] = {
                    METRIC_LABELS[metric]: {
                        "best_answer_score": _rounded(row[f"{metric}@{cutoff}"]),
                        "mean_top_k_score": _rounded(row[f"mean_{metric}@{cutoff}"]),
                        "ndcg": _rounded(row[f"{metric}_ndcg@{cutoff}"]),
                    }
                    for metric in METRICS
                }
            run_payload["effectiveness"][mode] = {"cutoffs": cutoffs_payload}
        runs_payload.append(run_payload)
    return {
        "report_description": "Human-readable VQA answer-generation scoring summary",
        "score_scale": "Higher is better for all effectiveness metrics",
        "definitions": {
            "best_answer_score": "At k=1, rank-1 score; at k=5/10, best score among the top k answers",
            "mean_top_k_score": "Mean score across the top k ranks; missing ranks count as zero",
            "ndcg": "Ranking quality within the submitted answers, using metric score as graded relevance",
        },
        "runs": runs_payload,
    }


def build_per_query_json(per_query_rows: Sequence[dict[str, Any]], max_rank: int) -> dict[str, Any]:
    grouped: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for row in per_query_rows:
        run_name = str(row["run_name"])
        mode = str(row["reference_mode"])
        ranks: list[dict[str, Any]] = []
        for rank in range(1, max_rank + 1):
            ranks.append(
                {
                    "rank": rank,
                    "answer": row[f"answer_r{rank}"],
                    "scores": {
                        METRIC_LABELS[metric]: _rounded(row[f"{metric}_r{rank}"])
                        for metric in METRICS
                    },
                }
            )
        cutoffs = {
            str(cutoff): {
                METRIC_LABELS[metric]: {
                    "best_answer_score": _rounded(row[f"{metric}@{cutoff}"]),
                    "mean_top_k_score": _rounded(row[f"mean_{metric}@{cutoff}"]),
                    "ndcg": _rounded(row[f"{metric}_ndcg@{cutoff}"]),
                }
                for metric in METRICS
            }
            for cutoff in CUTOFFS
        }
        grouped.setdefault(run_name, {}).setdefault(mode, []).append(
            {
                "Q_ID": row["Q_ID"],
                "Video_ID": row["Video_ID"],
                "query_submitted": bool(row["query_submitted"]),
                "valid_answer_count": row["valid_answer_count"],
                "ranks": ranks,
                "cutoffs": cutoffs,
            }
        )
    return {
        "report_description": "Per-query VQA answer-generation scores",
        "runs": [
            {
                "run_name": run_name,
                "reference_modes": modes,
            }
            for run_name, modes in grouped.items()
        ],
    }


def build_time_per_query_json(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["run_name"]), []).append(
            {key: _rounded(value) for key, value in row.items() if key != "run_name"}
        )
    return {
        "report_description": "Per-query submitted answer timing statistics",
        "runs": [{"run_name": name, "queries": queries} for name, queries in grouped.items()],
    }


def write_markdown_summary(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# VQA Answer-Generation Scoring Summary",
        "",
        "All effectiveness scores are macro-averaged over every eligible ground-truth query. Higher is better.",
        "Missing queries and ranks count as zero. Broken-video exclusions are omitted before calculation.",
        "",
    ]
    for run in payload["runs"]:
        coverage = run["coverage"]
        lines.extend(
            [
                f"## Run: {run['run_name']}",
                "",
                "### Coverage",
                "",
                "| Eligible | Submitted | Missing | Complete (all ranks) |",
                "| ---: | ---: | ---: | ---: |",
                f"| {coverage['eligible_queries']} | {coverage['submitted_queries']} | "
                f"{coverage['missing_queries']} | {coverage['queries_with_all_ranks']} |",
                "",
            ]
        )
        for mode, mode_payload in run["effectiveness"].items():
            label = "Correct answer only" if mode == "correct_only" else "Correct + alternative answers"
            lines.extend([f"### {label}", ""])
            for cutoff, metrics in mode_payload["cutoffs"].items():
                lines.extend(
                    [
                        f"#### Cutoff: top {cutoff}",
                        "",
                        "| Metric | Best answer | Mean top-k | NDCG |",
                        "| --- | ---: | ---: | ---: |",
                    ]
                )
                for metric_name, values in metrics.items():
                    lines.append(
                        f"| {metric_name} | {values['best_answer_score']:.6f} | "
                        f"{values['mean_top_k_score']:.6f} | {values['ndcg']:.6f} |"
                    )
                lines.append("")

        timing = run["timing"]
        lines.extend(["### Timing", "", "| Statistic | Value |", "| --- | ---: |"]) 
        for key in (
            "reported_answer_time_count",
            "missing_answer_time_count",
            "sum_answer_time",
            "mean_answer_time",
            "median_answer_time",
            "p90_answer_time",
            "p95_answer_time",
            "mean_top1_time",
            "median_top1_time",
            "p95_top1_time",
        ):
            value = timing.get(key, "")
            display = f"{value:.6f}" if isinstance(value, float) else str(value)
            lines.append(f"| {key.replace('_', ' ')} | {display} |")
        validation = run["validation"]
        lines.extend(
            [
                "",
                "### Validation",
                "",
                f"Errors: **{validation.get('error', 0)}**; "
                f"warnings: **{validation.get('warning', 0)}**.",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.max_rank < 1:
        raise InputError("--max-rank must be positive")
    if args.batch_size < 1:
        raise InputError("--batch-size must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    gt_items, excluded_items = load_ground_truth(args.ground_truth)
    gt_by_qid = {item.qid: item for item in gt_items}
    run_files = discover_run_files(args.runs_dir, args.run_file_name)
    runs = [load_run(name, path, gt_by_qid, args.max_rank) for name, path in run_files]

    candidates, references, keys = build_pair_work(runs, gt_items)
    scorer = QualityScorer(
        args.sts_model,
        args.bertscore_model,
        args.device,
        args.batch_size,
        args.download_nltk_data,
    )
    pair_scores = scorer.score_pairs(candidates, references)
    answer_scores = aggregate_pair_scores(keys, pair_scores)
    per_query, summary = calculate_reports(runs, gt_items, answer_scores, args.max_rank)
    per_query_time, time_summary = calculate_time_reports(runs, gt_items, args.max_rank)

    write_csv(args.output_dir / "effectiveness_summary.csv", summary)
    write_csv(args.output_dir / "effectiveness_per_query.csv", per_query)
    write_csv(args.output_dir / "time_summary.csv", time_summary)
    write_csv(args.output_dir / "time_per_query.csv", per_query_time)
    validation_rows = [issue for run in runs for issue in run.issues]
    if validation_rows:
        write_csv(args.output_dir / "validation_report.csv", validation_rows)
    else:
        validation_rows = [
            {"run_name": "ALL", "severity": "info", "code": "valid", "Q_ID": "", "detail": "No issues found"}
        ]
        write_csv(args.output_dir / "validation_report.csv", validation_rows)

    human_summary = build_human_summary(summary, time_summary, validation_rows)
    write_json(args.output_dir / "results_summary.json", human_summary)
    write_markdown_summary(args.output_dir / "results_summary.md", human_summary)
    write_json(
        args.output_dir / "effectiveness_per_query.json",
        build_per_query_json(per_query, args.max_rank),
    )
    write_json(
        args.output_dir / "time_per_query.json",
        build_time_per_query_json(per_query_time),
    )

    metadata_payload = {
        "scorer_version": SCORER_VERSION,
        "python": platform.python_version(),
        "ground_truth": str(args.ground_truth.resolve()),
        "ground_truth_sha256": file_sha256(args.ground_truth),
        "runs_directory": str(args.runs_dir.resolve()),
        "eligible_query_count": len(gt_items),
        "excluded_ground_truth_query_count": len(excluded_items),
        "excluded_ground_truth_pairs_found": [[item.qid, item.video_id] for item in excluded_items],
        "configured_exclusions": sorted([list(pair) for pair in EXCLUDED_QUERY_VIDEO_PAIRS]),
        "reference_modes": list(REFERENCE_MODES),
        "multi_reference_rule": "maximum score across Correct_Answer and deduplicated Alternative_Correct_Answers",
        "cutoffs": list(CUTOFFS),
        "max_rank": args.max_rank,
        "missing_query_and_rank_policy": "zero score; retained in denominator",
        "sts": {"model": args.sts_model, "measure": "cosine similarity; negative values clipped only for NDCG"},
        "meteor": {"implementation": "nltk.translate.meteor_score", "tokenizer": "lowercase Unicode word/punctuation regex", "wordnet": True},
        "bertscore": {
            "model": args.bertscore_model,
            "component": "F1",
            "rescale_with_baseline": False, # True for scaling scores between 0-1
            "lang": "en",
            "hash": scorer.bertscore_hash,
        },
        "ndcg": {"gain": "linear", "discount": "log2(rank+1)", "idcg": "all submitted ranks sorted, then truncated"},
        "time_policy": "Time is a per-answer, nonnegative numeric value; excluded queries omitted; no effectiveness weighting",
        "packages": {
            name: package_version(name)
            for name in ("sentence-transformers", "bert-score", "nltk", "numpy", "torch", "transformers")
        },
        "run_files": [
            {"run_name": run.run_name, "path": str(run.run_file.resolve()), "sha256": file_sha256(run.run_file)}
            for run in runs
        ],
    }
    with (args.output_dir / "scoring_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata_payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    error_count = sum(issue["severity"] == "error" for run in runs for issue in run.issues)
    print(f"Scored {len(runs)} run(s) over {len(gt_items)} eligible queries.")
    print(f"Excluded {len(excluded_items)} broken-video queries present in the ground truth.")
    print(f"Reports written to {args.output_dir}")
    if error_count:
        print(f"Validation found {error_count} error(s); see validation_report.csv", file=sys.stderr)
        return 0 if args.allow_incomplete_runs else 2
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (InputError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
