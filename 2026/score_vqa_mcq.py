"""                                                                                                                                                        
NIST-developed software is provided by NIST as a public service. You may use, copy, and distribute copies of the software in any medium, provided that you 
keep intact this entire notice. You may improve, modify, and create derivative works of the software or any portion of the software, and you may copy and  
distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of  
any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.                           
                                                                                                                                                           
NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT, OR ARISING BY OPERATION OF LAW,      
INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT, AND DATA ACCURACY. NIST        
NEITHR REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES    
NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY,  
RELIABILITY, OR USEFULNESS OF THE SOFTWARE.                                                                                                                
                                                                                                                                                           
You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use,    
including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment,     
and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury  
or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.                           
"""

#!/usr/bin/env python3
"""Score a ranked VQA-MCQ submission using Top-1 accuracy and MRR."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
import unicodedata
from pathlib import Path
from typing import Any


# Queries whose source YouTube videos are no longer available.  These Q_IDs are
# outside the evaluation set: submitted entries are ignored, and they do not
# participate in validation, completeness checks, per-query output, or metrics.
EXCLUDED_QUERIES: dict[str, str] = {
    "19": "9EyY7TK0mCA",
    "265": "_3mWTYMXSok",
    "470": "2ZefngWxXoE",
    "608": "I4BgRvGNpjM",
    "950": "UZDZb9Cijdo",
    "959": "APlvy8fawLM",
    "1242": "6f7tV70a_y4",
    "1381": "7cIqwelU2o4",
    "1407": "6T30Pikgiw4",
    "1505": "Axv5y-i_ieA",
    "1632": "bujAtvNcGiw",
    "2044": "BHvt4VcAIBQ",
    "2099": "8rzskun9Qyc",
    "2156": "LjGrmx69MQU",
    "2273": "8EX5QO6NDq0",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate and score a VQA-MCQ JSON run against a ground-truth CSV."
    )
    parser.add_argument("run", type=Path, help="Team submission JSON")
    parser.add_argument("queries", type=Path, help="Released query JSON containing all options")
    parser.add_argument("ground_truth", type=Path, help="Ground-truth CSV")
    parser.add_argument(
        "--summary",
        type=Path,
        help="Summary JSON path (default: <run_stem>.scores.json)",
    )
    parser.add_argument(
        "--per-query",
        type=Path,
        help="Per-query CSV path (default: <run_stem>.per_query.csv)",
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Allow missing queries; they remain in the denominator and score zero",
    )
    return parser.parse_args()


def qid_key(value: Any) -> str:
    return str(value).strip()


def normalize_answer(value: str) -> str:
    """Normalize encoding and whitespace only; preserve case and punctuation."""
    value = unicodedata.normalize("NFKC", value)
    return re.sub(r"\s+", " ", value).strip()


def load_ground_truth(
    path: Path,
) -> tuple[list[dict[str, str]], list[dict[str, Any]], list[dict[str, str]]]:
    errors: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        fields = set(reader.fieldnames or [])
        required = {"Q_ID", "Video_ID", "Correct_Answer"}
        missing = sorted(required - fields)
        if missing:
            raise ValueError(f"Ground truth is missing columns: {', '.join(missing)}")
        all_rows = list(reader)

    excluded_rows = [
        row for row in all_rows if qid_key(row.get("Q_ID", "")) in EXCLUDED_QUERIES
    ]
    rows = [
        row for row in all_rows if qid_key(row.get("Q_ID", "")) not in EXCLUDED_QUERIES
    ]

    seen: set[str] = set()
    for row_number, row in enumerate(rows, start=2):
        key = qid_key(row.get("Q_ID", ""))
        if not key:
            errors.append({"type": "empty_ground_truth_qid", "csv_row": row_number})
        elif key in seen:
            errors.append(
                {"type": "duplicate_ground_truth_qid", "Q_ID": key, "csv_row": row_number}
            )
        else:
            seen.add(key)
        if not normalize_answer(row.get("Correct_Answer", "")):
            errors.append({"type": "empty_correct_answer", "Q_ID": key, "csv_row": row_number})
    return rows, errors, excluded_rows


def load_run(path: Path) -> list[Any]:
    with path.open("r", encoding="utf-8-sig") as stream:
        run = json.load(stream)
    if not isinstance(run, list):
        raise ValueError("The run JSON root must be a list.")
    return run


def load_queries(path: Path) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    with path.open("r", encoding="utf-8-sig") as stream:
        rows = json.load(stream)
    if not isinstance(rows, list):
        raise ValueError("The query JSON root must be a list.")

    by_qid: dict[str, dict[str, Any]] = {}
    errors: list[dict[str, Any]] = []
    for query_index, query in enumerate(rows):
        if not isinstance(query, dict):
            errors.append({"type": "invalid_query_entry", "query_index": query_index})
            continue
        key = qid_key(query.get("Q_ID", ""))
        if key in EXCLUDED_QUERIES:
            continue
        if not key:
            errors.append({"type": "empty_query_qid", "query_index": query_index})
            continue
        if key in by_qid:
            errors.append({"type": "duplicate_query_qid", "Q_ID": key})
            continue
        options = query.get("Options")
        if not isinstance(options, list) or not options:
            errors.append({"type": "invalid_query_options", "Q_ID": key})
            continue
        option_texts: list[str] = []
        malformed = False
        for option_position, option in enumerate(options, start=1):
            if not isinstance(option, dict) or not isinstance(option.get("option"), str):
                errors.append(
                    {
                        "type": "invalid_query_option",
                        "Q_ID": key,
                        "option_position": option_position,
                    }
                )
                malformed = True
                continue
            text = normalize_answer(option["option"])
            if not text:
                errors.append(
                    {
                        "type": "empty_query_option",
                        "Q_ID": key,
                        "option_position": option_position,
                    }
                )
                malformed = True
            option_texts.append(text)
        duplicates = sorted({text for text in option_texts if option_texts.count(text) > 1})
        if duplicates:
            errors.append(
                {"type": "duplicate_query_options", "Q_ID": key, "options": duplicates}
            )
            malformed = True
        if not malformed:
            item = dict(query)
            item["_normalized_options"] = option_texts
            by_qid[key] = item
    return by_qid, errors


def validate_run(
    run: list[Any],
    gt_by_qid: dict[str, dict[str, str]],
    query_by_qid: dict[str, dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    valid: dict[str, dict[str, Any]] = {}
    seen_run_qids: set[str] = set()
    errors: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []

    for entry_index, entry in enumerate(run):
        location = {"entry_index": entry_index}
        if not isinstance(entry, dict):
            errors.append({"type": "invalid_run_entry", **location})
            continue

        key = qid_key(entry.get("Q_ID", ""))
        if not key:
            errors.append({"type": "empty_run_qid", **location})
            continue
        if key in EXCLUDED_QUERIES:
            continue
        if key in seen_run_qids:
            errors.append({"type": "duplicate_run_qid", "Q_ID": key, **location})
            continue
        seen_run_qids.add(key)
        gt = gt_by_qid.get(key)
        if gt is None:
            errors.append({"type": "unknown_qid", "Q_ID": key, **location})
            continue
        query = query_by_qid.get(key)
        if query is None:
            errors.append({"type": "qid_missing_from_query_file", "Q_ID": key, **location})
            continue

        video_id = entry.get("Video_ID")
        if not isinstance(video_id, str) or video_id != gt["Video_ID"]:
            errors.append(
                {
                    "type": "video_id_mismatch",
                    "Q_ID": key,
                    "submitted": video_id,
                    "expected": gt["Video_ID"],
                }
            )
            continue

        answers = entry.get("Answers")
        if not isinstance(answers, list) or not answers:
            errors.append({"type": "invalid_answers", "Q_ID": key})
            continue
        expected_options = query["_normalized_options"]
        expected_count = len(expected_options)
        if len(answers) != expected_count:
            errors.append(
                {
                    "type": "incorrect_answer_count",
                    "Q_ID": key,
                    "answer_count": len(answers),
                    "expected_answer_count": expected_count,
                }
            )
            continue

        ranked: dict[int, str] = {}
        normalized_seen: set[str] = set()
        entry_invalid = False
        for answer_index, item in enumerate(answers):
            if not isinstance(item, dict):
                errors.append(
                    {"type": "invalid_answer_entry", "Q_ID": key, "answer_index": answer_index}
                )
                entry_invalid = True
                continue
            rank = item.get("Rank")
            # bool is a subclass of int and must not be accepted as a rank.
            if isinstance(rank, bool) or not isinstance(rank, int) or not 1 <= rank <= expected_count:
                errors.append(
                    {
                        "type": "invalid_rank",
                        "Q_ID": key,
                        "answer_index": answer_index,
                        "Rank": rank,
                    }
                )
                entry_invalid = True
                continue
            if rank in ranked:
                errors.append({"type": "duplicate_rank", "Q_ID": key, "Rank": rank})
                entry_invalid = True
                continue
            answer = item.get("Answer")
            if not isinstance(answer, str) or not normalize_answer(answer):
                errors.append(
                    {"type": "invalid_answer_text", "Q_ID": key, "Rank": rank}
                )
                entry_invalid = True
                continue
            normalized = normalize_answer(answer)
            if normalized in normalized_seen:
                errors.append(
                    {"type": "duplicate_answer", "Q_ID": key, "Answer": answer}
                )
                entry_invalid = True
                continue
            normalized_seen.add(normalized)
            ranked[rank] = answer

        expected_ranks = list(range(1, len(answers) + 1))
        if sorted(ranked) != expected_ranks:
            errors.append(
                {
                    "type": "nonconsecutive_ranks",
                    "Q_ID": key,
                    "submitted_ranks": sorted(ranked),
                    "expected_ranks": expected_ranks,
                }
            )
            entry_invalid = True
        submitted_options = [normalize_answer(answer) for answer in ranked.values()]
        missing_options = sorted(set(expected_options) - set(submitted_options))
        unknown_options = sorted(set(submitted_options) - set(expected_options))
        if missing_options or unknown_options:
            errors.append(
                {
                    "type": "submitted_options_do_not_match_query_options",
                    "Q_ID": key,
                    "missing_options": missing_options,
                    "unknown_options": unknown_options,
                }
            )
            entry_invalid = True
        if entry_invalid:
            continue

        valid[key] = {"Video_ID": video_id, "ranked_answers": ranked}

    missing = [key for key in gt_by_qid if key not in valid]
    if missing:
        warnings.append(
            {
                "type": "missing_queries",
                "count": len(missing),
                "Q_IDs": missing,
            }
        )
    return valid, errors, warnings


def score(
    gt_rows: list[dict[str, str]],
    valid_run: dict[str, dict[str, Any]],
    excluded_rows: list[dict[str, str]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    per_query: list[dict[str, Any]] = []
    top1_sum = 0
    reciprocal_rank_sum = 0.0

    for gt in gt_rows:
        key = qid_key(gt["Q_ID"])
        correct = normalize_answer(gt["Correct_Answer"])
        submitted = valid_run.get(key)
        found_rank: int | None = None
        rank1_answer = ""
        if submitted is not None:
            ranked = submitted["ranked_answers"]
            rank1_answer = ranked.get(1, "")
            matches = [
                rank for rank, answer in ranked.items() if normalize_answer(answer) == correct
            ]
            if matches:
                found_rank = min(matches)

        top1 = int(found_rank == 1)
        reciprocal_rank = 0.0 if found_rank is None else 1.0 / found_rank
        top1_sum += top1
        reciprocal_rank_sum += reciprocal_rank
        per_query.append(
            {
                "Q_ID": gt["Q_ID"],
                "Video_ID": gt["Video_ID"],
                "Correct_Answer": gt["Correct_Answer"],
                "Submitted_Rank_1": rank1_answer,
                "Correct_Answer_Rank": "" if found_rank is None else found_rank,
                "Top_1": top1,
                "Reciprocal_Rank": reciprocal_rank,
                "Submitted": submitted is not None,
            }
        )

    denominator = len(gt_rows)
    top1_accuracy = top1_sum / denominator if denominator else math.nan
    mrr = reciprocal_rank_sum / denominator if denominator else math.nan
    metrics = {
        "ground_truth_query_count": denominator + len(excluded_rows),
        "excluded_query_count": len(excluded_rows),
        "excluded_qids": [qid_key(row["Q_ID"]) for row in excluded_rows],
        "query_count": denominator,
        "submitted_query_count": len(valid_run),
        "missing_query_count": denominator - len(valid_run),
        "top_1_correct_count": top1_sum,
        "top_1_accuracy": top1_accuracy,
        "reciprocal_rank_sum": reciprocal_rank_sum,
        "mrr": mrr,
    }
    return metrics, per_query


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def write_per_query(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "Q_ID",
        "Video_ID",
        "Correct_Answer",
        "Submitted_Rank_1",
        "Correct_Answer_Rank",
        "Top_1",
        "Reciprocal_Rank",
        "Submitted",
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    summary_path = args.summary or args.run.with_name(f"{args.run.stem}.scores.json")
    per_query_path = args.per_query or args.run.with_name(f"{args.run.stem}.per_query.csv")

    try:
        gt_rows, gt_errors, excluded_rows = load_ground_truth(args.ground_truth)
        query_by_qid, query_errors = load_queries(args.queries)
        run = load_run(args.run)
        gt_by_qid = {qid_key(row["Q_ID"]): row for row in gt_rows}
        consistency_errors: list[dict[str, Any]] = []
        for key, gt in gt_by_qid.items():
            query = query_by_qid.get(key)
            if query is None:
                consistency_errors.append({"type": "ground_truth_qid_missing_from_queries", "Q_ID": key})
                continue
            if query.get("Video_ID") != gt["Video_ID"]:
                consistency_errors.append(
                    {
                        "type": "query_ground_truth_video_id_mismatch",
                        "Q_ID": key,
                        "query_video_id": query.get("Video_ID"),
                        "ground_truth_video_id": gt["Video_ID"],
                    }
                )
            if normalize_answer(gt["Correct_Answer"]) not in query["_normalized_options"]:
                consistency_errors.append(
                    {"type": "correct_answer_missing_from_query_options", "Q_ID": key}
                )
        for key in query_by_qid.keys() - gt_by_qid.keys():
            consistency_errors.append({"type": "query_qid_missing_from_ground_truth", "Q_ID": key})

        valid_run, run_errors, warnings = validate_run(run, gt_by_qid, query_by_qid)
        errors = gt_errors + query_errors + consistency_errors + run_errors

        missing_count = len(gt_by_qid) - len(valid_run)
        if missing_count and not args.allow_partial:
            errors.append(
                {
                    "type": "incomplete_run",
                    "missing_query_count": missing_count,
                    "message": "Use --allow-partial to score missing queries as zero.",
                }
            )

        if errors:
            result = {
                "status": "failed_validation",
                "run_file": str(args.run),
                "query_file": str(args.queries),
                "ground_truth_file": str(args.ground_truth),
                "metrics": None,
                "warning_count": len(warnings),
                "error_count": len(errors),
                "warnings": warnings,
                "errors": errors,
            }
            write_json(summary_path, result)
            print(
                f"Validation failed with {len(errors)} error(s). See {summary_path}.",
                file=sys.stderr,
            )
            return 1

        metrics, per_query = score(gt_rows, valid_run, excluded_rows)
        result = {
            "status": "success",
            "run_file": str(args.run),
            "query_file": str(args.queries),
            "ground_truth_file": str(args.ground_truth),
            "answer_matching": "Unicode NFKC + collapsed whitespace; case and punctuation preserved",
            "metrics": metrics,
            "warning_count": len(warnings),
            "error_count": 0,
            "warnings": warnings,
            "errors": [],
        }
        write_json(summary_path, result)
        write_per_query(per_query_path, per_query)

        print(f"Queries in ground truth: {metrics['ground_truth_query_count']}")
        print(f"Queries excluded:        {metrics['excluded_query_count']}")
        print(f"Queries evaluated:       {metrics['query_count']}")
        print(f"Queries submitted:       {metrics['submitted_query_count']}")
        print(f"Top-1 accuracy:          {metrics['top_1_accuracy']:.6f}")
        print(f"MRR:                     {metrics['mrr']:.6f}")
        print(f"Summary:                 {summary_path}")
        print(f"Per-query results:       {per_query_path}")
        return 0
    except (OSError, ValueError, json.JSONDecodeError, csv.Error) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
