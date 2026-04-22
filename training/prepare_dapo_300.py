from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path


DAPO_PREFIX_RE = re.compile(
    r"^\s*Solve the following math problem step by step\."
    r"\s+The last line of your response should be of the form"
    r"\s+Answer:\s*\$Answer\s+\(without quotes\)"
    r"\s+where\s+\$Answer\s+is the answer to the problem\.\s+",
    re.IGNORECASE,
)
DAPO_SUFFIX_RE = re.compile(
    r"\s*Remember to put your answer on its own line\s+after\s+"
    r'"?Answer:?"?\s*\.?\s*$',
    re.IGNORECASE,
)


def strip_dapo_prefix(text: str) -> str:
    return DAPO_PREFIX_RE.sub("", text, count=1).strip()


def strip_dapo_wrappers(text: str) -> str:
    text = DAPO_PREFIX_RE.sub("", text, count=1)
    text = DAPO_SUFFIX_RE.sub("", text)
    return text.strip()


def _load_dapo_records(src: Path):
    try:
        import pyarrow.parquet as pq
    except ImportError:
        sys.exit(
            "missing pyarrow -- install with `uv pip install pyarrow` "
            "(usually pulled in by `datasets` already)"
        )

    parquet_files = sorted(src.rglob("*.parquet"))
    if not parquet_files:
        sys.exit(
            f"no .parquet files under {src}. Did the huggingface-cli "
            f"download finish? Expected layout: {src}/data/*.parquet"
        )
    print(f"[dapo] found {len(parquet_files)} parquet shard(s):")
    for f in parquet_files:
        print(f"  {f.relative_to(src)}")

    records = []
    for f in parquet_files:
        table = pq.read_table(str(f))
        rows = table.to_pylist()
        records.extend(rows)
    print(f"[dapo] total rows: {len(records)}")
    if records:
        print(f"[dapo] schema (first row keys): {list(records[0].keys())}")
    return records


def _extract_problem(row: dict) -> str | None:
    raw: str | None = None
    for key in ("problem", "question", "prompt"):
        v = row.get(key)
        if isinstance(v, str) and v.strip():
            raw = v.strip()
            break
    if raw is None:

        p = row.get("prompt")
        if isinstance(p, list) and p:
            for m in p:
                if isinstance(m, dict) and m.get("role") == "user":
                    content = m.get("content") or m.get("value")
                    if isinstance(content, str) and content.strip():
                        raw = content.strip()
                        break
    if raw is None:
        return None
    return strip_dapo_wrappers(raw)


def _extract_answer(row: dict) -> str | None:

    for key in ("answer", "final_answer", "ground_truth", "solution_answer"):
        v = row.get(key)
        if isinstance(v, str) and v.strip():
            return v.strip()

    rm = row.get("reward_model")
    if isinstance(rm, dict):
        for key in ("ground_truth", "answer"):
            v = rm.get(key)
            if isinstance(v, str) and v.strip():
                return v.strip()

    ei = row.get("extra_info")
    if isinstance(ei, dict):
        for key in ("answer", "gt_answer"):
            v = ei.get(key)
            if isinstance(v, str) and v.strip():
                return v.strip()
    return None


def _looks_judgeable(answer: str) -> bool:
    if not answer:
        return False

    if "\n" in answer.strip():
        return False

    if any(tok in answer.lower() for tok in [
        "step ", "therefore", "we have", "since ", "let ", "answer is"
    ]):
        return False

    if len(answer) > 100:
        return False
    return True


def _length_ok(problem: str, lo: int = 60, hi: int = 1500) -> bool:
    return lo <= len(problem) <= hi


def _self_grade_ok(problem: str, answer: str) -> bool:
    from grpo_reward import compute_reward
    fake_text = f"\\boxed{{{answer}}}"
    try:
        return compute_reward(fake_text, answer) >= 0.5
    except Exception:
        return False


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True,
                   help="Path to the downloaded DAPO-Math-17k directory.")
    p.add_argument("--out", default="grpo_train_dapo300.json",
                   help="Output JSON path.")
    p.add_argument("--n", type=int, default=300,
                   help="Number of problems to sample after filtering.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_problem_chars", type=int, default=1500)
    p.add_argument("--min_problem_chars", type=int, default=60)
    p.add_argument("--max_answer_chars", type=int, default=100)
    p.add_argument("--no_self_grade", action="store_true",
                   help="Skip the sympy round-trip check (faster, but "
                        "may include some answers the judge can't parse).")
    args = p.parse_args()

    random.seed(args.seed)
    src = Path(args.src)
    if not src.is_dir():
        sys.exit(f"{src} is not a directory")

    records = _load_dapo_records(src)


    pa: list[tuple[str, str]] = []
    miss_problem = miss_answer = 0
    for r in records:
        prob = _extract_problem(r)
        ans = _extract_answer(r)
        if not prob:
            miss_problem += 1
            continue
        if not ans:
            miss_answer += 1
            continue
        pa.append((prob, ans))
    print(
        f"[dapo] kept {len(pa)} after field extraction "
        f"(missing problem={miss_problem}, missing answer={miss_answer})"
    )


    seen = set()
    unique = []
    for prob, ans in pa:
        if prob in seen:
            continue
        seen.add(prob)
        unique.append((prob, ans))
    print(f"[dapo] after dedup: {len(unique)}")


    after_struct = []
    for prob, ans in unique:
        if not _length_ok(prob, args.min_problem_chars, args.max_problem_chars):
            continue
        if len(ans) > args.max_answer_chars:
            continue
        if not _looks_judgeable(ans):
            continue
        after_struct.append((prob, ans))
    print(f"[dapo] after structural filter: {len(after_struct)}")


    if not args.no_self_grade:
        kept = []
        n_drop = 0
        for i, (prob, ans) in enumerate(after_struct):
            if i % 1000 == 0:
                print(f"  self-grading {i}/{len(after_struct)}...", flush=True)
            if _self_grade_ok(prob, ans):
                kept.append((prob, ans))
            else:
                n_drop += 1
        print(
            f"[dapo] after sympy self-grade: kept {len(kept)} / "
            f"dropped {n_drop}"
        )
    else:
        kept = after_struct

    if len(kept) < args.n:
        sys.exit(
            f"only {len(kept)} problems survived filters; cannot sample "
            f"{args.n}. Try --no_self_grade or relax --max_problem_chars."
        )


    sample = random.sample(kept, args.n)
    out_data = [{"problem": prob, "answer": ans} for prob, ans in sample]
    Path(args.out).write_text(
        json.dumps(out_data, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


    plens = [len(d["problem"]) for d in out_data]
    alens = [len(d["answer"]) for d in out_data]
    print(f"\n[dapo] wrote {len(out_data)} problems to {args.out}")
    print(
        f"  problem len: min={min(plens)}, median={sorted(plens)[len(plens)//2]}, max={max(plens)}"
    )
    print(
        f"  answer  len: min={min(alens)}, median={sorted(alens)[len(alens)//2]}, max={max(alens)}"
    )
    print("\nSample (first 3):")
    for d in out_data[:3]:
        print(f"  Q: {d['problem'][:120]}")
        print(f"  A: {d['answer']}")
        print()


if __name__ == "__main__":
    main()
