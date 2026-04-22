from __future__ import annotations

import argparse
import asyncio
import json
import re
import time
from pathlib import Path


DEFAULT_BASE_URL = "http://localhost:8000/v1"
DEFAULT_MODEL = "Qwen/Qwen3-30B-A3B-Thinking-2507"
DEFAULT_INPUT_JSON = "qwen3-235b-a22b-math-train_lte4096.incorrect.json"
DEFAULT_SOLUTIONS_JSON = "math_full_minus_math500.json"
DEFAULT_MAX_TOKENS = 4096
DEFAULT_CONCURRENCY = 16


MATH_JUDGE_PROMPT = """You are an expert mathematics grader. Your job is to determine whether a student's final answer to a math problem is correct, by comparing it against a provided reference solution.

You will be shown:
1. The problem statement.
2. A reference solution that shows the correct approach and the correct final answer. Use this as your primary source of truth — read it carefully and extract the correct answer from it.
3. The student's response, which contains a step-by-step approach and a final answer (usually in a \\boxed{{}} or otherwise clearly stated at the end).

=== How to use the reference solution ===
- The reference solution is your PRIMARY source of truth. Read it carefully to understand what the correct answer is, what form it takes, and how many solutions are expected.
- If the problem asks for "all values" or "all solutions" and the reference solution finds multiple, then the complete set is the correct answer — even if a summary field elsewhere lists only one.
- Derive your understanding of the correct answer from the full reference solution, not from any single extracted field.

=== Grading rules ===
- Your PRIMARY focus is the student's final answer, not the reasoning or presentation.
- The student's intermediate reasoning is SECONDARY evidence. Do not ignore it entirely — use it as a tiebreaker or disambiguator when helpful, for example:
  * To identify which value is the intended final answer when the student boxes or states multiple candidates.
  * To confirm that a cleanly-stated final number is genuinely the conclusion of the student's work (vs. an intermediate value they happened to box by mistake).
  * To interpret an ambiguous or abbreviated final answer (e.g., "so k = -33/2" at the end of a short derivation).
  However, do NOT downgrade a correct final answer just because the stated reasoning looks thin, skips steps, or takes a different path from the reference solution. Correct answer with weak reasoning is still CORRECT.

- The student's answer is CORRECT if it is mathematically equivalent to the correct answer derived from the reference solution. In particular, ignore differences that do not change mathematical meaning, such as:
  * Equivalent algebraic forms: e.g. "1/2" vs "0.5" vs "\\frac{{1}}{{2}}"; "2\\sqrt{{2}}" vs "\\sqrt{{8}}"; "x^2+2x+1" vs "(x+1)^2".
  * Formatting/LaTeX differences: e.g. "\\dfrac" vs "\\frac"; extra/missing "$", spaces, or parentheses; "\\left(" vs "(".
  * Trivial labeling differences: e.g. reference "k = -33/2" vs student "-33/2"; "x = 3" vs "3".
  * Unit formatting when the numeric value is the same (unless the problem explicitly requires particular units).
  * Ordering of elements in an unordered set, tuple, or list of solutions.
  * Equivalent interval / set notation: e.g. "(-\\infty, 0) \\cup (0, \\infty)" vs "\\mathbb{{R}} \\setminus \\{{0\\}}".
  * Equivalent trigonometric / logarithmic forms: e.g. "\\ln 2" vs "\\log_e 2"; "\\cos(\\pi/3)" vs "1/2".
  * Reasonable numerical approximations of an exact answer: e.g. reference answer "3\\pi" vs student "9.42" or "9.4248"; reference "\\sqrt{{2}}" vs student "1.414"; reference "90\\pi" vs student "282.74". By default, accept decimal approximations of exact values as CORRECT. Mark as INCORRECT ONLY if the problem statement explicitly demands a specific form or precision (e.g. "give your answer in terms of \\pi", "to 4 decimal places", "exact form only") and the student violates that requirement.
  * Multiple-choice label vs. computed value: e.g. if the correct answer is "3" and the student writes "(C)" where option C corresponds to 3, treat as CORRECT.

=== Leniency for ambiguous student outputs ===
- First, determine whether the problem asks for a SINGLE answer or MULTIPLE answers (e.g., "find all values", "list all solutions", "enter all possible polynomials"). Use the problem statement and the reference solution to decide this.
- If the problem asks for MULTIPLE answers and the student provides multiple boxed or clearly-stated values, collect ALL of them as the student's complete final answer. Judge whether the full set matches the correct set of solutions (all required values present, no spurious extras that are not valid solutions). Ordering does not matter. The values may appear across multiple \\boxed{{}} expressions, or separated by commas/semicolons within a single box.
- If a problem asks for a single answer but the student boxes or states multiple values, consider all of those values when judging the response, rather than only the last one. Do not mark a response incorrect solely because it contains multiple boxed answers; mark it incorrect only if the student’s answer(s), taken as a whole, are not equivalent to the reference ground-truth answer(s).
- If the final answer line is missing or unclear but the step-by-step work unambiguously concludes with a specific value, you may use that concluding value as the student's final answer.

=== When the answer is INCORRECT ===
- It differs from the correct answer (as derived from the reference solution) in a way that is NOT covered by the equivalences above (wrong numerical value, wrong sign, wrong root selected, etc.).
- It is a different mathematical object (e.g. a function vs a number, an incorrect set of solutions).
- For a multi-solution problem, the student is missing required solutions or includes spurious extra ones that are NOT valid solutions of the problem. Ordering does not matter, but membership does.
- The student produced no identifiable final answer and their step-by-step work does not converge on one.
- If you are genuinely uncertain whether the two answers are mathematically equivalent, first try to simplify both to a canonical form and compare. If still uncertain, mark as INCORRECT (err on the side of rejecting).
- Do NOT re-solve the problem from scratch. Trust the provided reference solution.

=== Problem ===
{problem}

=== Ground-truth solution ===
{solution}

=== Student's response ===
{student_response}

=== Output format ===
Output ONLY your final verdict, in exactly this format and with no other text, explanation, or reasoning before or after:

<verdict>correct</verdict>

or

<verdict>incorrect</verdict>

Case does not matter (e.g., "Correct", "CORRECT", "correct" are all accepted). Do not output anything outside the <verdict>...</verdict> tags."""


VERDICT_RE = re.compile(
    r"<verdict>\s*(correct|incorrect)\s*</verdict>", re.IGNORECASE
)


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=(
            "Run an LLM math judge on student responses using a "
            "vLLM server."
        ),
    )
    parser.add_argument(
        "--input_json",
        type=str,
        default=DEFAULT_INPUT_JSON,
        help=(
            "Path to the input JSON or JSONL file containing student "
            "responses to judge."
        ),
    )
    parser.add_argument(
        "--solutions_json",
        type=str,
        default=DEFAULT_SOLUTIONS_JSON,
        help=(
            "Path to the secondary JSON file containing reference solutions, "
            "joined by the 'problem' field (exact string match)."
        ),
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default=None,
        help=(
            "Path to write the augmented JSON or JSONL. If omitted, writes "
            "back to the input file in the same format."
        ),
    )
    parser.add_argument(
        "--num_questions",
        type=int,
        default=None,
        help="Number of records to judge. If omitted, all records are processed.",
    )
    parser.add_argument(
        "--start_index",
        type=int,
        default=0,
        help="Index of the first record to judge (0-based).",
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip records that already have a populated _judge_verdict field.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print the full prompt sent to the judge for each record.",
    )
    parser.add_argument(
        "--base_url",
        type=str,
        default=DEFAULT_BASE_URL,
        help="Base URL of the vLLM OpenAI-compatible API.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help="Model name as registered in the vLLM server.",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help="Maximum tokens to generate per judge call.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=DEFAULT_CONCURRENCY,
        help="Maximum number of concurrent requests to the vLLM server.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (0.0 for greedy).",
    )
    return parser.parse_args()


def load_solutions_index(path: Path) -> dict[str, str]:
    if not path.exists():
        print(
            f"Warning: solutions file not found at {path}. "
            f"Reference solutions will be left blank."
        )
        return {}
    with path.open("r", encoding="utf-8") as f:
        rows = json.load(f)
    index: dict[str, str] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        problem = row.get("problem")
        solution = row.get("solution")
        if problem and solution:
            index[problem] = solution
    print(f"Loaded {len(index)} reference solutions from {path.name}.")
    return index


def load_records(path: Path) -> tuple[list[dict], bool]:
    if path.suffix.lower() == ".jsonl":
        records = []
        with path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise SystemExit(
                        f"Invalid JSON on line {line_no} of {path}: {exc}"
                    ) from exc
                if not isinstance(record, dict):
                    raise SystemExit(
                        f"Expected JSON object on line {line_no} of {path}, "
                        f"got {type(record).__name__}."
                    )
                records.append(record)
        return records, True

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise SystemExit(
            f"Expected a JSON list of records, got {type(data).__name__}."
        )
    return data, False


def render_judge_prompt(
    problem: str,
    solution: str,
    student_response: str,
) -> str:
    return MATH_JUDGE_PROMPT.format(
        problem=problem,
        solution=solution or "(no reference solution available)",
        student_response=student_response,
    )


def parse_verdict(raw_text: str) -> str:
    if not raw_text:
        return "unparseable"
    m = VERDICT_RE.search(raw_text)
    if not m:
        return "unparseable"
    return m.group(1).lower()


def write_records_atomic(data, out_file, *, as_jsonl: bool):
    out_file = Path(out_file)
    tmp_file = out_file.with_name(f"{out_file.name}.tmp")
    with tmp_file.open("w", encoding="utf-8") as f:
        if as_jsonl:
            for record in data:
                json.dump(record, f, ensure_ascii=False)
                f.write("\n")
        else:
            json.dump(data, f, indent=2, ensure_ascii=False)
            f.write("\n")
    tmp_file.replace(out_file)


async def judge_one(
    client,
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    semaphore: asyncio.Semaphore,
) -> tuple[str, float]:
    messages = [{"role": "user", "content": prompt}]

    async with semaphore:
        t0 = time.time()
        response = await client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        elapsed = time.time() - t0

    choice = response.choices[0].message

    content = (choice.content or "").strip()
    verdict = parse_verdict(content)

    return verdict, elapsed


async def main_async():
    args = parse_args()

    input_path = Path(args.input_json).expanduser().resolve()
    if not input_path.exists():
        raise SystemExit(f"Input JSON not found: {input_path}")

    solutions_path = Path(args.solutions_json).expanduser().resolve()
    output_path = (
        Path(args.output_json).expanduser().resolve()
        if args.output_json
        else input_path
    )

    print(f"Loading data from: {input_path}")
    data, input_is_jsonl = load_records(input_path)
    output_is_jsonl = (
        input_is_jsonl
        if args.output_json is None
        else output_path.suffix.lower() == ".jsonl"
    )

    solutions_index = load_solutions_index(solutions_path)

    total = len(data)
    start_idx = max(0, args.start_index)
    end_idx = total
    if args.num_questions is not None:
        if args.num_questions < 0:
            raise SystemExit("--num_questions must be >= 0.")
        end_idx = min(total, start_idx + args.num_questions)

    if start_idx >= total:
        raise SystemExit(
            f"--start_index ({start_idx}) is beyond dataset size ({total})."
        )


    work_items: list[int] = []
    for idx in range(start_idx, end_idx):
        record = data[idx]
        if not isinstance(record, dict):
            continue
        if args.skip_existing and record.get("_judge_verdict") in (
            "correct",
            "incorrect",
        ):
            print(
                f"[{idx + 1}/{total}] Skipping (already judged: "
                f"{record['_judge_verdict']})."
            )
            continue
        if not record.get("problem"):
            print(f"[{idx + 1}/{total}] Skipping (no 'problem' field).")
            continue
        if not record.get("answer"):
            print(f"[{idx + 1}/{total}] Skipping (no 'answer' field).")
            continue
        work_items.append(idx)

    n_work = len(work_items)
    print(f"Total records in file: {total}")
    print(f"Records to judge: {n_work}  (concurrency={args.concurrency})")
    print(f"vLLM server: {args.base_url}")

    if n_work == 0:
        print("Nothing to do.")
        return

    from openai import AsyncOpenAI

    client = AsyncOpenAI(base_url=args.base_url, api_key="unused")
    semaphore = asyncio.Semaphore(args.concurrency)

    start_time = time.time()
    processed = 0
    verdict_counts = {"correct": 0, "incorrect": 0, "unparseable": 0}

    for batch_start in range(0, n_work, args.concurrency):
        batch_indices = work_items[batch_start : batch_start + args.concurrency]

        tasks = []
        for idx in batch_indices:
            record = data[idx]
            prompt = render_judge_prompt(
                problem=record["problem"],
                solution=solutions_index.get(record["problem"], ""),
                student_response=record["answer"],
            )
            if args.verbose:
                print(f"\n===== PROMPT for record {idx + 1}/{total} =====")
                print(prompt)
                print("===== END PROMPT =====\n")
            tasks.append(
                judge_one(
                    client=client,
                    model=args.model,
                    prompt=prompt,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    semaphore=semaphore,
                )
            )

        results = await asyncio.gather(*tasks, return_exceptions=True)

        for idx, result in zip(batch_indices, results):
            record = data[idx]
            if isinstance(result, Exception):
                print(f"[{idx + 1}/{total}] ERROR: {result}")
                continue

            verdict, gen_seconds = result
            record["_judge_verdict"] = verdict
            verdict_counts[verdict] = verdict_counts.get(verdict, 0) + 1

            processed += 1
            elapsed = time.time() - start_time
            avg = elapsed / processed if processed else 0.0
            remaining = n_work - processed
            eta = avg * remaining
            print(
                f"[{idx + 1}/{total}] verdict={verdict} "
                f"in {gen_seconds:.1f}s "
                f"(avg {avg:.1f}s/q, ETA {eta / 60:.1f} min)"
            )


        write_records_atomic(data, output_path, as_jsonl=output_is_jsonl)

    total_seconds = time.time() - start_time
    print(f"\nFinished {processed} record(s) in {total_seconds / 60:.1f} min.")
    print(
        f"Verdict counts -> correct: {verdict_counts['correct']}, "
        f"incorrect: {verdict_counts['incorrect']}, "
        f"unparseable: {verdict_counts['unparseable']}"
    )
    print(f"Wrote results to: {output_path}")


def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
