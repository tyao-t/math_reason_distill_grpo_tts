import argparse
import asyncio
import json
import math
import time
from pathlib import Path

try:
    import httpx
except ImportError:
    httpx = None

try:
    from openai import AsyncOpenAI
except ImportError:
    AsyncOpenAI = None

from transformers import AutoTokenizer

IGNORE_INDEX = -100


def question_prompt(prompt):
    return (
        "You are a helpful math assistant.\n"
        "Answer the question and write the final result on a new line as:\n"
        "\\boxed{ANSWER}\n\n"
        f"Question:\n{prompt}\n\nAnswer:"
    )


def ensure_httpx_available():
    if httpx is None:
        raise RuntimeError(
            "httpx is required for HTTP requests. Install it with "
            "`pip install httpx` or run this script in the project environment."
        )


def ensure_openai_available():
    if AsyncOpenAI is None:
        raise RuntimeError(
            "The OpenAI Python SDK is required for vLLM requests. Install it "
            "with `pip install openai` or run this script in the project "
            "environment."
        )


def autodetect_model(base_url):
    ensure_httpx_available()
    url = base_url.rstrip("/") + "/models"
    with httpx.Client(timeout=10.0) as client:
        response = client.get(url, headers={"Authorization": "Bearer unused"})
        response.raise_for_status()
        data = response.json().get("data", [])
    if not data:
        raise RuntimeError(f"No models reported by {url}")
    return data[0]["id"]


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            problem = row.get("problem")
            answer = row.get("answer")
            if problem is None or answer is None:
                continue
            problem = str(problem)
            answer = str(answer).strip()
            if not problem or not answer:
                continue
            rows.append({"problem": problem, "answer": answer})
    return rows


def tokenize_example(row, tokenizer, max_seq_len):
    prompt_text = question_prompt(row["problem"])
    answer_text = row["answer"]
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    answer_ids = tokenizer(answer_text, add_special_tokens=False)["input_ids"]
    if not prompt_ids or not answer_ids:
        return None

    eos_id = tokenizer.eos_token_id
    input_ids = prompt_ids + answer_ids
    if eos_id is not None:
        input_ids = input_ids + [eos_id]
    if len(input_ids) < 2 or len(input_ids) > max_seq_len:
        return None

    answer_token_count = len(answer_ids) + (1 if eos_id is not None else 0)
    return {
        "input_ids": input_ids,
        "prompt_len": len(prompt_ids),
        "answer_len": answer_token_count,
    }


def extract_per_token_logprobs(prompt_lps, input_ids):
    out = []
    for pos, entry in enumerate(prompt_lps):
        if entry is None:
            out.append(None)
            continue
        target_id = input_ids[pos]
        token_entry = entry.get(str(target_id))
        if token_entry is None:
            token_entry = entry.get(target_id)
        if token_entry is None:

            token_entry = next(iter(entry.values()))
        if isinstance(token_entry, dict):
            out.append(token_entry["logprob"])
        else:
            out.append(float(token_entry))
    return out


async def score_one(
    client,
    model,
    row,
    ex,
    idx,
    semaphore,
    score_full_sequence,
):
    async with semaphore:


        resp = await client.completions.create(
            model=model,
            prompt=ex["input_ids"],
            max_tokens=1,
            temperature=0.0,
            logprobs=1,
            extra_body={"prompt_logprobs": 1},
        )

    choice = resp.choices[0]


    prompt_lps = getattr(choice, "prompt_logprobs", None)
    if prompt_lps is None and hasattr(choice, "model_extra"):
        prompt_lps = (choice.model_extra or {}).get("prompt_logprobs")
    if prompt_lps is None:
        raise RuntimeError(
            "Server response did not include 'prompt_logprobs'. Is your vLLM "
            "build new enough to support the prompt_logprobs request param?"
        )

    logprobs = extract_per_token_logprobs(prompt_lps, ex["input_ids"])

    if score_full_sequence:
        scored = [lp for lp in logprobs[1:] if lp is not None]
    else:
        scored = [lp for lp in logprobs[ex["prompt_len"]:] if lp is not None]

    if not scored:
        return idx, {
            "index": idx,
            "problem": row["problem"],
            "error": "no scoreable tokens in scored region",
        }

    sum_lp = float(sum(scored))
    n = len(scored)
    avg_lp = sum_lp / n
    return idx, {
        "index": idx,
        "problem": row["problem"],
        "answer_preview": row["answer"][:120],
        "n_total_tokens": len(ex["input_ids"]),
        "n_prompt_tokens": ex["prompt_len"],
        "n_answer_tokens": ex["answer_len"],
        "n_scored_tokens": n,
        "sum_log_prob": sum_lp,
        "avg_log_prob": avg_lp,
        "perplexity": math.exp(-avg_lp),
    }


def eta_progress_message(processed, total, start_time, label="Progress"):
    progress = f"{label}: {processed}/{total}"
    pad_width = len(f"{label}: {total}/{total} | ETA: 00h 00m 00s")
    if processed <= 0:
        return progress.ljust(pad_width)
    elapsed = time.time() - start_time
    if elapsed <= 0:
        return progress.ljust(pad_width)
    remaining = max(total - processed, 0)
    eta_seconds = max(int(round((elapsed / processed) * remaining)), 0)
    minutes, sec = divmod(eta_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        eta = f"{hours}h {minutes:02d}m {sec:02d}s"
    elif minutes:
        eta = f"{minutes:02d}m {sec:02d}s"
    else:
        eta = f"{sec:02d}s"
    return f"{progress} | ETA: {eta}".ljust(pad_width)


async def run_perplexity(args, model_name, examples_with_rows):
    ensure_httpx_available()
    ensure_openai_available()
    base_url = f"http://{args.host}:{args.port}/v1"

    http_client = httpx.AsyncClient(
        timeout=args.request_timeout,
        limits=httpx.Limits(
            max_connections=max(args.concurrency * 2, 64),
            max_keepalive_connections=max(args.concurrency, 32),
        ),
    )
    client = AsyncOpenAI(
        base_url=base_url,
        api_key="unused",
        http_client=http_client,
    )

    semaphore = asyncio.Semaphore(args.concurrency)
    num_examples = len(examples_with_rows)
    start_time = time.time()
    results = [None] * num_examples

    try:
        tasks = [
            asyncio.create_task(
                score_one(
                    client,
                    model_name,
                    row,
                    ex,
                    idx,
                    semaphore,
                    score_full_sequence=args.score_full_sequence,
                )
            )
            for idx, (row, ex) in enumerate(examples_with_rows)
        ]

        for done_count, fut in enumerate(asyncio.as_completed(tasks), start=1):
            idx, record = await fut
            results[idx] = record
            print(
                eta_progress_message(
                    processed=done_count,
                    total=num_examples,
                    start_time=start_time,
                    label="Perplexity (vLLM)",
                ),
                end="\r",
                flush=True,
            )
    finally:
        await http_client.aclose()

    elapsed = time.time() - start_time
    print()

    valid = [r for r in results if r and "perplexity" in r]
    errors = [r for r in results if r and "error" in r]

    if valid:
        mean_ppl = sum(r["perplexity"] for r in valid) / len(valid)
        mean_alp = sum(r["avg_log_prob"] for r in valid) / len(valid)
        macro_ppl = math.exp(-mean_alp)
        total_tokens = sum(r["n_scored_tokens"] for r in valid)
        total_sumlp = sum(r["sum_log_prob"] for r in valid)
        micro_ppl = math.exp(-total_sumlp / max(total_tokens, 1))
    else:
        mean_ppl = macro_ppl = micro_ppl = float("nan")
        mean_alp = float("nan")
        total_tokens = 0

    print(
        f"Scored {len(valid)} / {num_examples} examples"
        f"  ({len(errors)} errors)  |  wall time: {elapsed:.1f}s"
    )
    print(f"  mean per-example perplexity:        {mean_ppl:.4f}")
    print(f"  macro perplexity (exp of avg log p): {macro_ppl:.4f}")
    print(f"  micro perplexity (token-weighted):   {micro_ppl:.4f}")

    if args.out_file:
        out_path = Path(args.out_file)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        summary = {
            "model": model_name,
            "data_path": args.data_path,
            "score_region": "full_sequence" if args.score_full_sequence else "answer_plus_eos",
            "max_seq_len": args.max_seq_len,
            "num_examples": num_examples,
            "num_scored": len(valid),
            "num_errors": len(errors),
            "wall_time_seconds": elapsed,
            "mean_per_example_perplexity": mean_ppl,
            "macro_perplexity": macro_ppl,
            "micro_perplexity_token_weighted": micro_ppl,
            "mean_avg_log_prob": mean_alp,
            "total_scored_tokens": total_tokens,
            "results": results,
        }
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
            f.write("\n")
        print(f"Wrote per-example results to {out_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=__doc__,
    )
    parser.add_argument("--data_path", type=str,
                        default="distillation_results_new_judged_correct_dedup.jsonl",
                        help="JSONL with 'problem'/'answer' rows.")
    parser.add_argument("--host", type=str, default="localhost",
                        help="Host where the vLLM server is reachable.")
    parser.add_argument("--port", type=int, default=8000,
                        help="Port the vLLM server is listening on.")
    parser.add_argument("--model", type=str, default=None,
                        help=("Served model name or local path passed to "
                              "`vllm serve`. If omitted, auto-detected from "
                              "/v1/models."))
    parser.add_argument("--tokenizer", type=str, default=None,
                        help=("HF tokenizer name or local path. Defaults to "
                              "--model. Must match the served model so the "
                              "token ids we send match what vLLM expects."))
    parser.add_argument("--max_seq_len", type=int, default=2560,
                        help="Skip examples whose prompt+answer+EOS exceed "
                             "this length (matches train_qwen3_06b_sft.py).")
    parser.add_argument("--concurrency", type=int, default=128,
                        help="Number of in-flight requests.")
    parser.add_argument("--request_timeout", type=float, default=600.0,
                        help="Per-request HTTP timeout in seconds.")
    parser.add_argument("--score_full_sequence", action="store_true",
                        help="Average log-prob over the entire sequence "
                             "instead of only answer+EOS tokens.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Process only the first N examples (sanity check).")
    parser.add_argument("--out_file", type=str, default=None,
                        help="If set, write per-example results to this JSON.")
    return parser.parse_args()


def main():
    args = parse_args()
    base_url = f"http://{args.host}:{args.port}/v1"
    model_name = args.model or autodetect_model(base_url)
    tokenizer_src = args.tokenizer or model_name

    print(f"Server:    {base_url}")
    print(f"Model:     {model_name}")
    print(f"Tokenizer: {tokenizer_src}")
    print(f"Data:      {args.data_path}")
    print(f"Score:     {'full sequence' if args.score_full_sequence else 'answer + EOS only'}")

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_src, use_fast=True)

    rows = load_jsonl(args.data_path)
    if args.limit is not None:
        rows = rows[: args.limit]

    examples_with_rows = []
    skipped = 0
    for row in rows:
        ex = tokenize_example(row, tokenizer, args.max_seq_len)
        if ex is None:
            skipped += 1
            continue
        examples_with_rows.append((row, ex))
    print(
        f"Loaded {len(rows)} rows; tokenized {len(examples_with_rows)} usable, "
        f"skipped {skipped} (empty or > {args.max_seq_len} tokens).\n"
    )

    asyncio.run(run_perplexity(args, model_name, examples_with_rows))


if __name__ == "__main__":
    main()
