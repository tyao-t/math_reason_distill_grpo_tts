import argparse
import asyncio
import math
import time

import httpx
from openai import AsyncOpenAI

from evaluate_math500_vllm import (
    autodetect_model,
    ensure_httpx_available,
    ensure_openai_available,
    ensure_sympy_available,
    eta_progress_message,
    extract_final_candidate,
    grade_answer,
    load_math500_test,
    normalize_text,
    render_prompt,
)


def cluster_key(pred_text):
    if pred_text is None:
        return ""
    return normalize_text(pred_text)


def _build_clusters(samples):
    clusters = {}
    for s in samples:
        key = cluster_key(s["prediction"])
        if not key:
            continue
        c = clusters.setdefault(
            key,
            {"count": 0, "weight": 0.0, "representative": s["prediction"]},
        )
        c["count"] += 1
        mean_lp = s.get("mean_logprob")
        if mean_lp is not None:
            c["weight"] += math.exp(mean_lp)
    return clusters


def _pick_cluster(clusters, primary):
    assert primary in ("count", "weight")
    secondary = "weight" if primary == "count" else "count"
    return min(
        clusters.keys(),
        key=lambda k: (
            -clusters[k][primary],
            -clusters[k][secondary],
            k,
        ),
    )


def majority_vote(samples):
    if not samples:
        return ""
    clusters = _build_clusters(samples)
    if not clusters:
        return samples[0]["prediction"]
    return clusters[_pick_cluster(clusters, primary="count")]["representative"]


def weighted_majority_vote(samples):
    if not samples:
        return ""
    clusters = _build_clusters(samples)
    if not clusters:
        return samples[0]["prediction"]
    return clusters[_pick_cluster(clusters, primary="weight")]["representative"]


def pass_at_n(samples, ground_truth):
    return any(grade_answer(s["prediction"], ground_truth) for s in samples)


def _mean_logprob_from_completions(choice):
    lp = getattr(choice, "logprobs", None)
    if lp is None:
        return None
    token_lps = getattr(lp, "token_logprobs", None) or []
    valid = [x for x in token_lps if x is not None]
    if not valid:
        return None
    return sum(valid) / len(valid)


def _mean_logprob_from_chat(choice):
    lp = getattr(choice, "logprobs", None)
    if lp is None:
        return None
    content = getattr(lp, "content", None) or []
    valid = [tok.logprob for tok in content if getattr(tok, "logprob", None) is not None]
    if not valid:
        return None
    return sum(valid) / len(valid)


async def sample_one_problem(
    client,
    model,
    row,
    idx,
    semaphore,
    mode,
    n_samples,
    max_new_tokens,
    temperature,
    top_p,
    use_logprobs,
    seed,
    best_of,
):
    prompt = render_prompt(row["problem"])
    async with semaphore:
        if mode == "completions":
            kwargs = dict(
                model=model,
                prompt=prompt,
                max_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                n=n_samples,
            )
            if use_logprobs:
                kwargs["logprobs"] = 1
            if seed is not None:
                kwargs["seed"] = seed
            if best_of is not None and best_of > n_samples:


                kwargs["extra_body"] = {"best_of": best_of}
            resp = await client.completions.create(**kwargs)
            samples = []
            for choice in resp.choices:
                gen_text = choice.text or ""
                pred = extract_final_candidate(gen_text)
                samples.append({
                    "generation": gen_text,
                    "prediction": pred,
                    "mean_logprob": (
                        _mean_logprob_from_completions(choice) if use_logprobs else None
                    ),
                })
        else:
            kwargs = dict(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                n=n_samples,
            )
            if use_logprobs:
                kwargs["logprobs"] = True
            if seed is not None:
                kwargs["seed"] = seed
            resp = await client.chat.completions.create(**kwargs)
            samples = []
            for choice in resp.choices:


                gen_text = (choice.message.content or "")
                pred = extract_final_candidate(gen_text)
                samples.append({
                    "generation": gen_text,
                    "prediction": pred,
                    "mean_logprob": (
                        _mean_logprob_from_chat(choice) if use_logprobs else None
                    ),
                })


    for s in samples:
        s["correct"] = bool(grade_answer(s["prediction"], row["answer"]))

    record = {
        "index": idx,
        "problem": row["problem"],
        "ground_truth": row["answer"],
        "samples": samples,
    }
    return idx, record


def aggregate_metrics(records):
    total = len(records)
    n_cons = n_wcons = n_pass = 0
    sum_avg1 = 0.0

    for r in records:
        gt = r["ground_truth"]
        samples = r["samples"]

        cons_pred = majority_vote(samples)
        r["cons_prediction"] = cons_pred
        r["cons_correct"] = bool(grade_answer(cons_pred, gt))
        n_cons += int(r["cons_correct"])

        wcons_pred = weighted_majority_vote(samples)
        r["wcons_prediction"] = wcons_pred
        r["wcons_correct"] = bool(grade_answer(wcons_pred, gt))
        n_wcons += int(r["wcons_correct"])

        r["pass_correct"] = pass_at_n(samples, gt)
        n_pass += int(r["pass_correct"])

        per_sample = [int(s["correct"]) for s in samples]
        r["avg1"] = sum(per_sample) / len(per_sample) if per_sample else 0.0
        sum_avg1 += r["avg1"]

    return {
        "cons_at_n": n_cons / total if total else 0.0,
        "wcons_at_n": n_wcons / total if total else 0.0,
        "pass_at_n": n_pass / total if total else 0.0,
        "avg_at_1": sum_avg1 / total if total else 0.0,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=__doc__,
    )
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--model", type=str, default=None,
        help="Served model name. Auto-detected from /v1/models if omitted.",
    )
    parser.add_argument(
        "--mode", type=str, default="completions", choices=["completions", "chat"],
        help="'completions' for base models, 'chat' for instruct/thinking models.",
    )
    parser.add_argument(
        "--input_path", default="math500.json",
        help="Path to MATH-500 JSON (default: local math500.json).",
    )
    parser.add_argument("--dataset_size", type=int, default=500)


    parser.add_argument(
        "--n_samples", type=int, default=8,
        help="N for cons@N / pass@N. 8/16/32/64 are typical.",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.7,
        help="Sampling temperature. Must be > 0 for self-consistency to work.",
    )
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument(
        "--seed", type=int, default=None,
        help="If set, passed to vLLM for reproducible sampling.",
    )
    parser.add_argument(
        "--no_logprobs", action="store_true",
        help="Skip requesting logprobs (disables wcons@N, smaller responses).",
    )
    parser.add_argument(
        "--best_of", type=int, default=None,
        help=(
            "vLLM-only server-side best_of. If > n_samples, vLLM internally "
            "samples this many and returns the n_samples with highest cumulative "
            "logprob. Use this for best@N-style scaling."
        ),
    )

    parser.add_argument("--max_new_tokens", type=int, default=4096)
    parser.add_argument(
        "--concurrency", type=int, default=128,
        help="Number of in-flight problems (each issues one n=N request).",
    )
    parser.add_argument("--request_timeout", type=float, default=1800.0)
    return parser.parse_args()


async def run_eval(args, model_name, math_data):
    if args.temperature <= 0 and args.n_samples > 1:
        print(
            "[warn] temperature=0 with n_samples>1 means N identical greedy "
            "samples; cons@N will degenerate to greedy. Set --temperature 0.7."
        )

    base_url = f"http://{args.host}:{args.port}/v1"
    http_client = httpx.AsyncClient(
        timeout=args.request_timeout,
        limits=httpx.Limits(
            max_connections=max(args.concurrency * 2, 64),
            max_keepalive_connections=max(args.concurrency, 32),
        ),
    )
    client = AsyncOpenAI(base_url=base_url, api_key="unused", http_client=http_client)

    semaphore = asyncio.Semaphore(args.concurrency)
    num_examples = len(math_data)
    start_time = time.time()
    use_logprobs = not args.no_logprobs

    try:
        tasks = [
            asyncio.create_task(
                sample_one_problem(
                    client,
                    model_name,
                    row,
                    idx,
                    semaphore,
                    mode=args.mode,
                    n_samples=args.n_samples,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    use_logprobs=use_logprobs,
                    seed=args.seed,
                    best_of=args.best_of,
                )
            )
            for idx, row in enumerate(math_data)
        ]
        records = [None] * num_examples
        for done_count, fut in enumerate(asyncio.as_completed(tasks), start=1):
            idx, record = await fut
            records[idx] = record
            print(
                eta_progress_message(
                    processed=done_count,
                    total=num_examples,
                    start_time=start_time,
                    show_eta=True,
                    label=f"MATH-500 cons@{args.n_samples}",
                ),
                end="\r",
                flush=True,
            )
    finally:
        await http_client.aclose()

    metrics = aggregate_metrics(records)
    elapsed = time.time() - start_time

    print()
    print(f"avg@1     : {metrics['avg_at_1']*100:6.2f}%   (no TTS, baseline)")
    print(f"cons@{args.n_samples:<3} : {metrics['cons_at_n']*100:6.2f}%   (majority vote)")
    if use_logprobs:
        print(f"wcons@{args.n_samples:<2} : {metrics['wcons_at_n']*100:6.2f}%   (logprob-weighted vote)")
    print(f"pass@{args.n_samples:<3} : {metrics['pass_at_n']*100:6.2f}%   (oracle upper bound)")
    print(f"wall time : {elapsed:.1f}s")


def main():
    args = parse_args()
    ensure_sympy_available()
    ensure_httpx_available()
    ensure_openai_available()

    base_url = f"http://{args.host}:{args.port}/v1"
    model_name = args.model or autodetect_model(base_url)

    print(f"Server   : {base_url}")
    print(f"Model    : {model_name}")
    print(f"Mode     : {args.mode}")
    print(f"TTS      : n={args.n_samples}, temperature={args.temperature}, "
          f"top_p={args.top_p}, logprobs={not args.no_logprobs}")

    math_data = load_math500_test(local_path=args.input_path)[: args.dataset_size]
    print(f"Dataset  : {len(math_data)} problems\n")

    asyncio.run(run_eval(args, model_name, math_data))


if __name__ == "__main__":
    main()
