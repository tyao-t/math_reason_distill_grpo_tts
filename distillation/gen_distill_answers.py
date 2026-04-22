import asyncio
import json
import argparse
from pathlib import Path
from openai import AsyncOpenAI
from tqdm import tqdm

DISTILLATION_PROMPT = (
    "You are a rigorous mathematical/logical reasoning engine and assistant. Solve the following problem directly.\n\n"
    "Strict Rules for your output:\n"
    "1. Do not restate or paraphrase the problem. Step 1 can and should be mathematical setup such as defining variables, translating conditions into equations, or the first deduction, etc. — not narration.\n"
    "2. DO NOT skip any core algebraic, geometric or logical deduction steps. Show complete intermediate derivations and calculations.\n"
    "3. Keep natural language text to a minimum — use it only to connect steps (e.g., 'Substitute x into equation 1:').\n"
    "4. DO NOT use conversational filler, greetings, or narrative commentary such as 'Let's see', 'Okay, let me consider...', or 'Hmm, wait actually', etc.\n"
    "5. Structure the solution strictly using 'Step 1:', 'Step 2:', etc.\n"
    "6. On the last line, put your final answer within \\boxed{{}}\n\n"
    "Problem:\n{prompt}\n\n"
)


SUSPICIOUS_CONTENT_LEN = 50


def extract_reasoning(msg):
    r = getattr(msg, "reasoning_content", None)
    if r is not None:
        return r
    extra = getattr(msg, "model_extra", None) or {}
    return extra.get("reasoning_content")


async def solve_one(
    item: dict,
    num_samples: int,
    sem: asyncio.Semaphore,
    client: AsyncOpenAI,
    model_name: str,
    max_tokens: int,
    enable_thinking: bool,
):
    problem_text = item["problem"]
    problem_idx = item["unique_id"]
    user_content = DISTILLATION_PROMPT.format(prompt=problem_text)

    try:
        async with sem:
            response = await client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": user_content}],
                n=num_samples,
                temperature=0.7,
                top_p=0.80,
                presence_penalty=0.0,
                max_tokens=max_tokens,
                extra_body={
                    "top_k": 20,
                    "min_p": 0.0,
                    "repetition_penalty": 1.0,
                    "chat_template_kwargs": {"enable_thinking": enable_thinking},
                },
            )

        minimal_samples = []
        suspicious_dumps = []

        for idx, choice in enumerate(response.choices):
            msg = choice.message
            content = msg.content
            reasoning = extract_reasoning(msg)

            sample_record = {
                "content": content,
                "finish_reason": choice.finish_reason,
            }
            if reasoning is not None:
                sample_record["reasoning_content"] = reasoning
            minimal_samples.append(sample_record)


            is_suspicious = (
                content is None
                or (isinstance(content, str) and len(content.strip()) < SUSPICIOUS_CONTENT_LEN)
            )
            if is_suspicious:
                suspicious_dumps.append({
                    "problem_idx": problem_idx,
                    "sample_idx": idx,
                    "problem_text": problem_text,
                    "finish_reason": choice.finish_reason,
                    "content": content,
                    "reasoning_content": reasoning,
                    "full_message_dump": msg.model_dump(),
                    "full_choice_dump": choice.model_dump(),
                })

        return {
            "problem_idx": problem_idx,
            "original_answer": item.get("answer", ""),
            "samples": minimal_samples,
        }, suspicious_dumps

    except Exception as e:
        tqdm.write(f"Error processing problem {problem_idx}: {e}")

        error_dump = {
            "problem_idx": problem_idx,
            "type": "api_error",
            "error_message": str(e)
        }
        return None, [error_dump]


async def main(args):
    input_path = Path(args.input)
    output_path = Path(args.output)
    diag_path = output_path.with_suffix(".diagnostics.jsonl")

    if not input_path.exists():
        print(f"Error: Could not find input file at {input_path}")
        return

    client = AsyncOpenAI(
        base_url=args.base_url,
        api_key="EMPTY",
    )

    with input_path.open("r", encoding="utf-8") as f:
        dataset = json.load(f)

    target_problems = dataset[:args.num_questions] if args.num_questions > 0 else dataset

    print(f"Server        : {args.base_url}")
    print(f"Model name    : {args.model}")
    print(f"Problems      : {len(target_problems)}")
    print(f"Samples/prob  : {args.num_samples}")
    print(f"Concurrency   : {args.concurrency}")
    print(f"Max new tokens: {args.max_tokens}")
    print(f"enable_thinking: {args.enable_thinking}")
    print(f"Diagnostics   : {diag_path.resolve()}")

    sem = asyncio.Semaphore(args.concurrency)

    output_path.write_text("", encoding="utf-8")
    diag_path.write_text("", encoding="utf-8")

    tasks = [
        solve_one(
            item,
            args.num_samples,
            sem,
            client,
            args.model,
            args.max_tokens,
            args.enable_thinking,
        )
        for item in target_problems
    ]

    success_count = 0
    finish_reason_counts = {}
    null_content_count = 0
    short_content_count = 0
    reasoning_present_count = 0
    total_samples = 0

    with output_path.open("a", encoding="utf-8") as fout, diag_path.open("a", encoding="utf-8") as fdiag:
        for future in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Processing Problems"):
            result, suspicious = await future


            for d in suspicious:
                fdiag.write(json.dumps(d, ensure_ascii=False, default=str) + "\n")
                fdiag.flush()


            if result is None:
                continue

            fout.write(json.dumps(result, ensure_ascii=False) + "\n")
            fout.flush()
            success_count += 1

            for s in result["samples"]:
                total_samples += 1
                fr = s.get("finish_reason") or "none"
                finish_reason_counts[fr] = finish_reason_counts.get(fr, 0) + 1
                if s.get("content") is None:
                    null_content_count += 1
                elif isinstance(s["content"], str) and len(s["content"].strip()) < SUSPICIOUS_CONTENT_LEN:
                    short_content_count += 1
                if "reasoning_content" in s:
                    reasoning_present_count += 1

    print(f"\n=== Summary ===")
    print(f"Records written: {success_count} → {output_path.resolve()}")
    print(f"Total samples: {total_samples}")
    print(f"finish_reason distribution: {finish_reason_counts}")
    print(f"Null-content samples: {null_content_count}")
    print(f"Short-content samples (<{SUSPICIOUS_CONTENT_LEN} chars): {short_content_count}")
    print(f"Samples with reasoning_content field: {reasoning_present_count}")
    if reasoning_present_count > 0:
        print("  ⚠  reasoning_content is being populated — a reasoning parser IS active on the server.")
    else:
        print("  ✓  reasoning_content never populated — no reasoning parser splitting output.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate math distillation data against an OpenAI-compatible server (vLLM / SGLang).",
    )

    parser.add_argument(
        "--base-url",
        type=str,
        default="http://localhost:8000/v1",
        help="OpenAI-compatible base URL. vLLM default port is 8000; SGLang default is 30000.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="qwen3-27b",
        help="Model name. Must match --served-model-name on vLLM, or the model ID on SGLang.",
    )

    parser.add_argument("--input", type=str, default="math_full_minus_math500.json")
    parser.add_argument("--output", type=str, default="distillation_results.jsonl")
    parser.add_argument("--num-questions", type=int, default=10)
    parser.add_argument("--num-samples", type=int, default=4)
    parser.add_argument("--concurrency", type=int, default=128)

    parser.add_argument("--max-tokens", type=int, default=2560)
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        help="Pass enable_thinking=True to the chat template (Qwen thinking mode). Off by default.",
    )

    args = parser.parse_args()
    asyncio.run(main(args))
