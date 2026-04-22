import argparse
import asyncio
import json
import re
import time
from pathlib import Path
from tokenize import TokenError

try:
    import httpx
except ImportError:
    httpx = None

try:
    from openai import AsyncOpenAI
except ImportError:
    AsyncOpenAI = None

try:
    from sympy import simplify
    from sympy.core.sympify import SympifyError
    from sympy.parsing import sympy_parser as spp
    from sympy.polys.polyerrors import PolynomialError
except ImportError:
    simplify = None
    SympifyError = Exception
    spp = None
    PolynomialError = Exception


RE_NUMBER = re.compile(
    r"-?(?:\d+/\d+|\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)"
)
RE_SPECIAL = re.compile(r"<\|[^>]+?\|>")
LATEX_FIXES = [
    (r"\\left\s*", ""),
    (r"\\right\s*", ""),
    (r"\\,|\\!|\\;|\\:", ""),
    (r"\\cdot", "*"),
    (r"\u00B7|\u00D7", "*"),
    (r"\\\^\\circ", ""),
    (r"\\dfrac", r"\\frac"),
    (r"\\tfrac", r"\\frac"),
    ("\u00b0", ""),
]
SUPERSCRIPT_MAP = {
    "\u2070": "0", "\u00b9": "1", "\u00b2": "2", "\u00b3": "3",
    "\u2074": "4", "\u2075": "5", "\u2076": "6", "\u2077": "7",
    "\u2078": "8", "\u2079": "9", "\u207a": "+", "\u207b": "-",
    "\u207d": "(", "\u207e": ")",
}
SUPERSCRIPT_CHARS = "".join(re.escape(ch) for ch in SUPERSCRIPT_MAP)


def get_last_boxed(text):
    boxed_start_idx = text.rfind(r"\boxed")
    if boxed_start_idx == -1:
        return None

    current_idx = boxed_start_idx + len(r"\boxed")
    while current_idx < len(text) and text[current_idx].isspace():
        current_idx += 1

    if current_idx >= len(text) or text[current_idx] != "{":
        return None

    current_idx += 1
    brace_depth = 1
    content_start_idx = current_idx

    while current_idx < len(text) and brace_depth > 0:
        char = text[current_idx]
        if char == "{":
            brace_depth += 1
        elif char == "}":
            brace_depth -= 1
        current_idx += 1

    if brace_depth != 0:
        return None
    return text[content_start_idx:current_idx - 1]


def extract_final_candidate(text, fallback="number_then_full"):
    result = ""

    if text:
        boxed = get_last_boxed(text.strip())
        if boxed:
            result = boxed.strip().strip("$ ")
        elif fallback in ("number_then_full", "number_only"):
            numbers = RE_NUMBER.findall(text)
            if numbers:
                result = numbers[-1]
            elif fallback == "number_then_full":
                result = text
    return result


def normalize_text(text):
    if not text:
        return ""
    text = RE_SPECIAL.sub("", text).strip()

    match = re.match(r"^[A-Za-z]\s*[.:]\s*(.+)$", text)
    if match:
        text = match.group(1)

    text = re.sub(r"\^\s*\{\s*\\circ\s*\}", "", text)
    text = re.sub(r"\^\s*\\circ", "", text)
    text = text.replace("\u00b0", "")

    match = re.match(r"^\\text\{(?P<x>.+?)\}$", text)
    if match:
        text = match.group("x")

    text = re.sub(r"\\\(|\\\)|\\\[|\\\]", "", text)

    for pattern, replacement in LATEX_FIXES:
        text = re.sub(pattern, replacement, text)

    def convert_superscripts(value, base=None):
        converted = "".join(
            SUPERSCRIPT_MAP[ch] if ch in SUPERSCRIPT_MAP else ch
            for ch in value
        )
        if base is None:
            return converted
        return f"{base}**{converted}"

    text = re.sub(
        rf"([0-9A-Za-z\)\]\}}])([{SUPERSCRIPT_CHARS}]+)",
        lambda match: convert_superscripts(match.group(2), base=match.group(1)),
        text,
    )
    text = convert_superscripts(text)

    text = text.replace("\\%", "%").replace("$", "").replace("%", "")
    text = re.sub(
        r"\\sqrt\s*\{([^}]*)\}",
        lambda match: f"sqrt({match.group(1)})",
        text,
    )
    text = re.sub(
        r"\\sqrt\s+([^\\\s{}]+)",
        lambda match: f"sqrt({match.group(1)})",
        text,
    )
    text = re.sub(
        r"\\frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}",
        lambda match: f"({match.group(1)})/({match.group(2)})",
        text,
    )
    text = re.sub(
        r"\\frac\s+([^\s{}]+)\s+([^\s{}]+)",
        lambda match: f"({match.group(1)})/({match.group(2)})",
        text,
    )
    text = text.replace("^", "**")
    text = re.sub(
        r"(?<=\d)\s+(\d+/\d+)",
        lambda match: "+" + match.group(1),
        text,
    )
    text = re.sub(r"(?<=\d),(?=\d\d\d(\D|$))", "", text)

    return text.replace("{", "").replace("}", "").strip().lower()


def sympy_parser(expr):
    if expr is None or len(expr) > 2000:
        return None
    ensure_sympy_available()
    try:
        return spp.parse_expr(
            expr,
            transformations=(
                *spp.standard_transformations,
                spp.implicit_multiplication_application,
            ),
            evaluate=True,
        )
    except (
        SympifyError,
        SyntaxError,
        TypeError,
        AttributeError,
        IndexError,
        TokenError,
        ValueError,
        PolynomialError,
    ):
        return None


def ensure_sympy_available():
    if spp is None or simplify is None:
        raise RuntimeError(
            "SymPy is required for MATH-500 grading. Install it with "
            "`pip install sympy` or run this script in the project environment."
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


def equality_check(expr_gtruth, expr_pred):
    if expr_gtruth == expr_pred:
        return True

    gtruth, pred = sympy_parser(expr_gtruth), sympy_parser(expr_pred)
    if gtruth is not None and pred is not None:
        try:
            return simplify(gtruth - pred) == 0
        except (SympifyError, TypeError):
            pass
    return False


def split_into_parts(text):
    result = [text]

    if text:
        if (
            len(text) >= 2
            and text[0] in "(["
            and text[-1] in ")]"
            and "," in text[1:-1]
        ):
            items = [part.strip() for part in text[1:-1].split(",")]
            if all(items):
                result = items
    else:
        result = []

    return result


def grade_answer(pred_text, gt_text):
    result = False

    if pred_text is not None and gt_text is not None:
        gt_parts = split_into_parts(normalize_text(gt_text))
        pred_parts = split_into_parts(normalize_text(pred_text))

        if gt_parts and pred_parts and len(gt_parts) == len(pred_parts):
            result = all(
                equality_check(gt, pred)
                for gt, pred in zip(gt_parts, pred_parts)
            )
    return result


def render_prompt(prompt):
    return (
        "You are a helpful math assistant.\n"
        "Answer the question and write the final result on a new line as:\n"
        "\\boxed{ANSWER}\n\n"
        f"Question:\n{prompt}\n\nAnswer:"
    )


def load_math500_test(local_path="math500_test.json", save_copy=True):
    local_path = Path(local_path)
    url = (
        "https://raw.githubusercontent.com/rasbt/reasoning-from-scratch/"
        "main/ch03/01_main-chapter-code/math500_test.json"
    )
    candidates = [
        local_path,
        Path(__file__).resolve().parent / local_path,
        Path(__file__).resolve().parent.parent / "01_main-chapter-code" / local_path,
    ]

    for candidate in candidates:
        if candidate.exists():
            with candidate.open("r", encoding="utf-8") as f:
                return json.load(f)

    ensure_httpx_available()
    with httpx.Client(timeout=30.0, follow_redirects=True) as client:
        response = client.get(url)
        response.raise_for_status()
        data = response.json()

    if save_copy:
        local_path.parent.mkdir(parents=True, exist_ok=True)
        with local_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    return data


def eta_progress_message(
    processed,
    total,
    start_time,
    show_eta=False,
    label="Progress",
):
    progress = f"{label}: {processed}/{total}"
    pad_width = len(f"{label}: {total}/{total} | ETA: 00h 00m 00s")
    if not show_eta or processed <= 0:
        return progress.ljust(pad_width)

    elapsed = time.time() - start_time
    if elapsed <= 0:
        return progress.ljust(pad_width)

    remaining = max(total - processed, 0)
    avg_time = elapsed / processed
    eta_seconds = max(int(round(avg_time * remaining)), 0)
    minutes, rem_seconds = divmod(eta_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        eta = f"{hours}h {minutes:02d}m {rem_seconds:02d}s"
    elif minutes:
        eta = f"{minutes:02d}m {rem_seconds:02d}s"
    else:
        eta = f"{rem_seconds:02d}s"

    return f"{progress} | ETA: {eta}".ljust(pad_width)


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--host",
        type=str,
        default="localhost",
        help="Host where the vLLM server is reachable.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port the vLLM server is listening on (vLLM's default is 8000).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help=(
            "Served model name or local path passed to `vllm serve`. "
            "If omitted, auto-detected from /v1/models."
        ),
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="completions",
        choices=["completions", "chat"],
        help=(
            "'completions' for base models (no chat template). "
            "'chat' for instruction-tuned / thinking models."
        ),
    )
    parser.add_argument(
        "--dataset_size",
        type=int,
        default=500,
        help="Number of MATH-500 examples to evaluate (500 = full set).",
    )
    parser.add_argument(
        "--input_path",
        default="math500_test.json",
        help="Path to MATH-500 JSON. Default downloads if absent.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=4096,
        help="Max new tokens per request.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=128,
        help="Number of in-flight requests.",
    )
    parser.add_argument(
        "--request_timeout",
        type=float,
        default=600.0,
        help="Per-request HTTP timeout in seconds.",
    )
    parser.add_argument(
        "--out_file",
        type=str,
        default=None,
        help=(
            "If set, write per-problem results (problem, ground-truth answer, "
            "raw generation, extracted prediction, verdict) to this JSON file. "
            "Order matches math500_test.json."
        ),
    )
    return parser.parse_args()


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


async def solve_one(
    client,
    model,
    row,
    idx,
    semaphore,
    mode,
    max_new_tokens,
):
    prompt = render_prompt(row["problem"])
    async with semaphore:
        if mode == "completions":
            resp = await client.completions.create(
                model=model,
                prompt=prompt,
                max_tokens=max_new_tokens,
                temperature=0.0,
            )
            gen_text = resp.choices[0].text
        else:
            resp = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_new_tokens,
                temperature=0.0,
            )


            gen_text = resp.choices[0].message.content or ""

    extracted = extract_final_candidate(gen_text)
    is_correct = bool(grade_answer(extracted, row["answer"]))
    return idx, {
        "index": idx,
        "problem": row["problem"],
        "ground_truth": row["answer"],
        "generation": gen_text,
        "prediction": extracted,
        "boxed_prediction": f"\\boxed{{{extracted}}}",
        "correct": is_correct,
    }


async def run_eval(args, model_name, math_data):
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
    num_examples = len(math_data)
    start_time = time.time()

    try:
        tasks = [
            asyncio.create_task(
                solve_one(
                    client,
                    model_name,
                    row,
                    idx,
                    semaphore,
                    mode=args.mode,
                    max_new_tokens=args.max_new_tokens,
                )
            )
            for idx, row in enumerate(math_data)
        ]

        results = [None] * num_examples
        num_correct = 0
        for done_count, fut in enumerate(asyncio.as_completed(tasks), start=1):
            idx, record = await fut
            results[idx] = record
            num_correct += int(record["correct"])
            print(
                eta_progress_message(
                    processed=done_count,
                    total=num_examples,
                    start_time=start_time,
                    show_eta=True,
                    label="MATH-500 (vLLM)",
                ),
                end="\r",
                flush=True,
            )
    finally:
        await http_client.aclose()

    elapsed = time.time() - start_time
    acc = num_correct / num_examples if num_examples else 0.0
    print(
        f"\nAccuracy: {acc*100:.2f}% ({num_correct}/{num_examples})"
        f"  |  wall time: {elapsed:.1f}s"
    )

    if args.out_file:
        out_path = Path(args.out_file)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        summary = {
            "model": model_name,
            "mode": args.mode,
            "max_new_tokens": args.max_new_tokens,
            "num_examples": num_examples,
            "num_correct": num_correct,
            "accuracy": acc,
            "wall_time_seconds": elapsed,
            "results": results,
        }
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
            f.write("\n")
        print(f"Wrote per-problem results to {out_path}")


def main():
    args = parse_args()
    ensure_sympy_available()
    base_url = f"http://{args.host}:{args.port}/v1"
    model_name = args.model or autodetect_model(base_url)

    print(f"Server: {base_url}")
    print(f"Model:  {model_name}")
    print(f"Mode:   {args.mode}")

    math_data = load_math500_test(local_path=args.input_path)[: args.dataset_size]
    print(f"Dataset: {len(math_data)} problems\n")

    asyncio.run(run_eval(args, model_name, math_data))


if __name__ == "__main__":
    main()
