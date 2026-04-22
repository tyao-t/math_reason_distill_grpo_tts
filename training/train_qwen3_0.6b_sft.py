import argparse
import json
import math
import os
import random
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_constant_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
)


IGNORE_INDEX = -100


_REQUIRED_LOCAL_CONFIG_FILES = ("config.json",)
_TOKENIZER_CANDIDATES = (
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "vocab.json",
)
_WEIGHT_CANDIDATES = (
    "model.safetensors",
    "model.safetensors.index.json",
    "pytorch_model.bin",
    "pytorch_model.bin.index.json",
)


def resolve_model_source(model_name_or_path):

    candidate = Path(model_name_or_path).expanduser()


    looks_like_path = (
        candidate.is_absolute()
        or model_name_or_path.startswith(("./", "../", "~"))
        or os.sep in model_name_or_path
        or candidate.exists()
    )

    if not looks_like_path:
        return model_name_or_path, False

    if not candidate.exists():
        raise FileNotFoundError(
            f"Local model path does not exist: {candidate}"
        )
    if not candidate.is_dir():
        raise NotADirectoryError(
            f"Local model path is not a directory: {candidate}"
        )

    missing_required = [
        name for name in _REQUIRED_LOCAL_CONFIG_FILES
        if not (candidate / name).exists()
    ]
    if missing_required:
        raise FileNotFoundError(
            f"Local model directory {candidate} is missing required files: "
            f"{missing_required}"
        )

    if not any((candidate / name).exists() for name in _TOKENIZER_CANDIDATES):
        raise FileNotFoundError(
            f"Local model directory {candidate} is missing a tokenizer file "
            f"(expected one of {list(_TOKENIZER_CANDIDATES)})."
        )

    if not any((candidate / name).exists() for name in _WEIGHT_CANDIDATES):
        raise FileNotFoundError(
            f"Local model directory {candidate} is missing model weights "
            f"(expected one of {list(_WEIGHT_CANDIDATES)})."
        )

    return str(candidate.resolve()), True


def question_prompt(prompt):
    return (
        "You are a helpful math assistant.\n"
        "Answer the question and write the final result on a new line as:\n"
        "\\boxed{ANSWER}\n\n"
        f"Question:\n{prompt}\n\nAnswer:"
    )


def load_answer_pair_jsonl(path, strict=False, max_bad_examples=5):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    rows = []
    bad_rows = 0
    bad_json = 0
    bad_examples = []

    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                if strict:
                    raise ValueError(
                        f"Invalid JSON on line {line_no}: {exc}"
                    ) from exc
                bad_json += 1
                if len(bad_examples) < max_bad_examples:
                    bad_examples.append((line_no, f"JSONDecodeError: {exc}"))
                continue

            problem = row.get("problem")
            answer = row.get("answer")
            if problem is None or answer is None:
                bad_rows += 1
                if len(bad_examples) < max_bad_examples:
                    bad_examples.append((line_no, "missing problem/answer field"))
                continue
            problem = str(problem)
            answer = str(answer).strip()
            if not problem or not answer:
                bad_rows += 1
                if len(bad_examples) < max_bad_examples:
                    bad_examples.append((line_no, "empty problem/answer field"))
                continue
            rows.append({"problem": problem, "answer": answer})

    if bad_json or bad_rows:
        print(
            f"[load_answer_pair_jsonl] skipped {bad_json} malformed JSON lines "
            f"and {bad_rows} rows with missing/empty fields in {path}",
            flush=True,
        )
        for line_no, reason in bad_examples:
            print(f"  line {line_no}: {reason}", flush=True)

    return rows, bad_rows, bad_json


class SFTDataset(Dataset):
    def __init__(self, examples):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


@dataclass
class SFTDataCollator:
    pad_token_id: int
    max_length: int | None = None
    pad_to_multiple_of: int = 8

    def __call__(self, features):
        max_len = self.max_length or max(len(x["input_ids"]) for x in features)
        if self.pad_to_multiple_of:
            multiple = self.pad_to_multiple_of
            max_len = ((max_len + multiple - 1) // multiple) * multiple

        batch_size = len(features)
        input_ids = torch.full(
            (batch_size, max_len),
            self.pad_token_id,
            dtype=torch.long,
        )
        attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)
        labels = torch.full(
            (batch_size, max_len),
            IGNORE_INDEX,
            dtype=torch.long,
        )

        for row_idx, feature in enumerate(features):
            ids = torch.tensor(feature["input_ids"], dtype=torch.long)
            labs = torch.tensor(feature["labels"], dtype=torch.long)
            seq_len = ids.numel()

            input_ids[row_idx, :seq_len] = ids
            attention_mask[row_idx, :seq_len] = 1
            labels[row_idx, :seq_len] = labs

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def tokenize_examples(rows, tokenizer, max_seq_len):
    examples = []
    dropped_too_long = 0
    dropped_empty = 0

    eos_id = tokenizer.eos_token_id
    for row in rows:
        prompt_text = question_prompt(row["problem"])
        answer_text = row["answer"]


        prompt_ids = tokenizer(
            prompt_text,
            add_special_tokens=False,
            truncation=False,
        )["input_ids"]
        answer_ids = tokenizer(
            answer_text,
            add_special_tokens=False,
            truncation=False,
        )["input_ids"]

        if not prompt_ids or not answer_ids:
            dropped_empty += 1
            continue

        input_ids = prompt_ids + answer_ids
        labels = [IGNORE_INDEX] * len(prompt_ids) + list(answer_ids)
        if eos_id is not None:
            input_ids.append(eos_id)
            labels.append(eos_id)

        if len(input_ids) > max_seq_len:
            dropped_too_long += 1
            continue
        if len(input_ids) < 2:
            dropped_empty += 1
            continue

        examples.append(
            {
                "input_ids": input_ids,
                "labels": labels,
                "length": len(input_ids),
                "answer_tokens": len(answer_ids) + (1 if eos_id is not None else 0),
            }
        )

    stats = {
        "kept": len(examples),
        "dropped_too_long": dropped_too_long,
        "dropped_empty": dropped_empty,
    }
    return examples, stats


def seed_everything(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pick_device(allow_cpu=False):
    if torch.cuda.is_available():
        return torch.device("cuda")
    if allow_cpu:
        return torch.device("cpu")
    raise RuntimeError(
        "CUDA was not found. This training script is intended for one NVIDIA "
        "B200 or H100. Pass --allow_cpu only for a tiny smoke test."
    )


def format_eta(seconds):
    if seconds is None or math.isinf(seconds) or seconds < 0:
        return "--"
    seconds = int(seconds)
    minutes, sec = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:d}:{minutes:02d}:{sec:02d}"


def _slugify(value):
    value = str(value).strip().lower()


    cleaned = []
    for ch in value:
        if ch.isalnum() or ch in ("-", "_", "."):
            cleaned.append(ch)
        else:
            cleaned.append("-")
    slug = "".join(cleaned)

    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug.strip("-") or "run"


def auto_output_dir(args):
    model_basename = Path(args.model_name_or_path).name or args.model_name_or_path
    model_slug = _slugify(model_basename)
    data_slug = _slugify(Path(args.data_path).stem)
    parts = [
        model_slug,
        "sft",
        data_slug,
        f"bs{args.per_device_train_batch_size}",
    ]
    if args.gradient_accumulation_steps > 1:
        parts.append(f"ga{args.gradient_accumulation_steps}")
    if args.gradient_checkpointing:
        parts.append("gc")
    parts.append(_slugify(args.attn_implementation))
    base = Path(args.output_base_dir).expanduser() if args.output_base_dir else Path(".")
    return str(base / "-".join(parts))


def freeze_for_last_n_layers(model, num_trainable_layers):
    if num_trainable_layers is None:
        return None
    if num_trainable_layers < 0:
        raise ValueError("--num_trainable_layers must be >= 0.")

    base = getattr(model, "model", None)
    if base is None or not hasattr(base, "layers"):
        raise RuntimeError(
            "Could not locate `model.model.layers`; this layer-freezing helper "
            "assumes a LLaMA/Qwen-style decoder layout."
        )
    layers = base.layers
    total_layers = len(layers)
    if num_trainable_layers > total_layers:
        raise ValueError(
            f"--num_trainable_layers={num_trainable_layers} exceeds the model's "
            f"{total_layers} transformer layers."
        )

    tied = bool(getattr(model.config, "tie_word_embeddings", False))


    for param in model.parameters():
        param.requires_grad = False


    if hasattr(base, "norm"):
        for param in base.norm.parameters():
            param.requires_grad = True


    if hasattr(model, "lm_head") and not tied:
        for param in model.lm_head.parameters():
            param.requires_grad = True


    if num_trainable_layers > 0:
        for layer in layers[-num_trainable_layers:]:
            for param in layer.parameters():
                param.requires_grad = True


    if tied:
        if hasattr(base, "embed_tokens"):
            for param in base.embed_tokens.parameters():
                param.requires_grad = False
        if hasattr(model, "lm_head"):
            for param in model.lm_head.parameters():
                param.requires_grad = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    head_status = (
        "lm_head FROZEN (tied with input embeddings)"
        if tied
        else "lm_head trainable"
    )
    print(
        f"Layer freezing active: keeping the last {num_trainable_layers} of "
        f"{total_layers} transformer blocks trainable, plus final norm; "
        f"{head_status}",
        flush=True,
    )
    print(
        f"Trainable parameters: {trainable:,} / {total:,} "
        f"({100.0 * trainable / max(total, 1):.2f}%)",
        flush=True,
    )
    return trainable, total


def save_model(model, tokenizer, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir, safe_serialization=True, max_shard_size="5GB")
    tokenizer.save_pretrained(output_dir)


def parse_args():
    parser = argparse.ArgumentParser(
        description="SFT Qwen3-0.6B-Base on problem/answer JSONL pairs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="qwen3-235b-a22b-distillation-answer-pairs.jsonl",
        help="Path to JSONL data with 'problem' and 'answer' fields.",
    )
    parser.add_argument(
        "--strict_jsonl",
        action="store_true",
        help=(
            "Raise on the first malformed JSON line in --data_path instead of "
            "skipping it. By default, malformed lines and rows missing fields "
            "are skipped and counted."
        ),
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="Qwen/Qwen3-0.6B-Base",
        help=(
            "HuggingFace model id (e.g. 'Qwen/Qwen3-0.6B-Base') OR a local "
            "directory containing a downloaded checkpoint. When the value "
            "resolves to an existing local directory with a valid config, "
            "tokenizer, and weights, the script loads fully offline and will "
            "not attempt any network access."
        ),
    )
    parser.add_argument(
        "--force_offline",
        action="store_true",
        help=(
            "Force offline mode regardless of how --model_name_or_path is "
            "interpreted. Sets HF_HUB_OFFLINE=1 and TRANSFORMERS_OFFLINE=1 "
            "for this process so HuggingFace never reaches the network."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help=(
            "Full directory where the final HuggingFace checkpoint is saved. "
            "If omitted, a name is auto-generated from the model basename, "
            "data filename stem, batch size, gradient checkpointing flag, and "
            "attention backend, placed under --output_base_dir. When set, this "
            "value is used as-is and --output_base_dir is ignored."
        ),
    )
    parser.add_argument(
        "--output_base_dir",
        type=str,
        default=".",
        help=(
            "Parent directory that holds auto-generated run folders. Only used "
            "when --output_dir is not provided. Defaults to the current "
            "directory; pass e.g. 'checkpoints' to group runs under "
            "./checkpoints/<auto-name>."
        ),
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--per_device_train_batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=2560,
        help="Drop examples whose prompt + answer + EOS token length exceeds this.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument(
        "--save_every_epoch",
        action="store_true",
        help=(
            "Also save a checkpoint after each epoch. By default, only the "
            "final checkpoint is saved."
        ),
    )
    parser.add_argument(
        "--no_save_every_epoch",
        dest="save_every_epoch",
        action="store_false",
        default=False,
        help="Disable per-epoch checkpoints; only save the final checkpoint.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Enable gradient checkpointing to reduce activation memory.",
    )
    parser.add_argument(
        "--num_trainable_layers",
        type=int,
        default=None,
        help=(
            "If set to an integer n (n >= 0), freeze every parameter and then "
            "unfreeze only the last n transformer blocks plus the final "
            "RMSNorm. The output head (lm_head) is also unfrozen IFF the "
            "model does not tie input/output embeddings. For Qwen3-0.6B-Base "
            "(which has tie_word_embeddings=True), lm_head.weight IS "
            "model.embed_tokens.weight, so the head is kept FROZEN to avoid "
            "implicitly training the ~156M-parameter embedding matrix; the "
            "tied head still produces logits via the (frozen) embedding and "
            "the last-n blocks + final norm carry the trainable capacity. "
            "When n=0 with tied weights, only the final RMSNorm is trained. "
            "When n equals the total number of transformer layers, every "
            "block is trainable (the optimizer is still rebuilt to skip any "
            "frozen tensors). If this flag is not provided, all parameters "
            "are trained as before."
        ),
    )
    parser.add_argument(
        "--compile_model",
        action="store_true",
        help=(
            "Wrap the model with torch.compile after moving it to the target "
            "device. This can speed up long H100/B200 runs, but the first "
            "steps are slower while kernels are compiled."
        ),
    )
    parser.add_argument(
        "--no_compile_model",
        dest="compile_model",
        action="store_false",
        default=False,
        help="Disable torch.compile. This is the default.",
    )
    parser.add_argument(
        "--compile_mode",
        type=str,
        default="max-autotune",
        choices=["default", "reduce-overhead", "max-autotune"],
        help="torch.compile mode used when --compile_model is set.",
    )
    parser.add_argument(
        "--compile_fullgraph",
        action="store_true",
        help="Pass fullgraph=True to torch.compile.",
    )
    parser.add_argument(
        "--pad_to_max_seq_len",
        action="store_true",
        help=(
            "Pad every batch to --max_seq_len. This wastes some compute but "
            "keeps sequence shapes stable, which can make torch.compile faster "
            "and avoid repeated recompilation."
        ),
    )
    parser.add_argument(
        "--attn_implementation",
        type=str,
        default="sdpa",
        choices=["eager", "sdpa", "flash_attention_2"],
        help="Attention backend passed to from_pretrained.",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="bf16-mixed",
        choices=["bf16-mixed", "fp32"],
        help="Use bf16 autocast with fp32 weights, or pure fp32.",
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="Pass trust_remote_code=True to HuggingFace loaders.",
    )
    parser.add_argument(
        "--allow_cpu",
        action="store_true",
        help="Allow running without CUDA for tiny smoke tests only.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.epochs <= 0:
        raise ValueError("--epochs must be positive.")
    if args.per_device_train_batch_size <= 0:
        raise ValueError("--per_device_train_batch_size must be positive.")
    if args.gradient_accumulation_steps <= 0:
        raise ValueError("--gradient_accumulation_steps must be positive.")
    if args.max_seq_len <= 1:
        raise ValueError("--max_seq_len must be greater than 1.")

    if not args.output_dir:
        args.output_dir = auto_output_dir(args)
        print(f"Auto-generated --output_dir: {args.output_dir}", flush=True)

    seed_everything(args.seed)
    device = pick_device(allow_cpu=args.allow_cpu)

    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    resolved_model_path, is_local_model = resolve_model_source(
        args.model_name_or_path
    )


    use_offline = is_local_model or args.force_offline
    if use_offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        print(
            f"Loading model fully offline from local path: {resolved_model_path}"
            if is_local_model
            else "Forcing offline mode for HuggingFace loaders.",
            flush=True,
        )

    tokenizer = AutoTokenizer.from_pretrained(
        resolved_model_path,
        use_fast=True,
        trust_remote_code=args.trust_remote_code,
        local_files_only=use_offline,
    )
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
        else:
            tokenizer.pad_token = tokenizer.eos_token

    rows, bad_rows, bad_json = load_answer_pair_jsonl(
        args.data_path, strict=args.strict_jsonl
    )
    examples, tokenization_stats = tokenize_examples(
        rows=rows,
        tokenizer=tokenizer,
        max_seq_len=args.max_seq_len,
    )
    if not examples:
        raise RuntimeError(
            "No training examples remained after tokenization/filtering."
        )

    avg_len = sum(x["length"] for x in examples) / len(examples)
    max_len = max(x["length"] for x in examples)
    avg_answer_len = sum(x["answer_tokens"] for x in examples) / len(examples)
    print(f"Loaded JSONL rows: {len(rows)}")
    print(f"Rows skipped for missing/empty fields: {bad_rows}")
    print(f"Rows skipped for malformed JSON: {bad_json}")
    print(f"Kept tokenized examples: {tokenization_stats['kept']}")
    print(f"Dropped empty tokenized examples: {tokenization_stats['dropped_empty']}")
    print(
        f"Dropped examples over max_seq_len={args.max_seq_len}: "
        f"{tokenization_stats['dropped_too_long']}"
    )
    print(f"Average total tokens: {avg_len:.1f}; max total tokens: {max_len}")
    print(f"Average supervised answer tokens: {avg_answer_len:.1f}")

    dataset = SFTDataset(examples)
    collator = SFTDataCollator(
        pad_token_id=tokenizer.pad_token_id,
        max_length=args.max_seq_len if args.pad_to_max_seq_len else None,
    )
    generator = torch.Generator()
    generator.manual_seed(args.seed)
    dataloader = DataLoader(
        dataset,
        batch_size=args.per_device_train_batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        generator=generator,
        drop_last=False
    )

    model = AutoModelForCausalLM.from_pretrained(
        resolved_model_path,
        torch_dtype=torch.float32,
        attn_implementation=args.attn_implementation,
        trust_remote_code=args.trust_remote_code,
        local_files_only=use_offline,
    )
    if len(tokenizer) > model.get_input_embeddings().num_embeddings:
        model.resize_token_embeddings(len(tokenizer))
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False


    freeze_for_last_n_layers(model, args.num_trainable_layers)

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()


        if (
            args.num_trainable_layers is not None
            and hasattr(model, "enable_input_require_grads")
        ):
            model.enable_input_require_grads()
    model.to(device)
    if args.compile_model:
        if not hasattr(torch, "compile"):
            raise RuntimeError("torch.compile is unavailable in this PyTorch build.")
        compile_kwargs = {
            "mode": args.compile_mode,
            "fullgraph": args.compile_fullgraph,
            "dynamic": not args.pad_to_max_seq_len,
        }
        print(f"Compiling model with torch.compile({compile_kwargs})", flush=True)
        model = torch.compile(model, **compile_kwargs)
    model.train()


    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params:
        raise RuntimeError(
            "No parameters are trainable. Check --num_trainable_layers."
        )
    try:
        optimizer = torch.optim.SGD(
            trainable_params,
            lr=args.lr,

            fused=(device.type == "cuda"),
        )
    except TypeError:
        optimizer = torch.optim.SGD(
            trainable_params,
            lr=args.lr,

        )

    steps_per_epoch = math.ceil(len(dataloader) / args.gradient_accumulation_steps)
    total_update_steps = steps_per_epoch * args.epochs
    warmup_steps = int(total_update_steps * args.warmup_ratio)
    scheduler = get_constant_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,

    )

    use_bf16 = args.precision == "bf16-mixed" and device.type == "cuda"

    def autocast_context():
        if use_bf16:
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        return nullcontext()

    print(
        f"Model: {resolved_model_path} "
        f"({'local checkpoint, offline' if is_local_model else 'remote / hub id'})"
    )
    print(f"Device: {device}")
    print(f"Attention implementation: {args.attn_implementation}")
    print(f"Precision: {args.precision}")
    print(f"torch.compile enabled: {args.compile_model}")
    print(f"Pad every batch to max_seq_len: {args.pad_to_max_seq_len}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.per_device_train_batch_size}")
    print(f"Gradient accumulation steps: {args.gradient_accumulation_steps}")
    print(f"Optimizer update steps: {total_update_steps}")
    print(f"Warmup steps: {warmup_steps}")
    print(f"Output directory: {args.output_dir}")

    optimizer.zero_grad(set_to_none=True)
    global_step = 0
    running_loss = 0.0
    running_updates = 0
    running_tokens = 0
    start_time = time.time()
    last_log_time = start_time

    for epoch in range(1, args.epochs + 1):
        epoch_loss_sum = 0.0
        epoch_updates = 0
        accum_loss_sum = 0.0
        accum_batches = 0

        for batch_idx, batch in enumerate(dataloader, start=1):
            batch = {
                key: value.to(device, non_blocking=True)
                for key, value in batch.items()
            }
            supervised_tokens = batch["labels"].ne(IGNORE_INDEX).sum().item()

            with autocast_context():
                outputs = model(**batch)
                loss = outputs.loss
                scaled_loss = loss / args.gradient_accumulation_steps

            scaled_loss.backward()
            accum_loss_sum += loss.detach().float().item()
            accum_batches += 1
            running_tokens += supervised_tokens

            should_update = (
                batch_idx % args.gradient_accumulation_steps == 0
                or batch_idx == len(dataloader)
            )
            if not should_update:
                continue

            if args.max_grad_norm and args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            global_step += 1
            epoch_updates += 1
            running_updates += 1
            loss_value = accum_loss_sum / max(accum_batches, 1)
            epoch_loss_sum += loss_value
            running_loss += loss_value
            accum_loss_sum = 0.0
            accum_batches = 0

            if args.log_every and global_step % args.log_every == 0:
                now = time.time()
                elapsed = now - start_time
                step_rate = global_step / max(elapsed, 1e-9)
                remaining_steps = total_update_steps - global_step
                eta = remaining_steps / max(step_rate, 1e-9)
                tok_per_sec = running_tokens / max(now - last_log_time, 1e-9)
                avg_loss = running_loss / max(running_updates, 1)
                lr = scheduler.get_last_lr()[0]
                print(
                    f"[epoch {epoch}/{args.epochs} step "
                    f"{global_step}/{total_update_steps}] "
                    f"loss={avg_loss:.4f} lr={lr:.3e} "
                    f"tok/s={tok_per_sec:.1f} eta={format_eta(eta)}",
                    flush=True,
                )
                running_loss = 0.0
                running_updates = 0
                running_tokens = 0
                last_log_time = now

        epoch_avg_loss = epoch_loss_sum / max(epoch_updates, 1)
        print(
            f"Epoch {epoch} complete. avg_train_loss={epoch_avg_loss:.4f}",
            flush=True,
        )
        if args.save_every_epoch:
            checkpoint_dir = Path(args.output_dir) / f"checkpoint-epoch-{epoch}"
            save_model(model, tokenizer, checkpoint_dir)
            print(f"Saved epoch checkpoint to {checkpoint_dir}", flush=True)

    save_model(model, tokenizer, args.output_dir)
    print(f"Saved final model to {args.output_dir}", flush=True)

    if torch.cuda.is_available():
        peak_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
        print(f"Peak CUDA memory allocated: {peak_gb:.2f} GB", flush=True)


if __name__ == "__main__":
    main()
