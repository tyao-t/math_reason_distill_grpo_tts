from __future__ import annotations

import argparse
import json
import math
import os
import random
import statistics
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from grpo_reward import compute_reward, render_prompt


@dataclass
class Rollout:

    token_ids: List[int]
    logprobs: List[float]
    text: str
    finish_reason: str

    @property
    def length(self) -> int:
        return len(self.token_ids)


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)


    p.add_argument("--checkpoint_path", required=True,
                   help="HF dir of the SFT checkpoint to start GRPO from.")
    p.add_argument("--output_dir", required=True,
                   help="Where to write checkpoints, metrics, samples.")
    p.add_argument("--data_path", default="math_train.json",
                   help="JSON list with at least 'problem' and 'answer' (training prompts).")
    p.add_argument("--eval_data_path", default=None,
                   help="Optional separate JSON for held-out eval. If set, "
                        "the trainer takes the FIRST --eval_holdout entries "
                        "from this file as the held-out shard, and uses ALL "
                        "of --data_path for training. If not set, falls back "
                        "to splitting --data_path's last --eval_holdout rows.")


    p.add_argument("--eval_holdout", type=int, default=20,
                   help="Hold this many problems out as an in-loop eval set "
                        "(taken from the *end* of the file). 0 disables.")
    p.add_argument("--shuffle_train", action="store_true",
                   help="Shuffle the training shard each epoch.")


    p.add_argument("--num_rollouts", type=int, default=8,
                   help="G: completions per prompt for advantage computation.")
    p.add_argument("--prompts_per_step", type=int, default=2,
                   help="M: distinct prompts per optimizer step (M*G total "
                        "rollouts in the training batch). Bigger = more "
                        "diverse gradient, more memory.")
    p.add_argument("--max_new_tokens", type=int, default=4096)
    p.add_argument("--max_prompt_tokens", type=int, default=1024)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top_p", type=float, default=1.0)


    p.add_argument("--eps_low", type=float, default=0.2,
                   help="PPO clip lower epsilon (1-eps_low).")
    p.add_argument("--eps_high", type=float, default=0.28,
                   help="PPO clip upper epsilon (1+eps_high). DAPO 'clip-higher'.")
    p.add_argument("--no_std_normalize", action="store_true",
                   help="Dr.GRPO: don't divide group advantages by std.")
    p.add_argument("--kl_coef", type=float, default=0.0,
                   help=">0 enables a KL penalty toward a frozen reference.")
    p.add_argument("--max_resample", type=int, default=2,
                   help="If a group has zero reward variance, draw a fresh "
                        "prompt up to this many times before giving up.")
    p.add_argument("--inner_epochs", type=int, default=1,
                   help="Optimizer steps to take per fresh batch of rollouts.")


    p.add_argument("--lr", type=float, default=5e-7)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--warmup_steps", type=int, default=10)
    p.add_argument("--gradient_checkpointing", action="store_true")
    p.add_argument("--attn_impl", default="sdpa",
                   choices=["eager", "sdpa", "flash_attention_2"])
    p.add_argument("--micro_batch_size", type=int, default=4,
                   help="Split the (M*G) rollouts into micro-batches of this "
                        "size and accumulate gradients. Default 4 keeps peak "
                        "logits memory in check. Set to 0 to disable splitting.")


    p.add_argument("--max_steps", type=int, default=500,
                   help="Hard upper bound on step count. Training stops "
                        "at min(max_steps, target_prompts_trained reached).")
    p.add_argument("--target_prompts_trained", type=int, default=0,
                   help="Stop training when cumulative trained prompts "
                        "reaches this number. 0 = use max_steps only.")
    p.add_argument("--save_every", type=int, default=25,
                   help="Save checkpoint every K steps. Ignored if "
                        "--save_every_prompts > 0.")
    p.add_argument("--save_every_prompts", type=int, default=0,
                   help="Save checkpoint each time cumulative trained "
                        "prompts crosses a multiple of K. Overrides "
                        "--save_every when > 0.")
    p.add_argument("--eval_every", type=int, default=25,
                   help="Run held-out eval (greedy, in-process) every K steps. 0=off.")
    p.add_argument("--eval_max_new_tokens", type=int, default=4096)
    p.add_argument("--log_every", type=int, default=1)
    p.add_argument("--print_samples_every", type=int, default=10)


    p.add_argument("--seed", type=int, default=442)
    return p.parse_args()


def load_math_train(path: str) -> List[dict]:
    with open(path) as f:
        data = json.load(f)
    out = []
    for row in data:
        problem = row.get("problem")
        answer = row.get("answer")
        if problem and answer is not None:
            out.append({"problem": problem, "answer": str(answer)})
    return out


def build_prompt_token_ids(
    tokenizer, problem: str, max_prompt_tokens: int
) -> List[int]:
    text = render_prompt(problem)
    ids = tokenizer(text, add_special_tokens=False, truncation=False)["input_ids"]
    if len(ids) > max_prompt_tokens:
        ids = ids[-max_prompt_tokens:]
    return list(ids)


def group_relative_advantages(
    rewards: Sequence[float], normalize_std: bool
) -> torch.Tensor:
    r = torch.tensor(rewards, dtype=torch.float32)
    centered = r - r.mean()
    if normalize_std:
        centered = centered / (r.std(unbiased=False) + 1e-4)
    return centered


@dataclass
class GRPOBatch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    completion_mask: torch.Tensor
    old_logprobs: torch.Tensor
    advantages: torch.Tensor


def build_batch(
    prompts: List[List[int]],
    rollouts_per_prompt: List[List[Rollout]],
    advantages_per_prompt: List[torch.Tensor],
    pad_token_id: int,
    device: torch.device,
) -> GRPOBatch:
    rows: List[Tuple[List[int], List[int], List[float]]] = []
    advs: List[float] = []
    for prompt_ids, group_rollouts, group_advs in zip(
        prompts, rollouts_per_prompt, advantages_per_prompt
    ):
        for rollout, a in zip(group_rollouts, group_advs.tolist()):
            rows.append((prompt_ids, rollout.token_ids, rollout.logprobs))
            advs.append(a)

    N = len(rows)
    max_len = max(len(p) + len(c) for p, c, _ in rows)

    input_ids = torch.full((N, max_len), pad_token_id, dtype=torch.long)
    attn_mask = torch.zeros((N, max_len), dtype=torch.long)
    comp_mask = torch.zeros((N, max_len), dtype=torch.float32)
    old_lp = torch.zeros((N, max_len), dtype=torch.float32)

    for i, (p_ids, c_ids, c_lp) in enumerate(rows):
        pl = len(p_ids)
        cl = len(c_ids)
        input_ids[i, :pl] = torch.tensor(p_ids, dtype=torch.long)
        if cl:
            input_ids[i, pl:pl + cl] = torch.tensor(c_ids, dtype=torch.long)
            old_lp[i, pl:pl + cl] = torch.tensor(c_lp, dtype=torch.float32)
            comp_mask[i, pl:pl + cl] = 1.0
        attn_mask[i, :pl + cl] = 1

    advantages = torch.tensor(advs, dtype=torch.float32)

    return GRPOBatch(
        input_ids=input_ids.to(device),
        attention_mask=attn_mask.to(device),
        completion_mask=comp_mask.to(device),
        old_logprobs=old_lp.to(device),
        advantages=advantages.to(device),
    )


def split_micro_batches(batch: GRPOBatch, mb_size: int) -> List[GRPOBatch]:
    if mb_size <= 0 or mb_size >= batch.input_ids.size(0):
        return [batch]
    out = []
    N = batch.input_ids.size(0)
    for s in range(0, N, mb_size):
        e = min(s + mb_size, N)
        out.append(GRPOBatch(
            input_ids=batch.input_ids[s:e],
            attention_mask=batch.attention_mask[s:e],
            completion_mask=batch.completion_mask[s:e],
            old_logprobs=batch.old_logprobs[s:e],
            advantages=batch.advantages[s:e],
        ))
    return out


def selected_token_logprobs(
    logits: torch.Tensor, targets: torch.Tensor
) -> torch.Tensor:
    log_probs = F.log_softmax(logits, dim=-1)
    return log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)


def selected_entropy(logits: torch.Tensor) -> torch.Tensor:
    log_probs = F.log_softmax(logits, dim=-1)
    probs = log_probs.exp()
    return -(probs * log_probs).sum(dim=-1)


def grpo_token_loss_and_metrics(
    new_logp: torch.Tensor,
    old_logp: torch.Tensor,
    shift_mask: torch.Tensor,
    advantages: torch.Tensor,
    eps_low: float,
    eps_high: float,
    return_token_count: bool = True,
):
    log_ratio = new_logp - old_logp

    log_ratio = log_ratio * shift_mask
    ratio = torch.exp(log_ratio)


    ratio = torch.clamp(ratio, max=1.0 + eps_high)

    adv_b = advantages.unsqueeze(1)
    unclipped = ratio * adv_b
    clipped = torch.clamp(ratio, 1.0 - eps_low, 1.0 + eps_high) * adv_b


    surrogate = torch.minimum(unclipped, clipped)

    masked = -surrogate * shift_mask
    token_count = shift_mask.sum().clamp(min=1.0)

    pg_loss_sum = masked.sum()

    with torch.no_grad():
        denom = token_count
        ratio_mean = float(((ratio * shift_mask).sum() / denom).item())

        approx_kl = ((ratio - 1) - log_ratio) * shift_mask
        approx_kl_val = float((approx_kl.sum() / denom).item())
        clip_frac = (
            ((ratio < (1 - eps_low)) | (ratio > (1 + eps_high))).float()
            * shift_mask
        ).sum() / denom

    metrics = {
        "ratio_mean": ratio_mean,
        "approx_kl": approx_kl_val,
        "clip_frac": float(clip_frac.item()),
    }
    if return_token_count:
        return pg_loss_sum, metrics, token_count
    return pg_loss_sum, metrics


def k3_kl_sum(
    new_logp: torch.Tensor, ref_logp: torch.Tensor, mask: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    d = ref_logp - new_logp
    per_tok = torch.exp(d) - d - 1.0
    masked = per_tok * mask
    return masked.sum(), mask.sum().clamp(min=1.0)


def _trim_at_first_eos(
    token_ids: List[int], logprobs: List[float], eos_id: int
) -> Tuple[List[int], List[float], str]:
    for j, t in enumerate(token_ids):
        if t == eos_id:
            return token_ids[: j + 1], logprobs[: j + 1], "stop"
    return token_ids, logprobs, "length"


@torch.no_grad()
def hf_sample_rollouts(
    model,
    tokenizer,
    prompt_token_ids: List[int],
    n: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    device: torch.device,
    seed: Optional[int] = None,
) -> List[Rollout]:
    was_training = model.training
    model.eval()

    prompt = torch.tensor(prompt_token_ids, device=device, dtype=torch.long)
    prompt_len = prompt.numel()
    input_ids = prompt.unsqueeze(0).expand(n, -1).contiguous()
    attn_mask = torch.ones_like(input_ids)

    pad_id = (
        tokenizer.pad_token_id
        if tokenizer.pad_token_id is not None
        else tokenizer.eos_token_id
    )
    eos_id = tokenizer.eos_token_id


    del seed

    out = model.generate(
        input_ids=input_ids,
        attention_mask=attn_mask,
        do_sample=True,
        temperature=max(float(temperature), 1e-6),
        top_p=float(top_p),
        max_new_tokens=int(max_new_tokens),
        return_dict_in_generate=True,
        output_scores=True,
        pad_token_id=pad_id,
        eos_token_id=eos_id,
        use_cache=True,
    )

    if was_training:
        model.train()

    sequences = out.sequences

    if not out.scores:
        return [Rollout([], [], "", "length") for _ in range(n)]


    gen_tokens = sequences[:, prompt_len:]
    gen_logp_steps: List[torch.Tensor] = []
    for t, score_t in enumerate(out.scores):
        lp_t = F.log_softmax(score_t.float(), dim=-1)
        tok_t = gen_tokens[:, t:t + 1]
        gen_logp_steps.append(lp_t.gather(-1, tok_t).squeeze(-1))
        del lp_t
    gen_logp = torch.stack(gen_logp_steps, dim=1)

    rollouts: List[Rollout] = []
    for i in range(n):
        seq = gen_tokens[i].tolist()
        lp = gen_logp[i].tolist()
        seq, lp, finish = _trim_at_first_eos(seq, lp, eos_id)
        text = tokenizer.decode(seq, skip_special_tokens=False)
        rollouts.append(Rollout(token_ids=seq, logprobs=lp, text=text, finish_reason=finish))

    return rollouts


@torch.no_grad()
def hf_greedy_one(
    model,
    tokenizer,
    prompt_token_ids: List[int],
    max_new_tokens: int,
    device: torch.device,
) -> Rollout:
    was_training = model.training
    model.eval()

    prompt = torch.tensor(prompt_token_ids, device=device, dtype=torch.long).unsqueeze(0)
    attn_mask = torch.ones_like(prompt)
    pad_id = (
        tokenizer.pad_token_id
        if tokenizer.pad_token_id is not None
        else tokenizer.eos_token_id
    )
    eos_id = tokenizer.eos_token_id

    out = model.generate(
        input_ids=prompt,
        attention_mask=attn_mask,
        do_sample=False,
        max_new_tokens=int(max_new_tokens),
        return_dict_in_generate=True,
        pad_token_id=pad_id,
        eos_token_id=eos_id,
        use_cache=True,
    )

    if was_training:
        model.train()

    seq = out.sequences[0, prompt.size(1):].tolist()
    seq, _, finish = _trim_at_first_eos(seq, [0.0] * len(seq), eos_id)
    text = tokenizer.decode(seq, skip_special_tokens=False)
    return Rollout(token_ids=seq, logprobs=[], text=text, finish_reason=finish)


def run_eval_local(
    model,
    tokenizer,
    eval_data: List[dict],
    max_prompt_tokens: int,
    max_new_tokens: int,
    device: torch.device,
) -> Tuple[float, int, int, float]:
    correct = 0
    total = 0
    total_len = 0
    for row in eval_data:
        prompt_ids = build_prompt_token_ids(tokenizer, row["problem"], max_prompt_tokens)
        r = hf_greedy_one(model, tokenizer, prompt_ids, max_new_tokens, device)
        score = compute_reward(r.text, row["answer"])
        correct += int(score >= 0.5)
        total += 1
        total_len += len(r.token_ids)
    acc = correct / max(1, total)
    mean_len = total_len / max(1, total)
    return acc, correct, total, mean_len


class CSVLogger:
    def __init__(self, path: Path, fields: List[str]):
        self.path = path
        self.fields = fields
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            with path.open("w") as f:
                f.write(",".join(fields) + "\n")

    def write(self, row: dict):
        with self.path.open("a") as f:
            f.write(",".join(str(row.get(k, "")) for k in self.fields) + "\n")


class CollapseMonitor:
    def __init__(
        self,
        len_window: int = 50,
        len_floor: int = 32,
        entropy_window: int = 50,
        entropy_floor: float = 0.2,
        var_window: int = 100,
        var_floor: float = 0.01,
    ):
        self.len_q = deque(maxlen=len_window)
        self.ent_q = deque(maxlen=entropy_window)
        self.rew_q = deque(maxlen=var_window)
        self.len_floor = len_floor
        self.entropy_floor = entropy_floor
        self.var_floor = var_floor

    def update(self, mean_len: float, mean_ent: float, rewards: Sequence[float]):
        self.len_q.append(mean_len)
        self.ent_q.append(mean_ent)
        self.rew_q.extend(rewards)

    def warnings(self) -> List[str]:
        msgs = []
        if (
            len(self.len_q) >= self.len_q.maxlen
            and statistics.mean(self.len_q) < self.len_floor
        ):
            msgs.append(
                f"WARN: mean response length over last {self.len_q.maxlen} "
                f"steps is {statistics.mean(self.len_q):.1f} < {self.len_floor}"
                " -- possible mode collapse"
            )
        if (
            len(self.ent_q) >= self.ent_q.maxlen
            and statistics.mean(self.ent_q) < self.entropy_floor
        ):
            msgs.append(
                f"WARN: mean per-token entropy over last {self.ent_q.maxlen} "
                f"steps is {statistics.mean(self.ent_q):.3f} < "
                f"{self.entropy_floor} -- entropy collapsed"
            )
        if len(self.rew_q) >= self.rew_q.maxlen:
            v = statistics.pvariance(self.rew_q)
            if v < self.var_floor:
                msgs.append(
                    f"WARN: reward variance over last {self.rew_q.maxlen} "
                    f"rollouts is {v:.4f} < {self.var_floor} -- no learning "
                    "signal"
                )
        return msgs


@dataclass
class SampledGroup:
    row: dict
    prompt_idx: int
    prompt_ids: List[int]
    rollouts: List[Rollout]
    rewards: List[float]
    advantages: torch.Tensor


def sample_one_group_with_resample(
    model,
    tokenizer,
    train_data: List[dict],
    next_idx: int,
    args,
    device: torch.device,
    seed: int,
) -> Tuple[Optional[SampledGroup], int, int]:
    last_group: Optional[SampledGroup] = None
    resamples = 0
    for attempt in range(args.max_resample + 1):
        idx = next_idx % len(train_data)
        next_idx += 1
        row = train_data[idx]
        prompt_ids = build_prompt_token_ids(
            tokenizer, row["problem"], args.max_prompt_tokens
        )
        rollouts = hf_sample_rollouts(
            model=model,
            tokenizer=tokenizer,
            prompt_token_ids=prompt_ids,
            n=args.num_rollouts,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            device=device,
            seed=seed + attempt,
        )
        rewards = [compute_reward(r.text, row["answer"]) for r in rollouts]
        advs = group_relative_advantages(rewards, normalize_std=not args.no_std_normalize)
        last_group = SampledGroup(
            row=row,
            prompt_idx=idx,
            prompt_ids=prompt_ids,
            rollouts=rollouts,
            rewards=rewards,
            advantages=advs,
        )
        if advs.abs().max().item() > 1e-6:
            return last_group, next_idx, resamples
        resamples += 1
    return None, next_idx, resamples


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("GRPO training expects a CUDA device")

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)


    print(f"[trainer] loading tokenizer + model from {args.checkpoint_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint_path,
        torch_dtype=torch.bfloat16,
        attn_implementation=args.attn_impl,
    )
    model.to(device)
    model.train()
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    model.config.use_cache = False


    ref_model = None
    if args.kl_coef > 0:
        print(f"[trainer] kl_coef={args.kl_coef} > 0: loading frozen reference")
        ref_model = AutoModelForCausalLM.from_pretrained(
            args.checkpoint_path,
            torch_dtype=torch.bfloat16,
            attn_implementation=args.attn_impl,
        )
        ref_model.to(device)
        ref_model.eval()
        for p in ref_model.parameters():
            p.requires_grad_(False)


    train_full = load_math_train(args.data_path)
    if args.eval_data_path:


        eval_full = load_math_train(args.eval_data_path)
        eval_data = (
            eval_full[: args.eval_holdout]
            if args.eval_holdout > 0
            else []
        )
        train_data = train_full
    elif args.eval_holdout > 0 and args.eval_holdout < len(train_full):

        train_data = train_full[: -args.eval_holdout]
        eval_data = train_full[-args.eval_holdout:]
    else:
        train_data = train_full
        eval_data = []
    print(f"[trainer] train shard: {len(train_data)}  |  eval shard: {len(eval_data)}")

    if args.shuffle_train:
        random.shuffle(train_data)


    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay,
    )

    def lr_at(step: int) -> float:
        if args.warmup_steps > 0 and step < args.warmup_steps:
            return args.lr * (step + 1) / args.warmup_steps
        return args.lr


    csv_fields = [
        "step", "n_prompts", "n_rollouts",
        "loss", "pg_loss", "kl_loss",
        "reward_mean", "reward_std", "advantage_std",
        "mean_response_len", "frac_truncated",
        "mean_entropy", "ratio_mean", "approx_kl", "clip_frac",
        "grad_norm", "lr", "resamples",
        "wall_s",
    ]
    csv_log = CSVLogger(out / "metrics.csv", csv_fields)
    eval_csv = CSVLogger(
        out / "eval.csv",
        ["step", "accuracy", "correct", "total", "mean_response_len", "wall_s"],
    )
    sample_log_path = out / "samples.jsonl"
    sample_log_path.parent.mkdir(parents=True, exist_ok=True)
    monitor = CollapseMonitor()


    next_idx = 0
    total_prompts_trained = 0
    total_rollouts_trained = 0
    last_save_prompts = 0
    t_start = time.time()
    try:
        for step in range(1, args.max_steps + 1):
            step_t0 = time.time()
            cur_lr = lr_at(step)
            for g in optimizer.param_groups:
                g["lr"] = cur_lr


            groups: List[SampledGroup] = []
            total_resamples = 0
            for k in range(args.prompts_per_step):
                grp, next_idx, resamples = sample_one_group_with_resample(
                    model=model,
                    tokenizer=tokenizer,
                    train_data=train_data,
                    next_idx=next_idx,
                    args=args,
                    device=device,
                    seed=args.seed + step * 100_003 + k * 1009,
                )
                total_resamples += resamples
                if grp is not None:
                    groups.append(grp)

            if not groups:
                wall_s = time.time() - step_t0
                csv_log.write({
                    "step": step,
                    "n_prompts": 0,
                    "n_rollouts": 0,
                    "lr": cur_lr,
                    "resamples": total_resamples,
                    "wall_s": round(wall_s, 3),
                })
                print(
                    f"[step {step:04d}] all {args.prompts_per_step} prompts had "
                    f"identical reward after {total_resamples} resamples; skipping"
                )
                continue


            batch = build_batch(
                prompts=[g.prompt_ids for g in groups],
                rollouts_per_prompt=[g.rollouts for g in groups],
                advantages_per_prompt=[g.advantages for g in groups],
                pad_token_id=tokenizer.pad_token_id,
                device=device,
            )
            num_rollouts_in_batch = batch.input_ids.size(0)
            total_prompts_trained += len(groups)
            total_rollouts_trained += num_rollouts_in_batch


            last_metrics: Dict[str, float] = {}
            last_grad_norm = 0.0
            for inner in range(args.inner_epochs):
                optimizer.zero_grad(set_to_none=True)

                micro_batches = split_micro_batches(batch, args.micro_batch_size)


                with torch.no_grad():
                    pg_token_total = sum(
                        mb.completion_mask[:, 1:].sum() for mb in micro_batches
                    ).clamp(min=1.0)
                    if ref_model is not None:
                        kl_token_total = pg_token_total
                    else:
                        kl_token_total = None


                sum_pg_loss = torch.zeros((), device=device)
                sum_kl_loss = torch.zeros((), device=device)
                sum_entropy_num = torch.zeros((), device=device)
                sum_ratio_w = 0.0
                sum_kl_w = 0.0
                sum_clip_w = 0.0
                w_sum = 0

                for mb in micro_batches:
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        outputs = model(
                            input_ids=mb.input_ids,
                            attention_mask=mb.attention_mask,
                            use_cache=False,
                        )
                        logits = outputs.logits
                        shift_logits = logits[:, :-1, :]
                        shift_targets = mb.input_ids[:, 1:]
                        shift_mask = mb.completion_mask[:, 1:]
                        shift_old = mb.old_logprobs[:, 1:]

                        new_logp = selected_token_logprobs(
                            shift_logits, shift_targets
                        )
                        pg_sum, ratio_metrics, mb_tok = grpo_token_loss_and_metrics(
                            new_logp=new_logp,
                            old_logp=shift_old,
                            shift_mask=shift_mask,
                            advantages=mb.advantages,
                            eps_low=args.eps_low,
                            eps_high=args.eps_high,
                        )


                        pg_loss_mb = pg_sum / pg_token_total

                        kl_loss_mb = torch.zeros((), device=device)
                        if ref_model is not None and args.kl_coef > 0:
                            with torch.no_grad():
                                ref_outputs = ref_model(
                                    input_ids=mb.input_ids,
                                    attention_mask=mb.attention_mask,
                                    use_cache=False,
                                )
                                ref_logits = ref_outputs.logits[:, :-1, :]
                                ref_logp = selected_token_logprobs(
                                    ref_logits, shift_targets
                                )
                            kl_sum, _ = k3_kl_sum(new_logp, ref_logp, shift_mask)
                            kl_loss_mb = kl_sum / kl_token_total

                        with torch.no_grad():
                            ent_per_tok = selected_entropy(shift_logits)
                            sum_entropy_num = sum_entropy_num + (
                                ent_per_tok * shift_mask
                            ).sum()

                        loss_mb = pg_loss_mb + args.kl_coef * kl_loss_mb

                    loss_mb.backward()

                    sum_pg_loss = sum_pg_loss + pg_loss_mb.detach()
                    sum_kl_loss = sum_kl_loss + kl_loss_mb.detach()

                    w = float(mb_tok.item())
                    sum_ratio_w += ratio_metrics["ratio_mean"] * w
                    sum_kl_w += ratio_metrics["approx_kl"] * w
                    sum_clip_w += ratio_metrics["clip_frac"] * w
                    w_sum += w


                    del outputs, logits, shift_logits, new_logp
                    if "ref_outputs" in locals():
                        del ref_outputs

                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), args.max_grad_norm
                )
                optimizer.step()

                last_metrics = {
                    "pg_loss": float(sum_pg_loss.item()),
                    "kl_loss": float(sum_kl_loss.item()),
                    "loss": float(sum_pg_loss.item() + args.kl_coef * sum_kl_loss.item()),
                    "mean_entropy": float(
                        (sum_entropy_num / pg_token_total).item()
                    ),
                    "ratio_mean": sum_ratio_w / max(1.0, w_sum),
                    "approx_kl": sum_kl_w / max(1.0, w_sum),
                    "clip_frac": sum_clip_w / max(1.0, w_sum),
                }
                last_grad_norm = float(
                    grad_norm.item() if torch.is_tensor(grad_norm) else grad_norm
                )


            all_rewards = [r for g in groups for r in g.rewards]
            all_advs = torch.cat([g.advantages for g in groups])
            comp_lens = [
                len(r.token_ids) for g in groups for r in g.rollouts
            ]
            truncated = [
                r.finish_reason == "length" for g in groups for r in g.rollouts
            ]
            mean_resp_len = float(sum(comp_lens) / max(1, len(comp_lens)))
            rew_t = torch.tensor(all_rewards)
            wall_s = time.time() - step_t0
            row_log = {
                "step": step,
                "n_prompts": len(groups),
                "n_rollouts": num_rollouts_in_batch,
                "loss": last_metrics["loss"],
                "pg_loss": last_metrics["pg_loss"],
                "kl_loss": last_metrics["kl_loss"],
                "reward_mean": float(rew_t.mean()),
                "reward_std": float(rew_t.std(unbiased=False)),
                "advantage_std": float(all_advs.std(unbiased=False)),
                "mean_response_len": mean_resp_len,
                "frac_truncated": float(sum(truncated) / max(1, len(truncated))),
                "mean_entropy": last_metrics["mean_entropy"],
                "ratio_mean": last_metrics["ratio_mean"],
                "approx_kl": last_metrics["approx_kl"],
                "clip_frac": last_metrics["clip_frac"],
                "grad_norm": last_grad_norm,
                "lr": cur_lr,
                "resamples": total_resamples,
                "wall_s": round(wall_s, 3),
            }
            csv_log.write(row_log)

            if step % args.log_every == 0:
                print(
                    f"[step {step:04d}] r={row_log['reward_mean']:.3f}±{row_log['reward_std']:.3f} "
                    f"len={mean_resp_len:.0f} "
                    f"ent={last_metrics['mean_entropy']:.3f} "
                    f"loss={last_metrics['loss']:.4f} "
                    f"ratio={last_metrics['ratio_mean']:.3f} "
                    f"clip={last_metrics['clip_frac']:.2f} "
                    f"gn={last_grad_norm:.2f} "
                    f"resmpl={total_resamples} "
                    f"prompts={len(groups)}/{args.prompts_per_step} "
                    f"{wall_s:.1f}s"
                )

            if args.print_samples_every and step % args.print_samples_every == 0:
                with sample_log_path.open("a") as f:
                    for g in groups[:1]:
                        f.write(json.dumps({
                            "step": step,
                            "prompt_idx": g.prompt_idx,
                            "problem": g.row["problem"],
                            "ground_truth": g.row["answer"],
                            "rollouts": [
                                {
                                    "text": r.text,
                                    "reward": g.rewards[i],
                                    "len": len(r.token_ids),
                                    "finish": r.finish_reason,
                                }
                                for i, r in enumerate(g.rollouts)
                            ],
                        }) + "\n")

            monitor.update(mean_resp_len, last_metrics["mean_entropy"], all_rewards)
            for w in monitor.warnings():
                print(w)


            if (
                args.eval_every
                and step % args.eval_every == 0
                and eval_data
            ):
                t_eval = time.time()
                acc, correct, total, mean_len = run_eval_local(
                    model=model,
                    tokenizer=tokenizer,
                    eval_data=eval_data,
                    max_prompt_tokens=args.max_prompt_tokens,
                    max_new_tokens=args.eval_max_new_tokens,
                    device=device,
                )
                eval_wall = time.time() - t_eval
                eval_csv.write({
                    "step": step,
                    "accuracy": f"{acc:.6f}",
                    "correct": correct,
                    "total": total,
                    "mean_response_len": f"{mean_len:.2f}",
                    "wall_s": round(eval_wall, 3),
                })
                print(
                    f"[step {step:04d}] HOLDOUT EVAL acc={acc*100:.2f}% "
                    f"({correct}/{total}) mean_len={mean_len:.0f} in {eval_wall:.1f}s"
                )


            reached_target = (
                args.target_prompts_trained > 0
                and total_prompts_trained >= args.target_prompts_trained
            )
            if args.save_every_prompts > 0:
                trigger_save = (
                    total_prompts_trained // args.save_every_prompts
                    > last_save_prompts // args.save_every_prompts
                )
            else:
                trigger_save = step % args.save_every == 0
            if trigger_save or step == args.max_steps or reached_target:
                ckpt_dir = out / (
                    f"step-{step:05d}-p{total_prompts_trained:05d}"
                    if args.save_every_prompts > 0
                    else f"step-{step:05d}"
                )
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                model.save_pretrained(
                    str(ckpt_dir), safe_serialization=True, max_shard_size="5GB"
                )
                tokenizer.save_pretrained(str(ckpt_dir))
                last_save_prompts = total_prompts_trained
                print(
                    f"[step {step:04d}] saved checkpoint to {ckpt_dir} "
                    f"(cumulative trained: {total_prompts_trained} prompts / "
                    f"{total_rollouts_trained} rollouts)"
                )

            if reached_target:
                print(
                    f"[step {step:04d}] reached target_prompts_trained="
                    f"{args.target_prompts_trained}; stopping"
                )
                break

    except KeyboardInterrupt:
        print("\n[trainer] KeyboardInterrupt; exiting without saving")
        return

    total_t = time.time() - t_start
    print(f"[trainer] done in {total_t/60:.1f} min")

    if torch.cuda.is_available():
        max_mem_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
        print(f"Max CUDA memory allocated: {max_mem_gb:.2f} GB")


if __name__ == "__main__":
    main()
