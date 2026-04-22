from __future__ import annotations

import argparse
import threading
import time
from pathlib import Path
from typing import List, Optional

import safetensors.torch
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from vllm import LLM, SamplingParams


class GenerateRequest(BaseModel):
    prompt_token_ids: List[int]
    n: int = 1
    temperature: float = 1.0
    top_p: float = 1.0
    max_tokens: int = 2048
    stop_token_ids: Optional[List[int]] = None
    seed: Optional[int] = None


class SyncWeightsRequest(BaseModel):
    path: str


def _reload_weights_in_place(llm: LLM, path: str) -> None:
    path_str = str(path)

    def _do_reload(model):

        from pathlib import Path as _Path
        import safetensors.torch as _st

        p = _Path(path_str)
        shards = sorted(p.glob("*.safetensors"))
        if not shards:
            raise FileNotFoundError(
                f"no *.safetensors under {p} -- trainer didn't save yet?"
            )
        total = 0
        for shard in shards:
            state = _st.load_file(str(shard), device="cpu")
            model.load_weights(iter(state.items()))
            total += len(state)
        return total

    llm.llm_engine.apply_model(_do_reload)


    try:
        engine = llm.llm_engine
        if hasattr(engine, "reset_prefix_cache"):
            engine.reset_prefix_cache()
    except Exception:
        pass


def build_app(llm: LLM, lock: threading.Lock) -> FastAPI:
    app = FastAPI()
    state = {"ready": True, "last_sync_step": 0}

    @app.get("/health")
    def health():
        return {"ok": state["ready"]}

    @app.post("/generate")
    def generate(req: GenerateRequest):
        if not state["ready"]:
            raise HTTPException(503, "engine not ready")
        sp = SamplingParams(
            n=req.n,
            temperature=req.temperature,
            top_p=req.top_p,
            max_tokens=req.max_tokens,
            stop_token_ids=req.stop_token_ids,
            seed=req.seed,
            logprobs=1,
        )
        with lock:
            outputs = llm.generate(
                prompts=None,
                sampling_params=sp,
                prompt_token_ids=[req.prompt_token_ids],
                use_tqdm=False,
            )

        request_output = outputs[0]
        rollouts = []
        for completion in request_output.outputs:
            token_ids = list(completion.token_ids)


            logprobs_field = completion.logprobs or []
            lp_list: List[float] = []
            for tok_id, lp_dict in zip(token_ids, logprobs_field):
                if lp_dict is None:
                    lp_list.append(0.0)
                    continue
                entry = lp_dict.get(tok_id)
                if entry is None and lp_dict:


                    entry = next(iter(lp_dict.values()))
                lp_list.append(float(entry.logprob) if entry is not None else 0.0)
            rollouts.append(
                {
                    "token_ids": token_ids,
                    "logprobs": lp_list,
                    "text": completion.text,
                    "finish_reason": completion.finish_reason or "",
                }
            )
        return {"rollouts": rollouts}

    @app.post("/sync_weights")
    def sync_weights(req: SyncWeightsRequest):
        with lock:
            state["ready"] = False
            try:
                t0 = time.time()
                _reload_weights_in_place(llm, req.path)
                dt = time.time() - t0
                state["last_sync_step"] += 1
                state["ready"] = True
                return {
                    "ok": True,
                    "mode": "in_place_load_weights",
                    "elapsed_s": round(dt, 3),
                    "sync_count": state["last_sync_step"],
                }
            except Exception as err:
                state["ready"] = True
                raise HTTPException(500, f"sync_weights failed: {err!r}") from err

    return app


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="HF checkpoint dir to serve")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=4096,
        help="prompt + max_new_tokens cap (must match trainer's budget)",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.45,
        help="leave headroom for the trainer process if it shares the GPU",
    )
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument(
        "--enable-prefix-caching",
        action="store_true",
        help="off by default -- prefix caches go stale on weight sync",
    )
    args = parser.parse_args()

    print(f"[serve_vllm_grpo] booting vLLM with model={args.model}")
    llm = LLM(
        model=args.model,
        dtype=args.dtype,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        seed=args.seed,
        trust_remote_code=args.trust_remote_code,
        enable_prefix_caching=args.enable_prefix_caching,
    )
    print("[serve_vllm_grpo] engine ready")

    lock = threading.Lock()
    app = build_app(llm, lock)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
