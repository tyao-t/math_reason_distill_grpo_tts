from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List, Optional

import httpx


@dataclass
class Rollout:

    token_ids: List[int]
    logprobs: List[float]
    text: str
    finish_reason: str

    @property
    def length(self) -> int:
        return len(self.token_ids)


class RolloutClient:
    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8000",
        request_timeout: float = 600.0,
        sync_timeout: float = 600.0,
    ):
        self.base_url = base_url.rstrip("/")
        self._client = httpx.Client(timeout=request_timeout)
        self._sync_timeout = sync_timeout

    def close(self) -> None:
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def wait_until_ready(self, max_wait_s: float = 300.0) -> None:
        deadline = time.time() + max_wait_s
        last_err: Optional[Exception] = None
        while time.time() < deadline:
            try:
                resp = self._client.get(f"{self.base_url}/health", timeout=5.0)
                if resp.status_code == 200:
                    return
            except Exception as err:
                last_err = err
            time.sleep(2.0)
        raise RuntimeError(
            f"vLLM server at {self.base_url} not ready within {max_wait_s}s "
            f"(last error: {last_err!r})"
        )

    def generate(
        self,
        prompt_token_ids: List[int],
        n: int,
        temperature: float,
        top_p: float,
        max_tokens: int,
        stop_token_ids: Optional[List[int]] = None,
        seed: Optional[int] = None,
    ) -> List[Rollout]:
        payload = {
            "prompt_token_ids": list(prompt_token_ids),
            "n": int(n),
            "temperature": float(temperature),
            "top_p": float(top_p),
            "max_tokens": int(max_tokens),
        }
        if stop_token_ids is not None:
            payload["stop_token_ids"] = list(stop_token_ids)
        if seed is not None:
            payload["seed"] = int(seed)
        resp = self._client.post(f"{self.base_url}/generate", json=payload)
        resp.raise_for_status()
        data = resp.json()
        rollouts = []
        for r in data["rollouts"]:
            rollouts.append(
                Rollout(
                    token_ids=r["token_ids"],
                    logprobs=r["logprobs"],
                    text=r["text"],
                    finish_reason=r.get("finish_reason", ""),
                )
            )
        return rollouts

    def sync_weights(self, path: str) -> dict:
        resp = self._client.post(
            f"{self.base_url}/sync_weights",
            json={"path": path},
            timeout=self._sync_timeout,
        )
        resp.raise_for_status()
        return resp.json()
