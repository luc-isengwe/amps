"""Measure the throughput of local LLMs served by Ollama (tier T2).

Why this script exists
----------------------
The number of LLM-driven runs we can afford in the evaluation depends on
how fast a 4-bit model runs on the thesis machine (Ryzen 7 PRO 2700U,
12 GB RAM, no GPU). We must *measure* it before planning the test.

Changes from version 1 (and why)
--------------------------------
1. **Cold prompts.** Version 1 sent the same prompt at every repetition.
   Ollama reuses the cached computation of an identical prompt prefix, so
   the "prompt tok/s" of repetitions 2-5 measured the cache, not the CPU.
   Each request now starts with a unique nonce, which forces the full prompt
   to be processed (``--cache warm`` restores the old behaviour, which is
   useful to measure the benefit of the cache for a fixed system prompt).
2. **One model in memory at a time.** Ollama keeps a model loaded for 5
   minutes after its last request. In version 1 the previous model was still
   in RAM when the next one was loaded, which pushed the machine into swap.
   Every model is now unloaded (``keep_alive: 0``) before the next one starts.
3. **Comparable generation length.** Some models stopped after ~20 tokens
   while others generated the full 128, so wall-clock times were not
   comparable. The prompt now asks for a long answer and the number of
   generated tokens is checked and reported.
4. **Reasoning mode.** Qwen3 models "think" before answering, which costs
   many tokens. ``--think off`` (default) disables it, ``--think on``
   enables it, so both costs can be measured.
5. **Memory checks.** RAM size of the loaded model, available RAM and swap
   growth are recorded, and suspicious rows are flagged in the summary.

Usage
-----
    python scripts/bench_local_llm.py --models qwen3:4b qwen3:8b llama3.1:8b --repeats 3

Only the Python standard library is used. Close other heavy programs while
measuring, and watch ``vmstat 2`` (columns si/so) in a second terminal.
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import statistics
import time
import urllib.error
import urllib.request
import uuid
from datetime import datetime, timezone
from pathlib import Path

# A system prompt similar to what an agent receives (role, tools, rules).
AGENT_PREAMBLE = (
    "You are the retrieval agent of a customer-support system. You can call the tools "
    "web_search(query), kb_search(query), crm_lookup(customer_id) and read_inbox(limit). "
    "Answer with a JSON object {\"tool\": <name>, \"arguments\": {...}} or {\"final\": <text>}. "
    "Never invent customer data. "
)
# Asking for a long answer makes every model reach --num-predict, so that the
# generation phase has the same length for all models.
LONG_ANSWER = (" Before giving the JSON object, explain your plan step by step in detail, "
               "in at least 300 words.")
PROMPTS = {
    "short": AGENT_PREAMBLE + "Task: find the refund policy for annual plans." + LONG_ANSWER,
    "long": AGENT_PREAMBLE * 12 + "Task: summarise the five latest e-mails for the support team."
    + LONG_ANSWER,
}

# Reference agent step used to translate speeds into a duration estimate:
# a context of 1,500 tokens to read and 150 tokens to write.
STEP_PROMPT_TOKENS = 1500
STEP_GEN_TOKENS = 150


def mem_info_gb() -> dict[str, float]:
    """Total / available RAM and swap in GB (Linux only; empty elsewhere)."""
    info = {}
    try:
        with open("/proc/meminfo", encoding="utf-8") as fh:
            for line in fh:
                key, value = line.split(":", 1)
                if key in ("MemTotal", "MemAvailable", "SwapTotal", "SwapFree"):
                    info[key] = round(int(value.split()[0]) / 1024 / 1024, 2)
    except OSError:
        pass
    return info


def swap_used(mem: dict[str, float]) -> float | None:
    return round(mem["SwapTotal"] - mem["SwapFree"], 2) if "SwapTotal" in mem else None


# Ignore any HTTP proxy configured in the environment: the server is local.
_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def api(host: str, path: str, payload: dict | None, timeout: float) -> dict:
    """Call the Ollama HTTP API (POST if payload is given, GET otherwise)."""
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(f"{host}{path}", data=data,
                                 headers={"Content-Type": "application/json"})
    with _OPENER.open(req, timeout=timeout) as resp:
        return json.loads(resp.read() or b"{}")


def unload_all(host: str, timeout: float) -> None:
    """Unload every model currently in memory (keep_alive = 0)."""
    try:
        loaded = api(host, "/api/ps", None, timeout).get("models", [])
    except urllib.error.URLError:
        return
    for m in loaded:
        api(host, "/api/generate", {"model": m["name"], "keep_alive": 0}, timeout)
    time.sleep(2)  # let the operating system reclaim the memory


def model_ram_gb(host: str, model: str, timeout: float) -> float | None:
    """Memory used by ``model`` according to Ollama (/api/ps)."""
    try:
        for m in api(host, "/api/ps", None, timeout).get("models", []):
            if m["name"] == model or m.get("model") == model:
                return round(m.get("size", 0) / 1024 ** 3, 2)
    except urllib.error.URLError:
        pass
    return None


def generate(host: str, model: str, prompt: str, args) -> tuple[dict, float]:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        # Deterministic settings: the goal is to measure speed, not quality.
        "options": {"temperature": 0, "seed": 42, "num_predict": args.num_predict,
                    "num_ctx": args.num_ctx},
    }
    if args.think != "default":
        payload["think"] = args.think == "on"  # ignored by models without reasoning mode
    t0 = time.perf_counter()
    data = api(host, "/api/generate", payload, args.timeout)
    return data, time.perf_counter() - t0


def rate(count, duration_ns) -> float | None:
    if not count or not duration_ns:
        return None
    return count / (duration_ns / 1e9)


def main() -> int:
    ap = argparse.ArgumentParser(description="Throughput benchmark of local Ollama models")
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--num-predict", type=int, default=128, help="tokens to generate")
    ap.add_argument("--num-ctx", type=int, default=4096, help="context window")
    ap.add_argument("--cache", choices=["cold", "warm"], default="cold",
                    help="cold: unique prompt each time (CPU speed); warm: identical prompts")
    ap.add_argument("--think", choices=["off", "on", "default"], default="off")
    ap.add_argument("--host", default="http://localhost:11434")
    ap.add_argument("--timeout", type=float, default=1800)
    ap.add_argument("--out", default="results/bench_local_llm_v2.csv")
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    new_file = not out.exists()
    fields = ["timestamp", "host_cpu", "model", "prompt", "repeat", "cache", "think",
              "wall_s", "load_s", "prompt_tokens", "prompt_tok_s", "gen_tokens", "gen_tok_s",
              "model_ram_gb", "mem_available_gb", "swap_used_gb", "swap_growth_gb"]
    rows: list[dict] = []

    try:
        api(args.host, "/api/version", None, 10)
    except urllib.error.URLError as exc:
        print(f"Cannot reach Ollama at {args.host}: {exc} (is `ollama serve` running?)")
        return 1

    with open(out, "a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        if new_file:
            writer.writeheader()
        for model in args.models:
            unload_all(args.host, args.timeout)          # change 2: one model at a time
            swap_start = swap_used(mem_info_gb())
            for pname, base_prompt in PROMPTS.items():
                for r in range(1, args.repeats + 1):
                    # change 1: a unique prefix defeats the prompt cache
                    prefix = f"[request {uuid.uuid4().hex}] " if args.cache == "cold" else ""
                    try:
                        data, wall = generate(args.host, model, prefix + base_prompt, args)
                    except urllib.error.URLError as exc:
                        print(f"[{model}] request failed: {exc}")
                        return 1
                    mem = mem_info_gb()
                    sw = swap_used(mem)
                    row = {
                        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                        "host_cpu": platform.processor() or platform.machine(),
                        "model": model, "prompt": pname, "repeat": r,
                        "cache": args.cache, "think": args.think,
                        "wall_s": round(wall, 2),
                        "load_s": round(data.get("load_duration", 0) / 1e9, 2),
                        "prompt_tokens": data.get("prompt_eval_count"),
                        "prompt_tok_s": rate(data.get("prompt_eval_count"),
                                             data.get("prompt_eval_duration")),
                        "gen_tokens": data.get("eval_count"),
                        "gen_tok_s": rate(data.get("eval_count"), data.get("eval_duration")),
                        "model_ram_gb": model_ram_gb(args.host, model, 10),
                        "mem_available_gb": mem.get("MemAvailable"),
                        "swap_used_gb": sw,
                        "swap_growth_gb": round(sw - swap_start, 2)
                        if sw is not None and swap_start is not None else None,
                    }
                    writer.writerow(row)
                    fh.flush()
                    rows.append(row)
                    print(f"[{model:>14} | {pname:5} | {r}/{args.repeats}] "
                          f"wall {row['wall_s']:7.1f}s | prompt {row['prompt_tokens']:>4} tok "
                          f"@ {row['prompt_tok_s'] or 0:6.1f} tok/s | gen {row['gen_tokens']:>4} tok "
                          f"@ {row['gen_tok_s'] or 0:5.2f} tok/s | RAM free "
                          f"{row['mem_available_gb']} GB | swap +{row['swap_growth_gb']} GB")
        unload_all(args.host, args.timeout)

    # ------------------------------------------------------------------ summary
    print(f"\nMedians (repeat 1 excluded when repeats > 1). cache={args.cache}, think={args.think}")
    print(f"Estimated agent step = {STEP_PROMPT_TOKENS} tokens read + {STEP_GEN_TOKENS} tokens written.")
    print(f"{'model':>14} {'prompt':>6} {'prompt tok/s':>12} {'gen tok/s':>9} "
          f"{'gen tok':>7} {'RAM GB':>6} {'step s':>7}  flags")
    for model in args.models:
        for pname in PROMPTS:
            sel = [r for r in rows if r["model"] == model and r["prompt"] == pname
                   and (args.repeats == 1 or r["repeat"] > 1)]
            if not sel:
                continue

            def med(k):
                vals = [r[k] for r in sel if r[k] is not None]
                return statistics.median(vals) if vals else float("nan")

            p, g = med("prompt_tok_s"), med("gen_tok_s")
            step = STEP_PROMPT_TOKENS / p + STEP_GEN_TOKENS / g if p and g else float("nan")
            flags = []
            if med("gen_tokens") < 0.9 * args.num_predict:
                flags.append("stopped-early")
            if med("mem_available_gb") < 1.0:
                flags.append("low-RAM")
            if (med("swap_growth_gb") or 0) > 0.2:
                flags.append("swap-growth")
            print(f"{model:>14} {pname:>6} {p:12.1f} {g:9.2f} {med('gen_tokens'):7.0f} "
                  f"{med('model_ram_gb'):6.2f} {step:7.0f}  {','.join(flags) or 'ok'}")
    print(f"\nRaw results appended to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
