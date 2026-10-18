#!/usr/bin/env python3
"""QUELL Step 14 — end-to-end evaluation of the deployed quantized detector.

Runs the fine-tuned backbone as GGUF through llama.cpp (llama-server, last-token pooling),
applies the trained classification head to the pooled hidden state, and reports:
  * macro-F1 of the deployed detector (all classes / present classes)
  * agreement with the HF-FP16 classifier on the same samples (proxy-validity check)
  * batch-1 latency at the REAL prompt length (per-sample token counts, p50/p95)
Same script on the GPU host (accuracy) and on the Jetson (accuracy + latency; run
tegrastats alongside for power, see scripts/07_jetson_bench.py).

Start the server first (one model at a time), e.g.:
  llama-server -m model-q8_0.gguf --embeddings --pooling last -ngl 99 \
               -c 2112 -b 2112 -ub 2112 --port 8088
Then:
  python 14_gguf_eval.py --export <export_dir> --tag q8_0 --device jetson [--limit 1000]

Note: the head is linear without bias, so the argmax is invariant to the L2 normalisation
the server may apply to pooled embeddings.
"""
import argparse, json, time, urllib.request, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
from quell_common import macro_f1_both

ap = argparse.ArgumentParser()
ap.add_argument("--export", required=True)
ap.add_argument("--tag", required=True, help="f16 | q8_0 | q4_k_m")
ap.add_argument("--device", default="ypb")
ap.add_argument("--url", default="http://127.0.0.1:8088")
ap.add_argument("--limit", type=int, default=0, help="use only the first N subset rows (0 = all)")
ap.add_argument("--warmup", type=int, default=5)
ap.add_argument("--ids-file", default=None, help="cost-only run on these token sequences (e.g. another dataset's real lengths)")
ap.add_argument("--power", action="store_true", help="sample board power with tegrastats (Jetson)")
ap.add_argument("--endpoint", default="embedding", choices=["embedding", "completion"],
                help="completion = cost-only fallback: causal prefill of the whole sequence through /completion with n_predict 0 "
                     "(processed in ubatch chunks; used when the single-pass embedding buffer does not fit the device memory)")
a = ap.parse_args()

EXP = Path(a.export)
meta = json.load(open(EXP / "meta.json")); classes = np.array(meta["classes"])
head = np.load(EXP / "head.npy")
COST_ONLY = a.ids_file is not None
rows = [json.loads(l) for l in open(a.ids_file if COST_ONLY else EXP / "subset.jsonl")]
if a.limit:  # keep stratification: take every k-th row
    rows = rows[::max(1, len(rows) // a.limit)][:a.limit]


def embed(ids):
    if a.endpoint == "completion":
        req = urllib.request.Request(a.url + "/completion", data=json.dumps({"prompt": ids, "n_predict": 0, "cache_prompt": False}).encode(),
                                     headers={"Content-Type": "application/json"})
        r = json.loads(urllib.request.urlopen(req, timeout=600).read())
        n = (r.get("timings") or {}).get("prompt_n"); assert n is None or n >= len(ids) - 1, f"prompt only partly processed: {n}/{len(ids)}"
        return np.zeros(head.shape[1], dtype=np.float32)
    req = urllib.request.Request(a.url + "/embedding", data=json.dumps({"content": ids}).encode(),
                                 headers={"Content-Type": "application/json"})
    r = json.loads(urllib.request.urlopen(req, timeout=600).read())
    e = r[0]["embedding"] if isinstance(r, list) else r["embedding"]
    e = np.asarray(e, dtype=np.float32)
    return e[-1] if e.ndim == 2 else e  # pooled -> (H,), or per-token -> last token


import re, signal, subprocess
class Power:
    """tegrastats VDD_IN (board input power) and RAM, sampled every 100 ms"""
    def __init__(s): s.p = None; s.f = f"/tmp/_tegra_{a.tag}.log"
    def start(s):
        if a.power: s.p = subprocess.Popen(["tegrastats", "--interval", "100"], stdout=open(s.f, "w"), stderr=subprocess.STDOUT)
    def stop(s):
        if not s.p: return None
        s.p.send_signal(signal.SIGINT); s.p.wait(timeout=5)
        txt = open(s.f).read()
        w = [int(x) / 1000 for x in re.findall(r"VDD_IN (\d+)mW", txt)]
        ram = [int(x) for x in re.findall(r"RAM (\d+)/\d+MB", txt)]
        return dict(mean_w=round(float(np.mean(w)), 3) if w else None, n_samples=len(w),
                    ram_peak_mb=max(ram) if ram else None, ram_min_mb=min(ram) if ram else None)
PW = Power()

for r in rows[:a.warmup]: embed(r["ids"])
h = embed(rows[0]["ids"]); assert h.shape[0] == head.shape[1], f"hidden {h.shape} vs head {head.shape}"

preds, lat, ntok = [], [], []
PW.start(); t0 = time.time()
for k, r in enumerate(rows):
    t = time.perf_counter(); h = embed(r["ids"]); lat.append((time.perf_counter() - t) * 1000)
    preds.append(classes[int(np.argmax(head @ h))]); ntok.append(len(r["ids"]))
    if k % 200 == 0: print(f"  {k}/{len(rows)}  last {lat[-1]:.0f} ms ({ntok[-1]} tok)", flush=True)
wall = time.time() - t0; power = PW.stop()

lat = np.array(lat); ntok = np.array(ntok)
assert COST_ONLY or a.endpoint == "embedding", "the completion endpoint is cost-only"
res = dict(tag=a.tag, device=a.device, n=len(rows), dataset=meta["dataset"], model=meta["model"], seed=meta["seed"],
           max_len=meta["max_len"], cost_only_ids_file=a.ids_file, endpoint=a.endpoint)
if not COST_ONLY:
    yt = [r["label"] for r in rows]; hf = [r["hf_fp16_pred"] for r in rows]
    res.update(metrics=macro_f1_both(yt, preds, meta["classes_all"]), hf_fp16_on_same_rows=macro_f1_both(yt, hf, meta["classes_all"]),
               agreement_with_hf_fp16=round(float(np.mean(np.array(preds) == np.array(hf))), 4))
res.update(
           latency_ms=dict(mean=round(float(lat.mean()), 1), p50=round(float(np.percentile(lat, 50)), 1),
                           p95=round(float(np.percentile(lat, 95)), 1)),
           prompt_tokens=dict(mean=round(float(ntok.mean()), 1), p50=float(np.percentile(ntok, 50)),
                              p95=float(np.percentile(ntok, 95)), max=int(ntok.max())),
           wall_s=round(wall, 1), engine=("llama.cpp llama-server, --pooling last, batch 1" if a.endpoint == "embedding" else
                   "llama.cpp llama-server /completion n_predict=0 (causal prefill in ubatch chunks), batch 1"), power=power,
           energy_mj_per_inference=round(power["mean_w"] * float(lat.mean()), 1) if power and power["mean_w"] else None,
           llama_cpp_commit=(EXP / "llama_cpp_commit.txt").read_text().strip() if (EXP / "llama_cpp_commit.txt").exists() else None)
sfx = "" if not COST_ONLY else "_len-" + Path(a.ids_file).stem
out = EXP / f"gguf_eval_{a.tag}_{a.device}{sfx}.json"; json.dump(res, open(out, "w"), indent=2)
print(json.dumps(res, indent=2)); print("->", out)
