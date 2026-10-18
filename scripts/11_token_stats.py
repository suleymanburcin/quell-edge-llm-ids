#!/usr/bin/env python3
"""QUELL Step 11 (GPU host, CPU only, ~1 min) — token-length statistics of the serialized flows (Section III-D).

This is the exact code that was run as a notebook cell on 2026-10-15 to produce results/v4/token_stats.json,
saved here as a script for provenance. 4,000 random flows per dataset (random_state 42), Qwen2.5 tokenizer,
Edge-IIoTset with the identifier-free (noid) feature set; MAX_LEN = min(2048, p99 + 16).
Usage:  python scripts/11_token_stats.py
"""
import json, os
import numpy as np, pandas as pd
from transformers import AutoTokenizer

R = os.environ.get("QUELL_ROOT", os.path.expanduser("~/quell-edge-llm-ids"))
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B")
NOID = ["tcp.srcport", "tcp.dstport", "udp.port", "tcp.options", "tcp.payload", "http.file_data", "http.request.full_uri",
        "http.request.uri.query", "icmp.transmit_timestamp", "mqtt.msg", "udp.stream", "tcp.stream"]
LAB = {"label", "attack", "attack_type", "attack_label", "type", "class", "category", "marker", "__label__"}
rep = json.load(open(f"{R}/results/split_report.json")); out = {}
for ds, fs in [("edge_iiotset", "noid"), ("ciciot2023", ""), ("nbaiot", "")]:
    m = rep[ds]; df = pd.read_parquet(f"{R}/data/processed/{ds}.parquet")
    drop = {m["label_col"], *m.get("leaky_candidates", []), m.get("group_col"), m.get("time_col")}
    feats = [c for c in df.columns if c not in drop and c.lower() not in LAB and not (fs == "noid" and c in NOID)]
    s = df.sample(4000, random_state=42)
    T = ["Network traffic flow. " + ", ".join(f"{c}={round(float(v),4) if isinstance(v,(float,np.floating)) else v}"
         for c, v in zip(feats, r)) + " . Attack type:" for r in s[feats].values]
    L = np.array([len(tok(t)["input_ids"]) for t in T]); ML = int(min(2048, np.percentile(L, 99) + 16))
    out[ds] = dict(n_feat=len(feats), mean=round(L.mean(), 1), p50=int(np.median(L)), p95=int(np.percentile(L, 95)),
                   p99=int(np.percentile(L, 99)), max=int(L.max()), max_len=ML, trunc_pct=round(100 * (L > ML).mean(), 2),
                   example=T[int(np.argsort(L)[len(L) // 2])], example_label=str(s[m["label_col"]].values[int(np.argsort(L)[len(L) // 2])]))
    print(ds, {k: v for k, v in out[ds].items() if k != "example"}); del df
json.dump(out, open(f"{R}/results/v4/token_stats.json", "w"), indent=2, ensure_ascii=False)
print("\n", out["edge_iiotset"]["example"])
