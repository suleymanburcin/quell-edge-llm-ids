#!/usr/bin/env python3
"""QUELL Step 10 (CPU, minutes) — feature-leakage control on Edge-IIoTset.

The original pipeline (steps 03–09) removes IP addresses and the timestamp. The Edge-IIoTset authors' own
pre-processing additionally drops ports, raw payload/URI/option fields and timestamps that can
identify individual sessions. This script re-trains the classical detectors under the SAME matched
protocol as Table III (balanced 2,000 flows/class, full test partition, 5 seeds, identical RF/XGB
settings) with three feature sets:
  base    : features used in the paper
  ferrag  : base minus the additional fields dropped by the dataset authors
  strict  : ferrag minus raw TCP sequence/ack/checksum fields
If macro-F1 is essentially unchanged, the classical results do not depend on identifier-like fields.
Output: results/v4/leakage_check_edge_iiotset.json
"""
import json, sys, time
from pathlib import Path
import numpy as np, pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score
import xgboost as xgb
sys.path.insert(0, str(Path(__file__).parent))
import quell_common as qc

DATASET = "edge_iiotset"; SEEDS = [42, 1, 2, 3, 4]; TRAIN_CAP = 2000
FERRAG_EXTRA = ["tcp.srcport", "tcp.dstport", "udp.port", "tcp.options", "tcp.payload", "http.file_data",
                "http.request.full_uri", "http.request.uri.query", "icmp.transmit_timestamp", "mqtt.msg"]
STRICT_EXTRA = ["tcp.ack_raw", "tcp.seq", "tcp.ack", "tcp.checksum"]
STREAM_IDS = ["udp.stream", "tcp.stream"]   # Wireshark stream indices = session identifiers

R = qc.root(); RES = R / "results"; OUT = RES / "v4" / f"leakage_check_{DATASET}.json"
ONLY = [x for x in __import__("os").environ.get("QUELL_ONLY", "").split(",") if x]
OUT.parent.mkdir(parents=True, exist_ok=True)
meta = json.load(open(RES / "split_report.json"))[DATASET]
label = meta["label_col"]
df = pd.read_parquet(R / "data" / "processed" / f"{DATASET}.parquet").reset_index(drop=True)
sp = np.load(R / "splits" / f"{DATASET}_split.npz"); tr_idx, te_idx = sp["train"], sp["test"]
drop = {label} | set(meta.get("leaky_candidates", []))
for k in ("group_col", "time_col"):
    if meta.get(k): drop.add(meta[k])
drop |= {c for c in df.columns if c != label and c.lower() in qc.LABELISH}
base = [c for c in df.columns if c not in drop]
sets = {"base": base,
        "ferrag": [c for c in base if c not in FERRAG_EXTRA],
        "noid": [c for c in base if c not in FERRAG_EXTRA + STREAM_IDS],
        "strict": [c for c in base if c not in FERRAG_EXTRA + STRICT_EXTRA]}
y = df[label].astype(str).values; classes = sorted(pd.unique(y).tolist())
Xdf = df[base].copy()
for c in Xdf.columns:
    if not is_numeric_dtype(Xdf[c]): Xdf[c] = pd.factorize(Xdf[c])[0]
Xall = np.nan_to_num(Xdf.values.astype("float32")); col = {c: i for i, c in enumerate(base)}
yte = y[te_idx]
prev = json.load(open(OUT)) if OUT.exists() else {}
res = {"dataset": DATASET, "protocol": "balanced 2000/class train, full test partition, RF(300, balanced) / XGB(300, depth 8)",
       "seeds": SEEDS, "removed": {"ferrag": [c for c in FERRAG_EXTRA if c in base],
                                   "noid": [c for c in FERRAG_EXTRA + STREAM_IDS if c in base],
                                   "strict": [c for c in FERRAG_EXTRA + STRICT_EXTRA if c in base]},
       "n_features": {k: len(v) for k, v in sets.items()}, "results": prev.get("results", {})}
print(json.dumps(res["removed"], indent=1), res["n_features"], flush=True)
for name, feats in sets.items():
    if ONLY and name not in ONLY: continue
    X = Xall[:, [col[c] for c in feats]]; rf_s, xg_s = [], []; t = time.time()
    for seed in SEEDS:
        rng = np.random.default_rng(seed); sel = []
        for cls in pd.unique(y[tr_idx]):
            ids = tr_idx[y[tr_idx] == cls]
            if len(ids) > TRAIN_CAP: ids = rng.choice(ids, TRAIN_CAP, replace=False)
            sel += ids.tolist()
        sel = np.array(sorted(sel))
        rf = RandomForestClassifier(n_estimators=300, n_jobs=-1, class_weight="balanced", random_state=seed).fit(X[sel], y[sel])
        rf_s.append(float(f1_score(yte, rf.predict(X[te_idx]), average="macro", labels=classes, zero_division=0)))
        le = LabelEncoder().fit(y[sel])
        xg = xgb.XGBClassifier(n_estimators=300, max_depth=8, n_jobs=-1, tree_method="hist", random_state=seed,
                               num_class=len(le.classes_), objective="multi:softmax", eval_metric="mlogloss").fit(X[sel], le.transform(y[sel]))
        xg_s.append(float(f1_score(yte, le.inverse_transform(xg.predict(X[te_idx])), average="macro", labels=classes, zero_division=0)))
        if name == "base" and seed == SEEDS[0]:
            imp = sorted(zip(feats, rf.feature_importances_), key=lambda t: -t[1])[:15]
            res["rf_top15_importance_base"] = [[f, round(float(v), 4)] for f, v in imp]
        print(f"  {name:6s} seed {seed}: RF={rf_s[-1]:.4f} XGB={xg_s[-1]:.4f}", flush=True)
    res["results"][name] = {"rf": {"mean": round(float(np.mean(rf_s)), 4), "std": round(float(np.std(rf_s)), 4), "seeds": [round(v, 4) for v in rf_s]},
                            "xgb": {"mean": round(float(np.mean(xg_s)), 4), "std": round(float(np.std(xg_s)), 4), "seeds": [round(v, 4) for v in xg_s]},
                            "time_s": round(time.time() - t)}
    json.dump(res, open(OUT, "w"), indent=2)
    print(f"== {name}: RF {res['results'][name]['rf']['mean']}±{res['results'][name]['rf']['std']}  "
          f"XGB {res['results'][name]['xgb']['mean']}±{res['results'][name]['xgb']['std']}", flush=True)
print("DONE ->", OUT)
