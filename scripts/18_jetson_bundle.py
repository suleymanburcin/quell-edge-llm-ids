#!/usr/bin/env python3
"""QUELL Step 18 (GPU host, CPU only, ~10-20 min) — build the bundle that is copied to the Jetson.

Contents of <bundle>/ (default: ~/quell-edge-llm-ids/jetson_bundle):
  edge_llm/        GGUF Q8_0 + Q4_K_M of the identifier-free (noid) Edge-IIoTset detector, head.npy, meta.json,
                   subset.jsonl (stratified test flows with token ids, labels and the HF-FP16 prediction)
  lengths/         <dataset>_ids.jsonl: 300 real serialized test flows per dataset (token ids) for LLM cost at the
                   real prompt length of CICIoT2023 and N-BaIoT (cost depends on length, not on the weights)
  classic/         <dataset>_{rf.joblib, xgb.json, X.npy, y.npy, classes.json, rf.onnx}: RF (300 trees, balanced)
                   and XGBoost (300 trees, depth 8) trained under the matched protocol (2,000 flows/class, seed 42;
                   Edge-IIoTset noid), plus 3,000 stratified test inputs for on-device accuracy and cost
  scripts/         19_jetson_*.py / .sh + quell_common.py
Usage:  QUELL_ROOT=~/quell-edge-llm-ids python scripts/18_jetson_bundle.py
"""
import os, sys, json, shutil, subprocess
from pathlib import Path
import numpy as np, pandas as pd
from pandas.api.types import is_numeric_dtype
sys.path.insert(0, str(Path(__file__).parent))
import quell_common as qc

R = qc.root(); B = Path(os.environ.get("QUELL_BUNDLE", R / "jetson_bundle")); SEED = 42
EXP = R / "models" / "v4" / "edge_iiotset_Qwen2.5-1.5B_s42_noid"
for d in ("edge_llm", "lengths", "classic", "scripts"): (B / d).mkdir(parents=True, exist_ok=True)

# 1. deployed LLM detector
for f in ("model-q8_0.gguf", "model-q4_k_m.gguf", "head.npy", "meta.json", "subset.jsonl", "llama_cpp_commit.txt"):
    if (EXP / f).exists(): shutil.copy2(EXP / f, B / "edge_llm" / f)
    else: print("WARNING missing", EXP / f)

# 2. real prompt lengths + 3. classical detectors
from transformers import AutoTokenizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score
import joblib, xgboost as xgb
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B")
summary = {}
for ds, fs in (("edge_iiotset", "noid"), ("ciciot2023", ""), ("nbaiot", "")):
    os.environ["QUELL_FEATSET"] = fs
    d = qc.load(ds, SEED, with_texts=False); y, tr, ev, feats = d["y"], d["tr_sel"], d["eval_idx"], d["feats"]
    text = lambda j: d["row_to_text"](d["df"].iloc[j])
    sub = qc.stratified_subset(ev, y, 3000, SEED)
    rng = np.random.default_rng(SEED); lsub = np.sort(rng.choice(sub, min(300, len(sub)), replace=False))
    with open(B / "lengths" / f"{ds}_ids.jsonl", "w") as f:
        for j in lsub: f.write(json.dumps({"i": int(j), "ids": tok(text(j))["input_ids"], "label": y[j]}) + "\n")
    df = d["df"][feats].copy()
    for c in df.columns:
        if not is_numeric_dtype(df[c]): df[c] = pd.factorize(df[c])[0]
    X = np.clip(np.nan_to_num(df.values.astype("float64"), nan=0.0, posinf=0.0, neginf=0.0), -1e9, 1e9).astype("float32")
    rf = RandomForestClassifier(n_estimators=300, n_jobs=-1, class_weight="balanced", random_state=SEED).fit(X[tr], y[tr])
    le = LabelEncoder().fit(y[tr])
    xg = xgb.XGBClassifier(n_estimators=300, max_depth=8, n_jobs=-1, tree_method="hist", random_state=SEED,
                           num_class=len(le.classes_), objective="multi:softmax", eval_metric="mlogloss").fit(X[tr], le.transform(y[tr]))
    cd = B / "classic"
    joblib.dump(rf, cd / f"{ds}_rf.joblib"); xg.save_model(str(cd / f"{ds}_xgb.json"))
    np.save(cd / f"{ds}_X.npy", X[sub]); np.save(cd / f"{ds}_y.npy", y[sub].astype(str))
    json.dump(dict(classes=list(le.classes_), classes_all=d["classes_all"], features=feats, featset=fs or "paper"),
              open(cd / f"{ds}_classes.json", "w"), indent=2)
    try:  # optional optimized runtime for the random forest
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'skl2onnx', 'onnx'], capture_output=True)
        from skl2onnx import to_onnx
        onx = to_onnx(rf, X[:1], target_opset=15, options={id(rf): {"zipmap": False}})
        open(cd / f"{ds}_rf.onnx", "wb").write(onx.SerializeToString())
    except Exception as e: print("ONNX export skipped:", e)
    f_rf = f1_score(y[sub], rf.predict(X[sub]), average="macro", labels=d["classes_all"], zero_division=0)
    f_xg = f1_score(y[sub], le.inverse_transform(xg.predict(X[sub])), average="macro", labels=d["classes_all"], zero_division=0)
    summary[ds] = dict(n_features=len(feats), n_sub=int(len(sub)), rf_macro_f1_sub=round(float(f_rf), 4), xgb_macro_f1_sub=round(float(f_xg), 4),
                       prompt_tokens_p50=float(np.median([len(tok(text(j))["input_ids"]) for j in lsub])))
    print(ds, summary[ds], flush=True)
json.dump(summary, open(B / "bundle_summary.json", "w"), indent=2)

# 4. scripts
for f in ("19_jetson_run.sh", "19_jetson_classic_cost.py", "14_gguf_eval.py", "quell_common.py"):
    shutil.copy2(Path(__file__).parent / f, B / "scripts" / f)
subprocess.run(["tar", "-cf", str(B) + ".tar", "-C", str(B.parent), B.name], check=True)
print("bundle ->", str(B) + ".tar", f"({Path(str(B) + '.tar').stat().st_size / 1e9:.2f} GB)")
