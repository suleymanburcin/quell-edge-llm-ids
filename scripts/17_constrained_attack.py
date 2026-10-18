#!/usr/bin/env python
"""QUELL Step 17 (GPU host) — constrained, format-controlled black-box evasion, per precision.

Perturbs only valid, attacker-controllable features, controls for serialization/format shift, uses an
established black-box attack, and reports robustness per precision (RQ4), for RF, XGBoost and the fine-tuned
Qwen2.5 detector at FP16 / INT8 / NF4:

  * attacker-controllable features: per-dataset list of size / timing / sequence fields the sender of the
    malicious flow can set (CTRL below); protocol, flag and semantic fields stay fixed;
  * validity + format control: every perturbed vector is projected onto the training-set domain of each
    feature (clipped to [min, max]; integer-valued features rounded) and serialized with the SAME formatter
    as training (4-decimal floats), so the LLM never sees out-of-format numbers;
  * (1) format-controlled noise sweep: Gaussian noise of magnitude eps*sigma on (a) all numeric features and
        (b) controllable features only, both projected -> separates format shift from feature perturbation;
  * (2) HopSkipJump (Chen et al., IEEE S&P 2020; ART implementation), decision-based black-box attack on the
        benign/attack decision, restricted to the controllable features; success rates reported at
        L-inf budgets of 0.1 / 0.3 / 1.0 sigma and unbounded, with the number of model queries.

Inputs : QUELL_DATASET (default edge_iiotset), QUELL_FEATSET (default noid), QUELL_SEED (42),
         QUELL_ADAPTER (default models/v4/<dataset>_Qwen2.5-1.5B_s<seed>[_noid]/adapter, written by 13_quantize_export.py),
         QUELL_PRECISIONS (fp16,int8,nf4), QUELL_N_ATTACK (60), QUELL_N_NOISE (2000)
Output : results/v4/constrained_attack_<dataset>_<tag>.json   (resumable per block)
"""
import os, sys, json, time, gc, subprocess
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))
import quell_common as qc

DATASET = os.environ.get("QUELL_DATASET", "edge_iiotset")
os.environ.setdefault("QUELL_FEATSET", "noid" if DATASET == "edge_iiotset" else "")
FEATSET = qc.featset()
SEED = int(os.environ.get("QUELL_SEED", "42"))
MODEL = os.environ.get("QUELL_MODEL", "Qwen/Qwen2.5-1.5B")
PRECS = [p for p in os.environ.get("QUELL_PRECISIONS", "fp16,int8,nf4").split(",") if p]
N_ATTACK = int(os.environ.get("QUELL_N_ATTACK", "60"))
N_NOISE = int(os.environ.get("QUELL_N_NOISE", "2000"))
EPS_GRID = [0.0, 0.1, 0.25, 0.5, 1.0]
BUDGETS = [0.1, 0.3, 1.0]
HSJ = dict(max_iter=10, max_eval=100, init_eval=20, init_size=100)

CTRL = {  # fields the sender of a malicious flow can set without changing the attack's protocol semantics
    "edge_iiotset": ["tcp.len", "tcp.seq", "tcp.ack", "tcp.ack_raw", "tcp.checksum", "icmp.checksum", "icmp.seq_le",
                     "http.content_length", "udp.time_delta", "dns.qry.name.len", "mqtt.len", "mqtt.topic_len",
                     "mbtcp.len", "mbtcp.trans_id"],
    "ciciot2023": ["flow_duration", "Header_Length", "Rate", "Srate", "Duration", "Tot sum", "Min", "Max", "AVG",
                   "Std", "Tot size", "IAT", "Number", "Magnitue", "Radius", "Covariance", "Variance", "Weight"],
}

R = qc.root(); RES = R / "results" / "v4"; RES.mkdir(parents=True, exist_ok=True)
TAG = f"{MODEL.split('/')[-1]}_s{SEED}" + (f"_{FEATSET}" if FEATSET else "")
OUT = RES / f"constrained_attack_{DATASET}_{TAG}.json"
ADAPTER = Path(os.environ.get("QUELL_ADAPTER", R / "models" / "v4" / f"{DATASET}_{TAG}" / "adapter"))


def log(m): print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {m}", flush=True)


try:
    _o = subprocess.check_output("nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits", shell=True, text=True)
    _f = [(int(x.split(",")[0]), int(x.split(",")[1])) for x in _o.strip().splitlines()]
    os.environ["CUDA_VISIBLE_DEVICES"] = str(max(_f, key=lambda t: t[1])[0]); log(f"GPU free(MiB): {_f}")
except Exception as e: log(f"GPU selection skipped: {e}")

import numpy as np, pandas as pd
from pandas.api.types import is_numeric_dtype, is_float_dtype
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import torch
try:
    from art.attacks.evasion import HopSkipJump
    from art.estimators.classification import BlackBoxClassifier
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "adversarial-robustness-toolbox"], check=True)
    from art.attacks.evasion import HopSkipJump
    from art.estimators.classification import BlackBoxClassifier
from transformers import AutoTokenizer, AutoModelForSequenceClassification, BitsAndBytesConfig
from peft import PeftModel

np.random.seed(SEED); torch.manual_seed(SEED)
res = json.load(open(OUT)) if OUT.exists() else {}
res.update(dataset=DATASET, featset=FEATSET or "paper", seed=SEED, model=MODEL, adapter=str(ADAPTER),
           hsj_config=HSJ, budgets_sigma_linf=BUDGETS, n_attack=N_ATTACK, n_noise=N_NOISE)
res.setdefault("noise", {}); res.setdefault("hsj", {})
def save(): json.dump(res, open(OUT, "w"), indent=2)

# ---------------- data (identical feature list and serialization to qc.load) ----------------
meta = json.load(open(R / "results" / "split_report.json"))[DATASET]
label = meta["label_col"]
df = pd.read_parquet(R / "data" / "processed" / f"{DATASET}.parquet").reset_index(drop=True)
sp = np.load(R / "splits" / f"{DATASET}_split.npz"); tr_idx, te_idx = sp["train"], sp["test"]
drop = {label} | set(meta.get("leaky_candidates", []))
for k in ("group_col", "time_col"):
    if meta.get(k): drop.add(meta[k])
drop |= {c for c in df.columns if c != label and c.lower() in qc.LABELISH}
feats = [c for c in df.columns if c not in drop]
if FEATSET == "noid": feats = [c for c in feats if c not in qc.NOID_DROP]
y = df[label].astype(str).values; classes_all = sorted(pd.unique(y).tolist())
benign = next(c for c in classes_all if c.strip().lower() in {"normal", "benign", "background"}
              or "normal" in c.lower() or "benign" in c.lower())
ctrl = [c for c in CTRL[DATASET] if c in feats and is_numeric_dtype(df[c])]
assert len(ctrl) >= 5, f"too few controllable features found: {ctrl}"
num = [c for c in feats if is_numeric_dtype(df[c])]
fi = {c: i for i, c in enumerate(feats)}
isfloat = np.array([is_float_dtype(df[c]) for c in feats])
raw = df[feats].values                                   # object array, used for serialization
X = df[feats].copy()
for c in feats:
    if not is_numeric_dtype(X[c]): X[c] = pd.factorize(X[c])[0]
X = np.clip(np.nan_to_num(X.values.astype("float64"), nan=0.0, posinf=0.0, neginf=0.0), -1e9, 1e9)
lo, hi = X[tr_idx].min(0), X[tr_idx].max(0)
isint = np.array([np.all(np.mod(X[tr_idx][:, j], 1) == 0) for j in range(len(feats))])
sigma = X[tr_idx].std(0) + 1e-9
log(f"{DATASET} [{TAG}] feats={len(feats)} numeric={len(num)} controllable={len(ctrl)}: {ctrl}")
res["controllable_features"] = ctrl


def project(Xp, cols):
    """valid domain: clip to training range, round integer-valued features"""
    j = [fi[c] for c in cols]; v = np.clip(Xp[:, j], lo[j], hi[j])
    v[:, isint[j]] = np.round(v[:, isint[j]]); Xp[:, j] = v; return Xp


def serialize(rows, Xp, changed):
    """same formatter as training (qc.load.row_to_text); only the perturbed columns are re-read from Xp"""
    out = []
    for r, i in enumerate(rows):
        parts = []
        for j, c in enumerate(feats):
            if j in changed:
                v = round(float(Xp[r, j]), 4) if isfloat[j] else int(round(Xp[r, j]))
            else:
                v = raw[i, j]
                if isinstance(v, (float, np.floating)): v = round(float(v), 4)
            parts.append(f"{c}={v}")
        out.append("Network traffic flow. " + ", ".join(parts) + " . Attack type:")
    return out

# sanity: unperturbed serialization must reproduce the training text exactly
_rows = te_idx[:50]
_ref = []
for i in _rows:
    parts = []
    for c in feats:
        v = df.at[i, c]
        if isinstance(v, (float, np.floating)): v = round(float(v), 4)
        parts.append(f"{c}={v}")
    _ref.append("Network traffic flow. " + ", ".join(parts) + " . Attack type:")
assert serialize(_rows, X[_rows].copy(), set()) == _ref, "serializer differs from the training formatter"

# ---------------- detectors ----------------
log("training RF / XGB on the full training partition (as in the noise experiment)...")
rf = RandomForestClassifier(n_estimators=300, n_jobs=-1, class_weight="balanced", random_state=SEED).fit(X[tr_idx], y[tr_idx])
lex = LabelEncoder().fit(y[tr_idx])
xg = xgb.XGBClassifier(n_estimators=300, max_depth=8, n_jobs=-1, tree_method="hist", random_state=SEED,
                       num_class=len(lex.classes_), objective="multi:softmax", eval_metric="mlogloss").fit(X[tr_idx], lex.transform(y[tr_idx]))
tok = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token is None: tok.pad_token = tok.eos_token
_mp = ADAPTER / "quell_meta.json" if (ADAPTER / "quell_meta.json").exists() else ADAPTER.parent / "meta.json"
ameta = json.load(open(_mp)) if _mp.exists() else {}
MAX_LEN = int(ameta.get("max_len", 2048)); LLM_CLASSES = ameta.get("classes")
assert LLM_CLASSES, f"adapter meta with class order not found in {ADAPTER}"


MERGED = ADAPTER.parent / "merged_hf"   # faithful QLoRA merge written by 13_quantize_export.py (QUELL_MERGE=dequant)


def load_llm(prec):
    """fp16 / int8: the de-quantized-merge classifier (as deployed); nf4: adapter on the NF4 base (training precision)"""
    if prec in ("fp16", "int8") and MERGED.exists():
        kw = dict(device_map={"": 0})
        if prec == "fp16": kw["torch_dtype"] = torch.float16
        else: kw["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        m = AutoModelForSequenceClassification.from_pretrained(str(MERGED), **kw); m.config.pad_token_id = tok.pad_token_id
        return m.eval()
    kw = dict(num_labels=len(LLM_CLASSES), device_map={"": 0})
    if prec == "fp16": kw["torch_dtype"] = torch.float16
    elif prec == "int8": kw["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    else: kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                         bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    b = AutoModelForSequenceClassification.from_pretrained(MODEL, **kw); b.config.pad_token_id = tok.pad_token_id
    m = PeftModel.from_pretrained(b, str(ADAPTER)); m = m.merge_and_unload() if prec == "fp16" else m
    return m.eval()


def llm_predict(m, texts, bs=32):
    dev = next(m.parameters()).device; order = np.argsort([len(t) for t in texts]); out = np.empty(len(texts), dtype=object)
    for s in range(0, len(texts), bs):
        sl = order[s:s + bs]
        enc = tok([texts[k] for k in sl], truncation=True, max_length=MAX_LEN, padding=True, return_tensors="pt").to(dev)
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out[sl] = [LLM_CLASSES[k] for k in m(**enc).logits.argmax(-1).cpu().tolist()]
    return out

QUERIES = {"n": 0}; MEMO = {}
HSJ_TIMEOUT = int(os.environ.get("QUELL_HSJ_TIMEOUT", "300"))   # seconds per attacked flow; a timeout counts as not evaded
import signal
class _Timeout(Exception): pass
def _alarm(*_): raise _Timeout()
signal.signal(signal.SIGALRM, _alarm)
def make_predict(name, llm=None):
    def f(rows, Xp, changed):
        QUERIES["n"] += len(rows)
        if name == "rf": return rf.predict(Xp)
        if name == "xgb": return lex.inverse_transform(xg.predict(Xp).astype(int))
        T = serialize(rows, Xp, changed)
        # memo: one fixed decision per serialized flow. LLM.int8() outputs depend slightly on which inputs share a
        # batch; without the memo a decision-based attack can see the same input flip and fail to converge.
        new = [t for t in dict.fromkeys(T) if t not in MEMO]
        if new:
            for t, p in zip(new, llm_predict(llm, new)): MEMO[t] = p
        return np.array([MEMO[t] for t in T], dtype=object)
    return f

from sklearn.metrics import f1_score
rng = np.random.default_rng(SEED); pick = []
for c in np.unique(y[te_idx]):
    ids = te_idx[y[te_idx] == c]; k = max(1, int(round(len(ids) * N_NOISE / len(te_idx))))
    pick += rng.choice(ids, min(k, len(ids)), replace=False).tolist()
noise_idx = np.array(sorted(pick))
att = te_idx[y[te_idx] != benign]
att_idx = np.array(sorted(np.random.default_rng(SEED + 7).choice(att, min(N_ATTACK, len(att)), replace=False)))


def run_detector(name, pred):
    # (1) format-controlled noise
    if name not in res["noise"]:
        cur = {}
        for cond, cols in (("all_numeric_valid", num), ("controllable_valid", ctrl)):
            f1s = []
            for eps in EPS_GRID:
                Xp = X[noise_idx].copy(); j = [fi[c] for c in cols]
                Xp[:, j] += eps * sigma[j] * np.random.default_rng(100 + int(eps * 1000)).standard_normal((len(noise_idx), len(j)))
                Xp = project(Xp, cols)
                f1s.append(round(float(f1_score(y[noise_idx], pred(noise_idx, Xp, set(j) if eps > 0 else set()), average="macro", labels=classes_all, zero_division=0)), 4))
            cur[cond] = dict(eps=EPS_GRID, macro_f1=f1s); log(f"  {name} noise {cond}: {f1s}")
        res["noise"][name] = cur; save()
    # (2) HopSkipJump on the controllable features
    if name in res["hsj"]: return
    jc = [fi[c] for c in ctrl]; JC = set(jc); span = np.maximum(hi[jc] - lo[jc], 1e-9)
    succ, dist, qs, already, timeouts = [], [], [], 0, 0
    MEMO.clear()
    for n_i, i in enumerate(att_idx):
        x0 = X[[i]].copy()
        if pred(np.array([i]), x0, set())[0] == benign: already += 1; continue
        def bb(U, _i=i):
            Xp = np.repeat(X[[_i]], len(U), 0); Xp[:, jc] = lo[jc] + U * span; Xp = project(Xp, ctrl)
            p = pred(np.full(len(U), _i), Xp, JC); return np.stack([(p == benign), (p != benign)], 1).astype(np.float32)
        clf = BlackBoxClassifier(bb, input_shape=(len(jc),), nb_classes=2, clip_values=(0.0, 1.0))
        u0 = np.clip((x0[:, jc] - lo[jc]) / span, 0, 1).astype(np.float32)
        q0 = QUERIES["n"]
        signal.alarm(HSJ_TIMEOUT)
        try:
            adv = HopSkipJump(clf, targeted=False, norm=np.inf, verbose=False, **HSJ).generate(u0, y=np.array([[0, 1]], np.float32))
            signal.alarm(0)
        except _Timeout:
            timeouts += 1; adv = u0; log(f"    flow {n_i + 1}: HSJ time limit ({HSJ_TIMEOUT}s) reached -> not evaded")
        Xa = x0.copy(); Xa[:, jc] = lo[jc] + adv * span; Xa = project(Xa, ctrl)
        ok = pred(np.array([i]), Xa, JC)[0] == benign
        d = float(np.max(np.abs(Xa[0, jc] - x0[0, jc]) / sigma[jc])) if ok else float("inf")
        succ.append(bool(ok)); dist.append(d); qs.append(QUERIES["n"] - q0)
        if (n_i + 1) % 10 == 0: log(f"    {name} HSJ {n_i + 1}/{len(att_idx)}: evaded so far {int(np.sum(succ))}, timeouts {timeouts}")
    dist = np.array(dist)
    r = dict(n_attacked=len(succ), already_benign=already, timeouts=timeouts, time_limit_s=HSJ_TIMEOUT, median_queries=int(np.median(qs)) if qs else 0,
             success_unbounded=round(float(np.mean(succ)), 4) if succ else None,
             success_at_budget={str(b): round(float(np.mean(dist <= b)), 4) for b in BUDGETS} if succ else None,
             median_linf_sigma_successful=round(float(np.median(dist[np.isfinite(dist)])), 4) if np.isfinite(dist).any() else None)
    res["hsj"][name] = r; save(); log(f"  {name} HSJ: {r}")


t0 = time.time()
run_detector("rf", make_predict("rf")); run_detector("xgb", make_predict("xgb"))
for prec in PRECS:
    nm = f"llm_{prec}"
    if nm in res["noise"] and nm in res["hsj"]: continue
    log(f"loading LLM {prec} ...")
    m = load_llm(prec); run_detector(nm, make_predict(nm, m)); del m; gc.collect(); torch.cuda.empty_cache()
res["total_s"] = round(time.time() - t0); save()
log(f"DONE -> {OUT}")
