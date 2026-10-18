#!/usr/bin/env python
# ============================================================================
# QUELL Step 15 — UNKNOWN-ATTACK GENERALIZATION (RF, XGBoost, LLM), any dataset.
# Generalizes scripts/re_08_generalization.py with the IDENTICAL protocol
# (leave-one-out over attacks, closed-set training on 1,500 flows/class, 1 epoch, seed 42,
# prediction reduced to benign/attack; an unseen flow counts as caught if it is called "attack")
# and ADDS: XGBoost, the `noid` feature set (identifier-field control), family-level
# hold-out for CICIoT2023 (7 attack families instead of 33 per-class folds), no truncation
# (MAX_LEN = p99 + 16, cap 2048).
# ENV: QUELL_DATASET (edge_iiotset | ciciot2023 | nbaiot), QUELL_FEATSET ("" | "noid"),
#      QUELL_HOLDOUT ("class" | "family"; default family for ciciot2023, class otherwise),
#      QUELL_CLASSIC_ONLY=1 (RF+XGB only, CPU), QUELL_MODEL, QUELL_SEED
# Output (fold-level resume): results/v4/generalization_<dataset>_<tag>.json
# ============================================================================
import os, sys, json, time, gc, subprocess
from pathlib import Path
from datetime import datetime

ROOT = Path.home()/"quell-edge-llm-ids"; RES = ROOT/"results"
DATASET = os.environ.get("QUELL_DATASET", "edge_iiotset")
FEATSET = os.environ.get("QUELL_FEATSET", "")
HOLDOUT = os.environ.get("QUELL_HOLDOUT", "family" if DATASET in ("ciciot2023", "nbaiot") else "class")
CLASSIC_ONLY = os.environ.get("QUELL_CLASSIC_ONLY", "") == "1"
MODEL = os.environ.get("QUELL_MODEL", "Qwen/Qwen2.5-1.5B"); SEED = int(os.environ.get("QUELL_SEED", "42"))
TRAIN_CAP, EPOCHS = 1500, 1                       # re_08 protocol
BENIGN_FPR_CAP, KNOWN_CAP, UNSEEN_CAP = 5000, 5000, 5000
MAXLEN_CAP = 2048
NOID_DROP = ["tcp.srcport", "tcp.dstport", "udp.port", "tcp.options", "tcp.payload", "http.file_data",
             "http.request.full_uri", "http.request.uri.query", "icmp.transmit_timestamp", "mqtt.msg",
             "udp.stream", "tcp.stream"]
CICIOT_FAMILIES = {   # CICIoT2023 attack taxonomy (Neto et al., 2023)
    "DDoS": lambda c: c.startswith("DDoS-"),
    "DoS": lambda c: c.startswith("DoS-"),
    "Mirai": lambda c: c.startswith("Mirai-"),
    "Recon": lambda c: c.startswith("Recon-") or c == "VulnerabilityScan",
    "Spoofing": lambda c: c in ("DNS_Spoofing", "MITM-ArpSpoofing"),
    "Web": lambda c: c in ("Backdoor_Malware", "BrowserHijacking", "CommandInjection", "SqlInjection",
                           "Uploading_Attack", "XSS"),
    "BruteForce": lambda c: c == "DictionaryBruteForce",
}
TAG = ("classic" if CLASSIC_ONLY else MODEL.split("/")[-1]) + f"_s{SEED}" + (f"_{FEATSET}" if FEATSET else "") \
      + ("_family" if HOLDOUT == "family" else "")
OUT = RES/"v4"/f"generalization_{DATASET}_{TAG}.json"


def log(m): print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {m}", flush=True)
def _load(p, d):
    try: return json.load(open(p))
    except Exception: return d


if "--status" in sys.argv:
    g = _load(OUT, {}); print(OUT.name, "|", len(g.get("folds", {})), "fold:", list(g.get("folds", {}))); sys.exit(0)

try:
    _o = subprocess.check_output("nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits", shell=True, text=True)
    _f = [(int(x.split(",")[0]), int(x.split(",")[1])) for x in _o.strip().splitlines()]
    _b = max(_f, key=lambda t: t[1]); os.environ["CUDA_VISIBLE_DEVICES"] = str(_b[0])
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"; log(f"selected GPU {_b[0]} | free(MiB): {_f}")
except Exception as e: log(f"GPU secim atlandi: {e}")

import numpy as np, pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
if not CLASSIC_ONLY:
    import torch
    from transformers import (AutoTokenizer, AutoModelForSequenceClassification, BitsAndBytesConfig,
                              TrainingArguments, Trainer, DataCollatorWithPadding)
    from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
    torch.manual_seed(SEED); BF16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
np.random.seed(SEED); FCLIP = 1e9

meta = json.load(open(RES/"split_report.json"))[DATASET]
label = meta["label_col"]
LABELISH = {"label", "attack", "attack_type", "attack_label", "type", "class", "category", "marker", "__label__"}
df = pd.read_parquet(ROOT/"data"/"processed"/f"{DATASET}.parquet").reset_index(drop=True)
sp = np.load(ROOT/"splits"/f"{DATASET}_split.npz"); tr_idx, te_idx = sp["train"], sp["test"]
drop = {label} | set(meta.get("leaky_candidates", []))
for k in ("group_col", "time_col"):
    if meta.get(k): drop.add(meta[k])
drop |= {c for c in df.columns if c != label and c.lower() in LABELISH}
feats = [c for c in df.columns if c not in drop]
if FEATSET == "noid": feats = [c for c in feats if c not in NOID_DROP]
y_all = df[label].astype(str).values; classes_all = sorted(pd.unique(y_all).tolist())
benign = next(c for c in classes_all if c.strip().lower() in {"normal", "benign", "background"}
              or "normal" in c.lower() or "benign" in c.lower())
attacks = [c for c in classes_all if c != benign]
NBAIOT_FAMILIES = {"gafgyt": lambda c: c.startswith("gafgyt"), "mirai": lambda c: c.startswith("mirai")}
if HOLDOUT == "family":
    FAM = CICIOT_FAMILIES if DATASET == "ciciot2023" else NBAIOT_FAMILIES
    folds_def = {f: [c for c in attacks if fn(c)] for f, fn in FAM.items()}
    miss = set(attacks) - {c for v in folds_def.values() for c in v}
    assert not miss, f"family mapping misses classes: {miss}"
else:
    folds_def = {c: [c] for c in attacks}

Xdf = df[feats].copy()
for c in Xdf.columns:
    if not is_numeric_dtype(Xdf[c]): Xdf[c] = pd.factorize(Xdf[c])[0]
X = np.clip(np.nan_to_num(Xdf.values.astype("float64"), nan=0.0, posinf=0.0, neginf=0.0), -FCLIP, FCLIP).astype("float32")
del Xdf
vals = df[feats].values
def text(i):   # identical to re_08 row_to_text
    parts = []
    for c, v in zip(feats, vals[i]):
        if isinstance(v, (float, np.floating)): v = round(float(v), 4)
        parts.append(f"{c}={v}")
    return "Network traffic flow. " + ", ".join(parts) + " . Attack type:"
log(f"{DATASET} [{TAG}]: benign='{benign}' | {len(folds_def)} folds ({HOLDOUT}) | features={len(feats)}")

MAX_LEN = None
if not CLASSIC_ONLY:
    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    _s = np.random.default_rng(SEED).choice(len(df), min(4000, len(df)), replace=False)
    _L = np.array([len(tok(text(j))["input_ids"]) for j in _s])
    MAX_LEN = int(min(int(np.percentile(_L, 99)) + 16, MAXLEN_CAP)); trunc = float(np.mean(_L > MAX_LEN))
    log(f"MAX_LEN={MAX_LEN} (p99={int(np.percentile(_L, 99))}, mean={_L.mean():.0f}) trunc_rate={trunc:.3f}")


def balanced(pool):
    rng = np.random.default_rng(SEED); sel = []
    for cls in pd.unique(y_all[pool]):
        ids = pool[y_all[pool] == cls]
        if len(ids) > TRAIN_CAP: ids = rng.choice(ids, TRAIN_CAP, replace=False)
        sel += ids.tolist()
    return np.array(sorted(sel))


def eval_rows(H):
    r = np.random.default_rng(SEED)
    h = te_idx[np.isin(y_all[te_idx], H)]
    if len(h) > UNSEEN_CAP: h = np.sort(r.choice(h, UNSEEN_CAP, replace=False))
    ben = te_idx[y_all[te_idx] == benign]
    if len(ben) > BENIGN_FPR_CAP: ben = r.choice(ben, BENIGN_FPR_CAP, replace=False)
    oth = te_idx[(~np.isin(y_all[te_idx], H)) & (y_all[te_idx] != benign)]
    if len(oth) > KNOWN_CAP: oth = r.choice(oth, KNOWN_CAP, replace=False)
    return h, ben, oth


def rates(p):
    det = lambda m: float(np.mean(m != benign)) if len(m) else float("nan")
    return dict(unseen_recall=round(det(p["H"]), 4), benign_fpr=round(det(p["ben"]), 4), known_recall=round(det(p["oth"]), 4))


def train_llm(sel, le):
    class DS(torch.utils.data.Dataset):
        def __len__(s): return len(sel)
        def __getitem__(s, i):
            j = sel[i]; e = tok(text(j), truncation=True, max_length=MAX_LEN)
            return {"input_ids": e["input_ids"], "attention_mask": e["attention_mask"], "labels": int(le.transform([y_all[j]])[0])}
    for BS, GA in [(8, 2), (4, 4), (2, 8)]:          # effective batch 16, as in re_08
        try:
            gc.collect(); torch.cuda.empty_cache()
            bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16,
                                     bnb_4bit_use_double_quant=True)
            base = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=len(le.classes_),
                                                                      quantization_config=bnb, device_map={"": 0})
            base = prepare_model_for_kbit_training(base); base.config.pad_token_id = tok.pad_token_id
            m = get_peft_model(base, LoraConfig(task_type=TaskType.SEQ_CLS, r=16, lora_alpha=32, lora_dropout=0.05,
                                                target_modules=["q_proj", "v_proj"], modules_to_save=["score"]))
            m.config.use_cache = False
            args = TrainingArguments(output_dir=str(ROOT/"models"/f"tmp_gen_{DATASET}"), per_device_train_batch_size=BS,
                                     gradient_accumulation_steps=GA, num_train_epochs=EPOCHS, learning_rate=2e-4,
                                     bf16=BF16, fp16=(not BF16), gradient_checkpointing=True, logging_steps=50,
                                     save_strategy="no", report_to=[], seed=SEED)
            Trainer(model=m, args=args, train_dataset=DS(),
                    data_collator=DataCollatorWithPadding(tokenizer=tok, padding="longest", return_tensors="pt")).train()
            return m
        except torch.cuda.OutOfMemoryError:
            log(f"OOM batch={BS} -> smaller"); m = base = None; gc.collect(); torch.cuda.empty_cache()
    raise RuntimeError("OOM at batch 2")


def llm_predict(m, idx):
    dev = next(m.parameters()).device; m.eval(); idx = np.asarray(idx)
    T = [text(j) for j in idx]; order = np.argsort([len(t) for t in T]); out = np.empty(len(idx), dtype=int)
    for s in range(0, len(idx), 16):
        sl = order[s:s+16]
        enc = tok([T[k] for k in sl], truncation=True, max_length=MAX_LEN, padding="longest", return_tensors="pt").to(dev)
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(dev.type == "cuda")):
            out[sl] = m(**enc).logits.argmax(-1).cpu().numpy()
    return out


res = _load(OUT, {"dataset": DATASET, "benign": benign, "model": None if CLASSIC_ONLY else MODEL, "featset": FEATSET or "paper",
                  "holdout": HOLDOUT, "config": {"train_cap": TRAIN_CAP, "epochs": EPOCHS, "max_len": MAX_LEN, "seed": SEED,
                  "unseen_cap": UNSEEN_CAP, "rf": "200 trees, balanced", "xgb": "300 trees, depth 8",
                  "note": "revision 1: re_08 protocol + XGBoost, no truncation"},
                  "fold_members": folds_def, "folds": {}})
for fi, (F, H) in enumerate(folds_def.items(), 1):
    if F in res["folds"]: log(f"[{fi}/{len(folds_def)}] {F} done, skip"); continue
    t0 = time.time(); sel = balanced(tr_idx[~np.isin(y_all[tr_idx], H)]); le = LabelEncoder().fit(y_all[sel])
    h, ben, oth = eval_rows(H)
    if len(h) == 0: log(f"[{fi}/{len(folds_def)}] {F} skipped (0 test rows)"); continue
    log(f"[{fi}/{len(folds_def)}] UNSEEN={F} {H if len(H) > 1 else ''} train={len(sel):,} H={len(h):,} benign={len(ben):,} known={len(oth):,}")
    rf = RandomForestClassifier(n_estimators=200, n_jobs=-1, class_weight="balanced", random_state=SEED).fit(X[sel], y_all[sel])
    r_rf = rates({"H": rf.predict(X[h]), "ben": rf.predict(X[ben]), "oth": rf.predict(X[oth])})
    xg = xgb.XGBClassifier(n_estimators=300, max_depth=8, n_jobs=-1, tree_method="hist", random_state=SEED,
                           num_class=len(le.classes_), objective="multi:softmax", eval_metric="mlogloss").fit(X[sel], le.transform(y_all[sel]))
    xp = lambda i: le.inverse_transform(xg.predict(X[i]).astype(int))
    r_xg = rates({"H": xp(h), "ben": xp(ben), "oth": xp(oth)})
    log(f"    RF : {r_rf}\n    XGB: {r_xg}")
    fold = {"n_unseen": int(len(h)), "rf": r_rf, "xgb": r_xg}
    if not CLASSIC_ONLY:
        m = train_llm(sel, le); lp = lambda i: le.inverse_transform(llm_predict(m, i))
        fold["llm"] = rates({"H": lp(h), "ben": lp(ben), "oth": lp(oth)}); log(f"    LLM: {fold['llm']}")
        del m; gc.collect(); torch.cuda.empty_cache()
    fold["sec"] = round(time.time() - t0); res["folds"][F] = fold
    (RES/"v4").mkdir(parents=True, exist_ok=True); json.dump(res, open(OUT, "w"), indent=2, ensure_ascii=False)

for k in ("rf", "xgb", "llm"):
    u = [v[k]["unseen_recall"] for v in res["folds"].values() if k in v]
    if u: res.setdefault("summary", {})[k] = {"mean_unseen_recall": round(float(np.mean(u)), 4), "min": round(float(np.min(u)), 4)}
json.dump(res, open(OUT, "w"), indent=2, ensure_ascii=False)
log(f"SUMMARY {res.get('summary')} -> {OUT} | DONE")
