#!/usr/bin/env python
# ============================================================================
# QUELL RE-RUN (truncation fix) — Edge headline, ARKA PLAN / KOPMAYA DAYANIKLI
# - VPN/tarayici kopsa bile calisir (nohup veya tmux ile baslat).
# - Resume: bitmis seedleri atlar; seed ORTASINDA kesilirse checkpoint'ten devam eder.
# - Ilerleme: results/rerun_progress.json + rerun.log dosyalarina yazilir.
#
# CALISTIRMA (YPB terminalinde):
#     cd ~/quell-edge-llm-ids/scripts      # veya bu .py nereye koyduysan
#     tmux new -s quell                    # (tmux varsa; en saglami)
#     python re_05_5_multiseed_edge.py
#     # detach: Ctrl-b sonra d  |  geri don: tmux attach -t quell
#   VEYA tmux yoksa:
#     nohup python re_05_5_multiseed_edge.py > rerun.log 2>&1 &
#
# ILERLEME GORMEK (istedigin an, ayri terminal/hucre):
#     python re_05_5_multiseed_edge.py --status
#   veya:  tail -f rerun.log
# ============================================================================
import os, sys, json, time, gc, shutil
from pathlib import Path
from datetime import datetime

ROOT = Path.home()/"quell-edge-llm-ids"
RES  = ROOT/"results"
OUT  = RES/"multiseed_report_fullctx.json"          # seed 42 & 1 burada; devam eder
PROG = RES/"rerun_progress.json"

# ===== AYARLAR (seed 42/1 ile AYNI olmali; degistirme) =====
DATASET   = "edge_iiotset"
MODEL     = "Qwen/Qwen2.5-1.5B"
SEEDS     = [42, 1, 2, 3, 4]     # 42 & 1 zaten kayitli -> atlanir; 2,3,4 kosulur (~28 saat)
TRAIN_CAP = 2000
EPOCHS    = 2
MAX_LEN   = 2048                 # SABIT — seed 42/1 bununla kosuldu
# ===========================================================

def log(msg):
    line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    print(line, flush=True)

def write_progress(status, extra=None):
    RES.mkdir(parents=True, exist_ok=True)
    done = []
    if OUT.exists():
        try: done = list(json.load(open(OUT)).get(DATASET, {}).get("seeds", {}).keys())
        except Exception: pass
    d = {"updated": f"{datetime.now():%Y-%m-%d %H:%M:%S}", "status": status,
         "seeds_done": done, "seeds_target": [str(s) for s in SEEDS]}
    if extra: d.update(extra)
    json.dump(d, open(PROG, "w"), indent=2, ensure_ascii=False)

# ---------- --status modu: sadece durumu yazdir, cikis ----------
if "--status" in sys.argv:
    print("=== QUELL re-run durum ===")
    if PROG.exists(): print(open(PROG).read())
    else: print("progress dosyasi yok (henuz baslamamis olabilir)")
    if OUT.exists():
        d = json.load(open(OUT))[DATASET]; S = d["seeds"]
        print(f"\nMAX_LEN={d.get('max_len')}  bitmis seed: {list(S.keys())}")
        import statistics as st
        for key in ["llm", "rf", "xgb"]:
            v = [S[k][key]["macro_f1"] for k in S]
            if v:
                m = sum(v)/len(v); sd = (st.pstdev(v) if len(v) > 1 else 0.0)
                print(f"  {key:4s} macro-F1  {m:.4f} +/- {sd:.4f}   {[round(x,4) for x in v]}")
    sys.exit(0)

# ---------- normal kosu ----------
try:
    import subprocess
    _o = subprocess.check_output("nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits", shell=True, text=True)
    _f = [(int(x.split(",")[0]), int(x.split(",")[1])) for x in _o.strip().splitlines()]
    _b = max(_f, key=lambda t: t[1]); os.environ["CUDA_VISIBLE_DEVICES"] = str(_b[0])
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    log(f"selected GPU {_b[0]} | free(MiB): {_f}")
except Exception as e:
    log(f"GPU secim atlandi: {e}")

import numpy as np, pandas as pd
from pandas.api.types import is_numeric_dtype
import torch
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          BitsAndBytesConfig, TrainingArguments, Trainer)
from transformers.trainer_utils import get_last_checkpoint
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
import xgboost as xgb

PROC = ROOT/"data"/"processed"; SPL = ROOT/"splits"
rep = json.load(open(RES/"split_report.json")); meta = rep[DATASET]
label = meta["label_col"]; group = meta.get("group_col"); tcol = meta.get("time_col")
LABELISH = {"label","attack","attack_type","attack_label","type","class","category","marker","__label__"}
df = pd.read_parquet(PROC/f"{DATASET}.parquet").reset_index(drop=True)
sp = np.load(SPL/f"{DATASET}_split.npz"); tr_idx, te_idx = sp["train"], sp["test"]
drop = set([label]) | set(meta.get("leaky_candidates", []))
if group: drop.add(group)
if tcol:  drop.add(tcol)
for c in df.columns:
    if c != label and c.lower() in LABELISH: drop.add(c)
feats = [c for c in df.columns if c not in drop]
y_all = df[label].astype(str).values
classes_all = sorted(pd.unique(y_all).tolist())
BF16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
log(f"{DATASET}: feature={len(feats)} class={len(classes_all)} test={len(te_idx):,} | MAX_LEN={MAX_LEN} | seeds={SEEDS}")

Xdf = df[feats].copy()
for c in Xdf.columns:
    if not is_numeric_dtype(Xdf[c]): Xdf[c] = pd.factorize(Xdf[c])[0]
X = np.nan_to_num(Xdf.values.astype("float32"))

def row_to_text(r):
    parts = []
    for c in feats:
        v = r[c]
        if isinstance(v, (float, np.floating)): v = round(float(v), 4)
        parts.append(f"{c}={v}")
    return "Network traffic flow. " + ", ".join(parts) + " . Attack type:"
log("feature->text ...")
texts_all = df.apply(row_to_text, axis=1).values

def balanced_train(seed):
    rng = np.random.default_rng(seed); sel = []
    for cls in pd.unique(y_all[tr_idx]):
        ids = tr_idx[y_all[tr_idx] == cls]
        if len(ids) > TRAIN_CAP: ids = rng.choice(ids, TRAIN_CAP, replace=False)
        sel += ids.tolist()
    return np.array(sorted(sel))

def scores(yt, pred):
    return (round(accuracy_score(yt, pred), 4),
            round(f1_score(yt, pred, average="macro", labels=classes_all, zero_division=0), 4))
yte = y_all[te_idx]

from transformers import TrainerCallback
class HeartbeatCB(TrainerCallback):
    def __init__(self, seed): self.seed = seed
    def on_log(self, args, state, control, logs=None, **kw):
        if logs and "loss" in logs:
            write_progress(f"seed {self.seed} egitiliyor",
                           {"epoch": round(state.epoch or 0, 2),
                            "step": state.global_step, "max_steps": state.max_steps,
                            "train_loss": round(logs["loss"], 4)})

def run_llm(tr_sel, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    le = LabelEncoder().fit(y_all[tr_sel]); K = len(le.classes_)
    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    class DS(torch.utils.data.Dataset):
        def __init__(s, idx): s.idx = idx
        def __len__(s): return len(s.idx)
        def __getitem__(s, i):
            j = s.idx[i]; e = tok(texts_all[j], truncation=True, max_length=MAX_LEN, padding="max_length", return_tensors="pt")
            it = {k: v.squeeze(0) for k, v in e.items()}
            it["labels"] = torch.tensor(int(le.transform([y_all[j]])[0])); return it
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    base = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=K, quantization_config=bnb, device_map={"": 0})
    base = prepare_model_for_kbit_training(base); base.config.pad_token_id = tok.pad_token_id
    model = get_peft_model(base, LoraConfig(task_type=TaskType.SEQ_CLS, r=16, lora_alpha=32, lora_dropout=0.05,
                                            target_modules=["q_proj", "v_proj"], modules_to_save=["score"]))
    model.config.use_cache = False
    ckpt_dir = str(ROOT/"models"/f"ckpt_ms_seed{seed}")
    args = TrainingArguments(
        output_dir=ckpt_dir, per_device_train_batch_size=2, gradient_accumulation_steps=8,
        num_train_epochs=EPOCHS, learning_rate=2e-4, bf16=BF16, fp16=(not BF16),
        gradient_checkpointing=True, logging_steps=50,
        save_strategy="steps", save_steps=200, save_total_limit=1,   # <-- seed-ici checkpoint
        report_to=[], seed=seed)
    trainer = Trainer(model=model, args=args, train_dataset=DS(tr_sel), callbacks=[HeartbeatCB(seed)])
    last = get_last_checkpoint(ckpt_dir) if os.path.isdir(ckpt_dir) else None
    if last:
        log(f"seed {seed}: checkpoint bulundu ({last}) -> devam ediliyor")
    trainer.train(resume_from_checkpoint=last)
    dev = next(model.parameters()).device; model.eval(); preds = []; bs = 8
    write_progress(f"seed {seed} cikarim (inference)")
    for s in range(0, len(te_idx), bs):
        js = te_idx[s:s+bs]
        enc = tok(list(texts_all[js]), truncation=True, max_length=MAX_LEN, padding=True, return_tensors="pt").to(dev)
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(dev.type == "cuda")):
            preds += model(**enc).logits.argmax(-1).cpu().tolist()
    p = le.inverse_transform(np.array(preds))
    del model, base; gc.collect(); torch.cuda.empty_cache()
    try: shutil.rmtree(ckpt_dir, ignore_errors=True)   # disk temizle
    except Exception: pass
    return scores(yte, p)

allm = json.load(open(OUT)) if OUT.exists() else {}
allm.setdefault(DATASET, {"model": MODEL, "train_cap": TRAIN_CAP, "epochs": EPOCHS,
                          "max_len": int(MAX_LEN), "note": "full-context re-run (truncation fix)", "seeds": {}})
S = allm[DATASET]["seeds"]

for seed in SEEDS:
    k = str(seed)
    if k in S and all(m in S[k] for m in ["llm", "rf", "xgb"]):
        log(f"seed {seed} zaten var, atlaniyor"); continue
    t0 = time.time(); tr_sel = balanced_train(seed)
    log(f"=== seed {seed} | train={len(tr_sel):,} basladi ===")
    write_progress(f"seed {seed} basladi (RF/XGB)")
    rf = RandomForestClassifier(n_estimators=300, n_jobs=-1, class_weight="balanced", random_state=seed).fit(X[tr_sel], y_all[tr_sel])
    rf_acc, rf_f1 = scores(yte, rf.predict(X[te_idx])); log(f"  RF : acc={rf_acc} macroF1={rf_f1}")
    le2 = LabelEncoder().fit(y_all[tr_sel])
    xg = xgb.XGBClassifier(n_estimators=300, max_depth=8, n_jobs=-1, tree_method="hist", random_state=seed,
                           num_class=len(le2.classes_), objective="multi:softmax", eval_metric="mlogloss").fit(X[tr_sel], le2.transform(y_all[tr_sel]))
    xg_acc, xg_f1 = scores(yte, le2.inverse_transform(xg.predict(X[te_idx]))); log(f"  XGB: acc={xg_acc} macroF1={xg_f1}")
    llm_acc, llm_f1 = run_llm(tr_sel, seed); log(f"  LLM: acc={llm_acc} macroF1={llm_f1}")
    S[k] = {"llm": {"acc": llm_acc, "macro_f1": llm_f1}, "rf": {"acc": rf_acc, "macro_f1": rf_f1},
            "xgb": {"acc": xg_acc, "macro_f1": xg_f1}, "sec": round(time.time()-t0, 0)}
    json.dump(allm, open(OUT, "w"), indent=2, ensure_ascii=False)
    write_progress(f"seed {seed} BITTI", {"llm_macro_f1": llm_f1})
    log(f"  kaydedildi ({time.time()-t0:.0f}s) -> {OUT.name}")

# ---- OZET ----
import statistics as st
def ms(key):
    v = [S[k][key]["macro_f1"] for k in S]; return (sum(v)/len(v), st.pstdev(v) if len(v) > 1 else 0.0, v)
log("================ COK-SEED OZET (Edge macro-F1) ================")
for key in ["llm", "rf", "xgb"]:
    m, s, v = ms(key); log(f"  {key:4s}  {m:.4f} +/- {s:.4f}   (seedler: {[round(x,3) for x in v]})")
lm, ls, _ = ms("llm"); rm, rs, _ = ms("rf")
log(f"  LLM - RF (macro-F1 fark) = {lm-rm:+.4f}   | LLM std={ls:.4f} RF std={rs:.4f}")
write_progress("TAMAMLANDI", {"llm_mean": round(lm, 4), "llm_std": round(ls, 4),
                              "rf_mean": round(rm, 4), "llm_minus_rf": round(lm-rm, 4)})
log("DONE.")
