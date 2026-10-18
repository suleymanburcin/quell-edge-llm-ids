#!/usr/bin/env python
# ================== QUELL re_06 — QUANTIZATION ACCURACY RETENTION (tam-baglam yeniden kosum) ==================
# FP16 (merged) vs INT8 (bitsandbytes) macro-F1/accuracy karsilastirmasi.
# Eski kosum MAX_LEN=256 idi (truncation bug). Burada MAX_LEN otomatik (p99+16, cap=2048).
# Protokol: 4-bit QLoRA egitim -> adapter kaydedip belle'i bos -> FP16 merged eval -> INT8 eval.
# Edge_iiotset tek dataset (orijinal 06 notebook ile ayni).
#
# KULLANIM (YPB terminalinde):
#   cd /workspace/project
#   nohup python re_06_quantize.py > ~/quant_log.txt 2>&1 &
#   python re_06_quantize.py --status
#   tail -f ~/quant_log.txt
# ==============================================================================================================
import os, subprocess, sys, json, time, gc
from pathlib import Path

ROOT = Path.home()/"quell-edge-llm-ids"
RES  = ROOT/"results"
OUT  = RES/"quant_report_fullctx.json"         # ESKIYI EZMEZ (yeni dosya)
PROG = RES/"quant_progress.json"
ADAPTER_DIR = ROOT/"models"/"qwen_quant_fullctx"

DATASET  = "edge_iiotset"
MODEL    = "Qwen/Qwen2.5-1.5B"
TRAIN_CAP = 2000           # orijinal 06 protokolu
EPOCHS    = 2              # orijinal 06 protokolu
EVAL_CAP  = 60000          # orijinal 06 protokolu
SEED      = 42
MAXLEN_CAP = 2048          # quantize icin tam baglam (tek kosum, tolere edilir)

# ---------- --status modu (agir importlardan once) ----------
def _load(p, d):
    try: return json.load(open(p))
    except Exception: return d

if "--status" in sys.argv:
    prog = _load(PROG, {})
    rep  = _load(OUT, {})
    print("="*64); print("QUELL re_06 quantize DURUM"); print("="*64)
    if prog:
        for k, v in prog.items(): print(f"  {k}: {v}")
    else:
        print("  (henuz progress yok)")
    print("-"*64)
    if rep:
        for ds, models in rep.items():
            for mname, precs in models.items():
                for prec, vals in precs.items():
                    print(f"  {ds} | {prec}: macro_f1={vals.get('macro_f1','?')}  acc={vals.get('acc','?')}")
    else:
        print("  (sonuc yok)")
    sys.exit(0)

# ---------- GPU sec ----------
try:
    _o = subprocess.check_output("nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits", shell=True, text=True)
    _f = [(int(x.split(",")[0]), int(x.split(",")[1])) for x in _o.strip().splitlines()]
    _b = max(_f, key=lambda t: t[1]); os.environ["CUDA_VISIBLE_DEVICES"] = str(_b[0])
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    print("selected GPU:", _b[0], "| free(MiB):", _f, flush=True)
except Exception as e:
    print("GPU secim atlandi:", e, flush=True)

import numpy as np, pandas as pd
from pandas.api.types import is_numeric_dtype
for pk in ["transformers", "peft", "accelerate", "bitsandbytes"]:
    try: __import__(pk)
    except Exception: subprocess.run([sys.executable, "-m", "pip", "install", "-q", pk])
import torch
from transformers import (AutoTokenizer, AutoModelForSequenceClassification, BitsAndBytesConfig,
                          TrainingArguments, Trainer, TrainerCallback, DataCollatorWithPadding)
from peft import LoraConfig, get_peft_model, PeftModel, TaskType, prepare_model_for_kbit_training
from sklearn.metrics import f1_score, accuracy_score
from sklearn.preprocessing import LabelEncoder

torch.manual_seed(SEED); np.random.seed(SEED)
BF16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
PROC, SPL = ROOT/"data"/"processed", ROOT/"splits"

def write_progress(**kw):
    cur = _load(PROG, {})
    cur.update(kw); cur["ts"] = time.strftime("%Y-%m-%d %H:%M:%S")
    json.dump(cur, open(PROG, "w"), indent=2, ensure_ascii=False)

class Heartbeat(TrainerCallback):
    def on_log(s, args, state, control, logs=None, **kw):
        write_progress(stage="egitim", step=int(state.global_step), max_step=int(state.max_steps),
                       loss=(round(float(logs.get("loss")), 4) if logs and "loss" in logs else None))

# ============================================================================================================
print(f"\n########## re_06 QUANTIZE — {DATASET} ##########", flush=True)
write_progress(stage="basladi", dataset=DATASET)

# --- Veri hazirligi ---
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
K = len(classes_all)
le = LabelEncoder().fit(classes_all)
print(f"sinif={K}, feats={len(feats)}, train={len(tr_idx)}, test={len(te_idx)}", flush=True)

# feature -> text
def row_to_text(r):
    parts = []
    for c in feats:
        v = r[c]
        if isinstance(v, (float, np.floating)): v = round(float(v), 4)
        parts.append(f"{c}={v}")
    return "Network traffic flow. " + ", ".join(parts) + " . Attack type:"

print("feature->text...", flush=True)
texts_all = df.apply(row_to_text, axis=1).values

# Otomatik MAX_LEN
tok0 = AutoTokenizer.from_pretrained(MODEL)
if tok0.pad_token is None: tok0.pad_token = tok0.eos_token
_samp = np.random.default_rng(SEED).choice(len(texts_all), min(4000, len(texts_all)), replace=False)
_lens = np.array([len(tok0(texts_all[j], truncation=False)["input_ids"]) for j in _samp])
MAX_LEN = int(min(int(np.percentile(_lens, 99)) + 16, MAXLEN_CAP))
trunc_rate = float(np.mean(_lens > MAX_LEN))
print(f"MAX_LEN={MAX_LEN} (p99={int(np.percentile(_lens,99))}, mean={np.mean(_lens):.0f}, "
      f"max={int(_lens.max())}) | KIRPMA={trunc_rate*100:.1f}%", flush=True)
write_progress(stage="veri_hazir", max_len=MAX_LEN, trunc_rate_pct=round(trunc_rate*100, 2),
               n_classes=K, n_feats=len(feats))

# --- Dengeli train subset ---
rng = np.random.default_rng(SEED); sel = []
for cls in classes_all:
    ids = tr_idx[y_all[tr_idx] == cls]
    if len(ids) > TRAIN_CAP: ids = rng.choice(ids, TRAIN_CAP, replace=False)
    sel += ids.tolist()
tr_sel = np.array(sorted(sel))
print(f"train subset: {len(tr_sel)} (cap {TRAIN_CAP}/sinif)", flush=True)

# --- Eval subset ---
if len(te_idx) > EVAL_CAP:
    eval_idx = rng.choice(te_idx, EVAL_CAP, replace=False)
else:
    eval_idx = te_idx
print(f"eval subset: {len(eval_idx)}", flush=True)

# ============================================================================================================
# ADIM 1: 4-bit QLoRA egitim + adapter kaydet
# ============================================================================================================
print("\n--- ADIM 1: QLoRA egitim ---", flush=True)
write_progress(stage="egitim_basladi")

tok = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token is None: tok.pad_token = tok.eos_token

class DS(torch.utils.data.Dataset):
    def __init__(s, idx): s.idx = idx
    def __len__(s): return len(s.idx)
    def __getitem__(s, i):
        j = s.idx[i]
        e = tok(texts_all[j], truncation=True, max_length=MAX_LEN)
        return {"input_ids": e["input_ids"], "attention_mask": e["attention_mask"],
                "labels": int(le.transform([y_all[j]])[0])}

collator = DataCollatorWithPadding(tokenizer=tok, padding="longest", return_tensors="pt")

# OOM auto-fallback
model = None
for BS, GA in [(8, 2), (4, 4), (2, 8)]:
    try:
        if model is not None:
            del model; gc.collect(); torch.cuda.empty_cache()
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                  bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
        base = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=K,
                                                                   quantization_config=bnb, device_map={"": 0})
        base = prepare_model_for_kbit_training(base); base.config.pad_token_id = tok.pad_token_id
        model = get_peft_model(base, LoraConfig(task_type=TaskType.SEQ_CLS, r=16, lora_alpha=32,
                                                 lora_dropout=0.05, target_modules=["q_proj","v_proj"],
                                                 modules_to_save=["score"]))
        model.config.use_cache = False
        args = TrainingArguments(output_dir=str(ROOT/"models"/"tmp_quant"),
                                  per_device_train_batch_size=BS, gradient_accumulation_steps=GA,
                                  num_train_epochs=EPOCHS, learning_rate=2e-4, bf16=BF16, fp16=(not BF16),
                                  gradient_checkpointing=True, logging_steps=50, save_strategy="no",
                                  report_to=[], seed=SEED)
        print(f"  batch={BS} grad_accum={GA} (efektif {BS*GA}) deneniyor...", flush=True)
        Trainer(model=model, args=args, train_dataset=DS(tr_sel),
                data_collator=collator, callbacks=[Heartbeat()]).train()
        break
    except torch.cuda.OutOfMemoryError:
        print(f"  OOM batch={BS} -> kucultuluyor", flush=True)
        try: del model, base
        except Exception: pass
        model = None; gc.collect(); torch.cuda.empty_cache()
        continue
else:
    raise RuntimeError("batch=2'de bile OOM")

# Adapter kaydet
model.save_pretrained(ADAPTER_DIR)
tok.save_pretrained(ADAPTER_DIR)
quell_meta = {"classes": classes_all, "features": feats, "max_len": MAX_LEN,
              "base_model": MODEL, "train_cap": TRAIN_CAP, "epochs": EPOCHS, "seed": SEED}
json.dump(quell_meta, open(ADAPTER_DIR/"quell_meta.json", "w"), indent=2)
print(f"  adapter kaydedildi: {ADAPTER_DIR}", flush=True)
write_progress(stage="adapter_kaydedildi", adapter_dir=str(ADAPTER_DIR))

# Egitim modelini bosalt
del model, base; gc.collect(); torch.cuda.empty_cache()
print("  egitim modeli bosaltildi", flush=True)

# ============================================================================================================
# Ortak eval fonksiyonu
# ============================================================================================================
def evaluate_model(model, tag):
    model.eval(); preds = []; bs = 64
    # uzunluga gore sirala -> padding optimal
    order = np.argsort([len(texts_all[j]) for j in eval_idx])
    all_preds = np.empty(len(eval_idx), dtype=int)
    for s in range(0, len(eval_idx), bs):
        sl = order[s:s+bs]; js = eval_idx[sl]
        enc = tok(list(texts_all[js]), truncation=True, max_length=MAX_LEN,
                  padding="longest", return_tensors="pt").to(model.device)
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            lo = model(**enc).logits
        all_preds[sl] = lo.argmax(-1).cpu().numpy()
    y_true = le.transform(y_all[eval_idx])
    acc = round(float(accuracy_score(y_true, all_preds)), 4)
    mf1 = round(float(f1_score(y_true, all_preds, average="macro", labels=list(range(K)), zero_division=0)), 4)
    print(f"  {tag}: acc={acc}  macro_f1={mf1}", flush=True)
    return {"acc": acc, "macro_f1": mf1}

# ============================================================================================================
# ADIM 2: FP16 merged eval
# ============================================================================================================
print("\n--- ADIM 2: FP16 eval (merged) ---", flush=True)
write_progress(stage="fp16_eval")

tok = AutoTokenizer.from_pretrained(ADAPTER_DIR)
fp16_base = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=K,
                                                                torch_dtype=torch.float16, device_map={"": 0})
fp16_base.config.pad_token_id = tok.pad_token_id
fp16_model = PeftModel.from_pretrained(fp16_base, ADAPTER_DIR)
fp16_model = fp16_model.merge_and_unload()
fp16_result = evaluate_model(fp16_model, "FP16")

# FP16 modeli bosalt
del fp16_model, fp16_base; gc.collect(); torch.cuda.empty_cache()
print("  FP16 model bosaltildi", flush=True)

# ============================================================================================================
# ADIM 3: INT8 eval
# ============================================================================================================
print("\n--- ADIM 3: INT8 eval ---", flush=True)
write_progress(stage="int8_eval")

int8_bnb = BitsAndBytesConfig(load_in_8bit=True)
int8_base = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=K,
                                                                quantization_config=int8_bnb, device_map={"": 0})
int8_base.config.pad_token_id = tok.pad_token_id
int8_model = PeftModel.from_pretrained(int8_base, ADAPTER_DIR)
# INT8'de merge_and_unload yapilamaz, adapter'li olarak eval
int8_result = evaluate_model(int8_model, "INT8")

del int8_model, int8_base; gc.collect(); torch.cuda.empty_cache()
print("  INT8 model bosaltildi", flush=True)

# ============================================================================================================
# Sonuclari kaydet
# ============================================================================================================
result = {
    DATASET: {
        MODEL: {
            "fp16": fp16_result,
            "int8": int8_result
        },
        "config": {
            "max_len": MAX_LEN,
            "trunc_rate_pct": round(trunc_rate * 100, 2),
            "token_mean": round(float(np.mean(_lens)), 1),
            "token_p99": int(np.percentile(_lens, 99)),
            "train_cap": TRAIN_CAP,
            "epochs": EPOCHS,
            "eval_n": len(eval_idx),
            "note": "full-context re-run (truncation fix)"
        }
    }
}

json.dump(result, open(OUT, "w"), indent=2, ensure_ascii=False)
write_progress(stage="TAMAMLANDI", fp16=fp16_result, int8=int8_result)

delta_f1 = int8_result["macro_f1"] - fp16_result["macro_f1"]
delta_acc = int8_result["acc"] - fp16_result["acc"]
print(f"\n{'='*64}", flush=True)
print(f"SONUC: FP16 macro_f1={fp16_result['macro_f1']}  INT8 macro_f1={int8_result['macro_f1']}  "
      f"Δ={delta_f1:+.4f}", flush=True)
print(f"       FP16 acc={fp16_result['acc']}  INT8 acc={int8_result['acc']}  "
      f"Δ={delta_acc:+.4f}", flush=True)
print(f"Kaydedildi: {OUT}", flush=True)
print(f"{'='*64}", flush=True)
