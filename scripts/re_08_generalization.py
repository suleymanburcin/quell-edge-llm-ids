#!/usr/bin/env python
# ================== QUELL re_08 — UNKNOWN-ATTACK GENERALIZATION (tam-baglam yeniden kosum) ==================
# leave-one-attack-out: her attack sinifini sirayla EGITIMDEN tamamen cikar, test'te "hic gorulmemis" goster.
# RF ve LLM kapali-kume egitilir; tahmin ikiliye indirgenir (benign / attack). Unseen attack "attack" derse YAKALANMIS.
#
# BU SURUM: truncation bug DUZELTMESI. Eski scriptlerde MAX_LEN=256/512 -> edge/nbaiot kesiliyordu.
# Burada MAX_LEN dataset basina OTOMATIK (token uzunlugu p99 + 16, tavan 2048). Protokolun geri kalani AYNEN eski
# generalization ile (TRAIN_CAP=1500, EPOCHS=1, SEED=42) -> tek degisen truncation, temiz karsilastirma.
#
# DAYANIKLILIK: nohup ile arkaplanda calisir (VPN kopsa da surer). Fold-seviyesi resume (JSON'da olan fold atlanir).
# nbaiot oturum-gruplu split -> bazi mirai foldlarinda 0 test satiri -> otomatik atlanir. Progress dosyasi + --status.
#
# KULLANIM (YPB terminalinde):
#   cd /workspace/project    (veya scriptin oldugu yer)
#   nohup python re_08_generalization.py > ~/gen_log.txt 2>&1 &
#   # ilerleme:
#   python re_08_generalization.py --status
#   tail -f ~/gen_log.txt
# ============================================================================================================
import os, subprocess, sys, json, time, gc
from pathlib import Path

ROOT = Path.home()/"quell-edge-llm-ids"
RES  = ROOT/"results"
OUT  = RES/"generalization_report_fullctx.json"          # ESKIYI EZMEZ (yeni dosya)
PROG = RES/"gen_progress.json"

DATASETS = ["edge_iiotset", "nbaiot"]                      # SIRALI: once edge, sonra nbaiot
MODEL    = "Qwen/Qwen2.5-1.5B"
TRAIN_CAP, EPOCHS, SEED = 1500, 1, 42                      # ESKI generalization protokolu AYNEN
BENIGN_FPR_CAP, KNOWN_CAP = 5000, 5000
MAX_FOLDS = None
MAXLEN_CAP = 1024                                          # tavan (kullanici karari: hiz/dogruluk dengesi)

# ---------- --status modu (agir importlardan once) ----------
def _load(p, d):
    try: return json.load(open(p))
    except Exception: return d

if "--status" in sys.argv:
    prog = _load(PROG, {})
    allg = _load(OUT, {})
    print("="*64); print("QUELL re_08 generalization DURUM"); print("="*64)
    if prog:
        for k, v in prog.items(): print(f"  {k}: {v}")
    else:
        print("  (henuz progress yok - baslamamis olabilir)")
    print("-"*64)
    for ds in DATASETS:
        folds = allg.get(ds, {}).get("folds", {})
        print(f"  {ds}: {len(folds)} fold tamam -> {list(folds)}")
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
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

torch.manual_seed(SEED); np.random.seed(SEED)
BF16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
PROC, SPL = ROOT/"data"/"processed", ROOT/"splits"
FCLIP = 1e9

def write_progress(**kw):
    cur = _load(PROG, {})
    cur.update(kw); cur["ts"] = time.strftime("%Y-%m-%d %H:%M:%S")
    json.dump(cur, open(PROG, "w"), indent=2, ensure_ascii=False)

class Heartbeat(TrainerCallback):
    def __init__(s, tag): s.tag = tag; s.t = time.time()
    def on_log(s, args, state, control, logs=None, **kw):
        write_progress(stage="egitim", detail=s.tag,
                       step=int(state.global_step), max_step=int(state.max_steps),
                       loss=(round(float(logs.get("loss")), 4) if logs and "loss" in logs else None))

# ============================================================================================================
def run_dataset(ds):
    print(f"\n########## DATASET: {ds} ##########", flush=True)
    rep = json.load(open(RES/"split_report.json")); meta = rep[ds]
    label = meta["label_col"]; group = meta.get("group_col"); tcol = meta.get("time_col")
    LABELISH = {"label","attack","attack_type","attack_label","type","class","category","marker","__label__"}
    df = pd.read_parquet(PROC/f"{ds}.parquet").reset_index(drop=True)
    sp = np.load(SPL/f"{ds}_split.npz"); tr_idx, te_idx = sp["train"], sp["test"]
    drop = set([label]) | set(meta.get("leaky_candidates", []))
    if group: drop.add(group)
    if tcol: drop.add(tcol)
    for c in df.columns:
        if c != label and c.lower() in LABELISH: drop.add(c)
    feats = [c for c in df.columns if c not in drop]
    y_all = df[label].astype(str).values
    classes_all = sorted(pd.unique(y_all).tolist())

    benign = None
    for c in classes_all:
        lc = c.strip().lower()
        if lc in {"normal","benign","background"} or "normal" in lc or "benign" in lc:
            benign = c; break
    assert benign is not None, f"benign class not found. classes: {classes_all}"
    attacks = [c for c in classes_all if c != benign]
    if MAX_FOLDS: attacks = attacks[:MAX_FOLDS]

    # RF icin sayisal matris (nbaiot asiri buyuk/inf kirp)
    Xdf = df[feats].copy()
    for c in Xdf.columns:
        if not is_numeric_dtype(Xdf[c]): Xdf[c] = pd.factorize(Xdf[c])[0]
    X = np.nan_to_num(Xdf.values.astype("float64"), nan=0.0, posinf=0.0, neginf=0.0)
    X = np.clip(X, -FCLIP, FCLIP).astype("float32")

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

    # ---- OTOMATIK MAX_LEN (truncation fix): token uzunlugu p99 + 16, tavan 2048 ----
    tok0 = AutoTokenizer.from_pretrained(MODEL)
    if tok0.pad_token is None: tok0.pad_token = tok0.eos_token
    _samp = np.random.default_rng(SEED).choice(len(texts_all), min(4000, len(texts_all)), replace=False)
    _lens = np.array([len(tok0(texts_all[j], truncation=False)["input_ids"]) for j in _samp])
    MAX_LEN = int(min(int(np.percentile(_lens, 99)) + 16, MAXLEN_CAP))
    trunc_rate = float(np.mean(_lens > MAX_LEN))   # bu MAX_LEN'de kirpilan ornek orani (durustluk icin)
    print(f"{ds}: benign='{benign}' | {len(attacks)} attack folds | features={len(feats)} | "
          f"MAX_LEN={MAX_LEN} (p99 token={int(np.percentile(_lens,99))}, mean={np.mean(_lens):.0f}, "
          f"max={int(_lens.max())}) | KIRPMA ORANI={trunc_rate*100:.1f}% (bu MAX_LEN'i asan ornek)", flush=True)

    def balanced_train(tr_pool):
        rng = np.random.default_rng(SEED); sel = []
        for cls in pd.unique(y_all[tr_pool]):
            ids = tr_pool[y_all[tr_pool] == cls]
            if len(ids) > TRAIN_CAP: ids = rng.choice(ids, TRAIN_CAP, replace=False)
            sel += ids.tolist()
        return np.array(sorted(sel))

    def eval_rows_for(H):
        r = np.random.default_rng(SEED)
        H_rows = te_idx[y_all[te_idx] == H]
        ben = te_idx[y_all[te_idx] == benign]
        if len(ben) > BENIGN_FPR_CAP: ben = r.choice(ben, BENIGN_FPR_CAP, replace=False)
        oth = te_idx[(y_all[te_idx] != H) & (y_all[te_idx] != benign)]
        if len(oth) > KNOWN_CAP: oth = r.choice(oth, KNOWN_CAP, replace=False)
        return H_rows, ben, oth

    def train_llm(tr_sel, le, tag):
        K = len(le.classes_)
        tok = AutoTokenizer.from_pretrained(MODEL)
        if tok.pad_token is None: tok.pad_token = tok.eos_token
        # DINAMIK PADDING: her ornek 2048'e degil, batch'teki en uzuna kadar doldurulur.
        # Truncation yine MAX_LEN'de; sonuc birebir ayni (padding maskeli), sadece bosa hesap kesilir.
        class DS(torch.utils.data.Dataset):
            def __init__(s, idx): s.idx = idx
            def __len__(s): return len(s.idx)
            def __getitem__(s, i):
                j = s.idx[i]
                e = tok(texts_all[j], truncation=True, max_length=MAX_LEN)   # padding YOK
                return {"input_ids": e["input_ids"], "attention_mask": e["attention_mask"],
                        "labels": int(le.transform([y_all[j]])[0])}
        collator = DataCollatorWithPadding(tokenizer=tok, padding="longest", return_tensors="pt")
        # Efektif batch HEP 16 (eski generalization protokolu 8x2=16 ile birebir).
        # Hiz icin batch 8'den basla; GPU dolarsa OOM'da otomatik 4'e, sonra 2'ye dus.
        model = None
        for BS, GA in [(8, 2), (4, 4), (2, 8)]:
            try:
                if model is not None:
                    del model
                gc.collect(); torch.cuda.empty_cache()
                bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                         bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
                base = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=K,
                                                                          quantization_config=bnb, device_map={"": 0})
                base = prepare_model_for_kbit_training(base); base.config.pad_token_id = tok.pad_token_id
                model = get_peft_model(base, LoraConfig(task_type=TaskType.SEQ_CLS, r=16, lora_alpha=32,
                                                        lora_dropout=0.05, target_modules=["q_proj","v_proj"],
                                                        modules_to_save=["score"]))
                model.config.use_cache = False
                args = TrainingArguments(output_dir=str(ROOT/"models"/"tmp_gen"),
                                         per_device_train_batch_size=BS, gradient_accumulation_steps=GA,
                                         num_train_epochs=EPOCHS, learning_rate=2e-4, bf16=BF16, fp16=(not BF16),
                                         gradient_checkpointing=True, logging_steps=50, save_strategy="no",
                                         report_to=[], seed=SEED)
                print(f"    egitim batch={BS} grad_accum={GA} (efektif 16) deneniyor...", flush=True)
                Trainer(model=model, args=args, train_dataset=DS(tr_sel),
                        data_collator=collator, callbacks=[Heartbeat(tag)]).train()
                return model, tok
            except torch.cuda.OutOfMemoryError:
                print(f"    OOM batch={BS} -> kucultuluyor", flush=True)
                write_progress(stage="OOM_retry", batch=BS)
                try: del model, base
                except Exception: pass
                model = None; gc.collect(); torch.cuda.empty_cache()
                continue
        raise RuntimeError("batch=2'de bile OOM")

    def llm_predict(model, tok, idx):
        dev = next(model.parameters()).device; model.eval(); bs = 16
        idx = np.asarray(idx)
        # uzunluga gore sirala -> batch icinde dinamik padding minimal (hiz); sonra orijinal siraya geri koy
        order = np.argsort([len(t) for t in texts_all[idx]])
        preds = np.empty(len(idx), dtype=int)
        for s in range(0, len(idx), bs):
            sl = order[s:s+bs]; js = idx[sl]
            enc = tok(list(texts_all[js]), truncation=True, max_length=MAX_LEN, padding="longest", return_tensors="pt").to(dev)
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(dev.type=="cuda")):
                lo = model(**enc).logits
            preds[sl] = lo.argmax(-1).cpu().numpy()
        return preds

    def rates(pred_labels):
        def det(m): return float(np.mean(m != benign)) if len(m) else float("nan")
        return dict(unseen_recall=round(det(pred_labels["H"]), 4),
                    benign_fpr=round(det(pred_labels["ben"]), 4),
                    known_recall=round(det(pred_labels["oth"]), 4))

    allg = _load(OUT, {})
    allg.setdefault(ds, {"benign": benign, "model": MODEL,
                         "config": {"train_cap": TRAIN_CAP, "epochs": EPOCHS, "max_len": MAX_LEN,
                                    "trunc_rate_pct": round(trunc_rate*100, 2),
                                    "token_mean": round(float(_lens.mean()), 1),
                                    "token_p99": int(np.percentile(_lens, 99)),
                                    "note": "full-context re-run (truncation fix; MAX_LEN cap=1024)"}, "folds": {}})
    folds = allg[ds]["folds"]

    for fi, H in enumerate(attacks, 1):
        if H in folds:
            print(f"[{fi}/{len(attacks)}] {H} zaten var, atlaniyor", flush=True); continue
        t0 = time.time()
        tr_pool = tr_idx[y_all[tr_idx] != H]
        tr_sel = balanced_train(tr_pool)
        le = LabelEncoder().fit(y_all[tr_sel])
        H_rows, ben, oth = eval_rows_for(H)
        if len(H_rows) == 0:
            print(f"[{fi}/{len(attacks)}] {H} atlandi (0 test satiri; oturum-gruplu split)", flush=True); continue
        n_H = len(H_rows)
        tag = f"{ds}:{H}({fi}/{len(attacks)})"
        write_progress(stage="fold_basladi", dataset=ds, fold=H, fold_no=fi, total_folds=len(attacks),
                       n_unseen=int(n_H), max_len=MAX_LEN)
        print(f"\n[{fi}/{len(attacks)}] UNSEEN='{H}'  train={len(tr_sel):,}  test H={n_H:,} "
              f"benign={len(ben):,} known={len(oth):,}", flush=True)

        rf = RandomForestClassifier(n_estimators=200, n_jobs=-1, class_weight="balanced", random_state=SEED)
        rf.fit(X[tr_sel], y_all[tr_sel])
        rf_r = rates({"H": rf.predict(X[H_rows]), "ben": rf.predict(X[ben]), "oth": rf.predict(X[oth])})
        print(f"    RF : unseen_recall={rf_r['unseen_recall']}  benign_fpr={rf_r['benign_fpr']}  known_recall={rf_r['known_recall']}", flush=True)

        model, tok = train_llm(tr_sel, le, tag)
        lp = lambda idx: le.inverse_transform(llm_predict(model, tok, idx))
        llm_r = rates({"H": lp(H_rows), "ben": lp(ben), "oth": lp(oth)})
        print(f"    LLM: unseen_recall={llm_r['unseen_recall']}  benign_fpr={llm_r['benign_fpr']}  known_recall={llm_r['known_recall']}", flush=True)
        del model, tok; gc.collect(); torch.cuda.empty_cache()

        folds[H] = {"n_unseen": int(n_H), "rf": rf_r, "llm": llm_r, "sec": round(time.time()-t0, 0)}
        json.dump(allg, open(OUT, "w"), indent=2, ensure_ascii=False)   # her fold sonrasi kaydet
        write_progress(stage="fold_bitti", dataset=ds, fold=H, fold_no=fi, total_folds=len(attacks))
        print(f"    kaydedildi ({time.time()-t0:.0f}s) -> results/generalization_report_fullctx.json", flush=True)

    # dataset ozeti
    rf_u = [v["rf"]["unseen_recall"] for v in folds.values()]
    llm_u = [v["llm"]["unseen_recall"] for v in folds.values()]
    if rf_u:
        print(f"\n=== {ds} OZET (unseen recall): RF ort={np.mean(rf_u):.3f}  LLM ort={np.mean(llm_u):.3f}  "
              f"fark(LLM-RF)={np.mean(llm_u)-np.mean(rf_u):+.3f} ===", flush=True)

# ============================================================================================================
if __name__ == "__main__":
    write_progress(stage="basladi", datasets=DATASETS)
    for ds in DATASETS:
        try:
            run_dataset(ds)
        except Exception as e:
            import traceback
            print(f"!!! {ds} HATA: {e}", flush=True); traceback.print_exc()
            write_progress(stage="HATA", dataset=ds, error=str(e))
            continue
    write_progress(stage="TAMAMLANDI", datasets=DATASETS)
    print("\n================ HEPSI BITTI ================", flush=True)
    allg = _load(OUT, {})
    for ds in DATASETS:
        folds = allg.get(ds, {}).get("folds", {})
        rf_u = [v["rf"]["unseen_recall"] for v in folds.values()]
        llm_u = [v["llm"]["unseen_recall"] for v in folds.values()]
        if rf_u:
            print(f"{ds:14s}: {len(folds)} fold | RF unseen-recall ort={np.mean(rf_u):.3f} | "
                  f"LLM ort={np.mean(llm_u):.3f} | fark={np.mean(llm_u)-np.mean(rf_u):+.3f}")
    print("DONE.")
