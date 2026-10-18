#!/usr/bin/env python
# ============================================================================
# QUELL Detection RE-RUN (truncation fix) — Qwen2.5-1.5B, ciciot2023 + nbaiot
# - MAX_LEN veri setine gore OTOMATIK (tam baglam; ciciot kisa -> hizli, nbaiot ~2048)
# - IKI veri seti SIRAYLA (ayni anda calistirma; bu script ikisini de yapar)
# - Arka plan/kopmaya dayanikli: 200 adimda checkpoint + resume; bitmis dataset atlanir
# - Cikti: results/llm_report_fullctx.json  (eski llm_report.json EZILMEZ)
#
# CALISTIRMA:
#   cd /workspace/project
#   nohup python re_05_4_detection.py > ~/det.log 2>&1 &
# DURUM:  python re_05_4_detection.py --status   |   tail -f ~/det.log
# ============================================================================
import os, sys, json, time, gc, shutil
from pathlib import Path
from datetime import datetime

ROOT = Path.home()/"quell-edge-llm-ids"
RES  = ROOT/"results"
OUT  = RES/"llm_report_fullctx.json"     # YENI (eski llm_report.json ezilmez)
PROG = RES/"det_progress.json"

MODEL    = "Qwen/Qwen2.5-1.5B"
DATASETS = ["ciciot2023", "nbaiot"]      # sirayla
TRAIN_CAP=2000; EPOCHS=2; SEED=42; EVAL_CAP=60000
MAXLEN_CAP=2048                           # bellek tavani; MAX_LEN veri setine gore <=bu

def log(m): print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {m}", flush=True)
def _done_sets():
    if not OUT.exists(): return []
    try:
        d=json.load(open(OUT)); return [ds for ds in DATASETS if ds in d and MODEL in d[ds]]
    except Exception: return []
def write_progress(status, extra=None):
    RES.mkdir(parents=True, exist_ok=True)
    d={"updated":f"{datetime.now():%Y-%m-%d %H:%M:%S}","status":status,
       "datasets_done":_done_sets(),"datasets_target":DATASETS}
    if extra: d.update(extra)
    json.dump(d, open(PROG,"w"), indent=2, ensure_ascii=False)

if "--status" in sys.argv:
    print("=== detection re-run durum ===")
    if PROG.exists(): print(open(PROG).read())
    else: print("henuz baslamamis (progress yok)")
    if OUT.exists():
        d=json.load(open(OUT))
        for ds in DATASETS:
            if ds in d and MODEL in d[ds]:
                r=d[ds][MODEL]; print(f"  {ds}: macroF1={r['macro_f1']} acc={r['acc']} (MAX_LEN={r.get('max_len','?')})")
    sys.exit(0)

import subprocess
try:
    _o=subprocess.check_output("nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits",shell=True,text=True)
    _f=[(int(x.split(",")[0]),int(x.split(",")[1])) for x in _o.strip().splitlines()]
    _b=max(_f,key=lambda t:t[1]); os.environ["CUDA_VISIBLE_DEVICES"]=str(_b[0])
    os.environ["PYTORCH_CUDA_ALLOC_CONF"]="expandable_segments:True"
    log(f"selected GPU {_b[0]} | free(MiB): {_f}")
except Exception as e: log(f"GPU secim atlandi: {e}")

import numpy as np, pandas as pd
from pandas.api.types import is_numeric_dtype
import torch
from transformers import (AutoTokenizer, AutoModelForSequenceClassification, BitsAndBytesConfig,
                          TrainingArguments, Trainer, TrainerCallback)
from transformers.trainer_utils import get_last_checkpoint
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, classification_report

PROC=ROOT/"data"/"processed"; SPL=ROOT/"splits"
rep=json.load(open(RES/"split_report.json"))
BF16=torch.cuda.is_available() and torch.cuda.is_bf16_supported()
LABELISH={"label","attack","attack_type","attack_label","type","class","category","marker","__label__"}

class HeartbeatCB(TrainerCallback):
    def __init__(s,ds): s.ds=ds
    def on_log(s,args,state,control,logs=None,**kw):
        if logs and "loss" in logs:
            write_progress(f"{s.ds} egitiliyor",{"epoch":round(state.epoch or 0,2),"step":state.global_step,
                                                 "max_steps":state.max_steps,"train_loss":round(logs["loss"],4)})

def run_dataset(DATASET):
    torch.manual_seed(SEED); np.random.seed(SEED)
    meta=rep[DATASET]; label=meta["label_col"]; group=meta.get("group_col"); tcol=meta.get("time_col")
    df=pd.read_parquet(PROC/f"{DATASET}.parquet").reset_index(drop=True)
    sp=np.load(SPL/f"{DATASET}_split.npz"); tr_idx,te_idx=sp["train"],sp["test"]
    drop=set([label])|set(meta.get("leaky_candidates",[]))
    if group: drop.add(group)
    if tcol: drop.add(tcol)
    for c in df.columns:
        if c!=label and c.lower() in LABELISH: drop.add(c)
    feats=[c for c in df.columns if c not in drop]
    def row_to_text(r):
        parts=[]
        for c in feats:
            v=r[c]
            if isinstance(v,(float,np.floating)): v=round(float(v),4)
            parts.append(f"{c}={v}")
        return "Network traffic flow. "+", ".join(parts)+" . Attack type:"
    write_progress(f"{DATASET} feature->text")
    log(f"{DATASET}: feature->text ({len(df)} rows)...")
    texts_all=df.apply(row_to_text,axis=1).values; y_all=df[label].astype(str).values
    rng=np.random.default_rng(SEED); tr_sel=[]
    for cls in pd.unique(y_all[tr_idx]):
        ids=tr_idx[y_all[tr_idx]==cls]
        if len(ids)>TRAIN_CAP: ids=rng.choice(ids,TRAIN_CAP,replace=False)
        tr_sel+=ids.tolist()
    tr_sel=np.array(sorted(tr_sel))
    le=LabelEncoder().fit(y_all[tr_sel]); K=len(le.classes_); classes_all=sorted(pd.unique(y_all).tolist())
    tok=AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token is None: tok.pad_token=tok.eos_token
    # ---- TRUNCATION FIX: MAX_LEN veri setine gore (p99'u kapsa, cap MAXLEN_CAP) ----
    _s=np.random.RandomState(SEED).choice(len(texts_all),min(3000,len(texts_all)),replace=False)
    _ln=[len(tok(t)["input_ids"]) for t in texts_all[_s]]
    MAX_LEN=int(min(int(np.percentile(_ln,99))+16, MAXLEN_CAP))
    log(f"{DATASET}: feature={len(feats)} MAX_LEN={MAX_LEN} (p95={int(np.percentile(_ln,95))} p99={int(np.percentile(_ln,99))} max={max(_ln)}) train={len(tr_sel):,} test={len(te_idx):,} class={K}")

    class DS(torch.utils.data.Dataset):
        def __init__(s,idx): s.idx=idx
        def __len__(s): return len(s.idx)
        def __getitem__(s,i):
            j=s.idx[i]; enc=tok(texts_all[j],truncation=True,max_length=MAX_LEN,padding="max_length",return_tensors="pt")
            it={k:v.squeeze(0) for k,v in enc.items()}; it["labels"]=torch.tensor(int(le.transform([y_all[j]])[0])); return it
    bnb=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type="nf4",bnb_4bit_compute_dtype=torch.bfloat16,bnb_4bit_use_double_quant=True)
    model=AutoModelForSequenceClassification.from_pretrained(MODEL,num_labels=K,quantization_config=bnb,device_map={"":0})
    model=prepare_model_for_kbit_training(model); model.config.pad_token_id=tok.pad_token_id
    model=get_peft_model(model,LoraConfig(task_type=TaskType.SEQ_CLS,r=16,lora_alpha=32,lora_dropout=0.05,
        target_modules=["q_proj","v_proj"],modules_to_save=["score"])); model.config.use_cache=False
    ckpt=str(ROOT/"models"/f"ckpt_det_{DATASET}")
    args=TrainingArguments(output_dir=ckpt,per_device_train_batch_size=2,gradient_accumulation_steps=8,
        num_train_epochs=EPOCHS,learning_rate=2e-4,bf16=BF16,fp16=(not BF16),gradient_checkpointing=True,
        logging_steps=50,save_strategy="steps",save_steps=200,save_total_limit=1,report_to=[],seed=SEED)
    trainer=Trainer(model=model,args=args,train_dataset=DS(tr_sel),callbacks=[HeartbeatCB(DATASET)])
    _last=get_last_checkpoint(ckpt) if os.path.isdir(ckpt) else None
    if _last: log(f"{DATASET}: checkpoint bulundu ({_last}) -> devam")
    log(f"{DATASET}: training...")
    t=time.time(); trainer.train(resume_from_checkpoint=_last); log(f"{DATASET}: training done {time.time()-t:.0f}s")

    if len(te_idx)<=EVAL_CAP:
        eval_idx=te_idx; scope="tam test"
    else:
        r2=np.random.default_rng(SEED); pick=[]
        for cls in np.unique(y_all[te_idx]):
            ids=te_idx[y_all[te_idx]==cls]; k=max(1,int(round(len(ids)*EVAL_CAP/len(te_idx))))
            pick+=r2.choice(ids,min(k,len(ids)),replace=False).tolist()
        eval_idx=np.array(sorted(pick)); scope=f"stratified subset ({len(eval_idx):,})"
    write_progress(f"{DATASET} eval ({scope})")
    device=next(model.parameters()).device; model.eval(); bs=8; preds=[]; n=len(eval_idx); t=time.time()
    for s in range(0,n,bs):
        js=eval_idx[s:s+bs]
        enc=tok(list(texts_all[js]),truncation=True,max_length=MAX_LEN,padding=True,return_tensors="pt").to(device)
        with torch.no_grad(), torch.autocast(device_type="cuda",dtype=torch.bfloat16,enabled=(device.type=="cuda")):
            preds+=model(**enc).logits.argmax(-1).cpu().tolist()
        if (s//bs)%50==0: log(f"  {DATASET} eval ...{s:,}/{n:,}")
    pred=le.inverse_transform(np.array(preds)); yte=y_all[eval_idx]
    acc=accuracy_score(yte,pred); mf1=f1_score(yte,pred,average="macro",labels=classes_all,zero_division=0)
    wf1=f1_score(yte,pred,average="weighted",labels=classes_all,zero_division=0)
    rp=classification_report(yte,pred,labels=classes_all,target_names=classes_all,output_dict=True,zero_division=0)
    log(f"[{DATASET} / {MODEL}] acc={acc:.4f} macroF1={mf1:.4f} weightedF1={wf1:.4f} ({scope}, {time.time()-t:.0f}s)")
    allr=json.load(open(OUT)) if OUT.exists() else {}
    allr.setdefault(DATASET,{})[MODEL]={"acc":round(float(acc),4),"macro_f1":round(float(mf1),4),"weighted_f1":round(float(wf1),4),
        "eval_scope":scope,"max_len":MAX_LEN,"train_cap":TRAIN_CAP,"epochs":EPOCHS,"n_train":int(len(tr_sel)),"n_eval":int(len(eval_idx)),
        "note":"full-context re-run (truncation fix)","per_class_f1":{k:round(rp[k]['f1-score'],4) for k in classes_all}}
    json.dump(allr,open(OUT,"w"),indent=2,ensure_ascii=False)
    try:
        b=json.load(open(RES/"baseline_report.json"))[DATASET]["models"]
        log(f"  KIYAS {DATASET}: LLM={mf1:.3f} | RF={b['random_forest']['macro_f1']} XGB={b['xgboost']['macro_f1']}")
    except Exception: pass
    del model,trainer; gc.collect(); torch.cuda.empty_cache()
    try: shutil.rmtree(ckpt, ignore_errors=True)
    except Exception: pass
    write_progress(f"{DATASET} BITTI",{"macro_f1":round(float(mf1),4)})

for ds in DATASETS:
    if ds in _done_sets():
        log(f"{ds} zaten var, atlaniyor"); continue
    run_dataset(ds)

write_progress("TAMAMLANDI")
log("HEPSI BITTI -> results/llm_report_fullctx.json")
