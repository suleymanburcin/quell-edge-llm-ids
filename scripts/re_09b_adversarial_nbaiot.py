#!/usr/bin/env python
# ============================================================================
# QUELL Step 09b — ADVERSARIAL N-BaIoT (RF vs LLM) — RE-RUN (truncation fix)
# - MAX_LEN=2048 (edge multiseed ile TUTARLI; eski 256 idi, %100 kesiyordu)
# - Arka plan / kopmaya dayanikli: LLM egitimi 200 adimda checkpoint, resume eder
# - Ilerleme: results/adv_progress.json ; cikti: results/adversarial_report_fullctx.json
#   (eski adversarial_report.json EZILMEZ)
#
# CALISTIRMA (YPB terminal):
#   cd /workspace/project
#   nohup python re_09b_adversarial_nbaiot.py > ~/adv_nbaiot.log 2>&1 &
# DURUM:  python re_09b_adversarial_nbaiot.py --status   |   tail -f ~/adv.log
# ============================================================================
import os, sys, json, time, gc, shutil
from pathlib import Path
from datetime import datetime

ROOT = Path.home()/"quell-edge-llm-ids"
RES  = ROOT/"results"
OUT  = RES/"adversarial_report_nbaiot_fullctx.json"   # YENI cikti (eski adversarial_report_nbaiot.json ezilmez)
PROG = RES/"adv_nbaiot_progress.json"

# ===== AYARLAR (orijinal 09 ile AYNI; sadece MAX_LEN + batch degisti) =====
DATASET="nbaiot"; MODEL="Qwen/Qwen2.5-1.5B"
TRAIN_CAP=2000; EPOCHS=2; SEED=42
EPS_GRID=[0.0,0.1,0.25,0.5,1.0]
EVAL_SUB=8000
EVASION_N=20; EVASION_EPS=0.3; EVASION_SUB=800
MAX_LEN=2048                     # <-- TRUNCATION FIX (edge multiseed ile tutarli)
# =========================================================================

def log(m): print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {m}", flush=True)
def write_progress(status, extra=None):
    RES.mkdir(parents=True, exist_ok=True)
    d={"updated":f"{datetime.now():%Y-%m-%d %H:%M:%S}","status":status,"max_len":MAX_LEN,
       "done": OUT.exists()}
    if extra: d.update(extra)
    json.dump(d, open(PROG,"w"), indent=2, ensure_ascii=False)

if "--status" in sys.argv:
    print("=== 09b adversarial N-BaIoT re-run durum ===")
    if PROG.exists(): print(open(PROG).read())
    else: print("henuz baslamamis (progress yok)")
    if OUT.exists():
        print("\n--- SONUC (fullctx) ---"); print(open(OUT).read())
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
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          BitsAndBytesConfig, TrainingArguments, Trainer, TrainerCallback)
from transformers.trainer_utils import get_last_checkpoint
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

np.random.seed(SEED); torch.manual_seed(SEED)
PROC=ROOT/"data"/"processed"; SPL=ROOT/"splits"
rep=json.load(open(RES/"split_report.json")); meta=rep[DATASET]
label=meta["label_col"]; group=meta.get("group_col"); tcol=meta.get("time_col")
LABELISH={"label","attack","attack_type","attack_label","type","class","category","marker","__label__"}
df=pd.read_parquet(PROC/f"{DATASET}.parquet").reset_index(drop=True)
sp=np.load(SPL/f"{DATASET}_split.npz"); tr_idx,te_idx=sp["train"],sp["test"]
drop=set([label])|set(meta.get("leaky_candidates",[]))
if group: drop.add(group)
if tcol: drop.add(tcol)
for c in df.columns:
    if c!=label and c.lower() in LABELISH: drop.add(c)
feats=[c for c in df.columns if c not in drop]
y_all=df[label].astype(str).values
classes_all=sorted(pd.unique(y_all).tolist())
benign=next((c for c in classes_all if c.strip().lower() in {"normal","benign","background"} or "normal" in c.lower() or "benign" in c.lower()),None)
assert benign, f"benign yok: {classes_all}"
BF16=torch.cuda.is_available() and torch.cuda.is_bf16_supported()

num_mask=np.array([is_numeric_dtype(df[c]) for c in feats])
num_idx=np.where(num_mask)[0]
orig_vals=df[feats].values
baseX=np.zeros((len(df),len(feats)),dtype="float32")
for p,c in enumerate(feats):
    baseX[:,p]=df[c].values.astype("float32") if num_mask[p] else pd.factorize(df[c])[0].astype("float32")
FCLIP=1e9  # N-BaIoT'nin cok buyuk ozniteliklerini sinirla (float32 overflow/inf onle)
baseX=np.nan_to_num(baseX,nan=0.0,posinf=0.0,neginf=0.0)
baseX=np.clip(baseX,-FCLIP,FCLIP).astype("float32")
sigma_num=np.nan_to_num(baseX[tr_idx][:,num_idx].std(0),nan=0.0,posinf=0.0,neginf=0.0)
sigma_num=np.clip(sigma_num,0.0,FCLIP).astype("float64")+1e-9
log(f"{DATASET}: feature={len(feats)} (numeric={len(num_idx)}) class={len(classes_all)} benign='{benign}' MAX_LEN={MAX_LEN}")
write_progress("basladi (veri yuklendi)")

def perturb(idx,eps,rng,gaussian=True):
    base=baseX[idx][:,num_idx].astype("float64")   # float64'te hesapla -> tasma(inf) yok
    noise=rng.standard_normal(base.shape) if gaussian else rng.uniform(-1,1,base.shape)
    out=base+eps*sigma_num*noise
    out=np.nan_to_num(np.clip(out,-FCLIP,FCLIP),nan=0.0,posinf=FCLIP,neginf=-FCLIP)
    return out.astype("float32")
def build_X(idx,num_pert):
    Xp=baseX[idx].copy(); Xp[:,num_idx]=num_pert
    return np.nan_to_num(np.clip(Xp,-FCLIP,FCLIP),nan=0.0,posinf=FCLIP,neginf=-FCLIP).astype("float32")
def build_texts(idx,num_pert):
    texts=[]
    for r,i in enumerate(idx):
        parts=[]; ni=0
        for p,c in enumerate(feats):
            if num_mask[p]: v=round(float(num_pert[r,ni]),4); ni+=1
            else: v=orig_vals[i,p]
            parts.append(f"{c}={v}")
        texts.append("Network traffic flow. "+", ".join(parts)+" . Attack type:")
    return texts

log("Training RF...")
write_progress("RF egitiliyor")
rf=RandomForestClassifier(n_estimators=300,n_jobs=-1,class_weight="balanced",random_state=SEED).fit(baseX[tr_idx],y_all[tr_idx])

rng0=np.random.default_rng(SEED); tr_sel=[]
for cls in pd.unique(y_all[tr_idx]):
    ids=tr_idx[y_all[tr_idx]==cls]
    if len(ids)>TRAIN_CAP: ids=rng0.choice(ids,TRAIN_CAP,replace=False)
    tr_sel+=ids.tolist()
tr_sel=np.array(sorted(tr_sel))
le=LabelEncoder().fit(y_all[tr_sel]); K=len(le.classes_)
tok=AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token is None: tok.pad_token=tok.eos_token
class DS(torch.utils.data.Dataset):
    def __init__(s,idx): s.idx=idx
    def __len__(s): return len(s.idx)
    def __getitem__(s,i):
        j=s.idx[i]; t=build_texts([j],perturb([j],0.0,np.random.default_rng(0)))[0]
        e=tok(t,truncation=True,max_length=MAX_LEN,padding="max_length",return_tensors="pt")
        it={k:v.squeeze(0) for k,v in e.items()}; it["labels"]=torch.tensor(int(le.transform([y_all[j]])[0])); return it

class HeartbeatCB(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kw):
        if logs and "loss" in logs:
            write_progress("LLM egitiliyor",{"epoch":round(state.epoch or 0,2),"step":state.global_step,
                                             "max_steps":state.max_steps,"train_loss":round(logs["loss"],4)})

bnb=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type="nf4",bnb_4bit_compute_dtype=torch.bfloat16,bnb_4bit_use_double_quant=True)
base=AutoModelForSequenceClassification.from_pretrained(MODEL,num_labels=K,quantization_config=bnb,device_map={"":0})
base=prepare_model_for_kbit_training(base); base.config.pad_token_id=tok.pad_token_id
llm=get_peft_model(base,LoraConfig(task_type=TaskType.SEQ_CLS,r=16,lora_alpha=32,lora_dropout=0.05,
    target_modules=["q_proj","v_proj"],modules_to_save=["score"])); llm.config.use_cache=False
log("Training LLM...")
ckpt=str(ROOT/"models"/"ckpt_adv_nbaiot")
tr=Trainer(model=llm,args=TrainingArguments(output_dir=ckpt,per_device_train_batch_size=2,
    gradient_accumulation_steps=8,num_train_epochs=EPOCHS,learning_rate=2e-4,bf16=BF16,fp16=(not BF16),
    gradient_checkpointing=True,logging_steps=50,save_strategy="steps",save_steps=200,save_total_limit=1,
    report_to=[],seed=SEED), train_dataset=DS(tr_sel), callbacks=[HeartbeatCB()])
_last=get_last_checkpoint(ckpt) if os.path.isdir(ckpt) else None
if _last: log(f"checkpoint bulundu ({_last}) -> devam")
tr.train(resume_from_checkpoint=_last)
dev=next(llm.parameters()).device; llm.eval()

def llm_pred_texts(texts):
    preds=[]; bs=8
    for s in range(0,len(texts),bs):
        enc=tok(texts[s:s+bs],truncation=True,max_length=MAX_LEN,padding=True,return_tensors="pt").to(dev)
        with torch.no_grad(), torch.autocast(device_type="cuda",dtype=torch.bfloat16,enabled=(dev.type=="cuda")):
            preds+=llm(**enc).logits.argmax(-1).cpu().tolist()
    return le.inverse_transform(np.array(preds))

# (1) NOISE CURVE
r=np.random.default_rng(SEED); pick=[]
for cls in np.unique(y_all[te_idx]):
    ids=te_idx[y_all[te_idx]==cls]; k=max(1,int(round(len(ids)*EVAL_SUB/len(te_idx))))
    pick+=r.choice(ids,min(k,len(ids)),replace=False).tolist()
eval_idx=np.array(sorted(pick)); yte=y_all[eval_idx]
curve={"eps":[],"rf_macro_f1":[],"llm_macro_f1":[]}
log("[1] Noise-robustness curve:")
for eps in EPS_GRID:
    write_progress("noise curve",{"eps":eps})
    rr=np.random.default_rng(100+int(eps*1000))
    npz=perturb(eval_idx,eps,rr,gaussian=True)
    rf_f1=f1_score(yte,rf.predict(build_X(eval_idx,npz)),average="macro",labels=classes_all,zero_division=0)
    llm_f1=f1_score(yte,llm_pred_texts(build_texts(eval_idx,npz)),average="macro",labels=classes_all,zero_division=0)
    curve["eps"].append(eps); curve["rf_macro_f1"].append(round(float(rf_f1),4)); curve["llm_macro_f1"].append(round(float(llm_f1),4))
    log(f"  eps={eps:<4}  RF macroF1={rf_f1:.4f}   LLM macroF1={llm_f1:.4f}")

# (2) EVASION
r2=np.random.default_rng(SEED)
att=te_idx[y_all[te_idx]!=benign]
if len(att)>EVASION_SUB: att=np.array(sorted(r2.choice(att,EVASION_SUB,replace=False)))
succ_rf=np.zeros(len(att),bool); succ_llm=np.zeros(len(att),bool)
log(f"[2] Kara-kutu evasion: {len(att)} attack, N={EVASION_N}, eps={EVASION_EPS}")
for t in range(EVASION_N):
    rr=np.random.default_rng(500+t)
    npz=perturb(att,EVASION_EPS,rr,gaussian=False)
    succ_rf |= (rf.predict(build_X(att,npz))==benign)
    succ_llm|= (llm_pred_texts(build_texts(att,npz))==benign)
    if (t+1)%5==0:
        log(f"    tur {t+1}/{EVASION_N}: RF kacis={succ_rf.mean():.3f}  LLM kacis={succ_llm.mean():.3f}")
        write_progress("evasion",{"tur":t+1,"rf_kacis":round(float(succ_rf.mean()),3),"llm_kacis":round(float(succ_llm.mean()),3)})
ev_rf=float(succ_rf.mean()); ev_llm=float(succ_llm.mean())

out={"dataset":DATASET,"model":MODEL,"benign":benign,"max_len":MAX_LEN,
     "note":"full-context re-run (truncation fix)",
     "noise_curve":curve,
     "evasion":{"n_samples":int(len(att)),"tries":EVASION_N,"eps":EVASION_EPS,
                "rf_evasion_rate":round(ev_rf,4),"llm_evasion_rate":round(ev_llm,4)}}
json.dump(out,open(OUT,"w"),indent=2,ensure_ascii=False)
try: shutil.rmtree(ckpt, ignore_errors=True)
except Exception: pass
log("===== ADVERSARIAL OZET (fullctx) =====")
for i,eps in enumerate(curve["eps"]):
    log(f"   eps={eps:<4} RF={curve['rf_macro_f1'][i]:.3f}  LLM={curve['llm_macro_f1'][i]:.3f}")
log(f"  Evasion kacis (dusuk=iyi): RF={ev_rf:.3f}  LLM={ev_llm:.3f}")
write_progress("TAMAMLANDI",{"evasion_rf":round(ev_rf,4),"evasion_llm":round(ev_llm,4),
                             "noise_llm_eps0.1": curve["llm_macro_f1"][1] if len(curve["llm_macro_f1"])>1 else None})
log("saved -> results/adversarial_report_nbaiot_fullctx.json | DONE")
