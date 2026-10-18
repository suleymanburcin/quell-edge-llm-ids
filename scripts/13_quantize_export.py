#!/usr/bin/env python3
"""QUELL Step 13 (GPU host) — fine-tuning, accuracy retention under quantization, and export of the deployed detector.

1. Fine-tune the detector (identical recipe to the paper: QLoRA r=16, q/v_proj, 2000/class,
   2 epochs) with the FULL-CONTEXT sequence length — or reuse an existing adapter (--adapter).
2. Accuracy retention of the fine-tuned model WITH its trained classification head at
   FP16, INT8 (bitsandbytes LLM.int8) and 4-bit (bitsandbytes NF4) on the full stratified
   evaluation set.
3. Export what the on-device (GGUF) pipeline needs:
     <out>/backbone_hf/   merged decoder saved as a causal LM  -> convert to GGUF
     <out>/head.npy       trained score head (K x H), applied to the last-token hidden state
     <out>/subset.jsonl   stratified subset: truncated token ids, label, HF-FP16 prediction
     <out>/meta.json      classes, MAX_LEN, token-length stats

Usage:
  python 13_quantize_export.py --dataset edge_iiotset --model Qwen/Qwen2.5-1.5B --seed 42
  (optional) --adapter <dir>   reuse an already-trained fullctx adapter, skip training
"""
import os, subprocess, argparse, json, time, gc, sys
try:
    _o = subprocess.check_output("nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits", shell=True, text=True)
    _f = [(int(x.split(",")[0]), int(x.split(",")[1])) for x in _o.strip().splitlines()]
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(max(_f, key=lambda t: t[1])[0]))
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
except Exception as e:
    print("GPU selection skipped:", e)
from pathlib import Path
import numpy as np, torch
from transformers import (AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM,
                          BitsAndBytesConfig, TrainingArguments, Trainer)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training, PeftModel
from sklearn.preprocessing import LabelEncoder
sys.path.insert(0, str(Path(__file__).parent))
import quell_common as qc

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", default="edge_iiotset")
ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--adapter", default=None)
ap.add_argument("--subset", type=int, default=3000, help="size of the stratified GGUF/on-device subset")
ap.add_argument("--precisions", default="fp16,int8,nf4")
ap.add_argument("--epochs", type=int, default=2)
ap.add_argument("--max-len", type=int, default=None, help="force MAX_LEN (use the value the --adapter was trained with)")
a = ap.parse_args()

torch.manual_seed(a.seed); np.random.seed(a.seed)
R = qc.root(); RES = R / "results" / "v4"; RES.mkdir(parents=True, exist_ok=True)
TAG = f"{a.dataset}_{a.model.split('/')[-1]}_s{a.seed}" + (f"_{qc.featset()}" if qc.featset() else "")
OUT = R / "models" / "v4" / TAG; OUT.mkdir(parents=True, exist_ok=True)
t_all = time.time()

d = qc.load(a.dataset, a.seed)
texts, y, tr_sel, eval_idx, classes_all = d["texts"], d["y"], d["tr_sel"], d["eval_idx"], d["classes_all"]
tok = AutoTokenizer.from_pretrained(a.model)
if tok.pad_token is None: tok.pad_token = tok.eos_token
MAX_LEN, lenstats = qc.fullctx_maxlen(tok, texts, tr_sel, a.seed)
if a.max_len: MAX_LEN = lenstats["max_len"] = a.max_len
le = LabelEncoder().fit(y[tr_sel]); K = len(le.classes_)
if a.adapter and (Path(a.adapter) / "quell_meta.json").exists():
    _m = json.load(open(Path(a.adapter) / "quell_meta.json"))
    _c = _m.get("classes")
    assert _c is None or list(_c) == list(le.classes_), f"label order mismatch with adapter: {_c} vs {list(le.classes_)}"
    if _m.get("max_len") and not a.max_len: MAX_LEN = lenstats["max_len"] = int(_m["max_len"])
    print("adapter meta OK:", {k: _m.get(k) for k in ("max_len", "seed", "epochs")}, flush=True)
print(f"{a.model} | {a.dataset}: K={K} train={len(tr_sel):,} eval={len(eval_idx):,} MAX_LEN={MAX_LEN} {lenstats}", flush=True)


class DS(torch.utils.data.Dataset):
    def __init__(s, idx): s.idx = idx
    def __len__(s): return len(s.idx)
    def __getitem__(s, i):
        j = s.idx[i]
        enc = tok(texts[j], truncation=True, max_length=MAX_LEN, padding="max_length", return_tensors="pt")
        it = {k: v.squeeze(0) for k, v in enc.items()}
        it["labels"] = torch.tensor(int(le.transform([y[j]])[0])); return it


# ---------------- 1. train (or reuse) ----------------
ADAPTER = Path(a.adapter) if a.adapter else OUT / "adapter"
if not a.adapter:
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    base = AutoModelForSequenceClassification.from_pretrained(a.model, num_labels=K, quantization_config=bnb, device_map={"": 0})
    base = prepare_model_for_kbit_training(base); base.config.pad_token_id = tok.pad_token_id
    model = get_peft_model(base, LoraConfig(task_type=TaskType.SEQ_CLS, r=16, lora_alpha=32, lora_dropout=0.05,
                                            target_modules=["q_proj", "v_proj"], modules_to_save=["score"]))
    model.config.use_cache = False
    args = TrainingArguments(output_dir=str(OUT / "train"), per_device_train_batch_size=8, gradient_accumulation_steps=2,
                             num_train_epochs=a.epochs, learning_rate=2e-4, bf16=True, gradient_checkpointing=True,
                             logging_steps=50, save_strategy="no", report_to=[], seed=a.seed)
    t = time.time(); Trainer(model=model, args=args, train_dataset=DS(tr_sel)).train()
    print(f"training: {time.time()-t:.0f}s", flush=True)
    model.save_pretrained(str(ADAPTER)); tok.save_pretrained(str(ADAPTER))
    json.dump(dict(classes=list(le.classes_), max_len=MAX_LEN, seed=a.seed, epochs=a.epochs, featset=qc.featset()),
              open(ADAPTER / "quell_meta.json", "w"), indent=2)
    del model, base; gc.collect(); torch.cuda.empty_cache()


# ---------------- 2. accuracy retention per precision ----------------
def predict(m, idx, bs=32):
    m.eval(); dev = next(m.parameters()).device; out = []
    for s in range(0, len(idx), bs):
        js = idx[s:s + bs]
        enc = tok(list(texts[js]), truncation=True, max_length=MAX_LEN, padding=True, return_tensors="pt").to(dev)
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out += m(**enc).logits.argmax(-1).cpu().tolist()
        if (s // bs) % 200 == 0: print(f"    ...{s:,}/{len(idx):,}", flush=True)
    return le.inverse_transform(np.array(out))


MERGE = os.environ.get("QUELL_MERGE", "dequant")   # "dequant" (faithful QLoRA merge, see qlora_merge.py) | "plain"
MERGED = OUT / "merged_hf"


def load_prec(prec):
    """fp16: adapter merged into the DE-QUANTIZED NF4 base (saved to merged_hf/); int8: that merged model in 8 bit;
    nf4: adapter on the NF4 base (= training precision). MERGE=plain reproduces the earlier, unfaithful export."""
    if MERGE == "dequant":
        from qlora_merge import dequantized_merge, load_nf4_base
        if prec == "fp16":
            m = dequantized_merge(a.model, K, tok.pad_token_id, ADAPTER)
            m.save_pretrained(str(MERGED)); tok.save_pretrained(str(MERGED)); return m
        if prec == "int8":
            b = AutoModelForSequenceClassification.from_pretrained(str(MERGED), device_map={"": 0},
                                                                   quantization_config=BitsAndBytesConfig(load_in_8bit=True))
            b.config.pad_token_id = tok.pad_token_id; return b
        return PeftModel.from_pretrained(load_nf4_base(a.model, K, tok.pad_token_id), str(ADAPTER))
    if prec == "fp16":
        b = AutoModelForSequenceClassification.from_pretrained(a.model, num_labels=K, torch_dtype=torch.float16, device_map={"": 0})
    elif prec == "int8":
        b = AutoModelForSequenceClassification.from_pretrained(a.model, num_labels=K, device_map={"": 0},
                                                               quantization_config=BitsAndBytesConfig(load_in_8bit=True))
    elif prec == "nf4":
        b = AutoModelForSequenceClassification.from_pretrained(a.model, num_labels=K, device_map={"": 0},
                quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                       bnb_4bit_compute_dtype=torch.bfloat16))
    b.config.pad_token_id = tok.pad_token_id
    m = PeftModel.from_pretrained(b, str(ADAPTER))
    return m.merge_and_unload() if prec == "fp16" else m


sub_idx = qc.stratified_subset(eval_idx, y, a.subset, a.seed)
pos = {j: i for i, j in enumerate(eval_idx)}
report = dict(dataset=a.dataset, model=a.model, seed=a.seed, merge=MERGE, max_len=MAX_LEN, token_len=lenstats,
              n_train=int(len(tr_sel)), n_eval=int(len(eval_idx)), n_subset=int(len(sub_idx)), precisions={})
hf_fp16_sub = None
for prec in a.precisions.split(","):
    print(f"\n== {prec} ==", flush=True); t = time.time()
    m = load_prec(prec)
    p = predict(m, eval_idx)
    r = qc.macro_f1_both(y[eval_idx], p, classes_all)
    r["subset"] = qc.macro_f1_both(y[sub_idx], p[[pos[j] for j in sub_idx]], classes_all)
    r["eval_s"] = round(time.time() - t, 1)
    report["precisions"][prec] = r; print(prec, r, flush=True)
    if prec == "fp16":
        hf_fp16_sub = p[[pos[j] for j in sub_idx]]
        # ---- 3. export backbone + head for GGUF ----
        head = m.score.weight.detach().float().cpu().numpy()
        assert getattr(m.score, "bias", None) is None, "score head has a bias; update 14_gguf_eval.py accordingly"
        np.save(OUT / "head.npy", head)
        causal = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.float16)
        causal.model.load_state_dict(m.model.state_dict(), strict=True)
        causal.save_pretrained(str(OUT / "backbone_hf")); tok.save_pretrained(str(OUT / "backbone_hf"))
        # sanity: HF backbone last-token hidden @ head == classifier logits
        js = sub_idx[:4]
        enc = tok(list(texts[js]), truncation=True, max_length=MAX_LEN, return_tensors="pt", padding=True).to(m.device)
        with torch.no_grad():
            lg = m(**enc).logits.float().cpu().numpy()
            hs = m.model(**enc).last_hidden_state.float().cpu().numpy()
        last = enc["attention_mask"].sum(1).cpu().numpy() - 1
        lg2 = np.stack([hs[i, last[i]] @ head.T for i in range(len(js))])
        report["head_sanity_max_abs_diff"] = float(np.abs(lg - lg2).max()); print("head sanity:", report["head_sanity_max_abs_diff"])
        del causal
    del m; gc.collect(); torch.cuda.empty_cache()
    json.dump(report, open(RES / f"quant_retention_{TAG}.json", "w"), indent=2)

with open(OUT / "subset.jsonl", "w") as f:
    for k, j in enumerate(sub_idx):
        ids = tok(texts[j], truncation=True, max_length=MAX_LEN)["input_ids"]
        f.write(json.dumps({"i": int(j), "ids": ids, "label": y[j], "hf_fp16_pred": str(hf_fp16_sub[k])}) + "\n")
json.dump(dict(classes=list(le.classes_), classes_all=classes_all, max_len=MAX_LEN, model=a.model,
               dataset=a.dataset, seed=a.seed, token_len=lenstats), open(OUT / "meta.json", "w"), indent=2)
report["total_s"] = round(time.time() - t_all); json.dump(report, open(RES / f"quant_retention_{TAG}.json", "w"), indent=2)
print(f"\nDONE -> {RES/f'quant_retention_{TAG}.json'} | export -> {OUT}", flush=True)
