"""Shared data preparation for steps 10-22 (scripts/).

Reproduces EXACTLY the feature-to-text encoding, balanced training subset and
stratified evaluation subset used by the original notebooks (05_*, 06_*), with one
change: the full-context sequence length (truncation fix, see README "Versions").
"""
import json, os
from pathlib import Path
import numpy as np, pandas as pd

LABELISH = {"label", "attack", "attack_type", "attack_label", "type", "class",
            "category", "marker", "__label__"}
TRAIN_CAP = 2000
EVAL_CAP = 60000
MAXLEN_CAP = 2048          # same cap as the *_fullctx re-runs
# QUELL_FEATSET=noid: additionally drop identifier-like fields (Edge-IIoTset authors' list + Wireshark stream indices)
NOID_DROP = ["tcp.srcport", "tcp.dstport", "udp.port", "tcp.options", "tcp.payload", "http.file_data",
             "http.request.full_uri", "http.request.uri.query", "icmp.transmit_timestamp", "mqtt.msg",
             "udp.stream", "tcp.stream"]


def featset():
    return os.environ.get("QUELL_FEATSET", "")


def root():
    return Path(os.environ.get("QUELL_ROOT", Path.home() / "quell-edge-llm-ids"))


def load(dataset, seed, with_texts=True):
    R = root(); RES = R / "results"
    meta = json.load(open(RES / "split_report.json"))[dataset]
    label = meta["label_col"]; group = meta.get("group_col"); tcol = meta.get("time_col")
    df = pd.read_parquet(R / "data" / "processed" / f"{dataset}.parquet").reset_index(drop=True)
    sp = np.load(R / "splits" / f"{dataset}_split.npz"); tr_idx, te_idx = sp["train"], sp["test"]
    drop = {label} | set(meta.get("leaky_candidates", []))
    if group: drop.add(group)
    if tcol: drop.add(tcol)
    for c in df.columns:
        if c != label and c.lower() in LABELISH: drop.add(c)
    feats = [c for c in df.columns if c not in drop]
    if featset() == "noid": feats = [c for c in feats if c not in NOID_DROP]

    def row_to_text(r):
        parts = []
        for c in feats:
            v = r[c]
            if isinstance(v, (float, np.floating)): v = round(float(v), 4)
            parts.append(f"{c}={v}")
        return "Network traffic flow. " + ", ".join(parts) + " . Attack type:"

    if with_texts:
        print(f"feature->text ({len(df):,} rows)...", flush=True)
        texts = df.apply(row_to_text, axis=1).values
    else:
        texts = None
    y = df[label].astype(str).values

    rng = np.random.default_rng(seed); tr_sel = []
    for cls in pd.unique(y[tr_idx]):
        ids = tr_idx[y[tr_idx] == cls]
        if len(ids) > TRAIN_CAP: ids = rng.choice(ids, TRAIN_CAP, replace=False)
        tr_sel += ids.tolist()
    tr_sel = np.array(sorted(tr_sel))

    if len(te_idx) <= EVAL_CAP:
        eval_idx = te_idx
    else:
        r2 = np.random.default_rng(seed); pick = []
        for cls in np.unique(y[te_idx]):
            ids = te_idx[y[te_idx] == cls]
            k = max(1, int(round(len(ids) * EVAL_CAP / len(te_idx))))
            pick += r2.choice(ids, min(k, len(ids)), replace=False).tolist()
        eval_idx = np.array(sorted(pick))
    classes_all = sorted(pd.unique(y).tolist())
    return dict(texts=texts, y=y, tr_sel=tr_sel, eval_idx=eval_idx, feats=feats,
                label=label, classes_all=classes_all, df=df, row_to_text=row_to_text)


def fullctx_maxlen(tok, texts, idx, seed, n=3000):
    """p99 of token length on the training subset + 16, capped at MAXLEN_CAP."""
    rng = np.random.default_rng(seed)
    s = rng.choice(idx, min(n, len(idx)), replace=False)
    L = np.array([len(tok(texts[j])["input_ids"]) for j in s])
    ml = int(min(MAXLEN_CAP, np.percentile(L, 99) + 16))
    return ml, dict(p50=float(np.percentile(L, 50)), p95=float(np.percentile(L, 95)),
                    p99=float(np.percentile(L, 99)), max=int(L.max()), max_len=ml,
                    trunc_rate=float((L > ml).mean()))


def stratified_subset(eval_idx, y, n, seed, min_per_class=20):
    """Small stratified subset for the GGUF / on-device runs (every class represented)."""
    rng = np.random.default_rng(seed + 1000); pick = []
    classes, counts = np.unique(y[eval_idx], return_counts=True)
    for c, cnt in zip(classes, counts):
        ids = eval_idx[y[eval_idx] == c]
        k = max(min_per_class, int(round(cnt * n / len(eval_idx))))
        pick += rng.choice(ids, min(k, len(ids)), replace=False).tolist()
    return np.array(sorted(pick))


def macro_f1_both(y_true, y_pred, classes_all):
    """Report macro-F1 over ALL label classes (original definition) and over the
    classes actually present in the evaluated test set."""
    from sklearn.metrics import f1_score, accuracy_score
    present = sorted(set(y_true))
    return dict(acc=round(float(accuracy_score(y_true, y_pred)), 4),
                macro_f1_all=round(float(f1_score(y_true, y_pred, average="macro", labels=classes_all, zero_division=0)), 4),
                macro_f1_present=round(float(f1_score(y_true, y_pred, average="macro", labels=present, zero_division=0)), 4),
                n_classes_all=len(classes_all), n_classes_present=len(present))
