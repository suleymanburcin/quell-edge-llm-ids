#!/usr/bin/env python3
"""QUELL Step 19 (Jetson) — on-device accuracy and cost of the TRAINED classical detectors.

For each dataset in <bundle>/classic: random forest (scikit-learn; optionally ONNX Runtime) and XGBoost (native
C++ predictor) are loaded, their macro-F1 on the 3,000 stratified test inputs is checked, and the cost is measured
on the Jetson CPU at the pinned MAXN_SUPER operating point:
  * batch-1 latency: N single-row predict() calls (mean / p50 / p95), i.e. including library call overhead
  * batched throughput: predict() on all rows at once (samples/s, amortized ms/sample)
  * board power (tegrastats VDD_IN) during each block -> energy per inference; peak RAM
Usage (on the Jetson):  python 19_jetson_classic_cost.py --bundle ~/jetson_bundle [--n1 500]
Output: <bundle>/results/classic_cost_jetson.json
"""
import argparse, json, os, re, signal, subprocess, time, platform
from pathlib import Path
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--bundle", required=True); ap.add_argument("--n1", type=int, default=500)
ap.add_argument("--threads", type=int, default=os.cpu_count())
a = ap.parse_args()
B = Path(a.bundle); OUT = B / "results"; OUT.mkdir(exist_ok=True)
import joblib, sklearn, xgboost as xgb
from sklearn.metrics import f1_score
try:
    import onnxruntime as ort
except Exception:
    ort = None


class Power:
    def __init__(s, tag): s.f = f"/tmp/_tegra_classic_{tag}.log"; s.p = None
    def __enter__(s):
        try: s.p = subprocess.Popen(["tegrastats", "--interval", "100"], stdout=open(s.f, "w"), stderr=subprocess.STDOUT)
        except FileNotFoundError: s.p = None
        time.sleep(0.3); return s
    def __exit__(s, *e):
        s.res = None
        if s.p:
            s.p.send_signal(signal.SIGINT); s.p.wait(timeout=5); txt = open(s.f).read()
            w = [int(x) / 1000 for x in re.findall(r"VDD_IN (\d+)mW", txt)]; ram = [int(x) for x in re.findall(r"RAM (\d+)/\d+MB", txt)]
            s.res = dict(mean_w=round(float(np.mean(w)), 3) if w else None, n_samples=len(w), ram_peak_mb=max(ram) if ram else None)


def bench(tag, predict, X, n1):
    for i in range(10): predict(X[i:i + 1])                      # warm-up
    lat = []
    with Power(tag + "_b1") as p1:
        for i in range(n1):
            r = X[i % len(X)][None, :]; t = time.perf_counter(); predict(r); lat.append((time.perf_counter() - t) * 1000)
    reps = max(1, int(np.ceil(5000 / len(X))))
    with Power(tag + "_bat") as p2:
        t = time.perf_counter()
        for _ in range(reps): predict(X)
        dt = time.perf_counter() - t
    lat = np.array(lat); thr = reps * len(X) / dt
    out = dict(batch1_latency_ms=dict(mean=round(float(lat.mean()), 3), p50=round(float(np.percentile(lat, 50)), 3),
                                      p95=round(float(np.percentile(lat, 95)), 3)),
               batched_throughput_per_s=round(float(thr), 1), batched_ms_per_sample=round(1000 / thr, 4),
               power_batch1=p1.res, power_batched=p2.res)
    if p1.res and p1.res["mean_w"]: out["energy_mj_batch1"] = round(p1.res["mean_w"] * float(lat.mean()), 3)
    if p2.res and p2.res["mean_w"]: out["energy_mj_batched_per_sample"] = round(p2.res["mean_w"] * 1000 / thr, 4)
    return out


res = dict(device=platform.node(), machine=platform.machine(), threads=a.threads, sklearn=sklearn.__version__,
           xgboost=xgb.__version__, onnxruntime=getattr(ort, "__version__", None), datasets={})
for cf in sorted(B.glob("classic/*_classes.json")):
    ds = cf.name.replace("_classes.json", ""); meta = json.load(open(cf))
    X = np.load(B / "classic" / f"{ds}_X.npy"); y = np.load(B / "classic" / f"{ds}_y.npy", allow_pickle=True)
    rf = joblib.load(B / "classic" / f"{ds}_rf.joblib"); rf.set_params(n_jobs=a.threads)
    bst = xgb.XGBClassifier(); bst.load_model(str(B / "classic" / f"{ds}_xgb.json")); bst.set_params(n_jobs=a.threads)
    cls = np.array(meta["classes"]); lab = meta["classes_all"]
    r = dict(n_rows=int(len(X)), n_features=int(X.shape[1]), featset=meta.get("featset"))
    r["rf_sklearn"] = dict(macro_f1=round(float(f1_score(y, rf.predict(X), average="macro", labels=lab, zero_division=0)), 4),
                           **bench(f"{ds}_rf", rf.predict, X, a.n1))
    r["xgb_native"] = dict(macro_f1=round(float(f1_score(y, cls[bst.predict(X).astype(int)], average="macro", labels=lab, zero_division=0)), 4),
                           **bench(f"{ds}_xgb", bst.predict, X, a.n1))
    onx = B / "classic" / f"{ds}_rf.onnx"
    if ort is not None and onx.exists():
        so = ort.SessionOptions(); so.intra_op_num_threads = a.threads
        sess = ort.InferenceSession(str(onx), so, providers=["CPUExecutionProvider"]); iname = sess.get_inputs()[0].name
        po = lambda Z: sess.run(None, {iname: Z.astype(np.float32)})[0]
        r["rf_onnxruntime"] = dict(macro_f1=round(float(f1_score(y, po(X), average="macro", labels=lab, zero_division=0)), 4),
                                   **bench(f"{ds}_rf_ort", po, X, a.n1))
    res["datasets"][ds] = r; print(ds, json.dumps(r, indent=1), flush=True)
    json.dump(res, open(OUT / "classic_cost_jetson.json", "w"), indent=2)
print("->", OUT / "classic_cost_jetson.json")
