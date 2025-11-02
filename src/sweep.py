# scripts/sweep.py
import itertools, subprocess, os, sys, json
from datetime import date

PY = sys.executable
BASE = ["-m", "src.train", "--config", "configs/default.yaml",
        "--train_subset", "4000", "--eval_subset", "1000", "--test_subset", "1000"]

models = ["distilbert-base-uncased"]
lengths = [128]
lrs = [2e-5, 3e-5, 5e-5]
epochs = [1, 2]
wds = [0.0, 0.01]
warmups = [0.0, 0.06]

for model, L, lr, ep, wd, wr in itertools.product(models, lengths, lrs, epochs, wds, warmups):
    run_name = f"run-{model.split('/')[-1]}-L{L}-BS16-LR{lr}-E{ep}-WD{wd}-WR{wr}"
    cmd = [PY] + BASE + [
        "--model_name", model,
        "--epochs", str(ep),
        "--lr", str(lr),
        "--output_dir", f"models/tmp",
    ]
    env = os.environ.copy()
    env["RUN_NAME_OVERRIDE"] = run_name
    print(">>", " ".join(cmd), "RUN_NAME=", run_name)
    subprocess.run(cmd, env=env, check=True)
