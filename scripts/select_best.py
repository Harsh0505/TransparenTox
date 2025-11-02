# scripts/select_best.py
import json, pathlib

EXPS = pathlib.Path("experiments")
OUT  = pathlib.Path("app/best_model.json")

# Prefer PR-AUC (best for imbalance), then F1 at chosen threshold, then ROC-AUC.
PRIORITY = ["pr_auc_ap", "test_f1_at_best_t", "eval_f1", "roc_auc"]

def score_of(m):
    for k in PRIORITY:
        if k in m and m[k] is not None:
            try:
                return float(m[k]), k
            except Exception:
                pass
    return float("-inf"), None

best = None
for p in EXPS.rglob("reports/metrics.json"):
    try:
        m = json.loads(p.read_text())
        score, key = score_of(m)
        if key is None:
            continue
        if (best is None) or (score > best["score"]):
            best = {
                "score": score,
                "metric": key,
                "metrics_path": str(p),
                "reports_dir": str(p.parent),
                "model_dir": str((p.parent.parent / "model").resolve()),
                "best_threshold": float(
                    m.get("best_threshold_prec@rec>=0.5", m.get("best_threshold", 0.5))
                ),
                "run_name": m.get("run_name") or p.parent.parent.name,
            }
    except Exception:
        pass

if not best:
    raise SystemExit("No experiments found (missing reports/metrics.json). Train at least one run first.")

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(best, indent=2))
print("Wrote", OUT, "->", best)
