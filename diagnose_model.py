"""
diagnose_model.py  — robust version
Diagnoses whether xauusd_rf.pkl is intrinsically long-biased.

1. Loads the RF regressor.
2. Uses the newest cached feature parquet if found.
3. Otherwise tries to rebuild features from raw ticks+L2, locating
   Loader/feature functions automatically.

Requires:  pyarrow  (pip install pyarrow)
"""

from __future__ import annotations
import importlib, inspect, joblib, json, sys, types
from pathlib import Path
from pprint import pprint
import numpy as np
import pandas as pd

# ─── paths you may need to tweak ────────────────────────────────────────────
REPO_ROOT      = Path(__file__).resolve().parent          # repo root
MODEL_PATH     = REPO_ROOT / "models/trained/xauusd_rf.pkl"
CACHE_DIR      = REPO_ROOT / "data/cache"
RAW_TICK_DIR   = REPO_ROOT / "data/ticks"
RAW_L2_DIR     = REPO_ROOT / "data/orderbook"
DATE_TO_TEST   = "2023-07-05"
N_ROWS         = 500
# ────────────────────────────────────────────────────────────────────────────

sys.path.insert(0, str(REPO_ROOT))  # ensure `import src.…` works


# --------------------------------------------------------------------------- #
# helper: find first class whose name ends with "Loader" in data_ingestion
# --------------------------------------------------------------------------- #
def find_loader_class() -> type | None:
    din_path = REPO_ROOT / "src" / "data_ingestion"
    for py in din_path.glob("*.py"):
        mod_name = f"src.data_ingestion.{py.stem}"
        try:
            mod = importlib.import_module(mod_name)
        except ImportError:
            continue
        for name, obj in inspect.getmembers(mod, inspect.isclass):
            if name.lower().endswith("loader"):
                return obj
    return None


def find_feature_builder() -> types.FunctionType | None:
    feat_path = REPO_ROOT / "src" / "features"
    for py in feat_path.glob("*.py"):
        mod_name = f"src.features.{py.stem}"
        try:
            mod = importlib.import_module(mod_name)
        except ImportError:
            continue
        for name, obj in inspect.getmembers(mod, inspect.isfunction):
            if name in ("build_feature_matrix", "make_features"):
                return obj
    return None


def newest_cache_file() -> Path | None:
    if not CACHE_DIR.exists():
        return None
    files = sorted(CACHE_DIR.glob("*features*.parquet"))
    return files[-1] if files else None


# --------------------------------------------------------------------------- #
# main routine
# --------------------------------------------------------------------------- #
def main() -> None:
    print(f"\nLoaded model: {MODEL_PATH.name}")
    model = joblib.load(MODEL_PATH)
    print("Model type :", type(model).__name__)

    # 1⃣  try cached features
    feat_file = newest_cache_file()
    if feat_file:
        print(f"\n‣ Using cached features: {feat_file.name}")
        X = pd.read_parquet(feat_file).tail(N_ROWS)

    # 2⃣  else try to rebuild features on the fly
    else:
        loader_cls = find_loader_class()
        builder_fn = find_feature_builder()

        if not loader_cls or not builder_fn:
            print(
                "\n⚠️  Could not locate a Loader class or feature builder.\n"
                "   The simplest fix is to rerun your back-test once with\n"
                "   --cache-features yes\n"
                "   which will create a parquet file under data/cache/.\n"
            )
            sys.exit(1)

        print(
            f"\n‣ Rebuilding {N_ROWS} feature rows using "
            f"{loader_cls.__name__} + {builder_fn.__name__} …"
        )
        tick_file = RAW_TICK_DIR / f"{DATE_TO_TEST}.csv.gz"
        l2_file   = RAW_L2_DIR   / f"{DATE_TO_TEST}_L2.csv.gz"
        loader = loader_cls(tick_file, l2_file)
        df_raw = loader.load()
        X = builder_fn(df_raw).dropna().tail(N_ROWS)

    # 3⃣  sanity-check feature columns
    missing = set(model.feature_names_in_) - set(X.columns)
    if missing:
        print(f"\nERROR — Dataframe missing columns: {missing}")
        sys.exit(1)

    # 4⃣  predict
    y_hat = model.predict(X[model.feature_names_in_])
    signs = np.sign(y_hat).astype(int)

    print("\nPrediction summary on real data")
    print("--------------------------------")
    print(pd.Series(y_hat).describe())
    print("\nSign distribution:")
    print(pd.Series(signs, dtype="int").value_counts().sort_index())

    # 5⃣  thresholds in metadata?
    meta_path = MODEL_PATH.with_suffix(".json")
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        print(
            f"\nMetadata thresholds  "
            f"long>{meta.get('long_threshold')}   "
            f"short<{meta.get('short_threshold')}"
        )
    else:
        print("\n⚠️  No metadata JSON found.  Strategy layer will default to LONG.")

    print("\nDone.")


if __name__ == "__main__":
    main()
