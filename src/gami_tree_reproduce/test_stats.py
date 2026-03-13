from pathlib import Path
import re

import joblib
import pandas as pd


ROOT = Path.cwd()
CACHE_DIR = ROOT / "src" / "gami_tree_reproduce" / "cache"
OUTDIR = ROOT / "src" / "gami_tree_reproduce" / "tree_stats"


def node_depth(node) -> int:
    if node is None:
        return 0
    if node.t is None:
        return 0
    left_depth = node_depth(node.L)
    right_depth = node_depth(node.R)
    return 1 + max(left_depth, right_depth)


def node_splits(node) -> int:
    if node is None:
        return 0
    if node.t is None:
        return 0
    return 1 + node_splits(node.L) + node_splits(node.R)


def parse_cache_name(cache_name: str):
    m = re.match(
        r"^gamitree_(model\d+)_(n(?:50k|500k))_(rho(?:0|05))_(regression|classification)\.joblib$",
        cache_name,
    )
    if m is None:
        return "", "", "", ""
    return m.group(1), m.group(2), m.group(3), m.group(4)


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)

    per_tree_rows = []
    summary_rows = []

    cache_files = sorted(CACHE_DIR.glob("gamitree_*.joblib"))

    if not cache_files:
        raise FileNotFoundError(f"No cache files found in {CACHE_DIR}")

    for cache_file in cache_files:
        obj = joblib.load(cache_file)
        model = obj["model"]

        model_name, n_name, rho_name, task = parse_cache_name(cache_file.name)

        rows_this_model = []

        for i, tree in enumerate(model.main_trees):
            depth = node_depth(tree.root)
            splits = node_splits(tree.root)

            row = {
                "cache_file": cache_file.name,
                "model_name": model_name,
                "n_name": n_name,
                "rho_name": rho_name,
                "task": task,
                "tree_group": "main",
                "tree_index": i,
                "feature_1": f"X{tree.j + 1}",
                "feature_2": "",
                "depth": depth,
                "splits": splits,
            }
            per_tree_rows.append(row)
            rows_this_model.append(row)

        for i, tree in enumerate(model.int_trees):
            depth = node_depth(tree.root)
            splits = node_splits(tree.root)

            row = {
                "cache_file": cache_file.name,
                "model_name": model_name,
                "n_name": n_name,
                "rho_name": rho_name,
                "task": task,
                "tree_group": "interaction",
                "tree_index": i,
                "feature_1": f"X{tree.j + 1}",
                "feature_2": f"X{tree.k + 1}",
                "depth": depth,
                "splits": splits,
            }
            per_tree_rows.append(row)
            rows_this_model.append(row)

        df_one = pd.DataFrame(rows_this_model)

        if df_one.empty:
            summary_rows.append(
                {
                    "cache_file": cache_file.name,
                    "model_name": model_name,
                    "n_name": n_name,
                    "rho_name": rho_name,
                    "task": task,
                    "n_main_trees": 0,
                    "n_interaction_trees": 0,
                    "n_total_trees": 0,
                    "avg_main_depth": 0.0,
                    "avg_interaction_depth": 0.0,
                    "avg_total_depth": 0.0,
                    "max_main_depth": 0,
                    "max_interaction_depth": 0,
                    "max_total_depth": 0,
                    "avg_main_splits": 0.0,
                    "avg_interaction_splits": 0.0,
                    "avg_total_splits": 0.0,
                    "max_main_splits": 0,
                    "max_interaction_splits": 0,
                    "max_total_splits": 0,
                    "sum_main_splits": 0,
                    "sum_interaction_splits": 0,
                    "sum_total_splits": 0,
                }
            )
            continue

        df_main = df_one[df_one["tree_group"] == "main"]
        df_int = df_one[df_one["tree_group"] == "interaction"]

        summary_rows.append(
            {
                "model_name": model_name,
                "n_name": n_name,
                "rho_name": rho_name,
                "task": task,
                "n_main_trees": int(len(df_main)),
                "n_interaction_trees": int(len(df_int)),
                "n_total_trees": int(len(df_one)),
                "avg_main_depth": float(df_main["depth"].mean()) if not df_main.empty else 0.0,
                "avg_interaction_depth": float(df_int["depth"].mean()) if not df_int.empty else 0.0,

                "max_main_depth": int(df_main["depth"].max()) if not df_main.empty else 0,
                "max_interaction_depth": int(df_int["depth"].max()) if not df_int.empty else 0,

                "avg_main_splits": float(df_main["splits"].mean()) if not df_main.empty else 0.0,
                "avg_interaction_splits": float(df_int["splits"].mean()) if not df_int.empty else 0.0,

                "max_main_splits": int(df_main["splits"].max()) if not df_main.empty else 0,
                "max_interaction_splits": int(df_int["splits"].max()) if not df_int.empty else 0,

                "sum_main_splits": int(df_main["splits"].sum()) if not df_main.empty else 0,
                "sum_interaction_splits": int(df_int["splits"].sum()) if not df_int.empty else 0,
            }
        )

    per_tree_df = pd.DataFrame(per_tree_rows)
    summary_df = pd.DataFrame(summary_rows)

    reg_df = summary_df[summary_df["task"] == "regression"]
    clf_df = summary_df[summary_df["task"] == "classification"]

    reg_path = OUTDIR / "tree_stats_summary_regression.csv"
    clf_path = OUTDIR / "tree_stats_summary_classification.csv"

    reg_df.to_csv(reg_path, index=False)
    clf_df.to_csv(clf_path, index=False)

    print(f"saved: {reg_path}")
    print(f"saved: {clf_path}")

    if not summary_df.empty:
        print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()