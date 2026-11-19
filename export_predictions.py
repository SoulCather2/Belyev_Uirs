
import argparse
import pickle
from pathlib import Path

import pandas as pd

from Major import PipelineConfig, classify_samples_by_dct

DATA_DIR = Path(__file__).parent / "data" / "borealtc"


class _ModelUnpickler(pickle.Unpickler):
    """dct_models.pkl is written by running Major.py as a script, so its classes live in __main__."""

    def find_class(self, module: str, name: str):
        if module == "__main__":
            module = "Major"
        return super().find_class(module, name)


def load_models(path: Path) -> tuple[dict, dict]:
    with path.open("rb") as file:
        return _ModelUnpickler(file).load()


def build_predictions_dataset(cleaned: pd.DataFrame, dct_models: dict, dct_models_std: dict,
                              config: PipelineConfig, reset_memory: bool = False) -> pd.DataFrame:
    """Original columns plus sample_idx, predicted_terrain, raw_predicted_terrain, wrong, raw_wrong."""
    samples = classify_samples_by_dct(cleaned, dct_models, dct_models_std, config, reset_memory=reset_memory)
    added = ["sample_idx", "predicted_terrain", "raw_predicted_terrain", "wrong", "raw_wrong"]
    return samples[list(cleaned.columns) + added]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reset-memory", action="store_true", help="reset probability memory at the start of each run")
    parser.add_argument("--output", type=Path, default=DATA_DIR / "dataset_with_predictions.csv.gz")
    args = parser.parse_args()

    cleaned = pd.read_csv(DATA_DIR / "merged_interpolation_features_filtered.csv")
    dct_models, dct_models_std = load_models(DATA_DIR / "dct_models.pkl")
    dataset = build_predictions_dataset(cleaned, dct_models, dct_models_std, PipelineConfig(), args.reset_memory)
    dataset.to_csv(args.output, index=False)

    summary = dataset.groupby("terrain")["wrong"].agg(n="size", error_rate="mean")
    print(summary.round(3).to_string())
    print(f"Overall error rate: {dataset['wrong'].mean():.3f}")
    print(f"Saved {len(dataset)} rows to {args.output}")


if __name__ == "__main__":
    main()
