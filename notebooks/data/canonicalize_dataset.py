import pandas as pd

INPUT_PATH = "notebooks/data/heart_cleaned.csv"
OUTPUT_PATH = "notebooks/data/heart_cleaned.csv"  # overwrite


def normalize_cp(s: pd.Series) -> pd.Series:
    # Normalize cp to 0-3
    s = s.astype(int)
    if s.min() >= 1 and s.max() <= 4:  # 1-4 -> 0-3
        s = s - 1
    return s


def normalize_slope(s: pd.Series) -> pd.Series:
    # Normalize slope to 0-2
    s = s.astype(int)
    if s.min() >= 1 and s.max() <= 3:  # 1-3 -> 0-2
        s = s - 1
    return s


def normalize_thal(s: pd.Series) -> pd.Series:
    # Normalize thal to 0-3
    # If already 0-3, keep.
    s = s.astype(int)
    if set(s.unique()).issubset({0, 1, 2, 3}):
        return s

    # Map common UCI style: 3=normal, 6=fixed defect, 7=reversible defect
    mapping = {3: 1, 6: 2, 7: 3}
    return s.map(lambda x: mapping.get(x, 0)).astype(int)


def main() -> None:
    df = pd.read_csv(INPUT_PATH)

    df["cp"] = normalize_cp(df["cp"])
    df["slope"] = normalize_slope(df["slope"])
    df["restecg"] = df["restecg"].astype(int)  # keep 0-2
    df["ca"] = df["ca"].astype(int)            # keep 0-4 typically
    df["thal"] = normalize_thal(df["thal"])

    if "target" in df.columns:
        df["target"] = df["target"].astype(int)

    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Canonicalized dataset written to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
