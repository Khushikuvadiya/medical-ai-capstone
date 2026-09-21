from pathlib import Path

import pandas as pd


# =========================================================
# PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

TRAIN_CSV = PROJECT_ROOT / "data" / "train_split.csv"
VAL_CSV = PROJECT_ROOT / "data" / "val_split.csv"
TEST_CSV = PROJECT_ROOT / "data" / "test_split.csv"

OUTPUT_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "data_leakage_check.csv"
)


# =========================================================
# LOAD DATA
# =========================================================

train_df = pd.read_csv(TRAIN_CSV)
val_df = pd.read_csv(VAL_CSV)
test_df = pd.read_csv(TEST_CSV)


print("\nDataset sizes")
print("---------------------------")
print(f"Train:      {len(train_df)}")
print(f"Validation: {len(val_df)}")
print(f"Test:       {len(test_df)}")


# =========================================================
# IDENTIFY IMAGE PATH COLUMN
# =========================================================

possible_columns = [
    "filepath",
    "file_path",
    "path",
    "image_path",
    "filename"
]

path_column = None

for column in possible_columns:

    if column in train_df.columns:

        path_column = column
        break


if path_column is None:

    raise ValueError(
        "Could not find an image path column. "
        f"Available columns: {list(train_df.columns)}"
    )


print(
    f"\nUsing image column: {path_column}"
)


# =========================================================
# NORMALIZE PATHS
# =========================================================

def normalize_paths(dataframe):

    return set(
        dataframe[path_column]
        .astype(str)
        .str.replace("\\", "/", regex=False)
        .str.strip()
    )


train_paths = normalize_paths(
    train_df
)

val_paths = normalize_paths(
    val_df
)

test_paths = normalize_paths(
    test_df
)


# =========================================================
# CHECK DUPLICATION INSIDE EACH SPLIT
# =========================================================

train_duplicates = (
    len(train_df)
    - len(train_paths)
)

val_duplicates = (
    len(val_df)
    - len(val_paths)
)

test_duplicates = (
    len(test_df)
    - len(test_paths)
)


# =========================================================
# CHECK CROSS-SPLIT LEAKAGE
# =========================================================

train_val_overlap = (
    train_paths
    .intersection(
        val_paths
    )
)

train_test_overlap = (
    train_paths
    .intersection(
        test_paths
    )
)

val_test_overlap = (
    val_paths
    .intersection(
        test_paths
    )
)


# =========================================================
# RESULTS
# =========================================================

results = [
    {
        "check":
            "Duplicates inside train",

        "count":
            train_duplicates,

        "status":
            "PASS"
            if train_duplicates == 0
            else "FAIL"
    },

    {
        "check":
            "Duplicates inside validation",

        "count":
            val_duplicates,

        "status":
            "PASS"
            if val_duplicates == 0
            else "FAIL"
    },

    {
        "check":
            "Duplicates inside test",

        "count":
            test_duplicates,

        "status":
            "PASS"
            if test_duplicates == 0
            else "FAIL"
    },

    {
        "check":
            "Train-validation overlap",

        "count":
            len(
                train_val_overlap
            ),

        "status":
            "PASS"
            if len(
                train_val_overlap
            ) == 0
            else "FAIL"
    },

    {
        "check":
            "Train-test overlap",

        "count":
            len(
                train_test_overlap
            ),

        "status":
            "PASS"
            if len(
                train_test_overlap
            ) == 0
            else "FAIL"
    },

    {
        "check":
            "Validation-test overlap",

        "count":
            len(
                val_test_overlap
            ),

        "status":
            "PASS"
            if len(
                val_test_overlap
            ) == 0
            else "FAIL"
    }
]


results_df = pd.DataFrame(
    results
)


# =========================================================
# SAVE
# =========================================================

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

results_df.to_csv(
    OUTPUT_FILE,
    index=False
)


# =========================================================
# DISPLAY
# =========================================================

print(
    "\nData Leakage Check"
)

print(
    "---------------------------"
)

print(
    results_df.to_string(
        index=False
    )
)


overall_pass = (
    results_df["status"]
    .eq("PASS")
    .all()
)


print(
    "\nOverall result:",
    "PASS"
    if overall_pass
    else "FAIL"
)


print(
    f"\nResults saved to:\n{OUTPUT_FILE}"
)