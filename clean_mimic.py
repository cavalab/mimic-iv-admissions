import os
import pandas as pd
import warnings
import fire
import numpy as np

warnings.filterwarnings("ignore")


def merge_all(adm, ed, tri, pat):
    df = ed.merge(tri.drop("subject_id", axis=1), on="stay_id")
    df = df.merge(adm.drop("hadm_id", axis=1), on="subject_id")
    df = df.merge(pat.drop(["dod"], axis=1), on="subject_id")
    df = df.drop_duplicates(subset="stay_id")
    # sort by intime
    df = df.sort_values(by="intime")
    df = df.set_index("stay_id")
    return df


COLUMNS_TO_KEEP = [
    "subject_id",
    "hadm_id",
    "intime",
    "admission_type",
    "admission_location",
    "temperature",
    "heartrate",
    "resprate",
    "o2sat",
    "sbp",
    "dbp",
    "pain",
    "acuity",
    "ethnicity",
    "chiefcomplaint",
    "gender",
    "age_at_visit",
]


def adm_count(x):
    """Counts previous admissions for a subject's visits"""
    tmp = pd.Series(index=x.index)
    tmp.loc[x.index[0]] = 0
    tmp.iloc[1:] = (~x.iloc[:-1]["hadm_id"].isna()).cumsum()
    tmp.name = "prev_adm"

    return tmp


def remove_outliers(
    data,
    columns=[
        "temperature",
        "heartrate",
        "resprate",
        "o2sat",
        "sbp",
        "dbp",
        "pain",
        "acuity",
    ],
):
    min_temp = 95
    max_temp = 105
    min_hr = 30
    max_hr = 300
    min_rs = 2
    max_rs = 200
    min_o2 = 50
    max_o2 = 100

    min_sbp = 30
    max_sbp = 400

    min_dbp = 30
    max_dbp = 300

    pain_min = 0
    pain_max = 20

    acu_min = 1
    acu_max = 5

    min_l = [min_temp, min_hr, min_rs, min_o2, min_sbp, min_dbp, pain_min, acu_min]
    max_l = [max_temp, max_hr, max_rs, max_o2, max_sbp, max_dbp, pain_max, acu_max]
    l = len(columns)
    x = data.copy()
    for i in range(l):
        c = columns[i]
        low = min_l[i]
        high = max_l[i]
        x.loc[(x[c] < low) | (x[c] > high), c] = float("nan")
    return x


def age_at_visit(df):
    """Get the age of the patient at their visit.
    anchor_age: age at anchor year.
    anchor_year: randomized year to anchor patient data.
    intime: intime for the patient in the ed.

    age at visit is the anchor_age plus the intime year minus the anchor_year.
    """
    anchor_age = df["anchor_age"].values
    anchor_year = df["anchor_year"].copy()
    intime = pd.to_datetime(df["intime"].copy())
    year_delta = np.asarray([it.year - ay for it, ay in zip(intime, anchor_year)])
    age_at_visit = anchor_age + year_delta

    return age_at_visit


def process_data(
    mimic_path="./data",
    admissions_file="core/admissions.csv.gz",
    ed_file="ed/edstays.csv.gz",
    triage_file="ed/triage.csv.gz",
    patients_file="core/patients.csv.gz",
    results_path="final.csv",
):
    print("loading and processing mimic files...")
    adm = pd.read_csv(os.path.join(mimic_path, admissions_file))
    ed = pd.read_csv(os.path.join(mimic_path, ed_file))
    tri = pd.read_csv(os.path.join(mimic_path, triage_file))
    pat = pd.read_csv(os.path.join(mimic_path, patients_file))

    print("merging...")
    df = merge_all(adm, ed, tri, pat)

    df["age_at_visit"] = age_at_visit(df)

    print("dropping columns...")
    df = df[COLUMNS_TO_KEEP]

    print("adding columns..")
    ##########
    print("previous visits..")
    df.loc[:, "prev_visit"] = df.groupby("subject_id").cumcount()
    print("previous admissions..")
    tmp = df.groupby("subject_id").apply(adm_count)
    df = pd.merge(df, tmp, on="stay_id")

    df["y"] = ~df.hadm_id.isna()
    # filter observation admissions
    df = df.loc[~((df.y == 1) & (df.admission_type.str.contains("OBSERVATION"))), :]
    ##########

    print("removing outliers...")
    df = remove_outliers(df)
    print("patients:", df.groupby("y")["subject_id"].nunique())
    df = df.drop(
        columns=[
            "hadm_id",
            "subject_id",
            "intime",
            "admission_location",
            "admission_type",
        ]
    )

    print("finished processing dataset.")
    df["y"].sum() / ((~df["y"]).sum())
    print(f"size: {df.shape}, cases: {df.y.sum() / len(df)}")
    print("dataset columns:", df.columns)
    print(df.head())

    print("saving...")
    df.to_csv(results_path, index=False)
    print("done.")


if __name__ == "__main__":
    fire.Fire(process_data)
