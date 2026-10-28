"""
Ekstraksi 24 Fitur EWS dari Log LMS (format OULAD)

Logika di modul ini sama dengan bagian "Ekstraksi Fitur" pada model_ews.ipynb,
dipisahkan agar dapat dipakai ulang oleh pipeline simulasi dan inferensi.
Saat data Moodle STIKMAS tersedia, cukup ganti sumber tabel (mdl_logstore_standard_log,
mdl_assign_submission, mdl_lesson_attempts, mdl_grade_grades) ke format kolom yang sama.
"""

import numpy as np
import pandas as pd

CUT_OFF_DAY = 60
KEYS = ["code_module", "code_presentation", "id_student"]

FEATURE_COLS = [
    "total_clicks", "active_days", "days_inactive", "is_inactive_gt_5d", "is_inactive_gt_14d",
    "clicks_per_active_day", "clicks_last_14d_ratio", "unique_sites_accessed", "course_duration_hours",
    "missing_assignments", "late_submission_count", "avg_submission_gap", "procrastination_count",
    "lesson_attempts_count", "lesson_time_spent_min", "lesson_completion_ratio", "lesson_avg_score",
    "avg_score", "min_score", "score_rel_to_module", "failing_tasks_count", "grade_trend",
    "num_of_prev_attempts", "studied_credits",
]


def build_features(student_info, student_vle, student_assessment, assessments, cut_off=CUT_OFF_DAY):
    """Menghasilkan DataFrame berisi 24 fitur per siswa per mata pelajaran sampai hari ke-`cut_off`."""
    vle_early = student_vle[student_vle["date"] <= cut_off].copy()
    assessment_merged = pd.merge(student_assessment, assessments, on="id_assessment")
    assessment_early = assessment_merged[assessment_merged["date"] <= cut_off].copy()

    # A. Engagement dan estimasi durasi belajar
    vle_daily = vle_early.groupby(KEYS + ["date"])["sum_click"].sum().reset_index()
    vle_daily["daily_minutes"] = np.minimum(vle_daily["sum_click"] * 1.5, 120.0)
    course_duration = vle_daily.groupby(KEYS)["daily_minutes"].sum().reset_index(name="course_duration_minutes")
    course_duration["course_duration_hours"] = course_duration["course_duration_minutes"] / 60.0

    vle_features = vle_early.groupby(KEYS).agg(
        total_clicks=("sum_click", "sum"),
        active_days=("date", "nunique"),
        last_active_date=("date", "max"),
        unique_sites_accessed=("id_site", "nunique"),
    ).reset_index()
    vle_features["days_inactive"] = cut_off - vle_features["last_active_date"]
    vle_features["is_inactive_gt_5d"] = (vle_features["days_inactive"] > 5).astype(int)
    vle_features["is_inactive_gt_14d"] = (vle_features["days_inactive"] >= 14).astype(int)
    vle_features["clicks_per_active_day"] = vle_features["total_clicks"] / (vle_features["active_days"] + 1e-5)
    vle_features.drop(columns=["last_active_date"], inplace=True)

    vle_last_14d = vle_early[vle_early["date"] >= (cut_off - 14)].groupby(KEYS)["sum_click"].sum().reset_index(name="clicks_last_14d")
    vle_features = pd.merge(vle_features, vle_last_14d, on=KEYS, how="left")
    vle_features["clicks_last_14d"] = vle_features["clicks_last_14d"].fillna(0)
    vle_features["clicks_last_14d_ratio"] = vle_features["clicks_last_14d"] / (vle_features["total_clicks"] + 1e-5)
    vle_features = pd.merge(vle_features, course_duration[KEYS + ["course_duration_hours"]], on=KEYS, how="left")

    # B. Disiplin tugas dan aktivitas lesson
    assessment_early["submission_gap"] = assessment_early["date_submitted"] - assessment_early["date"]
    assessment_early["is_late"] = (assessment_early["submission_gap"] > 0).astype(int)
    assessment_early["is_procrastinated"] = ((assessment_early["submission_gap"] >= -1) & (assessment_early["submission_gap"] <= 0)).astype(int)
    assessment_early["is_failing"] = (assessment_early["score"] < 40).astype(int)

    assessment_features = assessment_early.groupby(KEYS).agg(
        tasks_submitted=("id_assessment", "count"),
        late_submission_count=("is_late", "sum"),
        avg_submission_gap=("submission_gap", "mean"),
        procrastination_count=("is_procrastinated", "sum"),
        avg_score=("score", "mean"),
        min_score=("score", "min"),
        failing_tasks_count=("is_failing", "sum"),
    ).reset_index()

    ordered = assessment_early.sort_values("date").groupby(KEYS)["score"]
    trend = pd.merge(ordered.first().reset_index(name="first_score"), ordered.last().reset_index(name="last_score"), on=KEYS)
    trend["grade_trend"] = trend["last_score"] - trend["first_score"]
    assessment_features = pd.merge(assessment_features, trend[KEYS + ["grade_trend"]], on=KEYS, how="left")

    cma_early = assessment_early[assessment_early["assessment_type"] == "CMA"]
    cma_features = cma_early.groupby(KEYS).agg(
        lesson_attempts_count=("id_assessment", "count"),
        lesson_avg_score=("score", "mean"),
    ).reset_index()
    cma_features["lesson_time_spent_min"] = cma_features["lesson_attempts_count"] * 25.0

    due = assessments[assessments["date"] <= cut_off]
    tasks_due = due.groupby(["code_module", "code_presentation"])["id_assessment"].nunique().reset_index(name="total_tasks_due")
    cma_due = due[due["assessment_type"] == "CMA"].groupby(["code_module", "code_presentation"])["id_assessment"].nunique().reset_index(name="total_cma_due")

    # C. Penggabungan dan pengisian nilai kosong (asumsi terburuk yang masuk akal)
    base_cols = KEYS + ["num_of_prev_attempts", "studied_credits", "final_result"]
    df = student_info[[c for c in base_cols if c in student_info.columns]].copy()
    for part, on in [(vle_features, KEYS), (assessment_features, KEYS), (cma_features, KEYS),
                     (tasks_due, ["code_module", "code_presentation"]), (cma_due, ["code_module", "code_presentation"])]:
        df = pd.merge(df, part, on=on, how="left")

    df["total_tasks_due"] = df["total_tasks_due"].fillna(0)
    df["tasks_submitted"] = df["tasks_submitted"].fillna(0)
    df["missing_assignments"] = np.maximum(0, df["total_tasks_due"] - df["tasks_submitted"])
    df["total_cma_due"] = df["total_cma_due"].fillna(0)
    for c in ["lesson_attempts_count", "lesson_avg_score", "lesson_time_spent_min", "total_clicks", "active_days",
              "course_duration_hours", "unique_sites_accessed", "clicks_per_active_day", "clicks_last_14d_ratio",
              "late_submission_count", "avg_submission_gap", "procrastination_count", "avg_score", "min_score",
              "failing_tasks_count", "grade_trend"]:
        df[c] = df[c].fillna(0)
    df["lesson_completion_ratio"] = np.where(df["total_cma_due"] > 0, df["lesson_attempts_count"] / df["total_cma_due"], 1.0)
    df["days_inactive"] = df["days_inactive"].fillna(cut_off)
    df["is_inactive_gt_5d"] = df["is_inactive_gt_5d"].fillna(1)
    df["is_inactive_gt_14d"] = df["is_inactive_gt_14d"].fillna(1)

    module_avg = df.groupby(["code_module", "code_presentation"])["avg_score"].transform("mean")
    df["score_rel_to_module"] = df["avg_score"] - module_avg
    if "final_result" in df.columns:
        df["is_at_risk"] = df["final_result"].isin(["Fail", "Withdrawn"]).astype(int)
    return df
