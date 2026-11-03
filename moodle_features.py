"""
Ekstraksi 24 Fitur EWS dari Database Moodle Asli (mdl_*)

Padanan langsung dari ews_features.py (format OULAD). Nama dan urutan 24 fitur
dipertahankan agar kontrak API dan aplikasi web BK-EWS tidak berubah.

Unit analisis : siswa x mata pelajaran (course Moodle).
Hari ke-0     : mdl_course.startdate; fitur dihitung sampai hari ke-`cut_off`.
Label         : is_at_risk = 1 jika nilai sementara mapel (rata-rata persentase semua
                kuis/tugas yang sudah dibuka sampai tanggal dump, tidak dikerjakan = 0)
                di bawah KKM. Semester belum selesai saat dump diambil, sehingga label
                ini adalah proksi dari nilai akhir.

Pemetaan sumber data:
  A. Engagement   <- mdl_logstore_standard_log (1 event = 1 klik), sesi untuk durasi belajar
  B. Disiplin     <- mdl_assign, mdl_assign_submission, mdl_quiz, mdl_quiz_attempts
     "Lesson"     <- kuis Moodle (mdl_quiz_attempts), padanan CMA (computer-marked) di OULAD
  C. Nilai        <- mdl_quiz_attempts (sumgrades) dan mdl_assign_grades, dalam persen 0-100
  D. Riwayat      <- num_of_prev_attempts tidak tercatat di Moodle (diisi 0);
                     studied_credits = jumlah mapel yang diikuti siswa (proksi beban belajar)
"""

import numpy as np
import pandas as pd

from ews_features import FEATURE_COLS

DAY = 86400
CUT_OFF_DAY = 45
DEFAULT_KKM = 75.0
SESSION_GAP_MIN = 30        # jeda > 30 menit dianggap sesi baru
MAX_ATTEMPT_MIN = 180       # batas durasi satu percobaan kuis
DEFAULT_DUE_DAYS = 7        # kuis tanpa timeclose dianggap jatuh tempo 7 hari setelah dibuka
MIN_STUDENTS = 20           # mapel dengan siswa lebih sedikit diabaikan


def _students(t):
    """Pasangan (userid, courseid) dengan peran student di konteks course."""
    ctx = t["mdl_context"]
    ra = t["mdl_role_assignments"]
    st = ra[ra.roleid == 5].merge(ctx[ctx.contextlevel == 50][["id", "instanceid"]],
                                  left_on="contextid", right_on="id")
    st = st.rename(columns={"instanceid": "courseid"})[["userid", "courseid"]].drop_duplicates()
    users = t["mdl_user"]
    active = users[(users.deleted == 0) & (users.suspended == 0)].id
    return st[st.userid.isin(active)]


def _items(t, courses):
    """Daftar kuis dan tugas (terlihat) per course beserta waktu buka & jatuh tempo."""
    cm = t["mdl_course_modules"].merge(t["mdl_modules"][["id", "name"]], left_on="module", right_on="id",
                                       suffixes=("", "_m"))
    cm = cm[(cm.visible == 1) & (cm.deletioninprogress == 0) & cm.course.isin(courses)]

    q = t["mdl_quiz"].merge(cm[cm.name == "quiz"][["instance", "added"]], left_on="id", right_on="instance")
    q = pd.DataFrame({"kind": "quiz", "instance": q.id, "courseid": q.course,
                      "open": np.where(q.timeopen > 0, q.timeopen, q.added),
                      "due": np.where(q.timeclose > 0, q.timeclose, 0), "max": q.sumgrades,
                      "maxgrade": q.grade})
    a = t["mdl_assign"].merge(cm[cm.name == "assign"][["instance", "added"]], left_on="id", right_on="instance")
    a = pd.DataFrame({"kind": "assign", "instance": a.id, "courseid": a.course,
                      "open": np.where(a.allowsubmissionsfromdate > 0, a.allowsubmissionsfromdate, a.added),
                      "due": np.where(a.duedate > 0, a.duedate, 0), "max": a.grade.astype(float),
                      "maxgrade": a.grade.astype(float)})
    items = pd.concat([q, a], ignore_index=True)
    items["due"] = np.where(items.due > 0, items.due, items.open + DEFAULT_DUE_DAYS * DAY)
    return items


def _student_work(t, items):
    """Satu baris per (siswa, item): waktu kumpul pertama, nilai %, jumlah & durasi percobaan."""
    qa = t["mdl_quiz_attempts"]
    qa = qa[(qa.state == "finished") & (qa.preview == 0) & (qa.timefinish > 0)]
    qi = items[items.kind == "quiz"]
    qa = qa.merge(qi[["instance", "max", "maxgrade"]], left_on="quiz", right_on="instance")
    qa["pct"] = np.where(qa["max"] > 0, qa.sumgrades / qa["max"] * 100, np.nan)
    qa["minutes"] = np.minimum((qa.timefinish - qa.timestart) / 60, MAX_ATTEMPT_MIN).clip(lower=0)
    quiz = pd.DataFrame({"kind": "quiz", "instance": qa.quiz, "userid": qa.userid, "t_submit": qa.timefinish,
                         "pct": qa.pct, "t_grade": qa.timefinish, "minutes": qa.minutes})

    sub = t["mdl_assign_submission"]
    sub = sub[(sub.status == "submitted") & (sub.latest == 1)]
    gr = t["mdl_assign_grades"]
    gr = gr[gr.grade >= 0].sort_values("attemptnumber").groupby(["assignment", "userid"]).last().reset_index()
    asg = sub[["assignment", "userid", "timemodified"]].merge(
        gr[["assignment", "userid", "grade", "timemodified"]], on=["assignment", "userid"], how="outer",
        suffixes=("_sub", "_grade"))
    ai = items[items.kind == "assign"][["instance", "max"]]
    asg = asg.merge(ai, left_on="assignment", right_on="instance")
    assign = pd.DataFrame({"kind": "assign", "instance": asg.assignment, "userid": asg.userid,
                           "t_submit": asg.timemodified_sub.fillna(asg.timemodified_grade),
                           "pct": np.where(asg["max"] > 0, asg.grade / asg["max"] * 100, np.nan),
                           "t_grade": asg.timemodified_grade, "minutes": 0.0})
    return pd.concat([quiz, assign], ignore_index=True)


def _sessions_minutes(ts):
    """Estimasi menit belajar dari stempel waktu event: jumlah jeda <= 30 menit + 1 menit per sesi."""
    ts = np.sort(ts.values)
    if len(ts) == 0:
        return 0.0
    gaps = np.diff(ts) / 60.0
    return float(gaps[gaps <= SESSION_GAP_MIN].sum() + 1 + (gaps > SESSION_GAP_MIN).sum())


def build_dataset(t, cut_off=CUT_OFF_DAY, kkm=DEFAULT_KKM, min_students=MIN_STUDENTS, label_after_cutoff=False):
    """Menghasilkan DataFrame: identitas + 24 fitur (sampai hari ke-cut_off) + label is_at_risk.
    Baris tanpa item bernilai (label tidak terdefinisi) dibuang."""
    course = t["mdl_course"].set_index("id")
    st = _students(t)
    n_per_course = st.groupby("courseid").userid.nunique()
    items_all = _items(t, n_per_course.index)
    graded_courses = items_all.courseid.unique()
    keep = n_per_course[(n_per_course >= min_students) & n_per_course.index.isin(graded_courses)].index
    st = st[st.courseid.isin(keep)].copy()
    items = items_all[items_all.courseid.isin(keep)].copy()

    log = t["mdl_logstore_standard_log"]
    dump_time = int(log.timecreated.max())
    st["t0"] = st.courseid.map(course.startdate)
    st["t_cut"] = st.t0 + cut_off * DAY
    KEY = ["userid", "courseid"]

    # A. Engagement dari log (hanya event siswa di course tersebut, sebelum cut-off)
    lg = log[log.courseid.isin(keep) & (log.anonymous == 0)][["userid", "courseid", "timecreated", "contextlevel",
                                                               "contextinstanceid"]]
    lg = lg.merge(st[KEY + ["t0", "t_cut"]], on=KEY)
    lg = lg[(lg.timecreated >= lg.t0 - 14 * DAY) & (lg.timecreated <= lg.t_cut)]
    lg["day"] = ((lg.timecreated - lg.t0) // DAY).clip(lower=0)
    g = lg.groupby(KEY)
    eng = g.agg(total_clicks=("timecreated", "size"), active_days=("day", "nunique"), last_day=("day", "max")).reset_index()
    sites = lg[lg.contextlevel == 70].groupby(KEY).contextinstanceid.nunique().rename("unique_sites_accessed")
    last14 = lg[lg.day >= cut_off - 14].groupby(KEY).size().rename("clicks_last_14d")
    dur = g.timecreated.apply(_sessions_minutes).rename("course_duration_minutes")
    eng = eng.merge(sites, on=KEY, how="left").merge(last14, on=KEY, how="left").merge(dur, on=KEY, how="left")
    eng["days_inactive"] = cut_off - eng.last_day
    eng["course_duration_hours"] = eng.course_duration_minutes / 60.0

    # B & C. Tugas, kuis ("lesson"), dan nilai
    work = _student_work(t, items).merge(items[["kind", "instance", "courseid", "open", "due"]], on=["kind", "instance"])
    work = work.merge(st[KEY + ["t_cut"]], on=KEY)

    early = work[work.t_submit <= work.t_cut].copy()
    first = early.sort_values("t_submit").groupby(KEY + ["kind", "instance"]).first().reset_index()
    first["gap_days"] = (first.t_submit - first.due) / DAY
    first["is_late"] = (first.gap_days > 0).astype(int)
    first["is_procrastinated"] = ((first.gap_days >= -1) & (first.gap_days <= 0)).astype(int)
    disc = first.groupby(KEY).agg(late_submission_count=("is_late", "sum"), avg_submission_gap=("gap_days", "mean"),
                                  procrastination_count=("is_procrastinated", "sum")).reset_index()

    # Item wajib sampai cut-off vs yang sudah dikumpulkan
    due_items = items.merge(st[KEY + ["t_cut"]], on="courseid")
    due_items = due_items[due_items.due <= due_items.t_cut]
    done = first[["userid", "courseid", "kind", "instance"]].assign(done=1)
    due_items = due_items.merge(done, on=["userid", "courseid", "kind", "instance"], how="left")
    miss = due_items.groupby(KEY).agg(total_tasks_due=("instance", "size"),
                                      tasks_done=("done", "sum")).reset_index()
    miss["missing_assignments"] = miss.total_tasks_due - miss.tasks_done
    qdue = due_items[due_items.kind == "quiz"].groupby(KEY).agg(total_quiz_due=("instance", "size"),
                                                                 quiz_done=("done", "sum")).reset_index()

    quiz_early = early[early.kind == "quiz"]
    les = quiz_early.groupby(KEY).agg(lesson_attempts_count=("instance", "size"),
                                      lesson_time_spent_min=("minutes", "sum")).reset_index()
    best = early[early.t_grade <= early.t_cut].dropna(subset=["pct"])
    best = best.groupby(KEY + ["kind", "instance"]).agg(pct=("pct", "max"), t_grade=("t_grade", "min")).reset_index()
    les_score = best[best.kind == "quiz"].groupby(KEY).pct.mean().rename("lesson_avg_score")
    best["is_failing"] = (best.pct < 40).astype(int)
    sc = best.groupby(KEY).agg(avg_score=("pct", "mean"), min_score=("pct", "min"),
                               failing_tasks_count=("is_failing", "sum")).reset_index()
    ordered = best.sort_values("t_grade").groupby(KEY).pct
    trend = (ordered.last() - ordered.first()).rename("grade_trend")

    # D. Riwayat & beban belajar
    load = st.groupby("userid").courseid.nunique().rename("studied_credits")

    df = st.copy()
    for part in [eng, disc, miss, qdue, les, les_score, sc, trend]:
        df = df.merge(part, on=KEY, how="left")
    df = df.merge(load, on="userid", how="left")

    # Pengisian nilai kosong, aturan sama dengan ews_features.py
    for c in ["total_clicks", "active_days", "course_duration_hours", "unique_sites_accessed", "clicks_last_14d",
              "late_submission_count", "avg_submission_gap", "procrastination_count", "lesson_attempts_count",
              "lesson_time_spent_min", "lesson_avg_score", "avg_score", "min_score", "failing_tasks_count",
              "grade_trend", "total_tasks_due", "missing_assignments", "total_quiz_due", "quiz_done"]:
        df[c] = df[c].fillna(0)
    df["days_inactive"] = df.days_inactive.fillna(cut_off).clip(lower=0)
    df["is_inactive_gt_5d"] = (df.days_inactive > 5).astype(int)
    df["is_inactive_gt_14d"] = (df.days_inactive >= 14).astype(int)
    df["clicks_per_active_day"] = df.total_clicks / (df.active_days + 1e-5)
    df["clicks_last_14d_ratio"] = df.clicks_last_14d / (df.total_clicks + 1e-5)
    df["lesson_completion_ratio"] = np.where(df.total_quiz_due > 0, df.quiz_done / df.total_quiz_due, 1.0)
    df["score_rel_to_module"] = df.avg_score - df.groupby("courseid").avg_score.transform("mean")
    df["num_of_prev_attempts"] = 0

    # Label: nilai sementara mapel per tanggal dump. Item dihitung bila sudah dibuka dan sudah dinilai
    # guru untuk minimal satu siswa. Tidak dikerjakan = 0; sudah dikumpulkan tetapi belum dinilai = diabaikan.
    final = work.dropna(subset=["pct"]).groupby(KEY + ["kind", "instance"]).pct.max().reset_index()
    graded = final[["kind", "instance"]].drop_duplicates()
    opened = items[items.open <= dump_time].merge(graded, on=["kind", "instance"]).merge(st[KEY + ["t_cut"]], on="courseid")
    if label_after_cutoff:  # uji prospektif: label hanya dari item yang baru dibuka setelah cut-off
        opened = opened[opened.open > opened.t_cut]
    opened = opened.merge(final, on=KEY + ["kind", "instance"], how="left")
    submitted = work[["userid", "courseid", "kind", "instance"]].drop_duplicates().assign(submitted=1)
    opened = opened.merge(submitted, on=KEY + ["kind", "instance"], how="left")
    opened = opened[~(opened.pct.isna() & (opened.submitted == 1))]
    opened["pct"] = opened.pct.fillna(0).clip(upper=100)
    nilai = opened.groupby(KEY).agg(nilai_sementara=("pct", "mean"), jumlah_item=("pct", "size")).reset_index()
    df = df.merge(nilai, on=KEY, how="inner")
    df["is_at_risk"] = (df.nilai_sementara < kkm).astype(int)

    # Identitas (untuk simulasi web; jangan dipublikasikan)
    users = t["mdl_user"].set_index("id")
    cohort = t["mdl_cohort_members"].merge(t["mdl_cohort"][["id", "name"]], left_on="cohortid", right_on="id")
    kelas = cohort.groupby("userid").name.first().str.replace("26/27_", "", regex=False)
    df["nis"] = df.userid.map(users.username)
    # Di Moodle Kokar, lastname berisi "kelas|NISN|tahun ajaran", jadi cukup firstname
    lastname = df.userid.map(users.lastname).fillna("")
    df["nama_siswa"] = (df.userid.map(users.firstname) + np.where(lastname.str.contains("|", regex=False), "",
                                                                  " " + lastname)).str.strip()
    df["kelas"] = df.userid.map(kelas).fillna("-")
    df["code_module"] = df.courseid.map(course.shortname)
    df["nama_mapel"] = df.courseid.map(course.fullname)

    id_cols = ["userid", "courseid", "nis", "nama_siswa", "kelas", "code_module", "nama_mapel"]
    return df[id_cols + FEATURE_COLS + ["nilai_sementara", "jumlah_item", "is_at_risk"]].reset_index(drop=True)
