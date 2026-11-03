"""
Pelatihan & Evaluasi Model EWS dengan Data Moodle Asli SMK Negeri 3 Sukawati (Kokar Bali)

Mengganti dataset OULAD dengan dump database Moodle sekolah (exp_kokar.sql).
Konfigurasi model sama dengan model_ews.ipynb (Random Forest + kalibrasi sigmoid),
sehingga perbandingan OULAD vs data asli bersifat apel-ke-apel.

Skenario evaluasi:
  A (utama)     : fitur s.d. hari ke-45, label = nilai sementara mapel < KKM (semua item).
  B (prospektif): fitur s.d. hari ke-28, label hanya dari kuis/tugas yang dibuka SETELAH
                  hari ke-28 (model benar-benar menebak masa depan, tanpa tumpang tindih).
Validasi: 5-fold StratifiedGroupKFold dikelompokkan per siswa (siswa yang sama tidak
pernah muncul di data latih dan uji sekaligus), ditambah leave-one-course-out.

Contoh:
    python train_kokar.py --dump ../exp_kokar.sql
"""

import argparse
import json
import os
import re
from datetime import date

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, roc_curve, brier_score_loss, confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import LeaveOneGroupOut, StratifiedGroupKFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ews_features import FEATURE_COLS
from moodle_dump import load_tables
from moodle_features import DEFAULT_KKM, build_dataset

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Proporsi berisiko data sekolah ~82%, sehingga threshold OULAD (0,40) menandai hampir semua siswa.
# SEDANG mengikuti titik Youden (~0,79) hasil validasi skenario A; TINGGI = probabilitas sangat yakin.
THRESHOLD = 0.80
THRESHOLD_TINGGI = 0.95
BK_RASIO_TINGGI = 0.75  # prioritas BK TINGGI bila >= 75% mapel siswa berstatus TINGGI
OULAD_THRESHOLD = 0.40
NAMA_MAPEL = {"MAT XII": "Matematika XII"}  # nama course memuat nama guru
SEED = 42


def make_rf():
    rf = RandomForestClassifier(n_estimators=200, max_depth=10, min_samples_leaf=5, min_samples_split=10,
                                class_weight="balanced", random_state=SEED, n_jobs=-1)
    return CalibratedClassifierCV(estimator=rf, method="sigmoid", cv=StratifiedKFold(5, shuffle=True, random_state=SEED))


def make_lr():
    return make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=2000))


def rule_baseline(X):
    """Aturan sederhana guru: ada tugas belum dikumpulkan atau rata-rata nilai di bawah KKM."""
    return ((X.missing_assignments > 0) | (X.avg_score < DEFAULT_KKM)).astype(float).values


def metrics(y, p, thr=THRESHOLD):
    pred = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    has_both = len(np.unique(y)) == 2
    return {"roc_auc": round(roc_auc_score(y, p), 3) if has_both else None,
            "akurasi": round(accuracy_score(y, pred), 3),
            "precision_berisiko": round(precision_score(y, pred, zero_division=0), 3),
            "recall_berisiko": round(recall_score(y, pred, zero_division=0), 3),
            "f1_berisiko": round(f1_score(y, pred, zero_division=0), 3),
            "recall_aman": round(tn / (tn + fp), 3) if tn + fp else None,
            "brier_score": round(brier_score_loss(y, np.clip(p, 0, 1)), 3),
            "confusion": {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)}}


def oof(model_fn, X, y, groups, splitter):
    p = np.zeros(len(y))
    for tr, te in splitter.split(X, y, groups):
        m = model_fn()
        m.fit(X.iloc[tr], y.iloc[tr])
        p[te] = m.predict_proba(X.iloc[te])[:, 1]
    return p


KALIBRASI_BINS = [0, 0.5, 0.8, 0.95, 1.0001]
KALIBRASI_LABEL = ["< 50%", "50-79%", "80-94%", ">= 95%"]


def calibration_table(y, p):
    """Rata-rata prediksi vs proporsi asli berisiko per rentang probabilitas (prediksi out-of-fold)."""
    d = pd.DataFrame({"y": y, "p": p})
    d["rentang"] = pd.cut(d.p, KALIBRASI_BINS, labels=KALIBRASI_LABEL, right=False)
    g = d.groupby("rentang", observed=False).agg(n=("y", "size"), p=("p", "mean"), y=("y", "mean"))
    return [{"rentang": k, "jumlah": int(r.n), "rata_rata_prediksi": round(float(r.p), 3),
             "proporsi_asli_berisiko": round(float(r.y), 3)} for k, r in g.iterrows()]


def evaluate(df, oulad_model, name, probs=None):
    """Metrik semua model; bila `probs` (dict) diberikan, probabilitas out-of-fold ikut disimpan di sana."""
    X, y = df[FEATURE_COLS], df["is_at_risk"]
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    p_rf = oof(make_rf, X, y, df.userid, sgkf)
    p_lr = oof(make_lr, X, y, df.userid, sgkf)
    res = {
        "skenario": name,
        "jumlah_baris": int(len(df)), "jumlah_siswa": int(df.userid.nunique()), "jumlah_mapel": int(df.courseid.nunique()),
        "proporsi_berisiko": round(float(y.mean()), 3),
        "random_forest_kalibrasi": metrics(y, p_rf),
        "logistic_regression": metrics(y, p_lr),
        "aturan_sederhana": metrics(y, rule_baseline(X), thr=0.5),
        "model_oulad_tanpa_latih_ulang": metrics(y, oulad_model.predict_proba(X)[:, 1], thr=OULAD_THRESHOLD),
    }
    # Generalisasi ke mapel yang belum pernah dilihat model
    p_loco = oof(make_rf, X, y, df.courseid, LeaveOneGroupOut())
    res["random_forest_leave_one_course_out"] = metrics(y, p_loco)
    if probs is not None:
        probs.update({"y": y.values, "RF dilatih ulang": p_rf, "Logistic Regression": p_lr,
                      "Model OULAD langsung": oulad_model.predict_proba(X)[:, 1], "RF leave-one-mapel-out": p_loco})
    per = []
    for cid, g in df.assign(p=p_rf).groupby("code_module"):
        per.append({"mapel": cid, "n": int(len(g)), "berisiko": round(float(g.is_at_risk.mean()), 3),
                    "roc_auc": round(roc_auc_score(g.is_at_risk, g.p), 3) if g.is_at_risk.nunique() == 2 else None})
    res["per_mapel_rf"] = per
    fpr, tpr, thr = roc_curve(y, p_rf)
    res["threshold_youden_rf"] = round(float(thr[np.argmax(tpr - fpr)]), 3)
    res["kalibrasi_rf"] = calibration_table(y.values, p_rf)
    return res


def kategori(code):
    return "Kejuruan/Produktif" if code.upper().startswith(("JUR", "K.", "KIK", "MAPIL", "DASPRO")) else "Umum/Normatif-Adaptif"


def course_mapping(tables, df, anonim):
    """course_mapping untuk web BK-EWS; guru pengampu diambil dari peran editingteacher di Moodle."""
    ra, ctx, users = tables["mdl_role_assignments"], tables["mdl_context"], tables["mdl_user"].set_index("id")
    teach = ra[ra.roleid == 3].merge(ctx[ctx.contextlevel == 50][["id", "instanceid"]], left_on="contextid", right_on="id")
    teach = teach.sort_values("id_x").groupby("instanceid").userid.first()
    rows = []
    for (cid, code), g in df.groupby(["courseid", "code_module"]):
        nama = NAMA_MAPEL.get(code) or g.nama_mapel.iloc[0].title()
        for akronim in ["Xii", "Xi", "Kik", "Tkkr", "Ipas", "Kka"]:
            nama = re.sub(rf"\b{akronim}\b", akronim.upper(), nama)
        uid = teach.get(cid)
        guru = f"Guru {nama}" if anonim or uid is None else \
            f"{users.firstname[uid]} {users.lastname[uid]}".replace(" *", "").strip()
        rows.append({"code_module": code, "nama_mapel": nama, "kategori_mapel": kategori(code),
                     "nama_guru_mapel": guru, "no_wa_guru": "", "kkm": int(DEFAULT_KKM)})
    return rows


def anonymize(df):
    """Nama & NIS diganti urutan per kelas agar aman ditampilkan di demo, tangkapan layar, dan laporan."""
    ids = df[["userid", "kelas"]].drop_duplicates().sort_values(["kelas", "userid"]).userid.tolist()
    seq = {u: i + 1 for i, u in enumerate(ids)}
    out = df.copy()
    out["nis"] = out.userid.map(lambda u: 900000 + seq[u])
    out["nama_siswa"] = out.userid.map(lambda u: f"Siswa {seq[u]:03d}")
    return out


def export_data(tables, df_a, out_dir=os.path.join(BASE_DIR, "data_kokar")):
    """Fitur & course_mapping versi anonim untuk pipeline web (identitas asli tidak disimpan ke file)."""
    os.makedirs(out_dir, exist_ok=True)
    anonymize(df_a).to_csv(os.path.join(out_dir, "fitur_kokar_hari45_anonim.csv"), index=False)
    pd.DataFrame(course_mapping(tables, df_a, anonim=True)).to_csv(
        os.path.join(out_dir, "course_mapping_kokar_anonim.csv"), index=False)


def fit_final(tables, df_a, out_model=os.path.join(BASE_DIR, "ews_kokar_model.pkl")):
    """Model final dilatih pada seluruh data skenario A; mengembalikan feature importance."""
    X, y = df_a[FEATURE_COLS], df_a["is_at_risk"]
    final = make_rf().fit(X, y)
    imp = pd.Series(np.mean([c.estimator.feature_importances_ for c in final.calibrated_classifiers_], axis=0),
                    index=FEATURE_COLS).sort_values(ascending=False)
    joblib.dump({"model": final, "feature_cols": FEATURE_COLS, "threshold": THRESHOLD, "versi": "2.1.0",
                 "threshold_tinggi": THRESHOLD_TINGGI, "bk_rasio_tinggi": BK_RASIO_TINGGI,
                 # pemicu durasi: di bawah median siswa yang tidak berisiko (batas OULAD 5 jam tidak cocok untuk SMK)
                 "batas_durasi_jam": round(float(df_a.loc[y == 0, "course_duration_hours"].median()), 2),
                 "istilah_lesson": "kuis",
                 "course_mapping": course_mapping(tables, df_a, anonim=True)}, out_model)
    return imp


PRESET_DEF = [
    # kunci, judul, deskripsi, filter kelompok (df berisi kolom p = probabilitas model final)
    ("aktif_aman", "Aktif & nilai baik", "Rutin membuka Moodle, kuis dan tugas dikerjakan dengan nilai baik.",
     lambda d: d[d.p < THRESHOLD]),
    ("mulai_waspada", "Mulai waspada", "Masih aktif, tetapi ada kuis/tugas yang terlewat atau nilai mulai turun.",
     lambda d: d[(d.p >= THRESHOLD) & (d.p < THRESHOLD_TINGGI)]),
    ("rajin_nilai_rendah", "Aktif tetapi nilai rendah", "Sering membuka Moodle, tetapi nilai kuis/tugas di bawah KKM.",
     lambda d: d[(d.p >= THRESHOLD_TINGGI) & (d.active_days >= 3) & (d.avg_score > 0) & (d.avg_score < DEFAULT_KKM)]),
    ("inaktif", "Tidak aktif di Moodle", "Hampir tidak pernah membuka mapel ini dan kuis/tugas tidak dikerjakan.",
     lambda d: d[(d.p >= THRESHOLD_TINGGI) & (d.days_inactive >= 14)]),
]


def export_sandbox(df_a, model_path=os.path.join(BASE_DIR, "ews_kokar_model.pkl"),
                   out_path=os.path.join(BASE_DIR, "sandbox_presets.json")):
    """Preset sandbox dari baris siswa asli (anonim, tanpa identitas) yang paling mewakili tiap kelompok,
    beserta rentang slider (persentil 99) agar sesuai sebaran data sekolah."""
    payload = joblib.load(model_path)
    d = anonymize(df_a)
    d["p"] = payload["model"].predict_proba(d[FEATURE_COLS])[:, 1]
    mapping = {c["code_module"]: c for c in payload["course_mapping"]}
    scale = d[FEATURE_COLS].std().replace(0, 1)
    presets = {}
    for key, judul, desc, pick in PRESET_DEF:
        g = pick(d)
        # baris terdekat ke median kelompok (jarak terstandar) = profil paling khas
        row = g.loc[((g[FEATURE_COLS] - g[FEATURE_COLS].median()) / scale).abs().sum(axis=1).idxmin()]
        c = mapping[row.code_module]
        presets[key] = {
            "label": f"{judul} ({row.p:.0%})", "description": desc, "kelompok_n": int(len(g)),
            "course": {"code_module": c["code_module"], "nama_mapel": c["nama_mapel"], "kategori": c["kategori_mapel"],
                       "guru_pengampu": c["nama_guru_mapel"], "kkm": c["kkm"]},
            "features": {f: round(float(row[f]), 3) for f in FEATURE_COLS},
        }
    q = d[FEATURE_COLS].quantile(0.99)
    ranges = {f: {"min": float(min(0, d[f].min())), "max": float(max(1, np.ceil(q[f])))} for f in FEATURE_COLS}
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"sumber": "Moodle sekolah (anonim)", "presets": presets, "ranges": ranges}, f,
                  ensure_ascii=False, indent=1)
    return presets


def write_card(hasil, imp, out_model=os.path.join(BASE_DIR, "ews_kokar_model.pkl")):
    card = {
        "nama_model": "EWS Moodle 2-Tier (Random Forest terkalibrasi sigmoid) - data asli",
        "versi": "2.1.0",
        "tanggal_latih": date.today().isoformat(),
        "berkas_model": os.path.basename(out_model),
        "dataset": "Dump database Moodle SMK Negeri 3 Sukawati (Kokar Bali), semester ganjil 2026/2027",
        "periode_log": "6 Juli - 29 Oktober 2026 (semester belum selesai)",
        "cut_off_hari": 45,
        "hari_ke_0": "mdl_course.startdate (umumnya 23 Juli 2026)",
        "label": f"nilai sementara mapel < KKM {int(DEFAULT_KKM)} (rata-rata % kuis/tugas yang sudah dinilai; tidak dikerjakan = 0)",
        "jumlah_fitur": len(FEATURE_COLS),
        "threshold": THRESHOLD,
        "level_risiko": {"TINGGI": f"p >= {THRESHOLD_TINGGI}", "SEDANG": f"{THRESHOLD} <= p < {THRESHOLD_TINGGI}",
                         "RENDAH": f"p < {THRESHOLD}"},
        "prioritas_bk_tinggi": f">= 2 mapel berisiko dan >= {int(BK_RASIO_TINGGI * 100)}% mapel berstatus TINGGI",
        "fitur_konstan": ["num_of_prev_attempts (tidak tercatat di Moodle)"],
        "evaluasi": hasil,
        "fitur_teratas": imp.head(8).round(3).to_dict(),
        "catatan": "Prototipe. Data satu sekolah, satu semester berjalan; label adalah proksi nilai akhir.",
    }
    with open(os.path.join(BASE_DIR, "model_card_kokar.json"), "w", encoding="utf-8") as f:
        json.dump(card, f, ensure_ascii=False, indent=2)



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", default=os.path.join(BASE_DIR, "..", "exp_kokar.sql"))
    ap.add_argument("--cache", default=os.path.join(BASE_DIR, "data_kokar", "tables"))
    ap.add_argument("--out-model", default=os.path.join(BASE_DIR, "ews_kokar_model.pkl"))
    args = ap.parse_args()

    tables = load_tables(args.dump, args.cache)
    oulad = joblib.load(os.path.join(BASE_DIR, "baseline_oulad", "ews_moodle_model.pkl"))["model"]

    df_a = build_dataset(tables, cut_off=45)
    df_b = build_dataset(tables, cut_off=28, label_after_cutoff=True)
    export_data(tables, df_a)

    hasil = {"A": evaluate(df_a, oulad, "A - hari ke-45, label nilai sementara"),
             "B": evaluate(df_b, oulad, "B - hari ke-28, label item setelah cut-off")}
    imp = fit_final(tables, df_a, args.out_model)
    export_sandbox(df_a, args.out_model)
    write_card(hasil, imp, args.out_model)

    for k, r in hasil.items():
        print("=" * 78)
        print(f" Skenario {r['skenario']} | n={r['jumlah_baris']} siswa={r['jumlah_siswa']} mapel={r['jumlah_mapel']}"
              f" | berisiko={r['proporsi_berisiko']}")
        for m in ["random_forest_kalibrasi", "logistic_regression", "aturan_sederhana", "model_oulad_tanpa_latih_ulang",
                  "random_forest_leave_one_course_out"]:
            v = r[m]
            print(f"  {m:36s} AUC={v['roc_auc']}  acc={v['akurasi']}  P={v['precision_berisiko']}  "
                  f"R={v['recall_berisiko']}  F1={v['f1_berisiko']}  R_aman={v['recall_aman']}  Brier={v['brier_score']}")
        print("  per mapel:", [(p["mapel"], p["n"], p["berisiko"], p["roc_auc"]) for p in r["per_mapel_rf"]])
    print("=" * 78)
    print(" Fitur teratas (model final):")
    print(imp.head(8).round(3).to_string())
    print(f" Model tersimpan: {args.out_model}")


if __name__ == "__main__":
    main()
