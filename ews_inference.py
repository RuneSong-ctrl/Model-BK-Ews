"""
Modul Inferensi Model EWS Moodle (24 Fitur)
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple

DEFAULT_MODEL_PATH = "ews_moodle_model.pkl"

PRESET_PROFILES: Dict[str, Dict[str, Any]] = {
    "teladan_aman": {
        "label": "Siswa Teladan (Aman - Rendah 16%)",
        "description": "Aktif harian, tugas tepat waktu, nilai asesmen tinggi, aktif modul Lesson.",
        "course": {
            "code_module": "AAA",
            "nama_mapel": "Pemrograman Web dan Perangkat Bergerak",
            "kategori": "Kejuruan",
            "guru_pengampu": "I Wayan Sudarma, S.Kom.",
            "kkm": 75
        },
        "features": {
            "total_clicks": 680,
            "active_days": 28,
            "days_inactive": 1,
            "is_inactive_gt_5d": 0,
            "is_inactive_gt_14d": 0,
            "clicks_per_active_day": 24.28,
            "clicks_last_14d_ratio": 0.45,
            "unique_sites_accessed": 35,
            "course_duration_hours": 16.5,
            "missing_assignments": 0,
            "late_submission_count": 0,
            "avg_submission_gap": -4.0,
            "procrastination_count": 0,
            "lesson_attempts_count": 4,
            "lesson_time_spent_min": 100.0,
            "lesson_completion_ratio": 1.0,
            "lesson_avg_score": 88.5,
            "avg_score": 85.0,
            "min_score": 80.0,
            "score_rel_to_module": 18.0,
            "failing_tasks_count": 0,
            "grade_trend": 6.0,
            "num_of_prev_attempts": 0,
            "studied_credits": 60
        }
    },
    "waspada_sedang": {
        "label": "Siswa Mulai Waspada (Sedang 54%)",
        "description": "Tugas terkumpul, namun inaktif 9 hari, waktu belajar rendah (4.2 jam), belum akses Lesson.",
        "course": {
            "code_module": "AAA",
            "nama_mapel": "Pemrograman Web dan Perangkat Bergerak",
            "kategori": "Kejuruan",
            "guru_pengampu": "I Wayan Sudarma, S.Kom.",
            "kkm": 75
        },
        "features": {
            "total_clicks": 170,
            "active_days": 10,
            "days_inactive": 9,
            "is_inactive_gt_5d": 1,
            "is_inactive_gt_14d": 0,
            "clicks_per_active_day": 17.0,
            "clicks_last_14d_ratio": 0.318,
            "unique_sites_accessed": 25,
            "course_duration_hours": 4.2,
            "missing_assignments": 0,
            "late_submission_count": 0,
            "avg_submission_gap": -2.5,
            "procrastination_count": 0,
            "lesson_attempts_count": 0,
            "lesson_time_spent_min": 0.0,
            "lesson_completion_ratio": 1.0,
            "lesson_avg_score": 0.0,
            "avg_score": 67.0,
            "min_score": 66.0,
            "score_rel_to_module": 2.2,
            "failing_tasks_count": 0,
            "grade_trend": 2.0,
            "num_of_prev_attempts": 0,
            "studied_credits": 60
        }
    },
    "krisis_inaktif": {
        "label": "Siswa Inaktif & Drop-out (Tinggi 94%)",
        "description": "Inaktif 18 hari, 3 tugas bolong, 2 kali terlambat, nilai drop drastis.",
        "course": {
            "code_module": "AAA",
            "nama_mapel": "Pemrograman Web dan Perangkat Bergerak",
            "kategori": "Kejuruan",
            "guru_pengampu": "I Wayan Sudarma, S.Kom.",
            "kkm": 75
        },
        "features": {
            "total_clicks": 45,
            "active_days": 3,
            "days_inactive": 18,
            "is_inactive_gt_5d": 1,
            "is_inactive_gt_14d": 1,
            "clicks_per_active_day": 15.0,
            "clicks_last_14d_ratio": 0.05,
            "unique_sites_accessed": 6,
            "course_duration_hours": 1.2,
            "missing_assignments": 3,
            "late_submission_count": 2,
            "avg_submission_gap": 3.5,
            "procrastination_count": 3,
            "lesson_attempts_count": 0,
            "lesson_time_spent_min": 0.0,
            "lesson_completion_ratio": 0.0,
            "lesson_avg_score": 0.0,
            "avg_score": 32.0,
            "min_score": 25.0,
            "score_rel_to_module": -31.0,
            "failing_tasks_count": 2,
            "grade_trend": -20.0,
            "num_of_prev_attempts": 1,
            "studied_credits": 60
        }
    },
    "kesulitan_akademik": {
        "label": "Siswa Rajin Kesulitan Materi (Tinggi 94%)",
        "description": "Akses LMS lumayan tapi 2 tugas belum kumpul, nilai di bawah KKM dan tren menurun.",
        "course": {
            "code_module": "BBB",
            "nama_mapel": "Basis Data",
            "kategori": "Kejuruan",
            "guru_pengampu": "Ni Made Rai Astuti, S.Pd.",
            "kkm": 75
        },
        "features": {
            "total_clicks": 280,
            "active_days": 18,
            "days_inactive": 3,
            "is_inactive_gt_5d": 0,
            "is_inactive_gt_14d": 0,
            "clicks_per_active_day": 15.5,
            "clicks_last_14d_ratio": 0.28,
            "unique_sites_accessed": 20,
            "course_duration_hours": 7.8,
            "missing_assignments": 2,
            "late_submission_count": 1,
            "avg_submission_gap": 1.2,
            "procrastination_count": 2,
            "lesson_attempts_count": 2,
            "lesson_time_spent_min": 40.0,
            "lesson_completion_ratio": 0.5,
            "lesson_avg_score": 42.0,
            "avg_score": 36.5,
            "min_score": 30.0,
            "score_rel_to_module": -28.5,
            "failing_tasks_count": 3,
            "grade_trend": -8.0,
            "num_of_prev_attempts": 0,
            "studied_credits": 60
        }
    }
}


class EWSInferenceEngine:
    def __init__(self, model_path: str = DEFAULT_MODEL_PATH):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        self.model_path = model_path
        self.payload = joblib.load(model_path)
        self.model = self.payload["model"]
        self.feature_cols: List[str] = self.payload["feature_cols"]
        self.threshold: float = float(self.payload.get("threshold", 0.40))
        self.course_mapping: List[Dict[str, Any]] = self.payload.get("course_mapping", [])
        card_path = os.path.join(os.path.dirname(os.path.abspath(model_path)), "model_card.json")
        self.model_card: Dict[str, Any] = {}
        if os.path.exists(card_path):
            with open(card_path, encoding="utf-8") as f:
                self.model_card = json.load(f)
        self.model_version: str = self.model_card.get("versi", "1.0.0")

    def predict_probability(self, features: Dict[str, Any]) -> float:
        row = {col: features.get(col, 0.0) for col in self.feature_cols}
        df = pd.DataFrame([row], columns=self.feature_cols)
        prob = float(self.model.predict_proba(df)[0, 1])
        return prob

    def determine_risk_level(self, prob: float) -> Tuple[str, str, str]:
        if prob >= 0.70:
            return "TINGGI", "MERAH", "Segera"
        if prob >= self.threshold:
            return "SEDANG", "KUNING", "Menengah"
        return "RENDAH", "HIJAU", "Rendah"

    def extract_trigger_factors(self, features: Dict[str, Any], prob: float) -> List[str]:
        triggers = []
        missing = int(features.get("missing_assignments", 0))
        if missing > 0:
            triggers.append(f"{missing} tugas belum dikumpulkan.")
        lesson_attempts = int(features.get("lesson_attempts_count", 0))
        lesson_score = float(features.get("lesson_avg_score", 0.0))
        if lesson_attempts == 0:
            triggers.append("Belum mengakses aktivitas Lesson interaktif sama sekali.")
        elif lesson_score < 50.0:
            triggers.append(f"Nilai rata-rata Lesson di bawah standar ({lesson_score:.1f}).")
        duration = float(features.get("course_duration_hours", 0.0))
        if duration < 5.0:
            triggers.append(f"Akumulasi waktu belajar sangat rendah ({duration:.1f} jam).")
        days_inactive = int(features.get("days_inactive", 0))
        if days_inactive >= 14:
            triggers.append(f"Inaktivitas kritis di kursus selama {days_inactive} hari berturut-turut.")
        elif days_inactive >= 5:
            triggers.append(f"Inaktif di kursus selama {days_inactive} hari.")
        late_sub = int(features.get("late_submission_count", 0))
        if late_sub > 0:
            triggers.append(f"{late_sub} kali mengumpulkan tugas melewati batas waktu.")
        avg_score = float(features.get("avg_score", 0.0))
        if avg_score < 40.0:
            triggers.append(f"Rata-rata nilai asesmen rendah ({avg_score:.1f} / 100).")
        fail_tasks = int(features.get("failing_tasks_count", 0))
        if fail_tasks > 0:
            triggers.append(f"{fail_tasks} tugas bernilai di bawah standar ketuntasan (< 40).")
        trend = float(features.get("grade_trend", 0.0))
        if trend <= -5.0:
            triggers.append(f"Tren performa nilai menurun sebesar {abs(trend):.1f} poin.")
        if not triggers:
            if prob >= self.threshold:
                triggers.append("Kombinasi pola interaksi terdeteksi berisiko oleh model.")
            else:
                triggers.append("Semua metrik keterlibatan dan kepatuhan dalam batas aman.")
        return triggers

    def generate_subject_teacher_payload(
        self,
        student_id: int,
        course_info: Dict[str, Any],
        features: Dict[str, Any],
        probability: float
    ) -> Dict[str, Any]:
        level, color, _ = self.determine_risk_level(probability)
        triggers = self.extract_trigger_factors(features, probability)
        rekomendasi = (
            "Perlu perhatian langsung dan konfirmasi tugas di kelas."
            if level == "TINGGI"
            else "Disarankan pemantauan berkala dan pendampingan materi."
            if level == "SEDANG"
            else "Aktivitas belajar terpantau normal."
        )

        return {
            "target_audience": "GURU_MATA_PELAJARAN",
            "id_siswa": student_id,
            "mata_pelajaran": {
                "kode_modul": course_info.get("code_module", "AAA"),
                "nama_mapel": course_info.get("nama_mapel", "Pemrograman Web dan Perangkat Bergerak"),
                "kategori": course_info.get("kategori", "Kejuruan"),
                "guru_pengampu": course_info.get("guru_pengampu", "I Wayan Sudarma, S.Kom."),
                "kkm": course_info.get("kkm", 75)
            },
            "analisis_risiko": {
                "tingkat_risiko": level,
                "status_warna": color,
                "probabilitas_risiko": round(probability, 4),
                "threshold": self.threshold
            },
            "metrik_24_fitur_model": {
                "engagement_dan_durasi": {
                    "total_clicks": int(features.get("total_clicks", 0)),
                    "active_days": int(features.get("active_days", 0)),
                    "days_inactive": int(features.get("days_inactive", 0)),
                    "is_inactive_gt_5d": int(features.get("is_inactive_gt_5d", 0)),
                    "is_inactive_gt_14d": int(features.get("is_inactive_gt_14d", 0)),
                    "clicks_per_active_day": round(float(features.get("clicks_per_active_day", 0.0)), 2),
                    "clicks_last_14d_ratio": round(float(features.get("clicks_last_14d_ratio", 0.0)), 3),
                    "unique_sites_accessed": int(features.get("unique_sites_accessed", 0)),
                    "course_duration_hours": round(float(features.get("course_duration_hours", 0.0)), 2)
                },
                "kepatuhan_tugas": {
                    "missing_assignments": int(features.get("missing_assignments", 0)),
                    "late_submission_count": int(features.get("late_submission_count", 0)),
                    "avg_submission_gap": round(float(features.get("avg_submission_gap", 0.0)), 2),
                    "procrastination_count": int(features.get("procrastination_count", 0))
                },
                "aktivitas_lesson": {
                    "lesson_attempts_count": int(features.get("lesson_attempts_count", 0)),
                    "lesson_time_spent_min": round(float(features.get("lesson_time_spent_min", 0.0)), 1),
                    "lesson_completion_ratio": round(float(features.get("lesson_completion_ratio", 0.0)), 2),
                    "lesson_avg_score": round(float(features.get("lesson_avg_score", 0.0)), 1)
                },
                "akademik": {
                    "avg_score": round(float(features.get("avg_score", 0.0)), 1),
                    "min_score": round(float(features.get("min_score", 0.0)), 1),
                    "score_rel_to_module": round(float(features.get("score_rel_to_module", 0.0)), 1),
                    "failing_tasks_count": int(features.get("failing_tasks_count", 0)),
                    "grade_trend": round(float(features.get("grade_trend", 0.0)), 1)
                },
                "konteks_siswa": {
                    "num_of_prev_attempts": int(features.get("num_of_prev_attempts", 0)),
                    "studied_credits": int(features.get("studied_credits", 60))
                }
            },
            "faktor_pemicu": triggers,
            "rekomendasi_tindakan": rekomendasi
        }

    def generate_bk_student_summary(
        self,
        student_id: int,
        courses_data: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        total_mapel = len(courses_data)
        berisiko = [c for c in courses_data if c["probability"] >= self.threshold]
        total_jam = sum(c["features"].get("course_duration_hours", 0) for c in courses_data)
        missing_tasks = sum(c["features"].get("missing_assignments", 0) for c in courses_data)
        late_tasks = sum(c["features"].get("late_submission_count", 0) for c in courses_data)
        max_inactive = max((c["features"].get("days_inactive", 0) for c in courses_data), default=0)

        if len(berisiko) >= 2:
            # Sama dengan logika generate_bk_student_summary pada model_ews.ipynb
            if missing_tasks >= 3 and late_tasks >= 2:
                profil = "Prokrastinasi Sistemik (Keterlambatan dan kelalaian tugas di banyak mapel)."
            elif max_inactive >= 14:
                profil = "Inaktivitas Menyeluruh (Absensi belajar berkepanjangan dari LMS)."
            else:
                profil = "Kendala Belajar Multimapel."
            prioritas = "TINGGI"
            rekomendasi = "Jadwalkan konseling individual dan koordinasi dengan wali kelas."
        elif len(berisiko) == 1:
            nama_mapel = berisiko[0]["course"].get("nama_mapel", "Mata Pelajaran")
            profil = f"Kesulitan Spesifik Mata Pelajaran ({nama_mapel})."
            prioritas = "SEDANG"
            rekomendasi = "Koordinasi dengan Guru Mata Pelajaran pengampu untuk evaluasi remedial."
        else:
            profil = "Profil Belajar Baik / Stabil."
            prioritas = "RENDAH"
            rekomendasi = "Pertahankan motivasi belajar siswa."

        rincian = []
        for c in courses_data:
            p = c["probability"]
            lvl, col, _ = self.determine_risk_level(p)
            feat = c["features"]
            rincian.append({
                "kode_modul": c["course"].get("code_module", ""),
                "nama_mapel": c["course"].get("nama_mapel", ""),
                "kategori": c["course"].get("kategori", ""),
                "guru_pengampu": c["course"].get("guru_pengampu", ""),
                "probabilitas": round(p, 3),
                "status_risiko": col,
                "tugas_belum_dikumpul": int(feat.get("missing_assignments", 0)),
                "durasi_belajar_jam": round(float(feat.get("course_duration_hours", 0)), 1),
                "lesson_attempts": int(feat.get("lesson_attempts_count", 0)),
                "days_inactive": int(feat.get("days_inactive", 0)),
                "avg_score": round(float(feat.get("avg_score", 0)), 1)
            })

        return {
            "target_audience": "GURU_BK",
            "id_siswa": student_id,
            "rekapitulasi_semester": {
                "total_mapel_diambil": total_mapel,
                "total_mapel_berisiko": len(berisiko),
                "total_jam_belajar": round(total_jam, 1),
                "total_tugas_belum_dikumpul": int(missing_tasks),
                "total_tugas_terlambat": int(late_tasks),
                "inaktivitas_terlama_hari": int(max_inactive)
            },
            "profil_karakter_belajar": profil,
            "prioritas_konseling": prioritas,
            "rekomendasi_tindakan": rekomendasi,
            "rincian_per_mata_pelajaran": rincian
        }
