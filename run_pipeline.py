"""
Pipeline Operasional EWS 2-Tier: Fitur Siswa -> Model -> JSON -> Aplikasi Web BK-EWS

Langkah yang dijalankan:
1. Membaca 24 fitur per siswa per mata pelajaran (prototipe: simulasi/kelas_stikmas_hari60.csv;
   penerapan nyata: hasil ekstraksi dari database Moodle sekolah).
2. Menghitung probabilitas risiko dengan ews_moodle_model.pkl.
3. Menyusun payload Tier 1 (Guru Mapel) dan Tier 2 (Guru BK).
4. Menyimpan JSON (--export) dan/atau mengirimkannya ke API Laravel (--push).

Contoh:
    python run_pipeline.py --export sample_output.json
    python run_pipeline.py --push http://127.0.0.1:8000 --token RAHASIA
"""

import argparse
import json
import os
import sys
from collections import Counter
from datetime import datetime

import pandas as pd

from ews_features import FEATURE_COLS
from ews_inference import EWSInferenceEngine

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_INPUT = os.path.join(BASE_DIR, "simulasi", "kelas_stikmas_hari60.csv")
DEFAULT_MODEL = os.path.join(BASE_DIR, "ews_moodle_model.pkl")


def course_info(engine: EWSInferenceEngine, code: str) -> dict:
    for c in engine.course_mapping:
        if c["code_module"] == code:
            return {"code_module": code, "nama_mapel": c["nama_mapel"], "kategori": c["kategori_mapel"],
                    "guru_pengampu": c["nama_guru_mapel"], "kkm": int(c["kkm"])}
    return {"code_module": code, "nama_mapel": code, "kategori": "-", "guru_pengampu": "-", "kkm": 75}


def run(input_csv: str, model_path: str) -> dict:
    engine = EWSInferenceEngine(model_path)
    df = pd.read_csv(input_csv)
    missing = [c for c in FEATURE_COLS + ["nis", "nama_siswa", "kelas", "code_module"] if c not in df.columns]
    if missing:
        raise ValueError(f"Kolom tidak ditemukan di {input_csv}: {missing}")

    probs = engine.model.predict_proba(df[engine.feature_cols])[:, 1]
    tier1, per_siswa = [], {}
    for (_, row), prob in zip(df.iterrows(), probs):
        feats = {c: float(row[c]) for c in FEATURE_COLS}
        info = course_info(engine, row["code_module"])
        nis = int(row["nis"])
        payload = engine.generate_subject_teacher_payload(nis, info, feats, float(prob))
        payload.update({"nama_siswa": row["nama_siswa"], "kelas": row["kelas"]})
        tier1.append(payload)
        per_siswa.setdefault(nis, {"nama": row["nama_siswa"], "kelas": row["kelas"], "courses": []})
        per_siswa[nis]["courses"].append({"course": info, "features": feats, "probability": float(prob)})

    tier2 = []
    for nis, s in per_siswa.items():
        summary = engine.generate_bk_student_summary(nis, s["courses"])
        summary.update({"nama_siswa": s["nama"], "kelas": s["kelas"]})
        tier2.append(summary)
    tier2.sort(key=lambda x: ({"TINGGI": 0, "SEDANG": 1, "RENDAH": 2}[x["prioritas_konseling"]],
                              -x["rekapitulasi_semester"]["total_mapel_berisiko"]))

    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "versi_model": engine.model_version,
        "sumber_data": os.path.basename(input_csv),
        "threshold": engine.threshold,
        "tier1_subject_teacher": tier1,
        "tier2_counselor_bk": tier2,
    }


def print_summary(out: dict) -> None:
    t1 = Counter(a["analisis_risiko"]["tingkat_risiko"] for a in out["tier1_subject_teacher"])
    t2 = Counter(s["prioritas_konseling"] for s in out["tier2_counselor_bk"])
    print("=" * 72)
    print(f" EWS 2-Tier | model v{out['versi_model']} | data: {out['sumber_data']} | {out['generated_at']}")
    print("=" * 72)
    print(f" Tier 1 (siswa x mapel) : {len(out['tier1_subject_teacher'])} baris -> "
          f"TINGGI {t1['TINGGI']}, SEDANG {t1['SEDANG']}, RENDAH {t1['RENDAH']}")
    print(f" Tier 2 (rekap siswa)   : {len(out['tier2_counselor_bk'])} siswa -> "
          f"prioritas TINGGI {t2['TINGGI']}, SEDANG {t2['SEDANG']}, RENDAH {t2['RENDAH']}")
    print("-" * 72)
    print(" Prioritas konseling teratas (Guru BK):")
    for s in out["tier2_counselor_bk"][:5]:
        r = s["rekapitulasi_semester"]
        print(f"  - {s['nama_siswa']:<26} {s['kelas']:<8} {r['total_mapel_berisiko']}/{r['total_mapel_diambil']} mapel berisiko"
              f" | {s['profil_karakter_belajar']}")
    print("=" * 72)


def push(out: dict, base_url: str, token: str) -> bool:
    import requests
    headers = {"X-EWS-Token": token, "Accept": "application/json"}
    ok = True
    for path, key in [("/api/ews/tier1/alerts", "tier1_subject_teacher"), ("/api/ews/tier2/summaries", "tier2_counselor_bk")]:
        url = base_url.rstrip("/") + path
        try:
            r = requests.post(url, json={key: out[key]}, headers=headers, timeout=120)
            msg = r.json().get("message", r.text[:200]) if r.headers.get("content-type", "").startswith("application/json") else r.text[:200]
            print(f" POST {path} -> {r.status_code} {msg}")
            ok = ok and r.ok
        except Exception as e:
            print(f" POST {path} gagal: {e}")
            ok = False
    return ok


def main() -> None:
    ap = argparse.ArgumentParser(description="Pipeline EWS 2-Tier: fitur -> model -> JSON -> web")
    ap.add_argument("--input", default=DEFAULT_INPUT, help="CSV 24 fitur per siswa per mapel")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--export", help="Simpan payload JSON ke path ini")
    ap.add_argument("--push", metavar="BASE_URL", help="Kirim payload ke API Laravel, mis. http://127.0.0.1:8000")
    ap.add_argument("--token", default=os.environ.get("EWS_API_TOKEN", ""), help="Token API (atau env EWS_API_TOKEN)")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    out = run(args.input, args.model)
    if not args.quiet:
        print_summary(out)
    if args.export:
        os.makedirs(os.path.dirname(os.path.abspath(args.export)), exist_ok=True)
        with open(args.export, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=1)
        print(f" JSON tersimpan: {os.path.abspath(args.export)}")
    if args.push:
        if not push(out, args.push, args.token):
            sys.exit(1)


if __name__ == "__main__":
    main()
