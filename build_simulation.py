"""
Membangun Data Simulasi Kelas STIKMAS untuk Demo Prototipe

Setiap baris (siswa simulasi x mata pelajaran) memakai 24 fitur asli dari satu baris
data uji OULAD kohor 2014 (data yang tidak dipakai saat pelatihan). Nama, NIS, dan
kelas adalah data fiktif. Komposisi siswa diatur agar demo memuat siswa aman,
siswa yang bermasalah di satu mapel (Tier 1), dan siswa yang bermasalah di banyak
mapel (Tier 2).

Jalankan sekali:
    python build_simulation.py
Hasil:
    simulasi/kelas_stikmas_hari60.csv
"""

import os
import numpy as np
import pandas as pd
from ews_features import build_features, FEATURE_COLS

SEED = 2026
OUT_DIR = "simulasi"
OUT_FILE = os.path.join(OUT_DIR, "kelas_stikmas_hari60.csv")
MAPEL_PER_SISWA = 3

NAMA = [
    "I Putu Adi Saputra", "Ni Kadek Ayu Lestari", "I Made Bagus Pranata", "Ni Komang Sri Wahyuni",
    "I Ketut Dharma Yasa", "Ni Putu Eka Pratiwi", "I Gede Wira Kusuma", "Ni Made Dewi Anggreni",
    "I Kadek Surya Wibawa", "Ni Luh Gede Purnami", "I Komang Arya Putra", "Ni Wayan Citra Dewi",
    "I Nyoman Rai Sudarsana", "Ni Kadek Intan Permata", "I Putu Gede Mahendra", "Ni Ketut Ratna Sari",
    "I Made Yoga Pradnyana", "Ni Putu Sinta Maharani", "I Wayan Agus Setiawan", "Ni Luh Putu Novitasari",
    "I Kadek Rian Ardana", "Ni Komang Tri Utami", "I Gede Bayu Prasetya", "Ni Made Ayu Kartika",
    "I Putu Krisna Wijaya", "Ni Kadek Mira Yanti", "I Komang Deva Saputra", "Ni Wayan Laksmi Dewi",
    "I Ketut Adi Wiguna", "Ni Putu Rahayu Astuti", "I Made Dwi Cahyadi", "Ni Luh Ade Paramita",
    "I Nyoman Satria Wibowo", "Ni Kadek Dian Pertiwi", "I Gede Arta Wiranata", "Ni Ketut Sekar Arum",
]

# (jumlah siswa, jumlah mapel berlabel berisiko dari 3 mapel)
KOMPOSISI = [(18, 0), (10, 1), (5, 2), (3, 3)]


def main():
    print("Memuat OULAD...")
    info = pd.read_csv("studentInfo.csv")
    vle = pd.read_csv("studentVle.csv")
    sa = pd.read_csv("studentAssessment.csv")
    ass = pd.read_csv("assessments.csv")
    mapping = pd.read_csv("course_mapping.csv")

    print("Mengekstrak 24 fitur (cut-off hari ke-60)...")
    df = build_features(info, vle, sa, ass)
    test = df[df["code_presentation"].str.startswith("2014")].copy()
    modules = sorted(set(test["code_module"]) & set(mapping["code_module"]))

    rng = np.random.default_rng(SEED)
    pools = {(m, r): test[(test["code_module"] == m) & (test["is_at_risk"] == r)].index.to_numpy() for m in modules for r in (0, 1)}
    used = set()

    def ambil(m, r):
        idx = rng.choice([i for i in pools[(m, r)] if i not in used])
        used.add(idx)
        return test.loc[idx]

    rows, no = [], 0
    for jumlah, n_berisiko in KOMPOSISI:
        for _ in range(jumlah):
            nama = NAMA[no]
            kelas = "X-RPL-1" if no % 2 == 0 else "X-RPL-2"
            nis = 26101 + no
            mapel = rng.choice(modules, size=MAPEL_PER_SISWA, replace=False)
            label = [1] * n_berisiko + [0] * (MAPEL_PER_SISWA - n_berisiko)
            for m, r in zip(mapel, label):
                src = ambil(m, r)
                row = {"nis": nis, "nama_siswa": nama, "kelas": kelas, "code_module": m}
                row.update({c: src[c] for c in FEATURE_COLS})
                row["hasil_akhir_oulad"] = src["final_result"]
                row["sumber_oulad"] = f"{src['code_module']}-{src['code_presentation']}-{int(src['id_student'])}"
                rows.append(row)
            no += 1

    out = pd.DataFrame(rows)
    os.makedirs(OUT_DIR, exist_ok=True)
    out.to_csv(OUT_FILE, index=False)
    print(f"Tersimpan: {OUT_FILE} ({out['nis'].nunique()} siswa, {len(out)} baris siswa-mapel)")


if __name__ == "__main__":
    main()
