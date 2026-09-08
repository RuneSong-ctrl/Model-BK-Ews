# Spesifikasi Teknis Sistem EWS Moodle 2-Tier
Dokumen arsitektur, spesifikasi API JSON lengkap (24 Fitur), dan pembagian kerja tim pengembang.

---

## 1. Arsitektur Sistem 2-Tier

Sistem Early Warning System (EWS) membagi deteksi risiko akademik siswa menjadi dua tingkatan:

```
                         [ Moodle LMS ]
                               |
                   [ Ingestion & Feature Eng ]
                               |
                    [ Model EWS (24 Fitur) ]
                               |
              +----------------+----------------+
              |                                 |
              v                                 v
   [ Tier 1: Per-Mata Pelajaran ]     [ Tier 2: Lintas Mapel ]
   Output: JSON Guru Mapel            Output: JSON Guru BK
   Aksi: Remedial / Tugas Kelas       Aksi: Konseling Siswa
```

1. **Tier 1 (Guru Mata Pelajaran)**:
   - Evaluasi performa belajar pada satu mata pelajaran spesifik.
   - Mengidentifikasi keterlambatan tugas, tugas belum dikumpul, durasi akses kursus, dan keterlibatan pada aktivitas *Lesson*.
   - Tindakan: Pembinaan di kelas, konfirmasi tugas, dan program remedial materi.

2. **Tier 2 (Guru Bimbingan Konseling)**:
   - Rekapitulasi profil belajar siswa secara holistik lintas seluruh mata pelajaran yang diambil pada semester aktif.
   - Mendiagnosis pola umum (misal: prokrastinasi merata di banyak mapel vs kesulitan belajar spesifik pada satu mapel kejuruan).
   - Tindakan: Pemanggilan konseling individual, koordinasi dengan wali kelas, dan komunikasi ke orang tua.

---

## 2. Referensi Pemetaan Mata Pelajaran (`course_mapping.csv`)

Kamus data eksternal untuk menghubungkan ID kursus teknis Moodle dengan struktur kurikulum sekolah:

| Kolom | Tipe Data | Keterangan | Contoh |
| :--- | :--- | :--- | :--- |
| `code_module` | String | Kode modul / ID kursus di Moodle | `AAA` |
| `nama_mapel` | String | Nama resmi mata pelajaran | Pemrograman Web dan Perangkat Bergerak |
| `kategori_mapel` | String | Kelompok mapel (`Kejuruan` / `Umum`) | Kejuruan/Produktif |
| `nama_guru_mapel` | String | Nama guru pengampu | I Wayan Sudarma, S.Kom. |
| `kontak_guru` | String | Kontak WhatsApp guru pengampu | +6281234567801 |
| `kkm` | Integer | Batas Kriteria Ketuntasan Minimal | 75 |

---

## 3. Spesifikasi Kebutuhan Web Dashboard

Aplikasi web mengonsumsi payload JSON dari model ML dan menyajikan data sesuai peran pengguna.

### 3.1 Antarmuka Guru Mata Pelajaran
- **Filter Kelas & Mapel**: Menampilkan kelas yang diampu guru yang sedang login.
- **Kartu Metrik Kursus**:
  - Jumlah siswa berstatus Aman (Hijau), Waspada (Kuning), dan Risiko Tinggi (Merah).
  - Rata-rata durasi belajar siswa pada kursus tersebut (dalam jam).
- **Tabel Monitoring Siswa**:
  - Kolom: ID Siswa, Nama Siswa, Skor Risiko, Durasi Belajar (Jam), Percobaan Lesson (Attempts), Rata-rata Nilai Lesson, Tugas Terlewat, Rekomendasi Tindakan.
- **Panel Rincian 24 Fitur**:
  - Menampilkan metrik lengkap: engagement, kepatuhan tugas, lesson, dan tren nilai.

### 3.2 Antarmuka Guru Bimbingan Konseling (BK)
- **Tabel Rekapitulasi Karakter Belajar**:
  - Menampilkan daftar siswa yang terindikasi kendala di minimal satu mata pelajaran.
  - Kolom: ID Siswa, Nama Siswa, Total Mapel Diambil, Jumlah Mapel Bermasalah, Inaktivitas Terlama (Hari), Diagnosis Karakter Belajar, Tingkat Prioritas, Aksi.
- **Penyaring Prioritas Konseling**:
  - Filter `Prioritas Tinggi`: Siswa dengan status merah pada >= 2 mata pelajaran.
  - Filter `Inaktif Kritis`: Siswa inaktif dari seluruh sistem > 14 hari.
  - Filter `Kendala Spesifik`: Siswa dengan masalah hanya pada 1 mata pelajaran.
- **Profil Detail Siswa**:
  - Matriks perbandingan performa di semua mata pelajaran (status risiko dan catatan tugas per mata pelajaran).
- **Jurnal Tindak Lanjut Konseling**:
  - Form pencatatan hasil konseling: tanggal pemanggilan, catatan kendala siswa, rencana tindak lanjut, dan status penanganan.

### 3.3 Antarmuka Administrator
- Modul sinkronisasi manual data Moodle (via REST API Moodle Web Service).
- Tabel upload dan update file `course_mapping.csv`.

---

## 4. Pembagian Kerja Tim Pengembang (2 Orang)

### Pengembang 1: Machine Learning & Data Pipeline
1. **Pipeline Data & Ekstraksi 24 Fitur**:
   - Memproses log Moodle dengan batasan cut-off day 60.
   - Mengekstrak 24 fitur (engagement, disiplin tugas, aktivitas lesson, akademik, dan konteks siswa).
2. **Pelatihan & Kalibrasi Model**:
   - Melatih model Random Forest terkalibrasi (`CalibratedClassifierCV`) dengan 5-fold Stratified CV.
   - Threshold klasifikasi optimal 0.40 (Recall >= 81%).
   - Menyimpan model ke `ews_moodle_model.pkl`.
3. **Query Ingestion Database Moodle**:
   - Menyusun query SQL ekstraksi tabel produksi Moodle: `mdl_lesson_attempts`, `mdl_lesson_timer`, `mdl_logstore_standard_log`, `mdl_assign_submission`, dan `mdl_grade_grades`.
4. **Modul Inferensi API**:
   - Menyediakan fungsi Python murni untuk menghasilkan output JSON Tier 1 (24 fitur) dan Tier 2 (agregasi holistik).

### Pengembang 2: LLM Service, API Backend & Web Application
1. **Generator Narasi Berbasis LLM (Opsional / Terpisah)**:
   - Membuat service pemanggil LLM (Groq API / Ollama lokal) yang menerima payload JSON 24 fitur untuk menghasilkan ringkasan laporan otomatis jika diminta pengguna.
2. **Backend REST API**:
   - Membangun endpoint API (FastAPI / Express / Laravel) untuk melayani data ke frontend web dashboard.
3. **Frontend Web Dashboard**:
   - Mengembangkan halaman antarmuka web untuk Guru Mapel, Guru BK, dan Admin.
4. **Database Aplikasi**:
   - Merancang skema tabel untuk riwayat inferensi, pengguna, dan jurnal konseling.

---

## 5. Kontrak Data Output JSON (Spesifikasi Lengkap 24 Fitur)

### 5.1 Endpoint Tier 1: Guru Mata Pelajaran (`generate_subject_teacher_payload`)
Payload memuat seluruh 24 fitur yang digunakan model dalam klasifikasi:

```json
{
  "target_audience": "GURU_MATA_PELAJARAN",
  "id_siswa": 28400,
  "mata_pelajaran": {
    "kode_modul": "AAA",
    "nama_mapel": "Pemrograman Web dan Perangkat Bergerak",
    "kategori": "Kejuruan",
    "guru_pengampu": "I Wayan Sudarma, S.Kom.",
    "kkm": 75
  },
  "analisis_risiko": {
    "tingkat_risiko": "TINGGI",
    "probabilitas_risiko": 0.832,
    "threshold": 0.40
  },
  "metrik_24_fitur_model": {
    "engagement_dan_durasi": {
      "total_clicks": 142,
      "active_days": 8,
      "days_inactive": 16,
      "is_inactive_gt_5d": 1,
      "is_inactive_gt_14d": 1,
      "clicks_per_active_day": 17.75,
      "clicks_last_14d_ratio": 0.05,
      "unique_sites_accessed": 12,
      "course_duration_hours": 4.5
    },
    "kepatuhan_tugas": {
      "missing_assignments": 2,
      "late_submission_count": 1,
      "avg_submission_gap": 2.5,
      "procrastination_count": 1
    },
    "aktivitas_lesson": {
      "lesson_attempts_count": 0,
      "lesson_time_spent_min": 0.0,
      "lesson_completion_ratio": 0.0,
      "lesson_avg_score": 0.0
    },
    "akademik": {
      "avg_score": 38.0,
      "min_score": 30.0,
      "score_rel_to_module": -18.5,
      "failing_tasks_count": 2,
      "grade_trend": -15.0
    },
    "konteks_siswa": {
      "num_of_prev_attempts": 0,
      "studied_credits": 60
    }
  },
  "faktor_pemicu": [
    "2 tugas belum dikumpulkan.",
    "Belum mengakses aktivitas Lesson interaktif.",
    "Akumulasi waktu belajar rendah (4.5 jam)."
  ],
  "rekomendasi_tindakan": "Perlu perhatian langsung dan konfirmasi tugas di kelas."
}
```

### 5.2 Endpoint Tier 2: Guru Bimbingan Konseling (`generate_bk_student_summary`)
```json
{
  "target_audience": "GURU_BK",
  "id_siswa": 28400,
  "rekapitulasi_semester": {
    "total_mapel_diambil": 2,
    "total_mapel_berisiko": 2,
    "total_jam_belajar": 8.2,
    "total_tugas_belum_dikumpul": 4,
    "total_tugas_terlambat": 2,
    "inaktivitas_terlama_hari": 16
  },
  "profil_karakter_belajar": "Prokrastinasi Sistemik (Keterlambatan dan kelalaian tugas di banyak mapel).",
  "prioritas_konseling": "TINGGI",
  "rekomendasi_tindakan": "Jadwalkan konseling individual dan koordinasi dengan wali kelas.",
  "rincian_per_mata_pelajaran": [
    {
      "kode_modul": "AAA",
      "nama_mapel": "Pemrograman Web dan Perangkat Bergerak",
      "kategori": "Kejuruan",
      "guru_pengampu": "I Wayan Sudarma, S.Kom.",
      "probabilitas": 0.832,
      "status_risiko": "MERAH",
      "tugas_belum_dikumpul": 2,
      "durasi_belajar_jam": 4.5,
      "lesson_attempts": 0,
      "days_inactive": 16,
      "avg_score": 38.0
    },
    {
      "kode_modul": "BBB",
      "nama_mapel": "Basis Data",
      "kategori": "Kejuruan",
      "guru_pengampu": "Ni Made Rai Astuti, S.Pd.",
      "probabilitas": 0.764,
      "status_risiko": "MERAH",
      "tugas_belum_dikumpul": 2,
      "durasi_belajar_jam": 3.7,
      "lesson_attempts": 1,
      "days_inactive": 12,
      "avg_score": 42.0
    }
  ]
}
```
