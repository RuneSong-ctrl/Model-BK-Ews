# Panduan Demo Prototipe EWS 2-Tier (Seminar KP)

Alur yang didemokan sama dengan alur nyata:
**fitur siswa → model ML → JSON Tier 1/Tier 2 → API Laravel → dasbor guru → tindak lanjut**.
Data yang dipakai adalah dump database Moodle asli sekolah pelanggan (`../exp_kokar.sql`): 374 siswa kelas XII,
8 mapel. Di web, nama siswa dan guru ditampilkan anonim ("Siswa 001", "Guru <mapel>").
Model prototipe OULAD disimpan di `baseline_oulad/` sebagai pembanding.

| Berkas | Isi |
|---|---|
| `model_ews.ipynb` | Notebook utama: data asli → 24 fitur → evaluasi → model final, plus perbandingan dengan OULAD |
| `train_kokar.py` | Versi skrip dari notebook utama (melatih ulang dan mengekspor data untuk web) |
| `moodle_dump.py`, `moodle_features.py` | Membaca dump SQL Moodle dan menghitung 24 fitur + label |
| `ews_kokar_model.pkl`, `model_card_kokar.json` | Model utama dan metriknya |
| `run_pipeline.py`, `ews_inference.py` | Fitur → model → JSON Tier 1/Tier 2 → API web |
| `sandbox_app.py`, `test_model.py` | Sandbox dan uji unit (memakai model pembanding OULAD) |
| `baseline_oulad/` | Dataset, model, dan notebook prototipe OULAD |
| `data_kokar/` | Cache tabel Moodle dan fitur anonim (tidak di-commit, berisi data siswa) |

## Persiapan (sekali saja)

```bash
# di folder EWS_Moodle
pip install -r requirements.txt
python test_model.py                       # semua harus PASS
python train_kokar.py                      # latih ulang dari ../exp_kokar.sql (±3 menit)
python run_pipeline.py --export data_kokar/output_kokar.json

# di folder Web-BK
php artisan optimize:clear
php artisan migrate:fresh --seed           # mengisi dasbor dari data_kokar/output_kokar.json
npm run build
php artisan serve
```

Akun demo (kata sandi `password`): `admin@gmail.com`, `guru1@gmail.com` (Guru Mapel),
`bk@gmail.com` (Guru BK), `kepsek@gmail.com`.

## Skenario demo (±7 menit)

1. **Model dan metrik.** Buka `model_ews.ipynb`: uji prospektif (Skenario B) ROC-AUC 0,879, precision 0,948,
   recall 0,846; model OULAD tanpa latih ulang hanya 0,737. Threshold SEDANG 0,80, TINGGI 0,95.
2. **Sandbox (model dihitung langsung).** `python sandbox_app.py`, buka http://127.0.0.1:5000.
   Pilih "Siswa Teladan", lalu geser *Inaktivitas Terakhir* dan *Tugas Belum Dikumpul*: status berubah HIJAU → KUNING → MERAH beserta faktor pemicunya.
3. **Pipeline ke web.** Login sebagai Admin, tekan **Tarik & Sinkronkan Sekarang**.
   Laravel menjalankan `run_pipeline.py` (model dihitung ulang), lalu hasilnya disimpan. Riwayat ada di tab *Log Sinkronisasi*.
   Alternatif dari terminal (memakai token API):
   ```bash
   python run_pipeline.py --push http://127.0.0.1:8000 --token <EWS_API_TOKEN di Web-BK/.env>
   ```
4. **Tier 1, Guru Mapel.** Login `guru1@gmail.com`, pilih mapel (mis. Matematika XII), lihat siswa TINGGI/SEDANG dan beri tindakan *Remedial*.
5. **Tier 2, Guru BK.** Login `bk@gmail.com`: siswa diurutkan menurut prioritas konseling dengan profil belajar
   (Inaktivitas Menyeluruh, Kendala Belajar Multimapel, Kesulitan Spesifik Mapel). Klik *Catat Jurnal BK*.
6. **Notifikasi.** Tunjukkan uji coba bot Telegram (dikerjakan rekan tim).

Tindakan guru (status remedial, jurnal konseling) tidak terhapus saat sinkronisasi diulang.

## Keamanan endpoint

- `POST /api/ews/tier1/alerts` dan `/tier2/summaries` wajib header `X-EWS-Token` (nilai `EWS_API_TOKEN` di `.env`).
- Endpoint baca (`/api/ews/teacher/*`, `/bk/*`, `/kepsek/*`) wajib login sesuai peran.

## Jika ada kendala

| Gejala | Penyebab dan solusi |
|---|---|
| Sinkronisasi gagal "Pipeline model gagal dijalankan" | Atur `EWS_PYTHON` di `Web-BK/.env` ke python yang memiliki paket di `requirements.txt` |
| Perubahan rute tidak berlaku | `php artisan optimize:clear` |
| Seeder: "File keluaran model tidak ditemukan" | Jalankan `python train_kokar.py` lalu `python run_pipeline.py` di folder EWS_Moodle |
| `.pkl` gagal dibuka | Versi scikit-learn harus 1.9.0 (lihat `requirements.txt`) |
