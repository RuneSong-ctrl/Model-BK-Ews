# Panduan Demo Prototipe EWS 2-Tier (Seminar KP)

Alur yang didemokan sama dengan alur nyata:
**fitur siswa → model ML → JSON Tier 1/Tier 2 → API Laravel → dasbor guru → tindak lanjut**.
Pada prototipe, data Moodle STIKMAS diganti data simulasi `simulasi/kelas_stikmas_hari60.csv`
(36 siswa fiktif, 3 mapel per siswa, fitur diambil dari data uji OULAD kohor 2014).

## Persiapan (sekali saja)

```bash
# di folder EWS_Moodle
pip install -r requirements.txt
python test_model.py                       # semua harus PASS
python build_simulation.py                 # opsional, membuat ulang data simulasi (±1 menit)
python run_pipeline.py --export sample_output.json

# di folder Web-BK
php artisan optimize:clear
php artisan migrate:fresh --seed           # mengisi dasbor dari sample_output.json
npm run build
php artisan serve
```

Akun demo (kata sandi `password`): `admin@gmail.com`, `guru1@gmail.com` (Guru Mapel),
`bk@gmail.com` (Guru BK), `kepsek@gmail.com`.

## Skenario demo (±7 menit)

1. **Model dan metrik.** Buka `model_card.json` atau notebook: ROC-AUC 0,872, recall berisiko 0,821, threshold 0,40.
2. **Sandbox (model dihitung langsung).** `python sandbox_app.py`, buka http://127.0.0.1:5000.
   Pilih "Siswa Teladan", lalu geser *Inaktivitas Terakhir* dan *Tugas Belum Dikumpul*: status berubah HIJAU → KUNING → MERAH beserta faktor pemicunya.
3. **Pipeline ke web.** Login sebagai Admin, tekan **Tarik & Sinkronkan Sekarang**.
   Laravel menjalankan `run_pipeline.py` (model dihitung ulang), lalu hasilnya disimpan. Riwayat ada di tab *Log Sinkronisasi*.
   Alternatif dari terminal (memakai token API):
   ```bash
   python run_pipeline.py --push http://127.0.0.1:8000 --token <EWS_API_TOKEN di Web-BK/.env>
   ```
4. **Tier 1, Guru Mapel.** Login `guru1@gmail.com`, pilih mapel (mis. Basis Data), lihat siswa TINGGI/SEDANG dan beri tindakan *Remedial*.
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
| Seeder: "File keluaran model tidak ditemukan" | Jalankan `python run_pipeline.py --export sample_output.json` di folder EWS_Moodle |
| `.pkl` gagal dibuka | Versi scikit-learn harus 1.9.0 (lihat `requirements.txt`) |
