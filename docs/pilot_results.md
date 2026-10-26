# Hasil pilot diagnostik end-to-end

Tanggal: 26 Oktober 2026. Run `diagnostic_20261026_05_complete`, review `validation12_v1`.

**Seluruh jalur pilot diagnostik telah dieksekusi dan pemeriksaan teknis lulus.** Data menggunakan RPC M4Human aktual pada Windows/CUDA RTX 3060. Target direct joints dan waktu nominal 12 Hz tetap provisional; `scientific_acceptance=false` dan G0 tetap false.

## Cakupan eksekusi

- 8 window train dari 8 subjek dan 52 window validation dari 4 subjek terpisah. Panel validation mencakup 12 aktivitas bersama ditambah 4 window awal; pemilihan memakai identitas sensor sebelum target dibaca.
- 100 update encoder/pose, 10 update untuk masing-masing projector M/S/F/E/R, evaluasi kontrol T/Maj/Rule, serta 50 update untuk masing-masing probe Y dan Z_R.
- 156 QA validation per kondisi, 13 pasangan donor aktual dalam 2 blok subjek, cache/forward equivalence, pembekuan Qwen dan projector pada evaluasi, checkpoint reload, loss/backward, resource dan integritas sumber.
- Normalisasi intensity/descriptor hanya berasal dari panel train. Target test tidak dibuka.
- Gain sensor 0,026 merupakan deviasi diagnostik yang dicatat; konfigurasi normatif H langsung tidak diganti.

## Training fisik

| Panel | L_phys awal | L_phys akhir |
|---|---:|---:|
| train | 4.855422 | 1.773253 |
| val | 4.936769 | 2.241553 |

Penurunan loss train: 63.48%. Ini meluluskan pemeriksaan penurunan 50% untuk pilot diagnostik, tanpa meluluskan G2 ilmiah dengan target yang belum disahkan.

| Metrik terhadap calon target | Train | Validation | Satuan |
|---|---:|---:|---|
| global_mpjpe | 0.899750 | 0.997347 | m |
| root_position_error | 0.986705 | 1.013333 | m |
| relative_mpjpe | 0.515823 | 0.550718 | m |
| absolute_velocity_error | 0.199611 | 0.389602 | m/s |
| relative_velocity_error | 0.209220 | 0.331784 | m/s |
| root_speed_mae | 0.028417 | 0.185658 | m/s |
| bone_length_mae | 0.014686 | 0.015386 | m |
| zero_velocity_error | 0.319607 | 0.351248 | m/s |

Agregasi menggunakan numerator/count seluruh window-endpoint-joint valid. Global MPJPE validation sekitar satu meter; visualisasi menunjukkan prediksi pose masih terkompresi dan meleset dari calon skeleton. Penurunan loss tidak membuktikan kualitas pose telah memadai.

![RPC aktual, calon target, dan prediksi pose](report_training/fresh_20261026_01/diagnostic_runs/diagnostic_20261026_05_complete/reviews/validation12_v1/pose_overlay.png)

## QA terhadap calon target

Semua skor berikut adalah diagnosis pada validation dengan target provisional, bukan hasil test ilmiah. Macro-F1 mempertahankan tiga kelas per tugas, termasuk kelas yang tidak hadir.

| Kondisi | S_atomic (%) | Accuracy weighted (%) | Schema valid (%) |
|---|---:|---:|---:|
| M | 23.29 | 61.73 | 100 |
| S | 24.52 | 65.31 | 100 |
| F | 24.20 | 64.80 | 100 |
| E | 23.96 | 63.78 | 100 |
| R | 24.38 | 65.31 | 100 |
| T | 0.00 | 0.00 | 100 |
| Maj | 27.52 | 74.49 | 100 |
| Rule | 26.78 | 70.41 | 100 |

R belum mengungguli S atau baseline mayoritas pada panel ini. Budget/panel sangat kecil dan target masih provisional; angka ini tidak menguji signifikansi atau membuktikan hipotesis penelitian.

Coverage pelvis: 35 `no_overall_trend`, 0 `speeding_up`, 0 `slowing_down`, dan 17 `unknown`. Lengan/kaki memiliki ketiga kelas substantive. Ceiling macro-F1 pelvis pada panel ini adalah 1/3; cakupan label harus ditinjau pada train/validation sebelum protokol final dikunci. Ambang label tidak diubah untuk memperbaiki hasil.

## Kontrol donor dan probe

Seluruh M/S/F/E/R/T memiliki PairBothCorrect **0/13**, dengan format kedua jawaban valid 13/13. Hash token dan embedding teks identik pada setiap pasangan; embedding sensor berubah pada kondisi sensor. Jadi jawaban yang valid belum menunjukkan grounding yang berhasil. Dua blok subjek hanya mendukung laporan deskriptif; donor pelvis tidak tercakup karena kelas acuan yang berbeda tidak tersedia.

| Probe | Update | S_atomic provisional (%) |
|---|---:|---:|
| Y | 50 | 27.70 |
| Z_R | 50 | 27.97 |

Checkpoint kedua probe dimuat ulang dan diperiksa; semua checkpoint tetap berstatus diagnostic.

## Kelayakan eksekusi

- Semua 21 pemeriksaan teknis pada review lulus. Cache dan forward langsung memiliki galat maksimum 0.0.
- Peak reserved CUDA training alignment: 9.32 GiB dari GPU 12 GiB. Resource guard fisik, alignment dan review lulus.
- Hash sumber dataset tidak berubah; parameter Qwen tetap sama. Tidak ada bukti untuk menjalankan dua proses training bersamaan pada GPU tersebut.
- 218 pengujian software lulus pada Windows/CPU dengan fixture sintetis. Bukti data aktual/CUDA berasal dari run diagnostik dan review terpisah ini.

## Tindak lanjut sebelum eksperimen utama

1. Sahkan target joint, koordinat RPC dan pemetaan waktu/recording menggunakan bukti pada [data authority](data_authority.md).
2. Diagnosis komponen loss dan prediksi pose yang terkompresi sebelum menjadikan checkpoint fisik sebagai sumber bukti QA. Gunakan train/validation dan simpan konfigurasi/hasil baru; hasil ini tidak memilih checkpoint test.
3. Tinjau distribusi label dan kelayakan tugas pelvis pada train/validation. Cakupan yang kosong harus dijelaskan sebelum menetapkan ambang/panel final.
4. Uji kestabilan alignment dan gain secara terkontrol dengan checkpoint fisik yang sama. Keberhasilan format dan PairBothCorrect menjadi pemeriksaan yang terpisah.
5. Setelah authority, pilot dan konfigurasi ditinjau, jalankan training penuh sebagai run terpisah. Proposal sudah dapat memakai temuan ini sebagai bukti kelayakan eksekusi beserta keterbatasannya.

## Bukti dan reproduksi

- [Ringkasan review lengkap](report_training/fresh_20261026_01/diagnostic_runs/diagnostic_20261026_05_complete/reviews/validation12_v1/summary.json)
- [Laporan pemeriksaan otomatis](report_training/fresh_20261026_01/diagnostic_runs/diagnostic_20261026_05_complete/reviews/validation12_v1/report.md)
- [Metrik fisik](report_training/fresh_20261026_01/diagnostic_runs/diagnostic_20261026_05_complete/reviews/validation12_v1/physical_metrics.json)
- [Matching donor](report_training/fresh_20261026_01/diagnostic_runs/diagnostic_20261026_05_complete/reviews/validation12_v1/donor_matching.json)
- [Training dan kurva loss](report_training/fresh_20261026_01/diagnostic_runs/diagnostic_20261026_05_complete/report.md)
- [Runbook langkah demi langkah](runbook.md#pilot-diagnostik-end-to-end)
- [Lokasi metadata, laporan, dan checkpoint](../ARTEFAK.md)
