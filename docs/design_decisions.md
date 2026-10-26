# Keputusan dan status penguncian

Konstanta rancangan mengikuti `planning_relasi_kinematik/spesifikasi.yaml`.
Nilai tersebut adalah default eksperimen, bukan hasil pencarian konfigurasi.
Belum ada keputusan ilmiah yang dikunci menggunakan label M4Human aktual.

| Pilihan | Dasar | Pemeriksaan aktual | Status |
|---|---|---|---|
| 64 frame | Skala observasi dalam rancangan | 63 interval, provisional 12 Hz = 5,25 s; gap indeks memutus segmen | Waktu sumber belum disahkan |
| Maksimum 150 titik | Anggaran encoder dan kesetaraan dukungan sensor semua kondisi | 99,993% frame train dan 99,998% val memiliki >150 titik; rata-rata fraksi yang dipertahankan 32,56%/31,60% | Kandidat; kualitas pose perlu diuji train/val |
| Signed log intensity | Mengurangi rentang energi, tetap mendukung nilai bertanda | Kanal aktual finite; transform/padding dan fit train-only diuji fixture | Statistik aktual belum di-fit karena G0 |
| Ambang 0,05 dan guard 0,01 | Definisi operasional tren/selisih kecepatan | Boundary, common support dan sensitivitas diuji fixture | Coverage aktual/near-boundary trajectories belum tersedia |
| Minimum 44 dari 58 endpoint | Cakupan minimum 75% setelah warm-up6 | Mask per joint dan turunan tujuh titik diuji | Coverage anotasi aktual belum diekspor |
| Fusion64, S/F/E/R berkapasitas sama | Endpoint primer mengisolasi bentuk penyajian state | 3.003.392 parameter setiap kondisi; M 2.761.728 | Kontrak perangkat lunak lulus |
| LM head `canonical_fp32` | Full/selected CE harus ekuivalen | Native BF16/FP16 head gagal toleransi gradien; FP32 dengan GEMM satu baris memberi error loss/gradien nol | Pilihan numerik eksplisit; biaya diprofilkan |

Profil fisik dan alignment memakai input sintetis. Profil bukan bukti bahwa
sampling150 mempertahankan informasi gerak, bahwa label acuan benar, atau bahwa
model menjawab sesuai sensor aktual. Audit label menyediakan sensitivitas
train/validation; tidak membaca test untuk memilih ambang.

Setelah G0 diselesaikan, tinjau distribusi/coverage label, lintasan dekat boundary,
perubahan intensity setelah normalisasi dan hasil pose train/val. Bila mengubah
N, transform, ambang, atau field ilmiah lain, buat spesifikasi/run ID baru dan
ulang kondisi pembanding secara setara. Jangan mengubah satu kondisi saja.
