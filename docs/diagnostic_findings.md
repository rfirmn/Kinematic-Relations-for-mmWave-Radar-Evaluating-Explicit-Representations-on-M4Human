# Temuan training diagnostik bertahap

Semua run berikut memakai RPC M4Human aktual pada Windows/CUDA RTX3060,
8 window train dari8 subjek dan4 window validation dari4 subjek terpisah.
Masing-masing menjalankan20 update fisik dari awal. Tidak ada label test dibuka.
Target first22 direct joints, transformasi kandidat dan waktu nominal12Hz
masih provisional. Ini bukan hasil akurasi ilmiah yang telah disahkan.

| Run | LR alignment | Warmup | Update per kondisi | Gain sensor | Temuan |
|---|---|---|---|---|---|
| diagnostic_20261026_01 | 3e-4 | 0 | 2 | 1 | M/S/F: generation terpotong; needs_revision |
| diagnostic_20261026_02 | 1e-5 | 2 | 2 | 1 | S dan sebagian R masih gagal schema; needs_revision |
| diagnostic_20261026_03 | 1e-6 | 5 | 10 | 1 | Penambahan update/LR kecil belum menstabilkan seluruh kondisi; needs_revision |
| diagnostic_20261026_04_gain | 1e-5 | 2 | 2 | 0.026 | Seluruh M/S/F/E/R:12/12 generation val valid; varian diagnostik |

Hasil run1–3 disimpan lengkap. Status eksekusi numerik dan keberhasilan format
dibedakan; optimizer yang berhasil berjalan tidak otomatis meluluskan generation.
`quality_review.json` menyimpan kriteria dan kegagalan setiap run. Parser strict,
token limit32 dan pertanyaan tidak diubah untuk memperbaiki hasil.

Probe pada checkpoint run3 menunjukkan prefix teacher/generation identik dan
hasil mask bool/integer sama pada contoh yang diperiksa. Argmax native head dan
FP32 head juga sama pada contoh tersebut. RMS H setelah LayerNorm sekitar1,
sedangkan RMS embedding token Qwen pada1000 token pertama sekitar0.02622.
Probe inference dengan gain0.02622/0.1 membuat contoh root S/F yang gagal kembali
menghasilkan JSON valid. Ini bukti terbatas untuk menguji skala, bukan pembuktian
penyebab tunggal atau evaluasi seluruh distribusi data.

## Status varian gain

Dokumen02 spesifikasi menetapkan LayerNorm weight1/bias0 dan input H langsung
ke Qwen. Karena itu, pengali0.026 sesudah projector pada run4 merupakan
**penyimpangan eksperimen yang eksplisit**, dicatat dalam plan sebagai
`experimental_spec_deviation`. Default `SensorProjector`, parameter count,
fusion64 dan berkas spesifikasi tidak diubah. Jumlah sensor embedding tetap64;
gain tidak menambah parameter dan gradien ke projector tetap mengalir.

Varian ini belum dipilih sebagai konfigurasi penelitian utama. Kesetaraan
hiperparameter, konfigurasi final dan studi gain1 versus kalibrasi memerlukan
protokol tersendiri sebelum full training. Parameter Qwen tetap beku pada semua
run. Format yang valid belum membuktikan penggunaan radar: kontrol T, Maj dan
Rule tetap dilaporkan dan dibutuhkan pemeriksaan grounding/donor lebih lanjut.

## Batas resep diagnostik

Panel dan budget sangat kecil. Akumulasi fisik8, alignment3 pertanyaan (satu per
task), fit intensity/descriptor hanya panel train, dan optimizer diagnostik
alignment memakai decay default pada seluruh parameter projector. Ini berbeda
dari resep ilmiah batch efektif16, sampling task0.5/0.25/0.25, decay hanya linear,
scheduler dan training penuh yang tersedia pada entry point ilmiah.

Seed dan panel sensor sama, tetapi model fisik di-fit ulang untuk setiap run;
bitwise determinisme GPU tidak diklaim. Maka perbedaan hasil antar-run ini
merupakan eksplorasi engineering, bukan ablation kausal satu faktor.

G0 tetap false. Sebelum hasil penelitian utama: selesaikan otoritas target,
koordinat dan mapping waktu; fit artefak kanonis train-only; tinjau kestabilan
alignment dan pilihan gain secara eksplisit; lakukan overfit/profil yang
disyaratkan; baru jalankan training ilmiah dan test lock. Training penuh tidak
dimulai oleh run diagnostik ini.

## Bukti

- [Runbook langkah demi langkah](runbook.md)
- [Laporan varian run4](report_training/fresh_20261026_01/diagnostic_runs/diagnostic_20261026_04_gain/report.md)
- [Probe prefix dan gain](report_training/fresh_20261026_01/diagnostic_runs/diagnostic_20261026_03/prefix_diagnosis.json)
- [Plan run4](report_training/fresh_20261026_01/diagnostic_runs/diagnostic_20261026_04_gain/plan.json)
- [Temuan run1](report_training/fresh_20261026_01/diagnostic_runs/diagnostic_20261026_01/quality_review.json)
