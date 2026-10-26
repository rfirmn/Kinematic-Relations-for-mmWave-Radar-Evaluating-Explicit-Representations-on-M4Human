# Kesiapan pilot dan eksperimen utama

Status 26 Oktober 2026, generasi `fresh_20261026_01`.

**Pilot diagnostik end-to-end sudah dijalankan sepenuhnya dengan data aktual
pada Windows/CUDA RTX 3060.** Run `diagnostic_20261026_05_complete` beserta review
`validation12_v1` menyelesaikan seluruh jalur komputasi dan 21 pemeriksaan teknis.
[Hasil dan tindak lanjut](pilot_results.md).

| Kegiatan | Bukti/status |
|---|---|
| Audit sumber aktual | 661.132 frame; sumber read-only dan hash tetap |
| Pengujian software | 218 tests lulus; fixture sintetis pada Windows/CPU |
| Training fisik diagnostik | 100 update; loss train 4,855422 → 1,773253; turun 63,48% |
| Alignment diagnostik | M/S/F/E/R masing-masing 10 update, Qwen tetap beku |
| Evaluasi QA aktual | 52 window validation; 156 QA per kondisi; seluruh schema valid |
| Kontrol | T/Maj/Rule, probe Y/Z_R, 13 pasangan donor aktual |
| Cache/checkpoint/resource | Pemeriksaan lulus pada perangkat sasaran |
| Target/waktu ilmiah | Masih provisional; G0 tetap false |
| Pilot ilmiah yang disahkan | Belum terpenuhi karena authority dan kualitas bukti |
| Training penuh/test ilmiah | Belum dijalankan; pekerjaan terpisah |

## Makna kelengkapan pilot

`execution_complete=true` berarti tahap persiapan, training fisik, alignment,
dan evaluasi benar-benar dieksekusi. Review `status=passed` berarti pemeriksaan
teknis yang ditetapkan berhasil, termasuk loss/backward, pembekuan, mask melalui
batch, splicing, parsing, cache, checkpoint, donor dan resource.

Kelengkapan eksekusi tidak mensyaratkan hipotesis R lebih baik dari S terbukti.
Pilot ini justru mengungkap masalah: global MPJPE validation sekitar satu meter,
prediksi pose terkompresi, R belum mengungguli S atau mayoritas, dan kedua sisi
donor tidak pernah sekaligus benar (0/13). Tugas pelvis belum memiliki contoh
`speeding_up`/`slowing_down` yang reference-defined pada panel ini. Temuan tersebut
menjadi pekerjaan berikutnya dan tidak disembunyikan oleh validitas JSON 100%.

## Status validitas

Target menggunakan first22 direct joints dengan transformasi kandidat. Waktu
memakai grid nominal 12 Hz. Gain sensor 0,026 adalah deviasi diagnostik dari
spesifikasi input H langsung. Panel train hanya delapan window, sehingga
normalizer panel ini tidak menjadi statistik train ilmiah penuh.

Belum ada bukti yang mengesahkan kesetaraan target terhadap aset SMPL-X/regressor
otoritatif, seluruh provenance sumbu/preprocessing RPC, atau pemetaan timestamp
dan recording sumber. Rincian wajib ada di [data authority](data_authority.md).
Seluruh checkpoint tetap diagnostic; label test belum dibuka dan G0–G6 tidak
diluluskan oleh pilot diagnostik ini.

## Langkah berikutnya

1. Baca [hasil pilot](pilot_results.md), khususnya overlay pose, coverage kelas,
   dan PairBothCorrect. Tidak perlu mengulang run lama untuk membaca hasil.
2. Lengkapi authority target/waktu dan diagnosis estimasi pose; tinjau label
   train/validation sebelum mengunci ambang atau memperluas training.
3. Uji gain/alignment secara terkontrol pada checkpoint fisik yang sama. Pilih
   konfigurasi berdasarkan evaluasi yang mencakup correctness dan grounding.
4. Setelah hasil ditinjau dan gate ilmiah terpenuhi, jalankan training penuh
   sebagai run terpisah. Pada RTX 3060 yang sama gunakan satu proses GPU;
   training alignment mencapai peak reserved sekitar 9,32 GiB dari 12 GiB.

Untuk reproduksi dari awal, ikuti [runbook pilot lengkap](runbook.md#pilot-diagnostik-end-to-end)
dengan run ID baru. Metadata/tensor, laporan dan checkpoint tetap disimpan di
dalam repo pada C; [indeks artefak](../ARTEFAK.md) menyediakan lokasinya.

Proposal dapat menuliskan kelayakan eksekusi aktual beserta temuan dan batas ini.
Belum ada hasil test ilmiah atau dukungan terhadap hipotesis R−S yang boleh
disimpulkan dari pilot tersebut.
