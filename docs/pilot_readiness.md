# Kesiapan pilot dan training penuh

Status diperiksa dari generasi `fresh_20261026_01` pada 26 Oktober 2026.
Dokumen ini merencanakan eksperimen; tidak meluluskan gate atau mengubah
konfigurasi penelitian.

| Kegiatan | Status saat ini |
|---|---|
| Penulisan proposal: rancangan, implementasi, audit, kelayakan komputasi awal | Dapat berjalan sekarang |
| Pengujian software dengan sensor sintetis | 210 tests lulus, overfit, profil CUDA, pilot M/S/R |
| Training diagnostik dengan RPC aktual dan calon target/waktu provisional | Empat run dijalankan; run1–3 perlu revisi, varian gain run4 lulus pemeriksaan format |
| Pilot supervised M4Human aktual | Belum siap: G0–G4 belum lulus |
| Training penuh dan evaluasi ilmiah | Belum dimulai; menunggu validasi data dan pilot aktual |

## Bukti yang sudah tersedia untuk proposal

- Audit aktual membaca 661.132 frame dari 20 subjek. Semua payload sensor
  berbentuk `[N,4]` float32 dengan kanal XYZ+intensity; intensity bukan Doppler.
- Pengujian software berjumlah 210 dan lulus. Fixture tidak membuktikan
  kesetaraan target, sinkronisasi waktu atau akurasi M4Human.
- Native Qwen, backward ke embedding sensor, parameter beku dan token splicing
  diperiksa pada Windows/CUDA RTX 3060. Cabang relasional masuk ke Qwen melalui
  64 embedding sensor.
- Overfit dan profil memakai sensor sintetis. Pilot integrasi CUDA yang sudah
  dijalankan mencakup M/S/R, satu update per kondisi. Ini belum merupakan training
  dasar pada dataset aktual atau evaluasi seluruh kondisi M/S/F/E/R/T.

Contoh paragraf penulisan:

> Implementasi awal telah melalui 210 pengujian perangkat lunak serta pengujian
> integrasi pada Windows/CUDA dengan GPU NVIDIA RTX 3060. Audit pembacaan dataset
> lokal mencakup 661.132 frame radar. Uji overfit, profil memori dan integrasi
> encoder–projector–model bahasa menggunakan sensor sintetis dan digunakan
> sebagai bukti kelayakan implementasi. Hasil ini belum merupakan pengukuran
> akurasi penelitian pada M4Human. Eksperimen supervised aktual akan dijalankan
> setelah provenance target, koordinat dan waktu tervalidasi.

Tambahan bukti engineering aktual: training kecil pada8 window train dan4
window validation telah dijalankan pada CUDA, dengan20 update fisik per run.
Tiga run input H langsung masih memiliki masalah generation. Varian diagnostik
gain0.026 menghasilkan format JSON valid untuk60/60 jawaban M/S/F/E/R pada val,
setelah2 update per kondisi. Ini bukan60 jawaban yang terbukti benar. Gain
merupakan deviasi eksperimen dari spesifikasi normatif; lihat
[temuan training](diagnostic_findings.md) sebelum memakai paragraf ini untuk
menjelaskan hasil awal.

Jangan menuliskan bahwa sistem telah berhasil mengenali gerak M4Human,
meningkatkan akurasi, atau memenuhi hipotesis R dibanding S berdasarkan bukti
software tersebut.

## Pekerjaan sebelum pilot aktual

1. Lengkapi dan periksa numerik aset SMPL-X/regressor yang cocok, urutan 22
   joint/pelvis, unit, dan transformasi radar. Lengkapi provenance RawPoints serta
   pemetaan frame/timestamp/recording asli. Jelaskan keterbatasan paket 993
   kelompok subjek/aksi dibanding 999 sequence pada paper. Detail bukti wajib
   tersedia dalam [data_authority.md](data_authority.md).
2. Pasangkan adapter waktu dan resolver authority dengan format bukti sumber
   yang tersedia. Adapter sekarang memakai indeks cache dengan waktu nominal
   provisional; mengganti flag G0 saja tidak menyelesaikan adapter ini.
3. Bangun ulang artefak training yang sah dari sumber: manifest dengan waktu
   tervalidasi, normalizer intensity train-only, target kanonis, QA dan audit
   label. Tinjau N150, normalisasi dan ambang menggunakan train/validation.
4. Runner `python -m relational64.diagnostic_pilot` tersedia dengan batas langkah,
   checkpoint per update dan jurnal kegagalan. Ini dapat menjalankan eksperimen
   engineering aktual sebelum G0 lulus, dengan target/waktu provisional dan
   scientific_acceptance=false. Sesudah authority lengkap, jalur pilot ilmiah
   tetap perlu memakai target/waktu yang telah disahkan. CLI `pilot` masih khusus
   `--synthetic`; dua entry point tersebut memiliki kontrak berbeda.

## Urutan pilot aktual yang direncanakan

Pilot memakai seed 42 dan hanya train/validation dari split subjek yang sudah
ditentukan. Pemilihan panel mengikuti identitas sensor, recording dan waktu;
anotasi yang hilang tetap dicatat sebagai mask. Test ilmiah belum dibuka.

1. Overfit delapan window aktual, lalu profil fisik pada Windows/CUDA sasaran.
2. Training fisik terbatas dan validasi pose/kecepatan. Simpan loss, mask,
   gradient, metrik fisik, status sumber target serta checkpoint pilot.
3. Ekstrak cache prediksi pilot dan fit descriptor hanya pada train. Periksa
   kesamaan cache dengan forward langsung dan pembekuan encoder/pose.
4. Profil alignment aktual, termasuk rentang panjang prompt dan I/O. Jalankan
   alignment terbatas untuk M/S/F/E/R serta kontrol T, Maj dan Rule sesuai
   kontrak masing-masing. Periksa loss, sensor splicing, parsing, QA metrics dan
   kontrol donor pada validation yang memenuhi syarat.
5. Tinjau kegagalan, konsumsi memori, lineage artefak dan pilihan eksperimen;
   tetapkan konfigurasi final sebelum training penuh tiga seed. Keberhasilan
   pilot membuktikan kelayakan eksekusi dan data, bukan otomatis hipotesis riset.

Budget pilot, panel dan ambang penerimaannya harus dicatat sebelum eksekusi;
dokumen ini tidak mengklaim budget tersebut sudah diterapkan atau dikunci.
Checkpoint pilot belum menjadi checkpoint terpilih untuk test ilmiah.

## Paralelisme dengan penulisan dan training penuh

Penulisan proposal dapat berjalan paralel dengan audit dan pilot. Bagian metode,
arsitektur, rancangan evaluasi dan bukti software sudah dapat disusun sekarang.
Hasil ilmiah aktual tetap ditulis setelah eksperimennya dijalankan.

Training penuh bergantung pada data dan hasil pilot yang sudah ditinjau. Menjalankan
keduanya sebelum masalah dasar selesai dapat mengharuskan seluruh training
diulang jika target atau waktu berubah.

Pada profil alignment sintetis R, peak allocated adalah 9.330.605.056 byte dan
peak reserved 9.946.791.936 byte, sekitar 9,26 GiB dari GPU 12 GiB. Bukti ini hanya
mendukung satu proses alignment yang diprofilkan; dua training pada GPU yang
sama belum diprofilkan. Jadwalkan pilot dan training penuh bergantian pada RTX
3060 tersebut. Komputasi paralel di perangkat lain memerlukan profil perangkat
dan konfigurasi yang sama, serta artefak run terpisah.

## Lokasi bukti

- [Laporan status](report_training/fresh_20261026_01/implementation_status.md)
- [201 pengujian](report_training/fresh_20261026_01/verify_all.json)
- [Profil alignment sintetis](report_training/fresh_20261026_01/synthetic_alignment_profile/profile_alignment_R.json)
- [Pilot sintetis](report_training/fresh_20261026_01/synthetic_pilot.json)
- [Indeks artefak](../ARTEFAK.md)
