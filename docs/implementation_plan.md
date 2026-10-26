# Rencana implementasi

1. P0: salinan spesifikasi byte-identik, provenance, resolver dan kontrak runtime.
2. P1: inventaris LMDB read-only, decoder ketat, recording/subject dan RPC audit.
3. P2: otoritas joint/unit/transform/time, bukti numerik/visual, target dan QA.
4. P3: split SHA subjek 12/4/4, window64, normalizer train-only dan coverage.
5. P4: encoder256 + pose, loss fisik, backward, overfit8, profil Windows/CUDA.
6. P5: cache prediksi, descriptor, M/S/F/E/R, frozen Qwen, splice dan alignment.
7. P6: evaluasi fisik/QA/donor, test lock, checkpoint/resume dan laporan otomatis.

Setiap tahap menyimpan laporan di `docs/report_training/`; data turunan memakai
`derived_data/m4human_relational64/`. Tahap yang bergantung pada authority
koordinat/waktu tidak boleh dijalankan dengan authority yang belum terbukti.
Implementasi dan fixture dapat dikembangkan secara independen dari gerbang data.

## Asal keputusan

| Kategori | Isi | Status |
|---|---|---|
| Fondasi mmMind menurut spesifikasi | Encoder256, spatial3/temporal8, heads4, FFN1024 | Diimplementasikan menurut definisi lokal, bukan reproduksi paper |
| Adaptasi M4Human | XYZ+intensity, joint22, pelvis0, nominal12Hz, frame64 | Format/unit/waktu harus diaudit dari paket aktual |
| Pilihan eksperimen | N150, signed-log intensity, ambang label, loss, fusion64, frozen Qwen1.5B | Default normatif; belum dianggap optimal |
| Hasil ilmiah | R−S pada S_atomic, donor, tiga seed | Belum tersedia; training penuh terpisah |

Normalisasi intensity serta coverage N150/ambang label ditinjau memakai train/val
sebelum dikunci. Perubahan scientific fields membuat config hash/run ID baru.
