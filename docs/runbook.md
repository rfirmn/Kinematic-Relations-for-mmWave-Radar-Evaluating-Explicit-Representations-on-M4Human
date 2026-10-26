# Menjalankan sistem

Semua perintah dijalankan dari root repo menggunakan PowerShell. Python lokal
berada di `.venv\Scripts\python.exe`. Alternatif sesudah setup adalah
`.venv\Scripts\uv.exe --cache-dir .cache/uv run --no-sync python -m relational64.cli ...`. Namespace
`eksperimen_model.relational64.cli` juga tersedia sesuai kontrak spesifikasi.

Generasi aktif `fresh_20261026_01` memakai output baru di dalam repo pada C.
`runtime.yaml` menetapkan `meta_root`, `derived_root`, `report_root`, dan
`checkpoint_root`. Tidak ada `metadata_root` eksternal atau dependency metadata
penelitian lama. [Kebijakan artefak](artifact_policy.md).

## Training diagnostik bertahap dengan data aktual

Runner berikut menjalankan training kecil memakai RPC M4Human aktual dan calon
target `params.joints[:22]` melalui transformasi kandidat yang diuji numerik.
Waktu memakai grid nominal 12 Hz yang masih provisional. Run ini menemukan
kegagalan implementasi sambil validasi ilmiah dilanjutkan; hasilnya tidak
meluluskan G0 dan tidak menjadi hasil akurasi penelitian yang telah disahkan.

Jalankan satu tahap sampai berhasil sebelum tahap berikutnya. Seluruh perintah
berikut memakai plan tetap: seed42, 8 window train dari8 subjek, 4 window val
dari4 subjek terpisah, 20 optimizer steps fisik dan2 steps per projector.
Varian diagnostik keempat memakai LR alignment1e-5, warmup2 langkah dan gain
sensor0.026. Gain ini menyimpang dari input H langsung dalam spesifikasi;
default arsitektur/resep ilmiah tidak berubah. [Temuan dan batas eksperimen](diagnostic_findings.md).
Pemilihan window memakai identitas sensor sebelum membuka target. Label test
tidak dibuka. Intensity hanya di-fit dari panel train run ini, sehingga
normalizer tersebut belum menggantikan statistik seluruh train ilmiah.

```powershell
$taskPilotArgs = @('--run-id','diagnostic_20261026_04_gain','--alignment-lr','0.00001','--alignment-warmup','2','--sensor-gain','0.026')
& .\.venv\Scripts\python.exe -m relational64.diagnostic_pilot @taskPilotArgs --stage prepare
& .\.venv\Scripts\python.exe -m relational64.diagnostic_pilot @taskPilotArgs --stage physical
& .\.venv\Scripts\python.exe -m relational64.diagnostic_pilot @taskPilotArgs --stage alignment
```

- `prepare`: periksa hash sumber/manifes, split dan mask; bentuk batch aktual,
  fit intensity panel train, dan bangun target/QA provisional.
- `physical`: encoder/pose dari awal, loss/backward, validasi train/val setiap
  update, serta simpan checkpoint dan optimizer setiap langkah.
- `alignment`: bekukan encoder/pose; ekstrak fitur prediksi, fit descriptor
  train-only; lakukan update M/S/F/E/R dengan pertanyaan train yang sama, lalu
  generate pada val. T, Maj dan Rule diperiksa pada panel yang sama. Qwen
  dibekukan dan hash parameternya harus tetap sama.

Output run ini berada pada subfolder `diagnostic_runs/diagnostic_20261026_04_gain/`
di bawah masing-masing `derived_root`, `report_root`, dan `checkpoint_root`.
Di report, baca `prepare.json`, `physical.json`, `alignment.json`, `summary.json`,
dan history langkah. Jika gagal, baca `failures.jsonl` beserta traceback dan
perbaiki penyebabnya. Generation yang terpotong/invalid menghasilkan status
`needs_revision` dan tidak diterima sebagai langkah yang lulus. Jalankan ulang
tahap yang gagal karena error eksekusi: checkpoint
`diagnostic.pt` terakhir dipulihkan; tahap dependen menolak input yang belum
lulus atau hashnya berubah. Jangan menjalankan dua proses penulis pada run ID
yang sama.

Budget dapat ditetapkan dengan `--steps` (1–100), `--alignment-steps` (1–10),
`--alignment-lr`, `--alignment-warmup`, `--sensor-gain` dan `--conditions M,S,F,E,R`;
berikan nilai yang sama pada semua perintah run.
Perubahan budget, panel, waktu, target atau konfigurasi membutuhkan run ID baru.
Revisi kode setelah tahap berhasil yang mempengaruhi hasil tahap tersebut juga
membutuhkan run ID baru atau pembangunan ulang tahap tersebut beserta seluruh
dependensinya; jangan mengubah JSON hasil secara manual. Jurnal kegagalan dan
hash kode setiap tahap dipertahankan.

Run1–3 disimpan lengkap dan ditandai `needs_revision` karena generation masih
tidak stabil. Run4 menguji gain embedding sebagai varian tersendiri dan mencapai
schema valid12/12 untuk setiap M/S/F/E/R pada panel val. Parser, token limit,
target dan waktu tidak diubah untuk memperbaiki angka tersebut. Keberhasilan
format bukan bukti akurasi/grounding dan belum meluluskan rancangan normatif.
Jika ingin menguji input H langsung lagi, gunakan run ID baru dan
`--sensor-gain 1`, serta budget yang dicatat.

Run diagnostik tidak menulis inventory/gate kanonis, target/QA kanonis, atau
checkpoint `best.pt`. Checkpoint berstatus `diagnostic` dan normalizer ditandai
`diagnostic_only`, sehingga tidak diterima sebagai artefak training ilmiah.
Metrik terhadap calon label tetap disimpan untuk debugging, tanpa klaim
kesetaraan SMPL-X, ketepatan waktu, signifikansi, atau generalisasi.

Setelah jalur ini stabil dan target/waktu disahkan, bangun artefak ilmiah dari
sumber dan jalankan tahapan berikut sesuai gate. Training penuh adalah run
terpisah; pada RTX3060 yang sama jadwalkan bergantian dengan alignment pilot.
Penulisan proposal dapat berjalan paralel.

## Pemeriksaan yang dapat dijalankan sekarang

```powershell
& .\.venv\Scripts\python.exe -m relational64.cli verify --suite all
& .\.venv\Scripts\python.exe tools/check_artifact_isolation.py
& .\.venv\Scripts\python.exe -m relational64.cli overfit --synthetic
& .\.venv\Scripts\python.exe -m relational64.cli profile --stage physical --synthetic
& .\.venv\Scripts\python.exe -m relational64.cli profile --stage alignment --condition R --synthetic
& .\.venv\Scripts\python.exe -m relational64.cli pilot --synthetic
& .\.venv\Scripts\python.exe -m relational64.cli report
```

Audit penuh membaca seluruh LMDB dan menghitung SHA-256; sumber tidak ditulis.
Inventory dan manifest generasi aktif sudah dibangun ulang. Perintah audit dan
prepare menolak tujuan yang sudah terisi. Untuk pembangunan ulang berikutnya,
tetapkan ID generasi baru yang sama pada runtime dan source_policy serta arahkan
seluruh root output ke folder generasi baru di repo, kemudian jalankan:

```powershell
& .\.venv\Scripts\python.exe -m relational64.cli audit --full
& .\.venv\Scripts\python.exe -m relational64.cli prepare --diagnostic
```

Destination target/cache/QA juga harus baru. Hasil diagnostik yang sudah tersedia
tidak perlu dibuat ulang untuk membaca laporannya. Pemeriksaan Qwen pretrained:

```powershell
& .\.venv\Scripts\python.exe -m relational64.verify_qwen --snapshot 'C:/Users/Rio Aslab/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306' --commit 989aa7980e4cf806f80c7fef2b1adb7bc71aa306 --head-mode canonical_fp32 --report docs/report_training/fresh_20261026_01/qwen_native_check.json
```

## Prasyarat data aktual

Lihat `data_authority.md`, `configs/data_authority.template.json`, dan
`configs/reference_target_provenance.template.json`. Diperlukan aset/regressor
SMPL-X yang cocok atau sertifikat equivalence numerik, provenance koordinat
RawPoints, serta mapping waktu/recording asli. Kode resmi dan seluruh berkas
lokal sudah diperiksa; artefak tersebut belum ditemukan. Menyalin flag
`audited:true` tidak menyelesaikan pemeriksaan ini.

Sebelum menandai G0 lulus, adapter waktu harus menggunakan mapping yang disahkan;
adapter sementara saat ini hanya merekonstruksi grid nominal dari indeks cache.
Resolver authority/timestamp untuk paket sumber final masih harus dipasangkan
dengan format mapping yang tersedia. Jalur training ilmiah tetap diblokir;
runner diagnostik di atas mencatat asumsi yang belum terverifikasi secara eksplisit.

Gerbang disimpan di `docs/report_training/fresh_20261026_01/acceptance.json`. Tiap gerbang lulus
memerlukan test IDs, jenis bukti aktual, hashes artefak, dan bukti Windows/CUDA
untuk G2/G4. Laporan saat ini sengaja belum meluluskan gerbang ilmiah.

## Tahapan sesudah authority dan gerbang diselesaikan

Contoh berikut adalah antarmuka implementasi yang telah diuji fixture. Ini belum
merupakan launcher yang disahkan untuk paket M4Human aktual saat ini.

```powershell
& .\.venv\Scripts\python.exe -m relational64.cli prepare
& .\.venv\Scripts\python.exe -m relational64.cli export-targets --target-method reference --target-provenance configs/reference_target_provenance.json
& .\.venv\Scripts\python.exe -m relational64.cli build-qa --target-provenance configs/reference_target_provenance.json
& .\.venv\Scripts\python.exe -m relational64.cli overfit
& .\.venv\Scripts\python.exe -m relational64.cli profile --stage physical
& .\.venv\Scripts\python.exe -m relational64.cli train-physical --seed 42
& .\.venv\Scripts\python.exe -m relational64.cli extract --split train,val --seed 42 --checkpoint checkpoints/m4human_relational64/fresh_20261026_01/physical/42/best.pt
& .\.venv\Scripts\python.exe -m relational64.cli fit-descriptors --condition R --seed 42
& .\.venv\Scripts\python.exe -m relational64.cli train-alignment --condition R --seed 42
& .\.venv\Scripts\python.exe -m relational64.cli train-probe --source Y --seed 42
```

S/F/E mengulangi fit dan alignment dengan condition masing-masing; M langsung
memakai Y tanpa normalizer descriptor. Semua mendapat stream QA seed/epoch sama.
`train-physical` dan `train-alignment` mendukung `--resume last.pt` pada batas epoch
yang telah selesai; optimizer/scheduler/scaler/RNG dan urutan sampler dipulihkan.
Belum ada penyimpanan periodik di tengah epoch; update setelah checkpoint terakhir
akan diulang. Training penuh dijalankan sebagai pekerjaan terpisah.

## Evaluasi dan test lock

`lock-test --plan <JSON>` memvalidasi checkpoint terpilih, seed, kondisi dan hashes
sebelum target/test cache dibuka. Test exports harus berada di direktori baru;
gunakan `--targets <direktori>` dan `--qa <qa_test.jsonl>` saat evaluasi. Resep target,
manifest dan label policy dipin sebelum lock; hash array target test dicatat
sesudah ekspor. Rincian ada pada `target_registry_contract.md`.

CLI menyediakan `evaluate`, `evaluate-physical`, `evaluate-controls`, `match-donors`,
`evaluate-donors`, dan `compare`. Semua evaluasi test memerlukan `--test-lock`.
`evaluate-controls --train-qa <qa_trainval.jsonl>` menghitung Maj hanya dari train;
Rule memakai prediksi pose tanpa mask GT. `compare --prediction-plan <JSON>`
menerima map kondisi ke path JSONL atau daftar path dengan urutan seed identik.
Donor menggunakan grid waktu teks bersama dan menyimpan hash token teks kedua sisi.
Dengan empat subjek test, paling banyak dua blok subjek terpisah tersedia, sehingga
donor dilaporkan deskriptif dan tidak mendapat CI inferensial lima blok.

## Batas implementasi yang masih terbuka

- Authority/mapping aktual, statistik intensity aktual, ekspor target dan QA
  aktual, serta penguncian ambang belum selesai.
- Profil alignment aktual dari cache train/val terpilih, rentang prompt terpanjang,
  cold/warm I/O dan telemetry kontinu belum menjadi gerbang penerimaan.
- Matriks training, pemilihan checkpoint, evaluasi test, dan hasil ilmiah belum ada.

Fixture sintetis dan pemeriksaan pretrained Qwen tidak menyelesaikan butir tersebut.
