# Artefak penelitian dibangun baru

Generasi aktif: **fresh_20261026_01**. Seluruh output penelitian disimpan pada
drive C di dalam repo ini. Pipeline tidak membaca metadata, split, target, QA,
normalizer, cache fitur atau checkpoint penelitian dari folder lama.

Input yang boleh dipakai bersama adalah dataset sumber immutable, paket
spesifikasi, kode sumber resmi sebagai referensi, pustaka perangkat lunak yang
terpasang, serta bobot pretrained Qwen pada commit yang dipin. Semuanya berbeda
dari checkpoint fisik/projector atau metadata yang dihasilkan penelitian.

| Isi | Lokasi dalam repo |
|---|---|
| Inventaris, schema, split audit, koordinat/waktu, provenance, registry | `derived_data/m4human_relational64/fresh_20261026_01/meta/` |
| Manifest dan seluruh data turunan baru | `derived_data/m4human_relational64/fresh_20261026_01/` |
| Pengujian, profil, log, laporan baru | `docs/report_training/fresh_20261026_01/` |
| Checkpoint penelitian yang kelak sah | `checkpoints/m4human_relational64/fresh_20261026_01/` |

Artefak terdahulu dipindahkan ke `archive/pre_fresh_20261026_01/` dan ditandai
tidak boleh dipakai untuk penerimaan penelitian. Arsip tidak menjadi dependency
runtime dan tidak dimuat otomatis. Tidak ada penghapusan dataset sumber.

Validator menolak path output/dependency artefak di luar repo atau di dalam
arsip, inventory berkontrak metadata lama, dan ID generasi berbeda pada
inventory, cache, normalizer, target, provenance bahasa dan checkpoint. Identitas
window mencakup generasi, sehingga cache temporal tidak dapat tertukar hanya
karena source frame-nya sama. Hash dan policy disimpan dalam registry baru.

Split serta statistik dapat memiliki nilai sama setelah dihitung ulang karena
dataset, seed dan algoritmenya deterministik. Kesamaan angka tidak berarti
berkas lama dipakai ulang; bukti asalnya adalah scan, eksekusi, ID generasi dan
hash input/output yang baru dicatat.

Membangun ulang tidak otomatis membuktikan kebenaran target atau waktu. Selama
G0 belum lulus, manifest tetap diagnostik. Target/QA kanonis, normalizer training
ilmiah, cache ilmiah dan checkpoint terpilih yang belum dapat disahkan dicatat sebagai
`not_built` dalam `meta/pending_artifacts.json`, bukan diisi dengan artefak lama
atau dilabeli valid tanpa bukti.

Training engineering memakai namespace `diagnostic_runs/<run_id>` terpisah.
RPC berasal dari M4Human aktual, tetapi first22 direct joints dan waktu nominal
tetap calon target/asumsi. Metadata, QA, normalizer dan checkpoint diberi flag
diagnostik; tidak menjadi bukti pelulusan gate atau input otomatis training
ilmiah. Eksperimen gain embedding sensor berbeda dari input H langsung pada
spesifikasi dan wajib ditandai `experimental_spec_deviation` dalam plan baru.
Spesifikasi normatif dan default projector tidak berubah karena eksperimen itu.
