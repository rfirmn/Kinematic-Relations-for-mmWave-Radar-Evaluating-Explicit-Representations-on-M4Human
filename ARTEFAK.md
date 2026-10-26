# Lokasi artefak penelitian

Generasi aktif **fresh_20261026_01**. Semua lokasi berikut berada pada drive C
di dalam repo ini. Metadata dan hasil penelitian dibangun baru; folder metadata
eksternal tidak digunakan.

| Isi | Buka |
|---|---|
| Metadata dan bukti audit aktual | [meta/](derived_data/m4human_relational64/fresh_20261026_01/meta/) |
| Inventaris sumber baru | [source_inventory.json](derived_data/m4human_relational64/fresh_20261026_01/meta/source_inventory.json) |
| Manifest frame/window dan turunan baru | [derived_data/](derived_data/m4human_relational64/fresh_20261026_01/) |
| Laporan hasil pemeriksaan | [implementation_status.md](docs/report_training/fresh_20261026_01/implementation_status.md) |
| Bukti isolasi artefak | [artifact_isolation_checks.json](docs/report_training/fresh_20261026_01/artifact_isolation_checks.json) |
| Daftar hash output | [artifact_registry.json](derived_data/m4human_relational64/fresh_20261026_01/meta/artifact_registry.json) |
| Checkpoint eksperimen aktual | [checkpoints/](checkpoints/m4human_relational64/fresh_20261026_01/) |
| Artefak yang belum dapat dibangun | [pending_artifacts.json](derived_data/m4human_relational64/fresh_20261026_01/meta/pending_artifacts.json) |
| Training diagnostik aktual varian gain | [report.md](docs/report_training/fresh_20261026_01/diagnostic_runs/diagnostic_20261026_04_gain/report.md) |
| Pilot diagnostik 100 update fisik dan 10 update/projector | [report.md](docs/report_training/fresh_20261026_01/diagnostic_runs/diagnostic_20261026_05_complete/report.md) |
| Review lengkap: fisik, QA, donor, probe, resource | [report.md](docs/report_training/fresh_20261026_01/diagnostic_runs/diagnostic_20261026_05_complete/reviews/validation12_v1/report.md) |
| Hasil pilot dan pekerjaan berikutnya | [pilot_results.md](docs/pilot_results.md) |
| Galeri radar, target, prediksi pose dan kurva training | [visual_evidence.md](docs/visual_evidence.md) |
| Asal, resolusi dan hash gambar publikasi | [manifest.json](docs/figures/fresh_20261026_01/manifest.json) |
| Catatan seluruh revisi | [diagnostic_findings.md](docs/diagnostic_findings.md) |
| Bukti temuan pada run pertama | [report.md](docs/report_training/fresh_20261026_01/diagnostic_runs/diagnostic_20261026_01/report.md) |

Checkpoint eksperimen ilmiah belum tersedia. Checkpoint training diagnostik
aktual berada di `checkpoints/.../diagnostic_runs/<run_id>/`,
berstatus `diagnostic`, dan tidak menjadi checkpoint terpilih untuk test.
Checkpoint pilot sintetis berada
di `derived_data/.../synthetic_pilot/` dan hanya membuktikan integrasi perangkat
lunak. Metadata/manifest aktual yang sudah terbentuk tetap diagnostik selama
authority target, koordinat dan waktu belum lulus G0. Generasi baru tidak
otomatis memberi validitas ilmiah.

[Kebijakan penyimpanan dan validasi](docs/artifact_policy.md) ·
[Konfigurasi aktif](configs/relational64/runtime.yaml) ·
[Perintah menjalankan sistem](docs/runbook.md)
