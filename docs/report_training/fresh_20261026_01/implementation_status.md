# Status implementasi relational64

Implementasi perangkat lunak tersedia. Gerbang data G0 belum lulus; training dan pilot ilmiah yang disahkan belum dijalankan. Training diagnostik dengan RPC aktual dan calon target/waktu provisional dilaporkan terpisah.

| Bukti | Status |
|---|---|
| Audit M4Human aktual | 661,132 frame; read-only |
| Fixture perangkat lunak | 218 tests; passed |
| Qwen pretrained Windows/CUDA | passed; sensor sintetis |
| Overfit8 sintetis | lulus |
| Profil fisik sintetis | tersimpan |
| Profil alignment sintetis | 10 warm-up +30 optimizer steps +30 generations tersimpan |
| Pilot M/S/R sintetis | passed |
| Training fisik diagnostik aktual | selesai; provisional, G0 tetap false |
| Alignment diagnostik aktual | selesai; provisional, G0 tetap false |

- Run diagnostik [diagnostic_20261026_01](diagnostic_runs/diagnostic_20261026_01/summary.json) — needs_revision
- Run diagnostik [diagnostic_20261026_02](diagnostic_runs/diagnostic_20261026_02/summary.json) — needs_revision
- Run diagnostik [diagnostic_20261026_03](diagnostic_runs/diagnostic_20261026_03/summary.json) — needs_revision
- Run diagnostik [diagnostic_20261026_04_gain](diagnostic_runs/diagnostic_20261026_04_gain/summary.json) — passed
- Run diagnostik [diagnostic_20261026_05_complete](diagnostic_runs/diagnostic_20261026_05_complete/summary.json) — passed

- Review diagnostik [diagnostic_20261026_05_complete/validation12_v1](diagnostic_runs/diagnostic_20261026_05_complete/reviews/validation12_v1/summary.json) — passed; scientific_acceptance=false

## Penghambat data

- Direct joints have no independently verified equivalence to hashed SMPL-X mesh/regressor reference.
- RawPoints producer axes/handedness, source offsets and upstream crop privilege not established.
- Producer renumbers union of radar/RGB filenames; contiguous output indices cannot prove absent original-frame gaps. Need raw filename/synchronization mapping.

## Gerbang

| Gate | Lulus | Alasan |
|---|---|---|
| G0 | False | Data unit/joint/frame/time authority unresolved |
| G1 | False | Actual train/val targets and label audit pending G0 |
| G2 | False | Actual8-window overfit and native resource acceptance pending G0/G1 |
| G3 | False | Selected physical checkpoint, cache equivalence and train descriptor statistics pending |
| G4 | False | Native Qwen software checks do not establish actual-cache alignment resource acceptance |
| G5 | False | 15 projectors/3 physical checkpoints and experiment choices have not been selected |
| G6 | False | Scientific test evaluation not run |

## Artefak

- [verify_all.json](verify_all.json)
- [qwen_native_check.json](qwen_native_check.json)
- [overfit.json](synthetic_overfit/overfit.json)
- [profile_physical.json](synthetic_profile/profile_physical.json)
- [profile_alignment_R.json](synthetic_alignment_profile/profile_alignment_R.json)
- [synthetic_pilot.json](synthetic_pilot.json)
- [source_inventory.json](../../../derived_data/m4human_relational64/fresh_20261026_01/meta/source_inventory.json)
- [coordinate_audit.json](../../../derived_data/m4human_relational64/fresh_20261026_01/meta/coordinate_audit.json)

Kecepatan/durasi belum dinyatakan terukur; manifest diagnostik memakai waktu nominal provisional. Fixture tidak membuktikan pairing RPC–target aktual atau hasil ilmiah. Training penuh adalah pekerjaan terpisah.
