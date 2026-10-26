# Kelengkapan pilot diagnostik aktual

Run `diagnostic_20261026_05_complete`, review `validation12_v1`: **passed**.

Seluruh hasil menggunakan calon target dan waktu nominal; scientific_acceptance=false, G0 tetap false.

| Pemeriksaan | Hasil |
|---|---|
| subject_split_disjoint | True |
| train_only_intensity_unchanged | True |
| test_labels_not_opened | True |
| physical_checkpoint_is_diagnostic | True |
| cache_forward_equivalent | True |
| physical_predictions_finite | True |
| physical_frozen_unchanged | True |
| physical_resource_guard | True |
| donor_has_actual_pairs | True |
| donor_actual_text_embeddings_identical | True |
| donor_sensor_embeddings_change | True |
| projectors_frozen_during_evaluation | True |
| frozen_Qwen_unchanged | True |
| alignment_resource_guard | True |
| all_QA_generation_schema_valid | True |
| donor_text_and_generation_valid | True |
| small_sample_loss_reduction_at_least_50_percent | True |
| checkpoint_reload | True |
| finite_gradients_and_losses | True |
| alignment_backward_and_loss_finite | True |
| source_snapshot_unchanged | True |

Panel: 8 train dan 52 validation windows; test tidak dibuka.
Penurunan loss train: 63.48%.

| Kondisi | Schema valid | S_atomic provisional |
|---|---|---|
| M | 1.0 | 0.2329365079365079 |
| S | 1.0 | 0.2452020202020202 |
| F | 1.0 | 0.24199834162520728 |
| E | 1.0 | 0.23955026455026454 |
| R | 1.0 | 0.2437908496732026 |
| T | 1.0 | 0.0 |
| Maj | 1.0 | 0.27520110608345905 |
| Rule | 1.0 | 0.2678300094966761 |

Donor aktual: 13 pasangan, hasil deskriptif. Validitas format bukan ketepatan jawaban.

Ringkasan JSON memuat galat fisik, coverage/kelas, probe, kontrol donor, resource, lingkungan, dan hashes input.
