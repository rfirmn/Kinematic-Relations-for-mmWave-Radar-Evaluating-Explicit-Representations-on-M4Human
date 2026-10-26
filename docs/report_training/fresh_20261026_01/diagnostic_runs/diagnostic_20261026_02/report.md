# Training diagnostik M4Human aktual

Run `diagnostic_20261026_02`; status **needs_revision**. RPC aktual, calon target direct joints, waktu nominal12Hz. G0 tetap false; hasil ini bukan akurasi penelitian yang disahkan.

| Tahap | Status |
|---|---|
| prepare | passed |
| physical | passed |
| alignment | needs_revision |

## Temuan yang perlu direvisi

- S: invalid_or_truncated_generation; valid schema rate=0.0.
- R: invalid_or_truncated_generation; valid schema rate=0.5.

| Panel | L_phys awal | L_phys akhir |
|---|---|---|
| train | 4.855422 | 2.866581 |
| val | 4.936769 | 2.813657 |

20 optimizer updates; checkpoint dimuat ulang: True.

![Loss diagnostik](physical_loss.png)

| Kondisi/kontrol | Optimizer updates | Schema valid rate pada val |
|---|---|---|
| M | 2 | 1.0 |
| S | 2 | 0.0 |
| F | 2 | 1.0 |
| E | 2 | 1.0 |
| R | 2 | 0.5 |
| T | 0 | 1.0 |
| Maj | 0 | 1.0 |
| Rule | 0 | 1.0 |

Hash parameter Qwen tetap sama: **True**.

Panel val hanya4 window. Label/kecepatan masih provisional; parser yang valid bukan bukti jawaban benar. Metrik numerik terhadap calon label tersimpan di alignment.json dan belum menjadi bukti generalisasi atau hipotesis riset.

Checkpoint per update dan optimizer berada di `C:\Users\Rio Aslab\Documents\ChatGPT\tugas akhir V2\checkpoints\m4human_relational64\fresh_20261026_01\diagnostic_runs\diagnostic_20261026_02`.

Jika suatu tahap gagal, baca failures.jsonl, perbaiki, dan ulangi tahap itu. Jangan meluluskan gate ilmiah atau memakai checkpoint diagnostic untuk pemilihan test.
