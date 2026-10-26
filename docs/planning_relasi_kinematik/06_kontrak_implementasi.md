# 06 — Kontrak implementasi dan artefak

## 1. Batas pekerjaan

Implementasi harus menyediakan komponen berikut. Nama path merupakan target organisasi modul; perintah pada dokumen ini adalah kontrak antarmuka yang harus tersedia setelah implementasi selesai. Paket spesifikasi sendiri tidak menjalankan training.

```text
eksperimen_model/relational64/
  cli.py                    # satu entry point subcommand
  config.py                 # schema, resolve, validation, hashing
  data/source.py            # RPC-only reader dan inventaris
  data/manifest.py          # frames, split, segment, window
  data/targets.py           # joint export dan annotation masks
  data/qa.py                # label policy dan QA registry
  data/batches.py           # SensorBatch / SupervisedBatch
  kinematics.py             # W derivative, root/local, descriptors
  models/encoder.py         # spatial/temporal encoder dan pose head
  models/projector.py       # M serta S/F/E/R
  models/language.py        # template, token splicing, generation
  models/probe.py           # classifier tiga task
  training/physical.py      # training fisik dan validation
  training/alignment.py     # training projector dan validation generation
  training/checkpoint.py    # save/resume/freeze audit
  evaluation/physical.py    # metric numerator/count
  evaluation/qa.py          # parser/confusion/task scores
  evaluation/donor.py       # matching dan donor generation
  evaluation/statistics.py  # paired cluster bootstrap
  reporting.py              # report, metrics, figures, INDEX
  resources.py              # profiler dan resource guard
tests/relational64/
configs/relational64/
derived_data/m4human_relational64/
docs/report_training/
```

File data/target dan checkpoint memiliki namespace protocol ID. Tidak menginfer konfigurasi dari nama folder atau suffix checkpoint. Semua operasi menerima resolved config dan memvalidasi provenance.

## 2. API minimum

| Komponen | Input | Output/invariant |
|---|---|---|
| `read_rpc(source_key)` | Key terdaftar, schema | float32 `[N,4]`, belum normalisasi, tidak membuka GT |
| `build_segments(frames,policy)` | Sensor metadata | Segment ID; annotation masks tidak diperiksa |
| `build_windows(segments,split)` | Urutan segmen | windows.jsonl deterministik dengan64 UID |
| `make_sensor_batch(window)` | RPC+timestamp+normalizer | X, point_mask, time_s, C, opaque UID |
| `make_derivative_operator(time_s)` | float64 `[64]` | W `[64,64]`, availability `[64]` |
| `RadarEncoder.forward(X,mask,time_s)` | Tensor sensor | Y `[B,64,256]` |
| `PoseHead.forward(Y)` | Y | P `[B,64,22,3]` |
| `derive_states(P,time_s)` | Prediksi saja | V,q,w,velocity_available |
| `make_descriptors(P,V,condition)` | Prediksi, S/F/E/R | A `[B,64,22,7]`, D `[B,64,29,14]` |
| `StructuredProjector.forward(Y,A,D,availability)` | Descriptor ternormalisasi | H `[B,64,d]`; optional Z untuk probe |
| `DirectProjector.forward(Y)` | Y | H `[B,64,d]` |
| `splice_chat(H,time_s,prompt,target?)` | Embedding sensor, teks | inputs_embeds, attention_mask, position_ids, labels? |
| `generate_answer(batch)` | Tidak ada target | Raw output tokens/text + stop reason |
| `parse_answer(text,task,stop_reason)` | Output mentah | enum/null, parse status, reason |
| `score_qa(predictions,qa_index)` | Join berdasarkan qa_uid | Metrik, confusion, counts; tidak inner-join drop missing |

`make_descriptors` tidak menerima label policy atau threshold, sehingga tidak dapat mengubah descriptor menurut kelas. `generate_answer` tidak menerima target, reference status, atau support mask anotasi. Inference interface tidak boleh mewarisi seluruh SupervisedBatch melalui `**kwargs`.

## 3. Konfigurasi dan identitas

Gunakan [spesifikasi.yaml](spesifikasi.yaml) sebagai nilai normatif. Runtime menerima patch hanya untuk field lingkungan: path absolut, source inventory, provenance, model commit, file hashes, precision hasil probe, dan microbatch yang mempertahankan effective batch. Perubahan scientific fields (T,N,condition,loss,layer,label,prompt,split) harus menghasilkan run ID/config hash berbeda dan tercatat sebagai kondisi eksperimen.

Canonical JSON untuk hashing: UTF-8, sorted keys, separator `,` dan `:`, tanpa NaN/Infinity, float finite, array mempertahankan urutan. Hash file biner memakai SHA-256 byte asli. `schema_id` memeriksa struktur; `policy_hash` memeriksa isi. Jangan menggunakan waktu pembuatan sebagai satu-satunya identitas data.

`runtime.lock.json` wajib memuat:

- dataset_root dan mapping database absolut, source_inventory hash, immutable_snapshot=true;
- coordinate/joint map/target source hashes, time_source, serializer/endian/sentinel;
- split dan exposure ledger hashes;
- model_id, model_commit (commit penuh), tokenizer_commit, chat_template_hash, config hash, safetensors manifest hash;
- OS, Python, dependency lock, torch/CUDA/driver/GPU identity, dtype, determinism flags;
- microbatch/accumulation aktual, total sequence limit, profil dan acceptance report hash.

Field null tidak diisi dengan string `unknown` lalu dianggap lulus. Validator membedakan audit data yang tidak membutuhkan Qwen dari alignment yang wajib memiliki seluruh identitas bahasa.

## 4. Cache per window

Simpan shard maksimum1.024 window sebagai array NumPy `.npy` yang dapat di-memory-map, bukan list pickle besar. Index JSONL menghubungkan UID ke shard,row. Array wajib:

| Array | Dtype/shape per window |
|---|---|
| Y | float32 `[64,256]` |
| P | float32 `[64,22,3]` |
| V | float32 `[64,22,3]` |
| time_s | float64 `[64]` |
| centroid | float32 `[3]` |
| velocity_available | bool `[64]` |

`cache.meta.json`: protocol/condition-independent physical seed, encoder+pose hash, windows/sampling/intensity-normalizer/coordinate hashes, output dtype, module eval state, shard hashes, count. Descriptor tidak perlu disimpan karena dapat dihitung murah dari P/V; normalizer descriptor disimpan terpisah per kondisi.

Y/P/V berjumlah sekitar99.328 byte per window, ditambah metadata. Estimasi disk = jumlah window aktual×ukuran tersebut; bukan jumlah frame publikasi. Cache tidak diduplikasi untuk tiga QA. Cache test dibuat setelah test lock; tidak dipakai menghitung normalizer.

## 5. Artefak prediksi

Satu row `predictions.jsonl` per QA expected, termasuk failure:

```json
{
  "qa_uid": "<sha256>",
  "window_uid": "<sha256>",
  "condition": "R",
  "seed": 42,
  "checkpoint_hash": "<sha256>",
  "prompt_hash": "<sha256>",
  "raw_text": "{\"answer\":\"tie\"}",
  "predicted_label": "tie",
  "parse_status": "valid",
  "stop_reason": "eos",
  "input_tokens": 0,
  "sensor_embeddings": 64,
  "generated_tokens": 0,
  "latency_ms": 0.0
}
```

Angka0 pada contoh adalah placeholder format, bukan pengukuran. File hasil wajib berisi nilai terukur. Output tidak memuat reference answer saat generation; evaluator menggabungkannya berdasarkan UID. Duplicate QA UID adalah error. Missing output dibuat row berstatus missing, sehingga denominator tidak turun. Prediction file tidak mengubah QA registry.

Donor output menambah pair_uid, side, donor policy/hash, source window UID, canonical time-grid hash, dan identical-text-token hash. Dua sisi disimpan walau salah satunya gagal.

## 6. CLI yang wajib disediakan

Semua command memakai `.venv` atau `uv`, bekerja dari root repository. Contoh antarmuka PowerShell:

```powershell
uv run python -m eksperimen_model.relational64.cli audit --config configs/relational64/runtime.yaml
uv run python -m eksperimen_model.relational64.cli prepare --config configs/relational64/runtime.yaml
uv run python -m eksperimen_model.relational64.cli verify --suite data
uv run python -m eksperimen_model.relational64.cli profile --stage physical --config configs/relational64/runtime.yaml
uv run python -m eksperimen_model.relational64.cli train-physical --seed 42 --config configs/relational64/runtime.yaml
uv run python -m eksperimen_model.relational64.cli extract --split train,val --seed 42 --config configs/relational64/runtime.yaml
uv run python -m eksperimen_model.relational64.cli fit-descriptors --condition R --seed 42 --config configs/relational64/runtime.yaml
uv run python -m eksperimen_model.relational64.cli profile --stage alignment --condition R --config configs/relational64/runtime.yaml
uv run python -m eksperimen_model.relational64.cli train-alignment --condition R --seed 42 --config configs/relational64/runtime.yaml
uv run python -m eksperimen_model.relational64.cli train-probe --source Y --seed 42 --config configs/relational64/runtime.yaml
uv run python -m eksperimen_model.relational64.cli lock-test --config configs/relational64/runtime.yaml
uv run python -m eksperimen_model.relational64.cli evaluate --split test --test-lock derived_data/m4human_relational64/test.lock.json
uv run python -m eksperimen_model.relational64.cli report --test-lock derived_data/m4human_relational64/test.lock.json
```

S/F/E/M mengulang perintah alignment dengan condition terkait; semua memiliki validation/prediction terpisah. `fit-descriptors` tidak diperlukan M. `prepare` membuat split, frames, windows, target dan QA secara bertahap dengan validasi; tidak menulis database sumber. `lock-test` menolak run yang belum mempunyai semua checkpoint rencana final. `evaluate` menjalankan ekstraksi test serta seluruh panel yang terdaftar, dengan rule yang sama pada semua kondisi.

Exit status:0 sukses,2 kontrak/config tidak sah,3 data/provenance tidak lengkap,4 resource tidak cukup,5 kegagalan numerik/runtime. Error wajib menyebut artefak/UID relevan tanpa membocorkan credential. Tidak melakukan fallback kondisi/model diam-diam.

## 7. Urutan pekerjaan dan definition of done

| Paket kerja | Deliverable | Ketergantungan | Diterima ketika |
|---|---|---|---|
| P0 Spec/config | Typed schema, resolver, validator, hashing | Paket ini | Semua konstanta konsisten dan null gate benar |
| P1 Data | RPC reader, inventory, split/frames/windows | P0 | Uji sentinel, segment, leakage, determinisme lulus |
| P2 Targets | Ekspor joint, W derivative, label/QA | P1 | Contoh numerik dan unit/mask lulus |
| P3 Physical model | Encoder, pose, loss, trainer | P1/P2 | Shape, mask, gradient, overfit kecil, profil CUDA lulus |
| P4 Features | Cache dan descriptor normalizer | P3 | Hash/cache fresh equivalence lulus |
| P5 Language | M,S,F,E,R; splice, parser, trainer | P4 | Parameter count/freeze/gradient/generation lulus |
| P6 Evaluation | Metric, donor, bootstrap, test lock | P2/P5 | Synthetic confusion/donor/CI checks lulus |
| P7 Pilot | Satu seed M/S/R dan laporan | P3–P6 | Feasibility dan coverage diketahui, tanpa test tuning |
| P8 Study | Seluruh seed/kondisi rencana | P7 | Checkpoint+prediksi+laporan lengkap dan dapat diaudit |

P3 dan P5 tidak diterima hanya karena forward menghasilkan tensor; bukti backward, freeze, loss masking, dan kemampuan overfit sampel kecil diperlukan. P8 tidak diterima sebagai bukti kualitas hanya karena unit test lulus.

## 8. Laporan per run

Setiap run menulis `docs/report_training/RUN_<UTC timestamp>_relational64_<stage>_<condition>_seed<n>/` dengan `report.md`, `metrics.json`, `config_snapshot.yaml`, `runtime.lock.json`, `environment.json`, `artifact_manifest.json`, kurva training/validation, metrik sumber, resource profile, dan link checkpoint. Tambahkan baris ke `docs/report_training/INDEX.md` tanpa menghapus laporan lain.

Laporan gagal tetap menyimpan status, alasan, langkah terakhir, dan artefak yang lengkap. Resume menambahkan provenance ke run yang sama atau child-run eksplisit. Artefak partial tidak didaftarkan sebagai hasil final.
