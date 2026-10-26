# 04 — Pelatihan, objective, dan pemilihan checkpoint

## 1. Urutan tahap

`Audit → split/window/target lock → training fisik → pemilihan checkpoint fisik → ekstraksi prediksi train/val → normalizer descriptor → alignment per kondisi → penguncian seluruh keputusan → ekstraksi/evaluasi test → laporan`.

Test tidak dipakai untuk profiling pemilihan model, label tuning, early stopping, atau debug semantik. Inventaris sumber test boleh memeriksa keberadaan/format/split tanpa membaca skor hasil model. Artefak sumber dan checkpoint yang berada di workspace tidak otomatis dimuat hanya karena namanya cocok; provenance harus memenuhi protokol.

## 2. Objective fisik

Prediksi P dan target P* memiliki offset sensor c yang sama. V dan V* memakai operator turunan pada 03. Semua loss dihitung float32 di luar autocast, dengan epsilon denominator hanya untuk proteksi zero count; bukan tambahan sampel.

Definisikan weighted L1 sebagai `sum(mask * joint_weight * time_weight * abs(error)) / (3*sum(mask * joint_weight * time_weight))` untuk vektor3D. Angka3 menghitung jumlah koordinat. Untuk scalar bone length tidak ada faktor3. Mask dapat berbeda per komponen loss.

| Komponen | Error | Mask | Bobot joint |
|---|---|---|---|
| L_pos | `(P−P*) / 1 meter` | Posisi target valid per joint/frame | Tabel01 |
| L_root | `(V_0−V*_0) / (1 meter/second)` | Tujuh posisi pelvis valid | 1 |
| L_local | `((V_j−V_0)−(V*_j−V*_0)) / (1 meter/second)` | Turunan joint j dan root valid, j=1..21 | Tabel01 |
| L_bone | `(norm(P_child−P_parent)−norm(P*_child−P*_parent)) / 1 meter` | Kedua posisi target valid | 1 |

Loss total `L_phys=L_pos+5*L_root+10*L_local+2*L_bone`. Koefisien ini dipakai bersama untuk semua kondisi; estimator kecepatan berbasis waktu dan satuan dinyatakan eksplisit sebagai keputusan protokol. Tidak mengklaim identik dengan setiap detail objective publikasi mmMind.

Time weight: t=0..53 bernilai1; t=54..63 bernilai `1+4*(t−54)/9`. Bobot itu dipakai pada seluruh komponen yang tersedia di endpoint tersebut. Indeks0..5 tidak ikut velocity loss. Mask posisi tidak boleh otomatis disamakan dengan mask velocity.

Jika sebuah komponen tidak mempunyai target dalam effective batch, komponen tersebut berkontribusi nol dan count0 dicatat. Jika seluruh komponen count0, skip optimizer step dan catat `no_supervision`; epoch/optimizer-step accounting tidak berpura-pura terjadi update. Window seperti itu tidak memberi supervisi fisik tetapi tetap dapat berada dalam manifest sensor.

Untuk accumulation dengan variasi target mask, hitung denominator tiap komponen untuk **seluruh effective batch** dari mask target terlebih dahulu. Backprop numerator tiap microbatch dibagi denominator kelompok yang sama; terapkan bobot loss setelah pembagian. Tidak merata-ratakan rasio microbatch yang denominator-nya berbeda. Validation mengakumulasi numerator dan denominator seluruh panel, kemudian menghitung setiap komponen dan totalnya.

## 3. Recipe fisik utama

| Parameter | Nilai |
|---|---|
| Initialization | Scratch untuk encoder dan pose head |
| Seeds rencana utama | 42,43,44; setiap seed memiliki encoder sendiri |
| Optimizer | AdamW, lr=6e-4, betas=(0.9,0.999), eps=1e-6 |
| Weight decay | 0.01 pada linear weights; 0 pada bias, LayerNorm, embedding/query |
| Effective batch | 16 window; awal microbatch1, accumulation16 |
| Epoch maksimum | 100 |
| Warm-up | 3 epoch pertama, linear optimizer-step dari0 ke lr maksimum |
| Setelah warm-up | Cosine tunggal hingga1e-6 pada akhir budget100 epoch |
| Gradient clipping | Global norm1.0 setelah unscale |
| Early stopping | 15 epoch validation tanpa improvement |
| Pemilihan | L_phys validation terkecil; tie≤1e-8 pilih epoch paling awal |
| Precision | bf16 autocast jika capability/test lulus; fp16+GradScaler fallback |
| Data order | Shuffle window train seeded setiap epoch; val urut UID |
| Augmentasi | Hanya pemilihan titik train; geometri none |

Scheduler melangkah sesudah optimizer step berhasil; overflow GradScaler yang membatalkan update tidak memajukan scheduler. Epoch terakhir/kelompok accumulation parsial memakai jumlah sampel dan denominator aktual. Tidak menggandakan window terakhir untuk memenuhi batch. Simpan RNG state untuk resume di batas optimizer step.

Pemilihan checkpoint tidak memakai loss bahasa atau test MPJPE. Laporan tetap menyertakan MPJPE relatif/global dan galat root/velocity, karena satu scalar L_phys dapat menyembunyikan tradeoff. Simpan `best` dan `last`, bukan setiap epoch penuh tanpa batas.

## 4. Ekstraksi fisik dan normalizer

Setelah best checkpoint dipilih: freeze encoder/pose, aktifkan eval, sampling titik deterministik, ekstrak Y float32 dan P float32 per window. V dihitung float32 dari W float64 dan P; simpan pula timestamp/centroid/UID dan source hash. Simpan V untuk mempermudah audit, walaupun dapat dibentuk ulang.

Cache utama tidak memakai stochastic point sampling train agar S/F/E/R mendapat bukti yang identik. Tidak boleh mencampurkan cache dari epoch/checkpoint/sampling yang berbeda. Satu cache per seed fisik, dibagi seluruh kondisi; tidak disalin per QA. Normalizer descriptor tiap kondisi hanya membaca cache train.

## 5. Objective bahasa

Semua kondisi sensor dilatih pada pasangan QA yang sama, termasuk reference-unknown, mengecualikan reference-undefined. Loss per sampel adalah rata-rata negative log-likelihood token **jawaban assistant dan end-of-turn**. Ambil rata-rata loss per sampel dalam effective batch, bukan rata-rata seluruh token campuran; ini menghindari label dengan token lebih panjang mendapat bobot lebih tinggi hanya karena panjang.

Untuk menghemat memori vocabulary, trainer dapat mengambil final hidden state Qwen hanya pada posisi k−1 yang target posisi k-nya tidak−100, lalu menerapkan LM head frozen yang sama pada state terpilih. CE dihitung float32 dan direduksi per contoh. Ini harus ekuivalen dengan full-logits causal LM loss pada fixture; final normalization backbone tetap diterapkan, tidak diganti layer lain. Jalur ini tidak mengurangi prefix/attention dan tidak mengubah token target. Gunakan jalur terpilih tersebut sebagai default jika equivalence lulus; jangan membentuk logits vocabulary penuh pada setiap posisi sensor hanya untuk membuangnya.

GT tidak menjadi masukan Qwen: token jawaban hanya masuk dalam teacher forcing pada posisi assistant dengan causal mask, sedangkan prefix sensor berasal dari cache prediksi. Evaluasi tidak memakai teacher forcing atau token jawaban acuan.

Task sampler:

1. Buat tiga daftar eligible QA train: root, arms, legs.
2. Sampling task dengan probabilitas0.50,0.25,0.25; dalam task, pilih QA uniform dengan replacement.
3. Panjang satu epoch = jumlah seluruh QA eligible di tiga daftar. Ini jumlah draw, bukan jumlah window unik.
4. Simpan urutan draw `(qa_uid,task)` per seed/epoch atau RNG state yang dapat memproduksinya persis. S/F/E/R/M memakai stream yang sama.
5. Jika suatu task tidak memiliki QA eligible, hentikan studi tiga-task; jangan menormalisasi diam-diam menjadi dua-task.

Tidak class-balance berdasarkan validation/test. Laporkan jumlah draw dan jumlah QA/window unik terlihat per run.

## 6. Recipe alignment utama

| Parameter | Nilai |
|---|---|
| Trainable | Hanya projector sesuai definisi02 |
| Frozen | Encoder, pose head, semua bobot Qwen termasuk embeddings/LM head |
| Optimizer | AdamW lr=3e-4, betas=(0.9,0.999), eps=1e-6 |
| Weight decay | 0.01 linear weights; bias/LN/embedding/query0 |
| Effective batch | 16 QA; awal microbatch1, accumulation16 |
| Budget | 20 epoch maksimum |
| Warm-up | 5% total optimizer steps budget20 epoch |
| Schedule | Linear warm-up, lalu cosine sampai lr=3e-6 |
| Gradient clipping | Global norm1.0 setelah unscale |
| Validation | Generasi seluruh eligible QA validation setiap epoch |
| Early stopping | 5 epoch tanpa peningkatan selection tuple |
| Selection tuple | S_atomic terbesar, lalu parse/schema-valid rate terbesar, lalu epoch paling awal |
| Toleransi tie | 1e-8 untuk skor dan parse rate |
| Qwen mode | eval; use_cache=False saat training, True pada generation yang lulus uji |
| Padding | Right pada training; inference microbatch1 default |
| Batas sequence | Training3072; generation input3040 + new tokens32 |

Validation score utama hanya reference-defined, tetapi generasi mencakup defined+unknown supaya panel sekunder dapat dihitung tanpa generasi ulang. Parse rate tie-break dihitung atas defined+unknown. Undefined tidak dipakai untuk seleksi. Model trained di titik early stopping yang berbeda tetap memakai budget maksimum dan aturan seleksi sama; laporkan actual optimizer steps.

Jangan menambah caption alignment stage dengan teks yang dihasilkan dari label aktivitas. Studi ini menguji QA terstruktur yang targetnya tersedia; tidak mengklaim menyalin seluruh kurikulum bahasa mmMind.

## 7. Matriks training

Rencana utama: tiga seed fisik42/43/44; untuk setiap seed latih M,S,F,E,R dengan seed alignment yang sama. Dengan demikian ada tiga encoder fisik dan15 projector. Bobot Qwen pretrained yang dipin identik pada seluruh run dalam backbone yang sama.

Mulai pilot satu seed dengan M,S,R untuk memeriksa kelayakan. Pilot bukan pengganti hasil tiga seed. Jika keterbatasan waktu hanya memungkinkan satu seed, laporkan studi sebagai eksploratif dan hapus klaim variasi lintas seed. Profil runtime menentukan jadwal, bukan alasan mengubah jumlah frame atau token pada satu kondisi saja.

Tidak tuning satu kondisi lebih lama karena hasilnya belum sesuai hipotesis. Bila mengadakan pencarian lr, semua kondisi memperoleh grid sama, misalnya {1e-4,3e-4,1e-3}, hanya validation, dan biaya seluruh pencarian dilaporkan terpisah. Default utama tidak menjalankan grid tersebut.

## 8. Probe dan rule

Rule memakai P,V prediksi dan rumus03 tanpa target mask sebagai input. Semua58 endpoint sensor tersedia. Pada test, skor Rule dibandingkan pada QA reference-defined yang sama dengan LLM; aturan ambiguitas prediksi boleh menghasilkan unknown dan tetap salah pada reference-defined.

Probe task memakai mean dan max sepanjang64 frame dari source feature256, concat512 → Linear512→128 → GELU → Linear128→12. Output dibagi tiga head empat kelas. Source `Y` dan `Z_R` merupakan run berbeda; tidak diberi P/V tambahan. Untuk Z_R, projector terpilih dibekukan; interpretasi menyertakan bahwa representasinya sudah dilatih dengan QA train.

Probe dilatih hanya pada QA eligible train, weighted task sampler sama, AdamW lr1e-3, batch128, wd0.01, maksimum50 epoch, patience8, selection S_atomic validation. Loss cross-entropy untuk head task terkait; kelas unknown sah. Seed mengikuti run. Probe bukan pembanding kapasitas identik dengan Qwen.

## 9. Checkpoint dan resume

Checkpoint berisi state_dict modul trainable, optimizer/scheduler/scaler, epoch, optimizer step, RNG CPU/CUDA/NumPy/sampler, best score, patience counter, config resolved, seluruh hash data/normalizer/target/model, environment lock, serta status `best`/`last`. Qwen frozen disimpan sebagai identitas model+commit+file hashes, bukan disalin ke setiap checkpoint.

Resume hanya jika seluruh input hash, seed, batch efektif, precision, kondisi, dan module shapes cocok. Simpan file sementara, fsync, lalu rename atomik. Checkpoint setengah tertulis tidak boleh dipilih sebagai best. Kegagalan resource menghasilkan run berstatus gagal/terhenti dengan laporan; tidak dianggap skor model nol tanpa penjelasan.
