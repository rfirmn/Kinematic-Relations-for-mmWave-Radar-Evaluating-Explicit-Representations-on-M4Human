# 05 — Evaluasi, statistik, dan interpretasi

## 1. Panel bersama

Seluruh kondisi dinilai pada QA UID yang sama. Primary panel memakai reference-defined. Secondary panel memakai defined+unknown dengan **generasi dan prompt yang sama**. Undefined hanya masuk coverage. Tidak menyaring panel berdasarkan keberhasilan parsing, confidence prediksi, atau apakah suatu model menghasilkan output.

Tiap seed menggunakan daftar QA sama dan checkpoint masing-masing yang dipilih di validation. Data test hanya dibuka setelah `test.lock.json` lengkap. Evaluator boleh menjalankan ulang metrik dari prediksi mentah dengan definisi yang sudah dikunci; tidak mengubah aturan untuk memperbaiki hasil.

## 2. Metrik QA

Untuk task k, tetapkan tiga kelas substantive secara tetap. Untuk setiap kelas c: `F1_c=2TP/(2TP+FP+FN)`; jika denominator0 maka0. `macroF1_k` adalah rata-rata ketiga kelas, termasuk kelas yang tidak hadir. Prediksi unknown/invalid/missing/truncated memberi FN pada kelas true dan tidak dianggap prediksi salah satu kelas substantive.

`S_atomic = 0.50*macroF1_root + 0.25*macroF1_arms + 0.25*macroF1_legs`.

Skor disimpan [0,1], tabel persen menggunakan100×skor. Delta dilaporkan dalam poin persentase, bukan persentase relatif. Endpoint primer `R−S`; M,R,F,E,S tetap dilaporkan lengkap untuk mencegah pemilihan pembanding yang menguntungkan.

Metrik wajib tambahan:

- Accuracy tiap task: jumlah jawaban tepat dibagi seluruh QA reference-defined task.
- Accuracy macro tiga task dan accuracy weighted0.50/0.25/0.25; label metrik harus membedakan keduanya.
- Macro-F1 empat kelas pada secondary panel, termasuk unknown; bobot task tetap sama.
- JSON parse rate, schema+enum valid rate, substantive answer rate, unknown rate, invalid/missing/truncated/runtime-error count.
- Confusion matrix primary: tiga baris true, kolom tiga kelas+unknown+invalid+missing/truncated. Secondary mempunyai empat baris.
- Support kelas, subjek, recording, window, QA; reference coverage per defined/unknown/undefined dan alasannya.

Jika kelas tidak hadir, ceiling macro-F1 maksimum menjadi `jumlah_kelas_hadir/3` pada primary. Tampilkan ceiling; jangan menghapus kelas. Jika suatu task sama sekali tidak memiliki reference-defined, `S_atomic` tidak terdefinisi, simpan null dan laporkan kegagalan cakupan. Jangan mengisi nilai kosong dengan nol untuk menyembunyikan task yang tidak diuji.

Contoh denominator:10 QA defined,8 benar,1 unknown,1 invalid → accuracy0.8, bukan1.0. Parse/schema-valid rate9/10 jika unknown valid; substantive rate8/10.

## 3. Metrik fisik

Sumber utama adalah kepala pose bersama dan turunan prediksinya. GT hanya pada evaluator.

| Metrik | Definisi |
|---|---|
| Global MPJPE | Mean `||P_pred_global−P_gt_global||₂` seluruh joint posisi valid |
| Relative MPJPE | Mean `||(P_pred_j−P_pred_0)−(P_gt_j−P_gt_0)||₂`, j=1..21, kedua GT valid |
| Root position error | Mean `||P_pred_0−P_gt_0||₂` |
| Absolute velocity error | Mean `||V_pred_j−V_gt_j||₂` pada target turunan valid |
| Relative velocity error | Mean `||(V_pred_j−V_pred_0)−(V_gt_j−V_gt_0)||₂`, j=1..21 |
| Root speed MAE | Mean `abs(||V_pred_0||−||V_gt_0||)` |
| Bone length MAE | Mean selisih absolut panjang edge tulang prediksi dan GT |

Posisi: laporkan m dan mm (mm=1000m). Velocity: m/s. Tidak Procrustes pada endpoint utama. Simpan numerator dan count, jangan rata-rata mean per batch dengan ukuran berbeda. Primary aggregation adalah per valid window-endpoint-joint; karena window overlap, observasi sumber bisa muncul berulang. Labeli ini secara eksplisit. Tambahkan macro per subjek untuk menunjukkan ketimpangan jumlah data.

Breakdown per joint, subject, dan task interval6..63. `ZeroVelocity` memberi vector0 pada semua sensor endpoint, dinilai dengan mask target yang sama; tidak mempunyai metrik posisi. Nonfinite prediksi bukan alasan mengurangi count: tandai evaluasi fisik gagal dan laporkan failure rate; hasil finite-subset hanya diagnostik berlabel subset, bukan metrik utama yang sah.

## 4. Matriks ablation dan diagnosis

| Kontras | Interpretasi yang boleh |
|---|---|
| R−S | Manfaat gabungan operasi global/lokal dan relasi eksplisit dengan kapasitas sama |
| F−S | Pengaruh pemisahan node global/lokal |
| R−F | Pengaruh deskriptor edge eksplisit ketika node sudah difaktorkan |
| E−S | Pengaruh edge eksplisit dengan node absolut |
| `(R−F)−(E−S)` | Interaksi kedua komponen |
| R−M | Perubahan kualitas sistem keseluruhan dengan tambahan state dan projector |
| R−T; R−Maj | Manfaat informasi sensor dibanding prior tertentu |
| R vs Rule | Alignment bahasa dibanding kalkulasi task langsung dari prediksi fisik |
| Probe(Y) vs Probe(Z_R) | Keterbacaan task oleh keluarga probe yang ditetapkan |

Kontras selain R−S merupakan sekunder; jangan memilih p-value terendah sebagai temuan primer. Sistem tidak harus menang pada setiap task. Periksa apakah manfaat root mengorbankan arms/legs atau sebaliknya.

Diagnosis opsional pada **validation**: jalankan Rule dengan GT yang sah untuk memeriksa generator label/parser (harus cocok pada defined), lalu jalankan projector terlatih dengan state GT sebagai pemeriksaan sensitivitas terhadap galat pose. Jalur GT tersebut diberi label oracle diagnostic, tidak dicampur dengan sistem radar dan tidak menjadi angka performa utama. Oracle distribution shift dapat membuat hasil memburuk, sehingga bukan upper bound matematis.

## 5. Prior baseline

TrainMajority menghitung kelas terbanyak **reference-defined train** per root/arms/legs. Tie memakai urutan kelas pada prompt. Secondary panel tetap memakai prediksi majority yang sama, tanpa mencari majority unknown terpisah. Kontrol T memakai Qwen pretrained yang sama, tidak dilatih pada QA; prompt dan parser sama kecuali radar block kosong sesuai02.

Tidak membuat baseline random yang kebetulan terlihat lemah sebagai satu-satunya pembanding. Maj,T,Rule,M,S wajib tersedia untuk menafsirkan R.

## 6. Donor sensor

Tujuan: pertanyaan tetap, bukti radar diganti, lalu periksa apakah kedua jawaban cocok dengan acuan masing-masing sumber. Ini penggantian rekaman offline, bukan intervensi kausal pada gerak manusia.

### 6.1 Syarat kandidat

Untuk pasangan window a,b dan task/part sama:

- Subjek dan recording berbeda, split sama, activity ID sama hanya untuk matching offline.
- Keduanya reference-defined dengan kelas berbeda dan support≥44/58.
- Joint map, coordinate convention, time_source, label policy, interval6..63 identik.
- Sensor frame mask sama, yaitu64 frame valid; mask padding point tidak harus sama karena jumlah deteksi merupakan bagian bukti radar.
- Relative duration difference `abs(d_a−d_b)/((d_a+d_b)/2)≤0.05`.
- Dengan waktu relatif τ, maksimum `abs(τ_a[t]−τ_b[t])≤0.25*median(concat(diff(t_a),diff(t_b)))` untuk seluruh64 timestamp.
- Window tidak dipakai kembali dalam pasangan lain di seluruh panel donor, termasuk task berbeda. Tidak memilih pasangan menurut prediksi model.

### 6.2 Algoritme deterministik

Urutkan ID subjek test menurut SHA-256 `donor/42/<id>`, pasangkan berurutan menjadi blok subjek yang saling lepas. Subject ganjil terakhir tidak dipasangkan dan dicatat. Di tiap blok, enumerasi kandidat yang memenuhi syarat; sort tuple `(hash(task,part,window_a,window_b),task,part,uid_a,uid_b)`. Ambil greedy jika kedua window belum pernah dipakai. Simpan jumlah kandidat dan alasan penolakan **sebelum** melihat prediksi.

Empat subjek test memberi maksimum dua blok independen; hasil donor primary split karena itu deskriptif. Jangan membuat banyak pasangan window seolah-olah menambah banyak pasangan subjek independen.

### 6.3 Menjaga prompt tetap

Prefix teks waktu untuk kedua sisi memakai **grid kanonik pasangan** `τ_pair=(τ_a+τ_b)/2`, diformat empat desimal, dan pertanyaan yang sama. Sensor encoder/turunan tetap memakai timestamp asli masing-masing. Dengan toleransi grid di atas, teks waktu adalah informasi pendekatan yang sama untuk pasangan; catat galat grid maksimum. Hanya64 embedding sensor yang berubah pada dua panggilan Qwen. Hasil donor ini dievaluasi ulang khusus panel donor, tidak disalin dari prediksi QA biasa yang menggunakan teks timestamp masing-masing.

Menggunakan masing-masing grid waktu sebagai teks pada donor akan mengubah dua masukan sekaligus. Karena itu, validasi hash seluruh token teks kedua sisi identik merupakan syarat donor. Tidak menyalin label/aktivitas pada prompt pasangan.

### 6.4 Metrik

`PairBothCorrect=mean(1[pred_a==true_a AND pred_b==true_b])` pada semua pasangan eligible. Invalid/unknown/missing tetap salah. Laporkan accuracy sisi A/B, both-parse-valid, both-correct, only-A, only-B, neither, serta matriks transisi true/pred. Pergantian jawaban tanpa benar pada kedua sisi tidak dihitung sebagai grounding berhasil.

Bandingkan R dengan S,M,T,Maj pada daftar pasangan sama. Laporkan coverage per activity/task/subjek dan bias seleksi donor. Tidak menggeneralisasi hasil pasangan discordant ini ke seluruh distribusi tanpa keterangan.

## 7. Ketidakpastian dan seed

Primary QA: percentile cluster bootstrap95%,5.000 draw, seed42. Unit resampling adalah subjek test dengan replacement. Setiap kali satu subjek terambil, seluruh QA-nya ikut; jika terambil dua kali bobot count-nya menjadi dua. Hitung ulang confusion/F1 dari count tertimbang. Kondisi yang dibandingkan menggunakan draw subjek yang sama; CI selisih dihitung langsung dari Δ per draw, bukan selisih ujung dua CI terpisah.

Untuk tiga seed, hitung metrik per seed terlebih dahulu. Endpoint gabungan adalah mean Δ tiga seed. Pada bootstrap, gunakan draw subjek yang sama untuk seluruh kondisi/seed dan rata-ratakan Δ seed dalam draw. Interval ini merefleksikan variasi subjek dengan checkpoint yang tersedia; bukan estimasi lengkap ketidakpastian training. Laporkan mean, standard deviation sample antar-seed, serta seluruh nilai seed. Jangan menyatukan prediksi tiga seed sebagai tiga kali jumlah subjek.

Kurang dari dua subjek: tidak ada CI. Dua sampai empat: CI diagnostik dengan keterbatasan eksplisit. Draw tanpa satu task defined menghasilkan nilai null; laporkan jumlah draw tidak terdefinisi, tidak mengisi nol. Jika >5% draw tidak terdefinisi, CI hanya dilabeli diagnostic incomplete-support, tanpa kesimpulan stabil.

Donor menggunakan bootstrap blok pasangan subjek, bukan window pair. Kurang dari lima blok: tampilkan angka deskriptif dan jumlah blok; tidak menampilkan klaim inferensial donor. Lima fold opsional meningkatkan cakupan out-of-fold, tetapi estimasi dari checkpoint/fold yang berbagi data train tetap tidak independen sepenuhnya.

## 8. Penguncian test

`test.lock.json` memuat hash paket spesifikasi, resolved config, split/windows/targets/QA/label policy, normalizer, donor policy, prompt+parser+metric registry, semua checkpoint terpilih, model/tokenizer commits, precision, generation settings, bootstrap policy, dan exposure ledger. Kunci semua seed/kondisi yang direncanakan sebelum evaluasi final.

Pembuatan daftar pasangan donor test berdasarkan label test dilakukan evaluator setelah lock dengan policy deterministik; tidak digunakan untuk mengubah policy. Hash daftar donor hasilnya ditambahkan sebagai artefak evaluasi, bukan input pemilihan model. Test evaluator menolak checkpoint yang tidak ada dalam lock.

## 9. Laporan hasil

Tabel utama: condition, seed, S_atomic, F1 root/arms/legs, accuracy, secondary4class, valid rate, trainable params, total tokens, peak VRAM, latency. Tabel selisih: R−S primer, kontras faktorial sekunder. Sertakan metrik fisik bersama, Rule, Probe, confusion, class support, reference coverage, donor coverage, serta plot per-subjek.

Kalimat kesimpulan harus mengikuti bukti: misalnya "R memberi Δ sebesar ... pada empat subjek test dengan ...", bukan "LLM memahami fisika". Jika kualitas lebih rendah tetapi biaya lebih kecil, laporkan tradeoff terukur. Jika membandingkan backbone, jangan menyimpulkan efek ukuran parameter saja karena keluarga/pretraining/tokenizer juga berubah.
