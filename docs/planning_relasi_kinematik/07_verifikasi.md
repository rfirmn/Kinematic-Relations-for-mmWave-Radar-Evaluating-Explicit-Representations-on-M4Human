# 07 — Verifikasi dan kriteria penerimaan

## 1. Tingkat bukti

Pisahkan empat status: **spesifikasi lengkap**, **fixture lulus**, **native Windows/CUDA lulus**, dan **eksperimen ilmiah selesai**. Satu status tidak menyiratkan status berikutnya. Uji laptop/CPU tidak membuktikan kapasitas VRAM, kinerja data nyata, atau kualitas QA.

Training panjang hanya dimulai setelah sanity end-to-end dan resource profile pada perangkat sasaran lulus. Seluruh uji di bawah memakai fixture sintetis atau train/validation; tidak menggunakan test untuk debug model atau definisi label.

## 2. Data dan provenance

| Uji | Fixture | Hasil wajib |
|---|---|---|
| RPC shape/endian | Header valid, salah ndim, byte terpotong, N berlebih | Hanya payload sah diterima |
| Sentinel | Titik nol valid dan sentinel eksplisit | Nol tidak otomatis dibuang; sentinel mengikuti schema |
| Nonfinite | NaN/Inf pada salah satu kanal | Row dibuang; count/reason tercatat |
| Point padding |8 point valid+142 padded | Padding tidak mengubah centroid/statistik/attention |
| Windows | Segmen F=63,64,80 | Counts train0,1,3; val0,1,2 |
| Gap | Delta0.126 s dan delta0.125 s | Yang pertama memutus; yang kedua diterima |
| Annotation missing | Hilangkan satu joint GT | Window UID/phase tetap; hanya target mask berubah |
| Subject split | Satu subjek muncul di dua split | Validator gagal |
| Source isolation | Params/bbox/RGB tak tersedia saat inference fixture | RPC inference tetap berfungsi |
| Hash mismatch | Ubah satu timestamp/checkpoint byte | Cache/lock ditolak |
| Windows spawn | Worker0 dan2 membaca sampel sama | Nilai sama, handle dibuka per proses, sumber tak ditulis |

Uji pembatasan akses dapat memakai reader stub yang langsung gagal jika modalitas selain RPC/metadata sensor dibuka. Ini lebih kuat daripada hanya memeriksa daftar nama argumen.

## 3. Turunan dan label: contoh numerik

Semua fixture memakai timestamp float64, toleransi derivatif1e-8 pada float64 dan1e-4 setelah cast float32.

1. `p(t)=(2,−1,0.5)` → V=(0,0,0) pada semua e≥6, awal6 invalid.
2. `p(t)=(1+0.3t,2−0.2t,0.5+0.1t)` → V=(0.3,−0.2,0.1), baik grid seragam1/12 maupun grid tidak seragam dengan gap≤0.125.
3. `p(t)=(0.5t²,0,0)` → V=(t,0,0), membuktikan pembagian waktu tepat sekali.
4. Hilangkan satu target pada e−2 → endpoint e untuk joint itu invalid; prediksi sensor tetap tersedia. Joint lain tidak ikut invalid tanpa alasan.
5. Posisi target di luar window tidak boleh mengubah enam warm-up endpoint menjadi valid.
6. Root bergerak dengan speed `0.2+0.1t` pada t≥0 → slope0.1, residual0 → speeding_up. Speed `1.2−0.1t` sepanjang5.25 s tetap positif → slowing_down. Speed0.3 konstan → no_overall_trend.
7. Score root0.05 → unknown;0.04 dan0.06 juga unknown. Score0.039999 → no_overall_trend jika residual sah.
8. Relative speed elbow/wrist kiri0.20, kanan0.10 pada semua endpoint → d0.10 → faster_left. Tukar joint kiri/kanan → faster_right. Kedua sisi0.20 → tie.
9. Tambahkan trajectory translasi yang sama ke semua joint: relative velocity/label limb tetap, root speed dapat berubah.
10. Dukungan43 → undefined;44 → dapat defined/unknown sesuai scalar; denominator coverage tetap58.

Uji transformasi di atas adalah uji matematika state/label, bukan klaim bahwa encoder radar invariant atau equivariant terhadap semua perubahan fisik sensor.

## 4. Encoder dan objective

- Shape lengkap sesuai02 untuk B1 dan B2; dtype/mask tidak tertukar.
- Pada eval, permutasi point valid dengan positional coordinates ikut dipermutasi menghasilkan Y/P sama dalam toleransi1e-5 float32 pada fixture N≤150. Tidak menguji efek perubahan subset sampling.
- Mengubah nilai padded point yang dimask menjadi angka besar tidak mengubah output.
- Satu batch hanya memiliki root GT: L_root/L_pos yang sah dihitung; L_local/bone tanpa support tidak membagi nol.
- Numerator/count reduction pada dua microbatch dengan jumlah target berbeda sama dengan perhitungan satu effective batch penuh.
- Backward loss velocity melalui W menghasilkan gradien finite nonzero pada pose head/encoder; perhitungan target tidak menerima gradien.
- Tidak ada ketergantungan dropout encoder pada cache eval; dua ekstraksi sama menghasilkan nilai identik pada konfigurasi deterministik atau toleransi yang dicatat untuk kernel tertentu.
- Overfit delapan train window yang disimpan tetap, tanpa point resampling: L_phys turun sedikitnya50% dari loss awal dalam maksimum500 optimizer steps. Bila gagal, diagnosis optimisasi, masks, skala, dan pairing target sebelum training penuh; jangan memakai ini sebagai bukti akurasi test.

## 5. Relasi dan kontrol kapasitas

- Daftar parent memiliki21 edge, bilateral8, total29; semua indeks0..21, tidak self-edge, orientasi sesuai01.
- Descriptor node7 dan edge14 tepat urutan02; pengujian memakai state yang komponennya berbeda agar pertukaran kanal terlihat.
- S/F menggunakan edge mentah identik bitwise; S/E node absolut identik; F/R node relatif identik; E/R edge eksplisit identik sebelum normalizer.
- Geometry pada warm-up tetap ada, kanal velocity di-zero-kan setelah normalisasi; availability false. Nilai nol tidak berubah menjadi `−mean/std`.
- Alignment untuk panjang atau relative speed nol menghasilkan0, tanpa NaN.
- Normalizer hanya mengakses train. Mengubah nilai validation/test tidak boleh mengubah file normalizer hash.
- S/F/E/R mempunyai parameter count dan shape state_dict identik. Untuk d1536: **3.003.392 parameter projector**. M: **2.761.728**. Tambahan terstruktur241.664. Perbedaan angka berarti detail modul/bias/LayerNorm tidak cocok dengan02.

## 6. Token splicing, freeze, dan generation

1. Terdapat tepat64 posisi sensor, urut frame00..63, dan timestamp teks sesuai source/canonical donor grid.
2. Labels seluruh posisi sensor/system/user/padding=-100. Hanya jawaban/end-of-turn berkontribusi CE.
3. Masukkan marker nol atau dua marker → batch ditolak, bukan menebak indeks. Splicing memakai pemisahan karakter sebelum tokenisasi segmen, sehingga tidak memotong token BPE.
4. Tokenize/detokenize jawaban kanonik mempertahankan enum dan boundary assistant yang benar. Prefix sebelum jawaban training identik dengan prefix generation; boundary tidak ditentukan dengan mencari kata jawaban dalam seluruh prompt. Terminasi assistant ditambahkan tepat sekali.
5. Setelah satu optimizer step alignment: projector berubah; hash encoder, pose, embedding Qwen, dan LM head Qwen tetap. Frozen parameters `.grad is None`; norm grad projector finite dan >0.
6. Ubah jawaban target sementara mempertahankan prefix: loss/grad projector berubah; forward inference tidak menerima jawaban target.
7. Uji jalur inputs_embeds terhadap input_ids biasa ketika seluruh embedding diisi embedding vocabulary asli: logits pada posisi terakhir cocok dengan toleransi dtype. Ini menguji wrapper tanpa mengasumsikan sensor embedding punya ID teks.
8. Bandingkan greedy decode langkah-per-langkah tanpa KV cache dengan jalur generation berkache selama sedikitnya8 token. Token identik pada fixture tanpa tie logits; position IDs dan prefix length sama. Jika tidak identik, perbaiki sebelum benchmark.
9. Sequence berlebih ditolak; tidak truncate atau memotong sensor. Generation terpotong32 token berstatus truncated.
10. Mengubah prompt hanya pada task menghasilkan pertanyaan benar tanpa mengubah H. Tidak ada task-conditioned sensor encoder.
11. CE dan gradient projector dari selected-position LM head sama dengan full-logits causal loss pada fixture kecil (toleransi1e-5 float32 atau1e-3 mixed precision). Per-example mean tidak berubah menjadi mean seluruh token batch.

## 7. Parser, metrik, dan donor

- Terima `{"answer":"tie"}` untuk limb; tolak key tambahan, duplicate answer, array, trailing explanation, Markdown fence, label root pada limb, atau case berbeda.
- Unknown valid secara schema tetapi salah pada reference-defined. Missing output masuk denominator.
- Confusion fixture yang semua prediksi benar menghasilkan F1=1 jika seluruh tiga kelas ada. Jika hanya satu kelas ada dan semuanya benar, macro-F1=1/3.
- Data tanpa satu task defined menghasilkan S_atomic=null, bukan0 atau skor dua-task.
- Delta kondisi identik menghasilkan semua bootstrap Δ=0; penggunaan draw subjek harus sama antar-kondisi.
- Menggandakan window pada satu subjek tidak menambah jumlah cluster; CI engine menerima multiplicity subjek secara benar.
- Donor tidak boleh memakai subjek/recording sama, label sama, window reused, atau gap/grid tidak cocok. Algoritme menghasilkan pairing identik tanpa bergantung urutan input file.
- Teks token donor a/b harus identik, termasuk waktu; yang berbeda hanya sensor embeddings. Hash divergence menggagalkan donor pair.
- Jika satu sisi benar dan sisi lain salah, PairBothCorrect=0. Pasangan tanpa output bukan dikeluarkan.
- Dua blok donor menghasilkan laporan deskriptif, tanpa CI inferensial.

## 8. Gerbang kelulusan

| Gate | Bukti | Tahap yang boleh dilanjutkan |
|---|---|---|
| G0 Data authority | Inventory/coordinate/time/split/exposure lengkap | Pembentukan target/window |
| G1 Targets | Numerical tests + label/coverage audit train/val | Training fisik |
| G2 Physical pipeline | Backward, mask, overfit, profiling native CUDA | Training fisik penuh |
| G3 Cache/projector | Hash/equivalence, descriptor, parameter equality | Alignment sanity |
| G4 Language | Freeze/grad/splice/parser/generation/resource lulus | Alignment penuh |
| G5 Experiment lock | Semua checkpoint/recipe/policy/metric dipin | Evaluasi test |
| G6 Report | Raw predictions+counts+CI+resource+provenance lengkap | Interpretasi ilmiah |

Gate lulus disimpan dalam JSON dengan test IDs, hasil, environment, tanggal, dan artifact hashes. Fixture yang memakai data sintetis harus ditandai demikian. Tidak ada threshold QA test yang dipakai untuk menentukan apakah software valid: hasil rendah tetap dilaporkan setelah seluruh kontrak benar.
