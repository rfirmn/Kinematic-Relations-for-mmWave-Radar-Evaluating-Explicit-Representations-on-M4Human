# 08 — Perangkat, resource, dan operasional

## 1. Perangkat sasaran

Informasi perangkat mengikuti [proposal sistem M4Human](../../skripsi/proposal_sistem_penelitian_m4human_h64.md): PC Windows, NVIDIA GeForce RTX3060 dengan VRAM12GB, AMD Ryzen7 7700 (8 core/16 thread), RAM16GB. Kapasitas disk bebas, versi driver/CUDA aktual, dan waktu training belum diasumsikan tersedia; ukur dan simpan dalam environment report.

Python3.12 memakai `.venv` atau `uv`. Operasi jaringan model/dataset dilakukan saat persiapan sesuai akses pengguna; training memakai snapshot lokal yang dipin. Model tidak dipilih lewat alias yang dapat berubah tanpa commit hash.

## 2. Dependensi

Dependensi inti: PyTorch CUDA dari wheel resmi yang cocok dengan driver, Transformers dengan dukungan Qwen2, Accelerate untuk low-memory loading, NumPy, PyYAML, safetensors, lmdb, msgpack bila metadata memerlukannya, psutil, matplotlib, dan pytest. Versi persis dipin dalam lock setelah installation/smoke test Windows; tidak menggunakan `latest` di resolved environment. Simpan `uv.lock` atau export lock ekuivalen dan hash-nya bersama run.

Seluruh encoder, attention, pooling, dan derivative memakai operasi PyTorch/NumPy standar. Gunakan PyTorch SDPA bila tersedia; fallback matematis harus lolos equivalence. Tidak membutuhkan pointnet2_ops, knn_cuda, FlashAttention eksternal, atau kompilasi C++/CUDA khusus. Jangan menambah dependency native untuk mengejar kecepatan tanpa keputusan terpisah yang terdokumentasi.

## 3. Strategi memori utama

- Training fisik dan alignment dijalankan sebagai proses terpisah. Qwen tidak dimuat ketika melatih encoder.
- Alignment menggunakan cache Y/P/V: encoder tidak perlu menetap di VRAM bersama Qwen.
- Mulai microbatch1 dengan accumulation16. Naikkan ke2/4 hanya setelah profil, sambil mempertahankan effective batch16 dan stream QA sama.
- Mulai DataLoader workers0, CPU threads4, prefetch factor2 hanya jika workers>0. Uji workers2 setelah spawn reader lulus; pin memory mengikuti profil.
- Cache dibaca mmap per shard, tidak seluruh dataset menjadi tensor RAM. Tidak menyimpan full attention maps saat training.
- Aktifkan gradient checkpointing non-reentrant pada blok besar bila diperlukan. Frozen Qwen masih perlu jalur aktivasi untuk gradient input pada alignment.
- `use_cache=False` saat training; jangan menahan KV cache lintas batch. Evaluasi microbatch1 dapat memakai KV cache setelah verifikasi.
- AMP bf16 dipilih setelah `torch.cuda.is_bf16_supported()` dan backward sanity; jika tidak, fp16 dengan GradScaler. FP32 dipakai loss, descriptor norms, dan operasi kinematik; float64 pada solve timestamp.
- Saat alignment, hitung LM-head vocabulary logits hanya pada posisi yang memprediksi jawaban menurut04 setelah equivalence test. Prefix tetap diproses penuh oleh Qwen; penghematan ini hanya menghindari logits yang tidak masuk loss.

Parameter projector Qwen1.5B sekitar3 juta. Dalam AdamW, bobot+grad+dua momentum float32 membutuhkan sekitar16 byte/parameter, sekitar48MB sebelum aktivasi. Bobot Qwen sekitar1.5 miliar pada dua byte/parameter berada pada orde3GB; ini **bukan** total memori training. Aktivasi, logits vocabulary, kernel workspace, buffer dan fragmentation tetap harus diukur.

## 4. Profil wajib

Profil menggunakan train/validation window dengan jumlah point mendekati maksimum dan prompt dengan token terpanjang. Jalankan10 langkah warm-up, lalu30 optimizer steps yang mencakup seluruh accumulation; jangan hanya forward. Sediakan juga30 generasi QA dan sampel cold/warm disk cache.

Catat: peak allocated/reserved CUDA, free VRAM minimum, process RSS/peak, system available RAM minimum, GPU utilization, I/O throughput, windows/s atau QA/s, time/optimizer-step, sequence token min/median/max, latency p50/p95, dtype, worker/batch/accumulation, model cache lokasi, dan GPU/device identity.

Default guard: free VRAM minimum1.024MiB dan available RAM minimum2.048MiB selama profil. Tidak ada OOM, NaN/Inf, swap thrashing, atau hang worker. Sisakan disk untuk raw artifacts+dua checkpoint+cache+laporan, dengan reserve sekurang-kurangnya10GiB setelah estimasi kebutuhan. Ukuran raw dataset tidak diasumsikan harus muat dalam RAM.

Jika gagal: kurangi microbatch dan tingkatkan accumulation agar efektif tetap16; workers menjadi0; gunakan checkpointing; hentikan proses lain milik eksperimen yang memang sudah selesai. Jika microbatch1 tetap gagal, laporkan resource gate gagal. Tidak mengurangi64 frame,150 titik, dimensi model, atau token hanya pada satu kondisi untuk membuat hasil tampak sebanding.

## 5. Qwen3-8B sebagai ekstensi

Checkpoint yang dimaksud adalah `Qwen/Qwen3-8B`, dengan dimensi hidden4096 yang diverifikasi dari config. Pada mode non-thinking, gunakan chat template dengan `enable_thinking=False`; tidak memberi budget reasoning tersembunyi yang berbeda antar-kondisi. Pin model/tokenizer/template untuk blok eksperimen ini. [Model card resmi](https://huggingface.co/Qwen/Qwen3-8B).

Bobot8B pada dua byte/parameter saja berada sekitar16GB, sehingga tidak muat sepenuhnya pada RTX3060 12GB. Recipe bf16/fp16 utama8B memerlukan perangkat dengan memori memadai yang profilnya lulus. Jangan menjanjikan training8B standar pada3060.

Quantization/offload adalah **kondisi eksperimen tambahan**, bukan fallback tersembunyi. Untuk membandingkan ukuran backbone pada precision berbeda, laporkan perubahan quantization/perangkat dan batasi interpretasi; lebih bersih memakai precision setara pada kedua backbone. Kemampuan backward terhadap input embedding melalui backend quantized harus diuji. Jika backend memerlukan extension native yang belum disetujui, ekstensi tersebut tidak dimasukkan ke recipe pure-PyTorch utama.

Tidak perlu8B untuk menjawab H-B pada1.5B. Klaim model kecil kompetitif terhadap8B baru dapat dibuat setelah kedua model dievaluasi pada panel sama dengan biaya dan perbedaan recipe transparan.

## 6. Estimasi waktu dan penyimpanan

Setelah manifest tersedia, hitung jumlah train window W dan eligible QA Q. Estimasi fisik = epoch aktual × ceil(W/16) × waktu optimizer-step terukur + validation. Alignment = epoch aktual × ceil(Q/16) × waktu step + biaya generasi seluruh validation per epoch. Kalikan jumlah seed/kondisi yang direncanakan, tambahkan ekstraksi dan evaluation.

Jangan menulis estimasi jam berdasarkan GPU name saja. Laporkan time budget hasil profil sebelum memulai matriks15 projector. Pilot satu seed memberi estimasi, bukan jaminan durasi final. Validation generation dapat mendominasi waktu sehingga harus masuk estimasi.

Cache Y/P/V float32 sekitar99KB/window ditambah waktu/mask/index. Cache dibagi antarkondisi. Checkpoint projector dan optimizer jauh lebih kecil daripada Qwen; gunakan satu snapshot Qwen yang diverifikasi. Prediksi mentah disimpan teks/JSONL; tidak perlu menyimpan logits vocabulary seluruh sequence untuk semua QA.

## 7. Risiko operasional dan respons

| Gejala | Diagnosis pertama | Respons yang menjaga protokol |
|---|---|---|
| Root baik, velocity buruk | Unit waktu, pairing GT, D1 noise | Audit07, laporkan galat; jangan ubah threshold dengan test |
| R baik pada loss tetapi JSON gagal | Labels shift, chat template, splicing | Fixture language dan validation generation |
| R unggul M tetapi sama S | State/capacity menjelaskan manfaat | Laporkan hipotesis relasi belum didukung |
| Unknown mendominasi | Boundary/support/label policy atau prediksi | Coverage/reference vs prediction dipisahkan |
| Prefix sama memberi output berbeda | Dropout, RNG, cache, kernel | Eval state, deterministic settings, log toleransi |
| RAM habis | Preload, process spawn, model double-load | Mmap, workers0, low-memory loading, profile ulang |
| Run terputus | Checkpoint atomic dan RNG | Resume hanya provenance cocok |

## 8. Pelaporan yang dapat diaudit

Laporan menyebut apakah berjalan di CPU fixture, Windows/CUDA sasaran, atau perangkat tambahan. Sertakan environment lock, acceptance gates, parameter count, selected epoch, optimizer steps, exposures, seluruh counts, model/model-tokenizer hashes, dan link artefak. Menjalankan spesifikasi atau sanity tidak boleh ditulis sebagai hasil eksperimen kompetitif.
