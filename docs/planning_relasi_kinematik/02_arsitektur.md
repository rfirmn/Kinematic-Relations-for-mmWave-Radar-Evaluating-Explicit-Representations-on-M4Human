# 02 — Spesifikasi arsitektur

## 1. Notasi dan batas model

`B` = microbatch window; `T=64`; `N=150`; `J=22`; `G=29` edge; `D=256`; `D_n=128`; `d` = dimensi embedding model bahasa. Dimensi d dibaca dari konfigurasi model, dengan assert d=1536 untuk Qwen2.5-1.5B-Instruct. Posisi/kecepatan komputasi fisik menggunakan float32; timestamp dan solusi turunan menggunakan float64. Mixed precision hanya pada layer jaringan yang aman.

| Nama tensor | Shape | Arti |
|---|---|---|
| X | `[B,64,150,4]` | XYZ meter terpusat c dan intensity ternormalisasi |
| point_mask | `[B,64,150]` bool | Titik benar-benar teramati |
| time_s | `[B,64]` float64 | Timestamp sensor |
| C | `[B,3]` | Centroid awal; z=0 |
| Y | `[B,64,256]` | Fitur radar setelah temporal encoder |
| P | `[B,64,22,3]` | Prediksi posisi di radar frame dengan offset c dikurangkan |
| V | `[B,64,22,3]` | Turunan P; enam frame awal nol secara komputasional |
| velocity_available | `[B,64]` bool | False pada indeks 0–5, True pada 6–63 |
| A | `[B,64,22,7]` | Fitur node, tergantung kondisi |
| D_edge | `[B,64,29,14]` | Fitur pasangan, tergantung kondisi |
| Z | `[B,64,256]` | Fitur fusion projector |
| H | `[B,64,d]` | Embedding sensor untuk Qwen |

Tidak ada tensor target atau status anotasi dalam signature inferensi. `velocity_available` diturunkan dari dukungan **sensor dan indeks waktu**, tidak dari GT. Semua primary window lengkap; anotasi yang hilang tidak mengubah bentuk tensor atau attention mask sensor.

Parameter jaringan trainable disimpan float32; autocast mengatur komputasi layer. Qwen frozen dimuat dalam dtype16-bit yang lulus profil. Sebelum concatenation teks/sensor, cast H ke dtype embedding Qwen melalui operasi `.to(dtype)` yang tetap mempertahankan autograd, tanpa detach. Positional phase RoPE dihitung minimal float32 dari selisih waktu float64; loss/state fisik tidak dihitung langsung dalam fp16.

## 2. Encoder radar bersama

### 2.1 Embedding titik

- Feature MLP: `Linear(4,256) → GELU → Linear(256,256)`.
- Positional MLP: `Linear(3,256) → GELU → Linear(256,256)`, input XYZ dari X.
- Jumlahkan kedua hasil; padded point di-zero-kan.
- Semua linear memiliki bias. LayerNorm memakai `eps=1e-5`. GELU exact, dropout utama 0.1 pada encoder.

### 2.2 Tiga blok spasial

Reshape batch spasial menjadi `[B*T,N,256]`. Setiap blok:

1. `U = X + Dropout(MHA(LN(X)))`.
2. `X_next = U + Dropout(Linear(1024,256)(Dropout(GELU(Linear(256,1024)(LN(U))))))`.

MHA memiliki empat head, lebar head64, Q/K/V dan output linear256. Attention point bersifat bidirectional. Key/value padding di-mask sebelum softmax. Query padded output di-zero-kan sesudah setiap residual. Minimal delapan point menjamin tidak ada baris dengan semua key masked. Tidak memakai BatchNorm yang dapat memasukkan padding ke statistik.

### 2.3 Pooling frame

Satu query trainable `[1,256]` dibroadcast ke setiap frame. Terapkan MHA empat head, Q=query, K=V=LN(output spasial), dengan key padding mask. Hasil `f_t = LN(query + attention_output)` menjadi satu token256 per frame. Ini tidak memakai joint GT untuk menentukan bagian tubuh.

### 2.4 Delapan blok temporal

Masukan `[B,64,256]`. Struktur pre-LN dan FFN sama dengan blok spasial. Attention bidirectional di dalam window. RoPE pada Q dan K menggunakan pasangan dimensi genap-ganjil pada setiap head64:

`phase(t,k) = u_t / 10000^(2k/64)`, `k=0..31`, `u_t=12*(time_s[t]−time_s[0])`.

Untuk pasangan `(a,b)`, hasil rotasi `(a cosφ−b sinφ, a sinφ+b cosφ)`. Waktu nominal seragam menghasilkan u_t=t. RoPE hanya pada Q/K, tidak pada V. Final LayerNorm256 menghasilkan Y. Waktu nyata digunakan agar interval tidak dianggap seragam ketika timestamp bervariasi.

### 2.5 Kepala pose

`LN256 → Linear(256,256) → GELU → Linear(256,66)`, reshape `[B,64,22,3]`. Prediksi langsung seluruh joint absolut setelah pengurangan c; tidak memaksa panjang tulang tetap, tidak Procrustes, tidak menyediakan residual dari GT. Posisi global untuk pelaporan = `P+C`.

Encoder dan kepala pose dilatih bersama pada objective 04. Setelah checkpoint dipilih, keduanya `eval()` dan `requires_grad=False` untuk seluruh tahap alignment/probe. Dropout encoder tidak aktif pada ekstraksi fitur.

## 3. State kinematik dari radar

Untuk setiap joint dan endpoint e≥6, terapkan operator turunan tujuh titik pada P, persis seperti 03. Prediksi P selalu ada pada primary sensor window, sehingga tersedia 58 endpoint V meskipun beberapa target GT hilang. Kecepatan tidak berasal dari intensity. Turunan dilakukan setelah merekonstruksi output P dalam meter, bukan terhadap descriptor yang sudah distandardisasi.

Definisikan:

`r_t=P[t,0]`, `v_root=V[t,0]`, `q[t,j]=P[t,j]−r_t`, `w[t,j]=V[t,j]−v_root`.

`q[:,0]=w[:,0]=0` menurut definisi. Untuk e<6, semua velocity input dan fitur turunannya diisi nol **setelah normalisasi**; embedding availability memberi tahu projector bahwa nilai itu tidak teramati. Jangan menafsirkan warm-up nol sebagai tubuh diam.

State dibentuk dari prediksi fisik yang sama pada seluruh kondisi. Pengurangan centroid c tidak menghilangkan gerak root sebab c konstan sepanjang window. Root di sini pelvis, bukan center of mass.

## 4. Deskriptor node

Urutan tujuh kanal adalah `[position_x,position_y,position_z,velocity_x,velocity_y,velocity_z,speed]`.

- **S dan E:** `A_j=[P_j,V_j,||V_j||]` untuk seluruh joint.
- **F dan R:** node0 tetap `[r,v_root,||v_root||]`; node1..21 menggunakan `[q_j,w_j,||w_j||]`.

Root global tetap ada pada F/R; tidak semua fitur dibuat invariant terhadap translasi tubuh. Identitas anatomi kiri/kanan tidak bergantung orientasi tubuh menghadap radar.

## 5. Deskriptor edge

Urutan 29 edge mengikuti 01: 21 tulang lalu delapan bilateral. Orientasi selalu sama, tanpa edge balik tambahan.

### 5.1 Pasangan mentah pada S dan F

Untuk edge i→j, `D_ij = concat(a_abs_i,a_abs_j)` dengan `a_abs=[P,V,||V||]`, total14 kanal. F tetap memakai edge absolut ini agar perbandingan F−S hanya mengubah node global/lokal.

### 5.2 Relasi eksplisit pada E dan R

| Kanal | Isi | Unit |
|---|---|---|
| 0–2 | `d_ij=P_j−P_i` | m |
| 3–5 | `u_ij=V_j−V_i` | m/s |
| 6 | `ell=||d_ij||` | m |
| 7 | `relative_speed=||u_ij||` | m/s |
| 8 | `s_i=||V_i−v_root||` | m/s |
| 9 | `s_j=||V_j−v_root||` | m/s |
| 10 | `alignment=dot(d_ij,u_ij)/(ell*||u_ij||)` | tanpa unit |
| 11–13 | `center_rel=(q_i+q_j)/2` | m |

Alignment ditetapkan nol bila ell<1e-6 m atau ||u||<1e-6 m/s; selain itu clamp hasil ke [-1,1] untuk galat numerik. Alignment menyatakan apakah perpindahan relatif searah dengan kecepatan relatif, bukan sudut anatomi atau gaya.

Warm-up pada edge eksplisit: kanal 3–5 dan 7–10 tidak tersedia dan di-zero-kan setelah normalisasi. Kanal posisi, panjang, dan pusat relatif tetap tersedia. Pada edge mentah, kanal 3–6 dan 10–13 adalah kanal velocity/speed yang di-zero-kan.

Tidak menghitung kelas, threshold label, tren keseluruhan, median kelompok lengan/kaki, atau jawaban QA dalam projector. Descriptor memuat pengukuran kontinu per endpoint. Rule menggunakan agregasi task secara terpisah sebagai pembanding.

## 6. Normalisasi descriptor

Gunakan rata-rata dan simpangan baku per identitas node/edge dan per kanal dari prediksi cache **train**, dengan bobot sama per window/frame. Welford float64; hanya endpoint yang tersedia untuk kanal velocity. Simpangan baku di-floor `1e-6`; transform `(x−mean)/std`, clip [-10,10]. Parameter frozen untuk val/test.

S/F/E/R memiliki normalizer descriptor masing-masing; semua diturunkan dari checkpoint fisik sama. Bukan normalizer dari GT. File memuat condition, source checkpoint hash, train manifest hash, count per kanal, mean/std, availability map. Jangan menggunakan statistik batch atau seluruh split.

## 7. Projector terstruktur S/F/E/R

Keempat kondisi memakai **modul dan jumlah parameter trainable identik**. Semua parameter berikut termasuk projector dan hanya dilatih pada tahap alignment bahasa:

1. Node MLP bersama: `Linear(7,128) → GELU → Linear(128,128)`. Tambahkan embedding joint `[22,128]` dan embedding availability velocity `[2,128]`, lalu LayerNorm128.
2. Edge MLP bersama: `Linear(14,128) → GELU → Linear(128,128)`. Tambahkan embedding edge ID `[29,128]`, tipe `[2,128]` (bone/bilateral), dan availability `[2,128]` terpisah dari node, lalu LayerNorm128.
3. Pool node per frame: query trainable128, skor `dot(query,h_j)/sqrt(128)`, softmax atas22, weighted sum menghasilkan n_t128.
4. Pool edge analog atas29 dengan query128 lain, menghasilkan e_t128. Edge tidak digandakan berdasarkan pertanyaan.
5. Fusion: `concat(Y_t,n_t,e_t)`512 → Linear512→256 → GELU → Linear256→256; tambah residual Y_t; LayerNorm256 menghasilkan Z_t.
6. Pemetaan bahasa: `Linear(256,d) → LayerNorm(d) → GELU → Linear(d,d) → LayerNorm(d)` menghasilkan H_t.

Dropout projector=0.0 untuk membatasi perbedaan acak pada studi representasi. Linear memakai default Kaiming-uniform PyTorch beserta bias; embedding dan query `Normal(0,0.02)`; LayerNorm weight1/bias0. Seed dan urutan konstruksi modul sama pada S/F/E/R. Simpan hash parameter inisialisasi agar kesetaraan dapat diperiksa.

Tidak ada Q-Former, attention tambahan antarframe, token query dari pertanyaan, atau loss fisik tambahan yang hanya dimiliki R. Relasi temporal berasal dari Y dan V; perhatian bahasa menerima urutan64 yang utuh. Projector tidak menghasilkan 22 atau29 token bahasa per frame: pooling dilakukan sebelum pemetaan bahasa sehingga anggaran tetap64.

## 8. Pembanding M

Gunakan Y langsung ke pemetaan bahasa pada langkah7.6. Tidak membuka kepala pose atau menambahkan P/V pada jalur M. Encoder M tetap memperoleh supervisi fisik yang sama saat pretraining. Bobot awal pemetaan bahasa sama dengan bagian yang bersesuaian pada S/F/E/R untuk seed yang sama. Kapasitas berbeda dicatat, bukan disamarkan.

## 9. Antarmuka Qwen dan token splicing

Qwen menggunakan embedding dan LM head pretrained, tanpa menambah vocabulary dan tanpa LoRA. Gunakan chat template tokenizer yang dipin. Buat message user dengan satu marker teks internal `[[RADAR_BLOCK]]`; marker tidak pernah diteruskan sebagai kata kepada model.

Prosedur pasti:

1. Render system/user sesuai03 melalui chat template dengan `tokenize=False, add_generation_prompt=True`, tanpa isi jawaban. Hasil harus sudah berakhir pada pembuka respons assistant yang sah.
2. Assert string marker muncul tepat sekali. Pisahkan string menjadi bagian sebelum dan sesudah marker. Tokenize kedua bagian secara terpisah dengan `add_special_tokens=False`. Pemisahan di batas multimodal ini disengaja dan dipakai identik pada training/generation; tidak mencari marker berdasarkan perkiraan jumlah token atau offset yang memotong token BPE.
3. Di antara kedua bagian, sisipkan64 blok. Blok t berisi embedding token teks `Frame {t:02d} at {elapsed:.4f}s: `, satu embedding sensor H_t, kemudian embedding newline. Setiap segmen teks blok ditokenisasi dengan `add_special_tokens=False`. Elapsed memakai waktu sensor, tanpa GT.
4. H bukan ID vocabulary. Gabungkan embedding pretrained segmen teks dengan H melalui concatenate yang mempertahankan autograd. Untuk training saja, append token jawaban kanonik yang ditokenisasi terpisah dan satu token end-of-turn assistant dari template. Assert token terminasi tersebut adalah satu ID yang terdaftar; jangan menambahkan EOS/BOS kedua. Prefix training sebelum jawaban harus identik dengan prefix generation.
5. Semua prefix, system, user, dan padding diberi labels=-100. Hanya token isi jawaban assistant dan end-of-turn yang memiliki label bahasa. Gunakan logits pada posisi k−1 untuk target di posisi k, tepat sekali; perhitungan CE per contoh pada04 tidak boleh menggandakan shift bawaan model.
6. Posisi model bahasa berurutan menurut sequence hasil splice. Hasil splicing menghasilkan tepat64 embedding sensor. Jumlah keseluruhan token teks/sensor dicatat, batas sequence training3072; input generasi maksimum3040, output maksimum32.
7. Jika batas dilampaui, batch gagal dengan UID dan panjang. Tidak memotong frame atau jawaban. Batas tersebut cukup hanya setelah tokenizer yang dipin diverifikasi, bukan menurut perkiraan karakter.

Pembanding T memakai template pertanyaan sama tetapi mengganti radar block dengan teks tetap `No radar evidence is provided.`; ini kontrol prior, bukan perbandingan token budget yang identik. Untuk S/F/E/R/M, teks frame/waktu dan jumlah embedding identik.

Generasi wajib mengembalikan hanya token baru setelah prefix. Periksa jalur `inputs_embeds`, position IDs, attention mask, padding, dan KV cache dengan equivalence test pada 07. Jangan menganggap API generate otomatis menghitung panjang input dari input_ids dummy.

## 10. Pembekuan dan jalur gradien

| Tahap | Encoder/pose | Descriptor | Projector | Qwen |
|---|---|---|---|---|
| Fisik | Trainable | Turunan differentiable untuk loss | Tidak dimuat | Tidak dimuat |
| Ekstraksi | Frozen eval | Deterministik | Tidak dimuat | Tidak dimuat |
| Alignment | Frozen eval atau cache | Deterministik | Trainable | Frozen eval, autograd input aktif |
| Evaluasi | Frozen eval | Deterministik | Frozen eval | Frozen eval |

`torch.no_grad()` boleh membungkus encoder frozen saat ekstraksi. **Tidak** membungkus forward Qwen ketika alignment: gradien loss harus mengalir melalui operasi Qwen menuju H dan projector. Weight Qwen tidak menerima gradien; embedding H menerima gradien. Gradient checkpointing harus diuji dengan input embedding yang membutuhkan gradien; pilih non-reentrant bila tersedia.

Seluruh inference hanya memerlukan RPC, waktu, kalibrasi statis/normalizer, checkpoint fisik, projector, tokenizer, dan model bahasa. Tidak memerlukan asset body model atau params GT.
