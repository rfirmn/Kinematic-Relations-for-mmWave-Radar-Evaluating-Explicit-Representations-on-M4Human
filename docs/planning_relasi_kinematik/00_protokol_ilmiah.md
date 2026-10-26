# 00 — Protokol ilmiah

## 1. Masalah dan hipotesis

Posisi tubuh absolut mencampur perpindahan seluruh tubuh dan artikulasi anggota tubuh. Contohnya, translasi pelvis yang besar dapat menaikkan kelajuan kedua pergelangan meskipun hanya satu lengan bergerak terhadap badan. Pemetaan langsung fitur radar ke model bahasa mengharuskan tahap alignment menemukan pemisahan ini dari supervisi jawaban. Representasi relasional menyediakan pemisahan tersebut sebelum embedding masuk ke model bahasa.

**H-B:** pada sumber radar, estimator fisik, backbone bahasa, jumlah embedding sensor, kapasitas projector, dan paparan QA yang sama, penyajian gerak global, artikulasi relatif, serta relasi antarbagian secara eksplisit meningkatkan ketepatan QA dibanding penyajian state absolut dan pasangan state mentah.

Mekanisme yang diuji adalah **bentuk penyajian bukti kinematik kepada model bahasa**. Penelitian tidak menganggap pose, supervisi kecepatan, graph tubuh, atau Qwen berukuran kecil sebagai kebaruan tunggal. Representasi eksplisit dapat pula memperburuk hasil karena galat pose diperbesar oleh turunan; kemungkinan ini merupakan prediksi tandingan yang harus diuji.

### Pertanyaan penelitian

| ID | Pertanyaan | Bukti yang diperlukan |
|---|---|---|
| RQ1 | Apakah representasi eksplisit membantu QA dengan kontrol kapasitas dan akses state yang ketat? | `R − S`, skor per tugas, CI berkelompok, variasi seed |
| RQ2 | Apakah pemisahan root dan relasi geometris memberi kontribusi yang dapat dibedakan? | Faktorial `S,F,E,R`, terutama `F−S`, `R−F`, dan interaksi |
| RQ3 | Apakah jawaban mengikuti sensor ketika sumber radar diganti? | Correctness kedua sisi donor, kontrol tanpa sensor, parser coverage |
| RQ4 | Seberapa besar galat fisik dan biaya komputasi membatasi hasil? | MPJPE/velocity, rule berbasis prediksi, probe, latency/memori |

## 2. Fondasi metode

mmMind menggunakan pretraining pose untuk membentuk fitur radar kontinu, encoder spasial-temporal, dan projector menuju LLM. Konfigurasi encoder yang diadopsi berlebar 256 dengan tiga blok spasial, delapan blok temporal, empat head, serta FFN 1.024. Paper memakai 75 frame, 150 titik, XYZ dan kecepatan radial, serta pose 17 joint. Tahap instruksi akhirnya melatih projector dan LoRA Qwen3-8B. Penelitian ini menggunakan fondasi encoder tersebut, sedangkan protokol lokal memakai 64 frame M4Human, intensity, 22 joint, dan Qwen sepenuhnya beku. Karena perbedaan tersebut, pembanding lokal disebut **mmMind-style M4Human**, bukan reproduksi persis hasil publikasi. [Sumber mmMind](https://arxiv.org/html/2608.04127v2).

M4Human menyediakan RPC, anotasi tubuh, 20 subjek, 50 aktivitas, sekitar 661 ribu frame, dan 999 sekuens. Laju nominalnya sekitar 12 Hz. Benchmark HAR-nya memakai klip skeleton 64 frame; itu mendukung pemilihan skala observasi sekitar lima detik, tetapi tidak membuktikan optimalitas panjang tersebut untuk QA. Statistik paket penelitian harus dihitung dari inventaris, bukan disalin dari statistik publikasi. [Sumber M4Human](https://arxiv.org/html/2512.12378v3).

Konsep relasi antarelemen memiliki preseden, misalnya Relation Networks. Di sini operasi yang diuji ditentukan oleh kinematika: pengurangan root, selisih posisi/kecepatan, panjang, dan keselarasan perubahan. Tidak ada klaim bahwa model ini equivariant penuh atau bahwa semua relasi dipelajari tanpa asumsi. [Relation Networks](https://arxiv.org/abs/1706.01427).

## 3. Kondisi eksperimen

Semua kondisi sensor menggunakan checkpoint encoder/kepala pose dan window manifest yang sama untuk satu seed/fold. `Y` adalah fitur encoder; `P,V` adalah prediksi state dari radar.

| ID | Nama | Node | Edge | Projector | Peran |
|---|---|---|---|---|---|
| M | mmMind-style M4Human | Tidak digunakan | Tidak digunakan | `Y → MLP → Qwen` | Pembanding sistem langsung |
| S | Raw-state pair control | Posisi/kecepatan absolut | Konkatenasi state kedua endpoint | Projector terstruktur | Kontrol utama akses state dan kapasitas |
| F | Global/local factorization | Root global + joint relatif | Konkatenasi endpoint absolut seperti S | Identik S | Memisahkan kontribusi faktorisasi |
| E | Explicit-edge control | Absolut seperti S | Deskriptor relasi eksplisit | Identik S | Memisahkan kontribusi edge |
| R | Relational kinematics | Root global + joint relatif | Deskriptor relasi eksplisit | Identik S | Sistem utama H-B |
| T | Text-only Qwen | Tidak ada | Tidak ada | Tidak dilatih | Prior bahasa |
| Maj | TrainMajority | Tidak ada | Tidak ada | Tidak ada | Prior frekuensi kelas train |
| Rule | Predicted-state rule | P,V dari encoder | Rumus label yang sama | Tanpa Qwen | Memisahkan estimasi fisik dari alignment |
| Probe | Frozen-feature task probe | Y atau fitur sebelum embedding bahasa | Sesuai source ID | Classifier kecil | Keterbacaan oleh keluarga probe tertentu |

S sengaja sudah diberi identitas joint, daftar pasangan tubuh, dan seluruh state yang dibaca R. Jadi `R−S` tidak membuktikan manfaat graph dibanding tanpa graph; ia menguji **operasi kinematik eksplisit dibanding pasangan state mentah pada struktur yang sama**. Ini kontrol konservatif untuk menghindari penjelasan bahwa R hanya unggul karena memperoleh pose atau kapasitas tambahan.

`R−M` mengukur perbedaan sistem keseluruhan. Selisih itu mencampurkan tambahan akses prediksi pose, turunan, dan projector; jangan menyebutnya estimasi satu faktor. `R−F` menguji relasi eksplisit ketika pemisahan global/lokal sudah tersedia. Efek interaksi dilaporkan sebagai `(R−F)−(E−S)`.

## 4. Variabel terkendali

Wajib identik pada S/F/E/R: split, sumber frame, sampling titik evaluasi, checkpoint fisik, normalisasi yang prosedurnya sama, target/prompt, QA draw order, initialization seed, jumlah langkah optimizer maksimum, scheduler, early stopping, lebar setiap layer, jumlah parameter trainable, jumlah 64 embedding sensor, precision, decoding, dan batas konteks. Statistik normalisasi descriptor berbeda karena nilainya berbeda; semuanya dihitung dari prediksi **train** dengan prosedur yang sama.

M memakai encoder dan QA yang sama tetapi projector lebih sederhana. Laporkan jumlah parameter dan biaya M apa adanya. Tidak menambah parameter mati untuk membuat tabel terlihat setara.

## 5. Kriteria interpretasi

- Endpoint primer tunggal: `Δ = S_atomic(R) − S_atomic(S)` pada test reference-defined.
- Selisih positif dengan CI seluruhnya di atas nol mendukung arah H-B secara terbatas pada protokol/populasi ini. Dengan empat subjek test, CI tetap diagnostik; jangan menggunakan banyak window untuk mengklaim kepastian populasi besar.
- Selisih positif kecil atau CI melintasi nol berarti bukti belum meyakinkan; laporkan ukuran efek, jangan mengubah endpoint setelah melihat hasil.
- Efek praktis dideskripsikan dalam poin persentase dan biaya komputasi. Tidak ada ambang keberhasilan praktis yang dikarang tanpa dasar penggunaan.
- Jika R mengalahkan M tetapi tidak S, tambahan state/projector lebih konsisten sebagai penjelasan daripada relasi eksplisit.
- Jika Rule menyamai atau melampaui Qwen, kontribusi yang didukung adalah penyajian/alignment bukti terukur, bukan penalaran bahasa yang melampaui perhitungan fisik.
- Jika improvement hanya terjadi pada validation, tidak ada bukti generalisasi test.

## 6. Ruang lingkup dan bahasa klaim

Task primer: tren kelajuan pelvis, perbandingan kelajuan relatif lengan, dan perbandingan kelajuan relatif kaki. Tidak termasuk forecasting, estimasi gaya/torsi, diagnosis, niat, center of mass, atau kausalitas dunia nyata. Encoder melihat keseluruhan window secara bidirectional; sistem menjawab setelah frame ke-64 tersedia, bukan sistem streaming kausal.

64 timestamp memuat 63 interval: pada 12 Hz rentang endpoint adalah 5,25 detik. Durasi aktual ditulis dari timestamp. Satu window memproduksi tiga QA kanonik; tiga QA itu bukan tiga pengamatan independen.

Perbandingan output LLM sah bila seluruh model dievaluasi pada task dan panel yang sama. Dataset tetap berpengaruh pada input, label, tingkat kesulitan, dan validitas eksternal. Karena itu, skor M4Human tidak dibandingkan secara numerik langsung dengan skor mmMind-Bench. Captioning, multi-turn, dan unseen-action mmMind tidak otomatis tercakup oleh tiga task ini.

Model utama adalah **Qwen2.5-1.5B-Instruct**: 2.5 merupakan nama keluarga, 1.5B adalah skala parameter. Mengganti ukuran model saja tidak membuktikan kebaruan. Pembandingan terhadap Qwen3-8B merupakan studi tambahan dalam blok eksperimen terpisah dengan semua kondisi sensor dilatih ulang untuk backbone tersebut. [Konfigurasi Qwen2.5](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/blob/main/config.json), [model Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B).
