# 03 — Target kinematik, QA, dan kontrak bahasa

## 1. Operator turunan tujuh titik

Untuk endpoint e=6..63 dan joint j, gunakan posisi pada indeks e−6..e. Tidak memakai posisi di luar window. Semua tujuh posisi GT harus valid untuk target joint tersebut; tidak mengganti nilai hilang dengan interpolasi, nol, atau prediksi sensor.

Definisikan `h=t_e−t_(e−6)>0` dan `x_i=(t_i−t_e)/h`. Bentuk matriks A berukuran7×3 dengan baris `[1,x_i,x_i²]`. Selesaikan least squares `A b = p` untuk tiga koordinat sekaligus melalui QR/SVD float64. `velocity=b[1]/h`, unit m/s. Ini ekuivalen dengan koefisien linear polinom pada waktu aktual. Skalasi x menjaga kondisi numerik tanpa mengubah satuan akhir.

Syarat sah: seluruh timestamp finite, meningkat, setiap delta≤0.125 s, rank(A)=3, condition number(A)≤1e6, hasil finite. Tidak memakai ridge atau forward difference sebagai fallback. Simpan alasan kegagalan. Enam endpoint awal memiliki `velocity_valid=False`.

Operator linear dapat ditulis sebagai matriks W `[64,64]`, dengan baris e≥6 berisi row koefisien linear pseudoinverse A dibagi h pada kolom e−6..e. Baris0..5 nol. Untuk loss training, `V_pred=W*P_pred` tetap differentiable; W tidak memerlukan gradien. Perhitungan W float64, hasil digunakan float32 di perangkat. Mask GT tidak digunakan saat menghitung V_pred; ia hanya menentukan elemen loss yang dinilai.

Target root velocity berasal dari joint0. Target relative velocity `V_rel[j]=V_abs[j]−V_abs[0]`; validity mengharuskan kedua turunan valid. Estimasi kausal terhadap trajectory tidak menjadikan encoder kausal: P_pred telah membaca seluruh window.

## 2. Dukungan task

- Penyebut tetap 58 endpoint kandidat: indeks6..63.
- Minimum dukungan:44 endpoint, yaitu ceil(0.75×58).
- Root menggunakan endpoint dengan velocity GT root valid.
- Arms menggunakan **irisan endpoint bersama** root, left_elbow18, right_elbow19, left_wrist20, right_wrist21. Semua empat joint anggota dan root harus valid di endpoint itu.
- Legs memakai irisan root, left_knee4, right_knee5, left_ankle7, right_ankle8.
- Tidak membandingkan sisi kiri dan kanan dari waktu atau jumlah joint yang berbeda. Jangan mengizinkan satu joint mendominasi karena joint pasangannya hilang.

Jika support<44, reference status=`undefined`, answer=null, reason=`insufficient_velocity_support`. Window tetap ada dalam manifest; QA tidak masuk loss maupun skor, tetapi masuk coverage.

## 3. Tren kelajuan root

Pada endpoint sah, `s_e=||V_root[e]||`. Hitung semua slope pasangan `b_ab=(s_b−s_a)/(t_b−t_a)`, a<b. Maksimum1.653 pasangan per window. `b=median(b_ab)`. Median untuk jumlah genap adalah rata-rata dua nilai tengah. Intersep `a=median(s_e−b*(t_e−t_6))`. Residual `r_e=s_e−(a+b*(t_e−t_6))`; dispersi `m=1.4826*median(abs(r_e−median(r)))`.

Default label policy: `delta_root=0.05 m/s²`, `guard_root=0.01 m/s²`, `max_root_residual=0.15 m/s`.

Urutan keputusan:

1. Dukungan/numerik tidak sah → undefined, tanpa kelas.
2. m>0.15 → unknown, reason=`nonlinear_or_unstable_speed_trend`.
3. b>0.06 → speeding_up.
4. b<−0.06 → slowing_down.
5. |b|<0.04 → no_overall_trend.
6. Selainnya, termasuk persis batas0.04/0.06 → unknown, reason=`trend_boundary_band`.

`no_overall_trend` berarti slope keseluruhan berada dalam deadband dengan dispersi yang memenuhi policy, bukan identik diam atau kecepatan konstan setiap saat.

## 4. Perbandingan gerak relatif anggota

Gunakan endpoint irisan bersama task. Untuk setiap joint, hitung `m_j=median_e(||V_abs[e,j]−V_root[e]||)`.

Arms: `s_L=(m_18+m_20)/2`, `s_R=(m_19+m_21)/2`. Legs: `s_L=(m_4+m_7)/2`, `s_R=(m_5+m_8)/2`. Nilai task `d=s_L−s_R`, unit m/s.

Default: `delta_limb=0.05 m/s`, `guard_limb=0.01 m/s`.

- d>0.06 → faster_left.
- d<−0.06 → faster_right.
- |d|<0.04 → tie.
- Selainnya termasuk batas → unknown, reason=`limb_boundary_band`.

Tie menyatakan kesetaraan agregat menurut deadband, bukan kedua sisi diam. Seluruh kelas ditentukan offline dari GT; tidak ada label aktivitas yang dipakai untuk menebak jawaban.

## 5. Status dan kalibrasi label

| Status reference | Jawaban | Alignment | Primary metric | Secondary metric |
|---|---|---|---|---|
| defined | Salah satu tiga kelas fisik task | Ya | Ya | Ya |
| unknown | `unknown` | Ya | Tidak | Ya |
| undefined | null | Tidak | Tidak | Tidak; coverage saja |

Nilai ambang di atas merupakan **keputusan operasional awal**, bukan angka yang diklaim berasal dari mmMind atau pengukuran noise mocap. Sebelum alignment, jalankan audit train/validation: distribusi b,d,m, coverage, kelas per subjek, contoh trajectory dekat boundary, serta sensitivity diagnostic δ pada `[0.025,0.05,0.10]` dengan guard=0.2δ. Root residual diagnostic `[0.075,0.15,0.30]`. Default tetap0.05/0.15 untuk eksperimen utama; hasil sensitivity dilaporkan sebagai ketergantungan pada definisi label.

Tidak memilih ambang berdasarkan QA test, skor model yang paling tinggi, atau keinginan menyeimbangkan kelas. Jika noise/unit target membuat default tidak dapat dipertanggungjawabkan, hentikan finalisasi label, dokumentasikan alasan dan nilai yang ditetapkan menggunakan train/validation, lalu kunci `label_policy.json` sebelum model alignment apa pun dievaluasi di test. Ini keputusan ilmiah yang harus transparan, bukan hyperparameter tersembunyi.

Kelas yang jarang atau tidak hadir tidak dihapus dari macro-F1. Unknown di sini adalah hasil policy ambiguitas, bukan estimasi confidence radar yang telah dikalibrasi. Undefined karena GT hilang tidak diajarkan sebagai unknown, agar model tidak belajar kelengkapan anotasi.

## 6. Pertanyaan dan jawaban kanonik

Bahasa prompt utama Inggris untuk konsistensi dengan backbone. Deskripsi hasil penelitian tetap dapat berbahasa Indonesia. Tidak menyertakan subject, action, target score, threshold angka, mask GT, atau status reference.

System prompt persis:

```text
You answer kinematic questions using the supplied radar evidence. Return exactly one JSON object with the key "answer" and one allowed label. Do not add explanations or other keys. Use "unknown" when the evidence does not support a choice.
```

User message memuat satu radar marker dan satu pertanyaan:

```text
Radar observations contain 64 consecutive frames numbered 00 to 63. The first six frames provide derivative context. The question concerns frames 06 to 63. Left and right refer to the person's anatomical sides. Root means pelvis. Relative limb motion excludes pelvis translation.
[[RADAR_BLOCK]]
{question}
Allowed answers: {choices}.
```

| task_id | part | question persis | choices berurutan |
|---|---|---|---|
| root_speed_trend | root | Over frames 06 to 63, does the pelvis speed show an increasing trend, a decreasing trend, or no overall trend? | speeding_up, slowing_down, no_overall_trend, unknown |
| relative_limb_motion | arms | Over frames 06 to 63, which arm has greater typical motion speed relative to the pelvis, using the elbows and wrists? | faster_left, faster_right, tie, unknown |
| relative_limb_motion | legs | Over frames 06 to 63, which leg has greater typical motion speed relative to the pelvis, using the knees and ankles? | faster_left, faster_right, tie, unknown |

Jawaban kanonik, tanpa spasi tambahan: `{"answer":"faster_left"}` atau enum lain yang sesuai. Token end-of-turn mengikuti chat template. Tidak menghasilkan penjelasan chain-of-thought atau target narasi yang tidak terukur.

Default train memakai pertanyaan kanonik yang sama, tanpa parafrasa, sehingga exposure mudah dikendalikan. Ekstensi parafrasa harus punya registry tetap dan tidak memperbanyak cache sensor atau jumlah window yang dilaporkan. Pertanyaan validation/test tetap satu per `(window_uid,task_id,part)`.

## 7. Record QA

Field wajib: `qa_uid`, `window_uid`, `task_id`, `part`, `prompt_id`, `interval=[6,63]`, `reference_status`, `answer`, `reason`, `support_count`, `candidate_count=58`, `scalar_value`, `residual_scale` untuk root, `label_policy_hash`, `target_hash`, `window_manifest_hash`.

`qa_uid=sha256(protocol_id/window_uid/task_id/part/prompt_hash/label_policy_hash)`. Scalar fisik, support count, alasan, dan answer hanya tersedia pada sisi target/evaluator; bukan teks user atau embedding sensor. `unknown` masih memiliki scalar yang sah. Undefined menyimpan scalar=null bila tidak dapat dihitung dengan dukungan wajib.

## 8. Decoding dan parser

Generasi utama greedy: `do_sample=False`, `num_beams=1`, `max_new_tokens=32`, tanpa constrained decoding/grammar repair, tanpa temperature/top-p sampling. EOS mengikuti tokenizer yang dipin. Tidak menilai argmax dari daftar likelihood jawaban sebagai pengganti generasi.

Parser setelah membuang special end-of-turn token:

1. Strip whitespace awal/akhir saja.
2. Parse tepat satu objek JSON, bukan mencari substring `{...}` dari penjelasan.
3. Tolak duplicate key, trailing content, Markdown fence, NaN/Infinity, nilai nonstring, key selain `answer`, atau enum yang tidak sah untuk task.
4. Enum case-sensitive; tidak fuzzy matching, tidak synonym expansion, tidak auto-repair.
5. Keluaran yang terkena batas token tetap ditandai truncated dan dihitung salah, meskipun substring JSON tampak lengkap tanpa terminasi yang diharapkan.

Status prediksi: `valid`, `invalid_json`, `invalid_schema`, `invalid_label`, `truncated`, `missing`, `runtime_error`. `valid` mencakup enum unknown. Pada reference-defined, semua status selain valid dengan label benar adalah kesalahan; unknown juga kesalahan. Simpan teks mentah dan alasan parser agar kegagalan format dapat dibedakan dari kesalahan kelas.
