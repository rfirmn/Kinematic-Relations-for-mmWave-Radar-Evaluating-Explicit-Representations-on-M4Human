# 01 — Data, koordinat, dan manifest

## 1. Input sumber dan adapter

Input sensor adalah point cloud RPC M4Human berbentuk `[N_raw,4]`, urutan kanal **x,y,z,intensity**. Intensity bukan Doppler, bukan SNR, dan tidak dikonversi menjadi kecepatan. Tidak membaca RGB, depth, mesh, bbox, aktivitas, atau validitas anotasi pada jalur inferensi.

Lokasi Windows disediakan sebagai path absolut `dataset_root`. Nama kandidat distribusi: `radar_pc.lmdb` untuk RPC; `params.lmdb`, `calib.lmdb`, dan `indicator.lmdb` untuk ekspor target/audit metadata. Nama fisik dapat berbeda, sehingga adapter memakai pemetaan eksplisit `source_inventory.json`. `RPC.lmdb` tidak boleh dianggap otomatis sama dengan `radar_pc.lmdb` tanpa inventaris. Dataset tidak dipindahkan atau diproses ulang otomatis.

### Audit serializer

Reader mendukung kontrak RPC berikut setelah contoh sumber membuktikannya:

1. Key string memuat triplet integer `(subject_id, action_id, source_frame_id)`; parser hanya menerima literal tuple/list tiga integer terbatas, tanpa evaluasi ekspresi.
2. Payload memiliki header tiga unsigned integer 32 bit: `ndim=2`, `N_raw`, `channels=4`; byte order tercatat dari producer.
3. Setelah header 12 byte, payload berupa `N_raw × 4` float32; panjang tepat `12 + 16*N_raw` byte.
4. Batas awal `N_raw≤100000`, payload≤1.600.012 byte; ukuran melampaui batas ditolak, bukan dipotong. Audit dapat menetapkan batas lebih ketat sebelum training.
5. Jika distribusi tidak memenuhi kontrak ini, tahap ingestion berhenti dengan laporan format. Implementasikan adapter format yang diaudit ke kontrak kanonik di bawah; jangan mencoba pickle/eval secara otomatis.

Manifest kanonik memisahkan key sumber, recording, urutan frame, dan timestamp. Nilai key tidak dianggap sebagai urutan waktu tanpa metadata. `action_id` hanya metadata offline dan tidak dikembalikan oleh `SensorBatch`.

LMDB dibuka per proses secara lazy: `readonly=True`, `create=False`, `writemap=False`, `readahead=False`, `subdir` sesuai tipe path. `lock=False` hanya untuk snapshot immutable tanpa writer; status ini wajib dibuktikan dalam inventaris. Handle tidak diwariskan melalui Windows spawn. Salin buffer sebelum transaksi ditutup bila hasil akan dipakai di luar transaksi. Tidak membuka semua modalitas melalui loader serbaguna.

## 2. Audit provenance

`source_inventory.json` harus memuat asal distribusi, waktu inventaris, manifest hash berkas, entry count, serializer, byte order, rentang N, statistik kanal, aturan sentinel, missing key, serta daftar metadata waktu/recording. Tidak perlu menahan seluruh LMDB di RAM atau menghitung hash ulang setiap batch.

Periksa apakah RPC distribusi dibentuk melalui cropping yang memakai GT tubuh/bbox atau informasi kamera. Bila iya, tulis bahwa input merupakan RPC terproses dengan bantuan informasi tersebut. Inferensi adapter yang hanya membaca RPC tidak menghapus privilege preprocessing sumber. Hasil penelitian tidak otomatis menunjukkan deployment tanpa bantuan informasi tersebut.

## 3. Sistem koordinat dan target tubuh

Koordinat kanonik: radar frame, meter, urutan XYZ sesuai transform kalibrasi yang dibuktikan. `coordinate_audit.json` memuat arah sumbu, handedness, sumbu vertikal, unit input, transform 4×4 `T_source_to_radar`, serta contoh round-trip. Konfigurasi utama mengharapkan z vertikal; bila paket berbeda, lakukan transform statis yang diaudit, bukan menebak melalui tanda koordinat.

Target absolut `P_gt[f,j,3]` dipetakan ke radar frame pada timestamp radar. Anotasi mocap berfrekuensi tinggi memakai pasangan sinkronisasi yang disediakan; jangan sekadar menyamakan indeks array. Tidak membuat interpolasi untuk menutup anotasi hilang. Bila ekspor melalui SMPL-X diperlukan, kunci asset hash, regressor, gender convention, parameter tubuh, urutan joint, dan transform. Parameter translasi body model tidak dianggap identik dengan posisi pelvis.

| j | Joint | Parent | Bobot fisik |
|---:|---|---:|---:|
| 0 | pelvis | -1 | 1 |
| 1 | left_hip | 0 | 2 |
| 2 | right_hip | 0 | 2 |
| 3 | spine1 | 0 | 1 |
| 4 | left_knee | 1 | 4 |
| 5 | right_knee | 2 | 4 |
| 6 | spine2 | 3 | 1 |
| 7 | left_ankle | 4 | 8 |
| 8 | right_ankle | 5 | 8 |
| 9 | spine3 | 6 | 1 |
| 10 | left_foot | 7 | 8 |
| 11 | right_foot | 8 | 8 |
| 12 | neck | 9 | 1 |
| 13 | left_collar | 9 | 2 |
| 14 | right_collar | 9 | 2 |
| 15 | head | 12 | 1 |
| 16 | left_shoulder | 13 | 2 |
| 17 | right_shoulder | 14 | 2 |
| 18 | left_elbow | 16 | 4 |
| 19 | right_elbow | 17 | 4 |
| 20 | left_wrist | 18 | 8 |
| 21 | right_wrist | 19 | 8 |

Parent list mendefinisikan 21 edge tulang dengan orientasi parent→child, urut child 1 sampai 21. Delapan edge bilateral berorientasi kiri→kanan: `(1,2),(4,5),(7,8),(10,11),(13,14),(16,17),(18,19),(20,21)`. Total 29 edge. Pelvis relatif adalah nol menurut definisi, sehingga tidak dihitung dalam MPJPE relatif 21 joint.

### 3.1 Ekspor anotasi secara operasional

Ekspor target adalah proses offline terpisah. Jalur reference memakai parameter body model dan regressor yang diaudit. Jalur joint langsung diperbolehkan hanya jika unit, frame, topology, dan kesetaraannya dengan reference dibuktikan; field bernama `joints` sendiri belum cukup.

Metadata MessagePack dibatasi16MiB/value dan kedalaman12. Array bertag `__nd__` harus memiliki dtype numerik non-object, shape integer positif, dan byte data tepat sesuai hasil kali shape×itemsize. Tolak extension arbitrary, array terlalu besar, dtype object, dan field yang tidak sesuai schema. Field parameter yang diperlukan:

| Field sumber | Shape | Pemakaian |
|---|---|---|
| betas | `[K]` | Shape coefficients; K ditetapkan dari metadata/asset yang cocok |
| pose_body | `[63]` |21 body joints dalam axis-angle radian |
| root_orient | `[3]` | Global orientation axis-angle radian |
| trans | `[3]` | Translasi body model, unit diaudit sendiri |
| gender | scalar/string | Lookup explicit ke asset male/female/neutral yang tersedia |

Gunakan SMPL-X offline dengan `use_pca=False`; input batch dimension ditambahkan. Mapping pemanggilan: betas→betas, pose_body→body_pose, root_orient→global_orient, trans yang telah dikonversi ke meter→transl. Parameter tangan masing-masing45, jaw/left-eye/right-eye masing-masing3, dan expression sebanyak dimensi asset harus berasal dari data atau konstanta default yang tercatat dalam target provenance. Nilai default nol hanya sah bila menjadi konvensi ekspor yang diaudit; tidak menebak gender atau mengganti bentuk tubuh yang hilang dengan neutral.

Ambil vertices V dalam meter dari output body model. Pastikan unit output melalui audit contoh; jangan mengalikan1000 hanya karena utilitas lain memakai millimeter. Bila vertices berada pada frame Vicon, transform column-vector:

`p_cam=R_cam_from_vicon*p_vicon+t_cam_from_vicon`

`p_radar=R_cam_from_radar^T*(p_cam−t_cam_from_radar)`.

Field kalibrasi yang dipetakan secara eksplisit: `vicon_to_cam_rotmatrix`, `vicon_to_cam_tvec`, `radar_to_cam_rotmatrix`, `radar_to_cam_tvec`. Masing-masing translation memiliki field unit sendiri; meter→faktor1, millimeter→0.001. Rotation3×3 harus finite, `R^T R≈I` dan determinant≈+1 dengan toleransi1e-5. Uji round-trip dengan toleransi1e-8 float64. Jika source sudah radar frame, transform identitas; tidak mengalibrasi parameter lalu mentransformasi vertices kedua kali.

Pilih22 baris `J_regressor` sesuai nama joint tabel01 dan row-index map yang disimpan. Jika regressor R_J berbentuk `[22,V_count]`, target `J_radar=R_J @ V_radar`. Setiap row sum harus≈1 dengan toleransi1e-5, ukuran vertex cocok, serta hash asset/regressor terkunci. Tidak sekadar mengambil22 baris pertama tanpa memeriksa map. Root adalah row pelvis hasil regressor, bukan trans.

Tidak menerapkan offset z konstan seperti−1.5m tanpa bukti bahwa itu bagian koordinat paket. Bila source memiliki offset yang terdokumentasi, masukkan inverse-nya dalam transform kanonik dan berlaku konsisten pada RPC/target; centroid neural c pada tahap sensor berbeda dari transform kalibrasi.

Missing/nonfinite anotasi satu frame menghasilkan mask target false dan alasan lokal; format/shape, asset, atau kalibrasi yang salah menggagalkan ekspor karena menandakan masalah sistematis. Output `.npy` ditulis float32 dengan invalid zero+mask; tidak mengubah sumber. Asset SMPL-X memerlukan distribusi yang sah dan dicatat, tidak diunduh otomatis. Inferensi dan training sesudah ekspor tidak perlu mengimpor SMPL-X.

## 4. Split subjek

Gunakan seluruh ID subjek unik hasil inventaris; ID diserialisasi sebagai string desimal integer tanpa leading zero. Untuk paket 20 subjek, urutkan menurut digest SHA-256 dari UTF-8 `m4human_relational64/split/42/<id>`, naik secara leksikografis; tie memakai ID numerik. Empat ID pertama menjadi test, empat berikutnya validation, dua belas sisanya train. Simpan daftar hasilnya dalam `split.lock.json` **sebelum membaca statistik label untuk memilih model/ambang**.

Jika jumlah subjek berbeda dari 20, validator berhenti untuk audit kelengkapan. Jangan mengecilkan split diam-diam. Semua recording seorang subjek berada pada satu split. Riwayat exposure subjek melalui checkpoint, threshold, atau hasil eksperimen harus dicatat. Checkpoint untuk studi ini dilatih hanya pada subjek train sesuai split ini. Jika tidak dapat membuktikan riwayat exposure, hindari klaim test yang sepenuhnya belum pernah terlihat.

Default ini adalah split penelitian, bukan klaim mengikuti pembagian resmi benchmark. Empat subjek test memberi ketidakpastian besar. Ekstensi lima fold dapat memakai lima kelompok empat subjek: fold k memakai kelompok k untuk test, `(k+1) mod 5` untuk validation, tiga lainnya untuk train. Setiap fold membutuhkan encoder, normalizer, dan projector sendiri; bukan hanya memindahkan window di tahap QA.

## 5. Timestamp dan segmentasi sensor

Prioritas timestamp: waktu radar terukur atau waktu sinkronisasi per frame yang terdokumentasi. Simpan float64 dalam detik. Nilai harus finite dan meningkat ketat. Tidak mengurutkan ulang frame bermasalah untuk menyembunyikan duplicate/nonmonotonic timestamp.

Jika distribusi hanya memiliki indeks frame, waktu nominal boleh diekspor sebagai `(source_frame_id−first_frame_id)/12` **hanya** setelah metadata memastikan indeks kontigu dan sampling nominal 12 Hz. Tulis `time_source=nominal_frame_index`; semua hasil kecepatan/durasi dinyatakan berdasarkan waktu nominal. Jangan menyebut timestamp hasil rekonstruksi sebagai pengukuran aktual. Tidak mencampur jenis waktu dalam satu recording; panel measured dan nominal dilaporkan terpisah bila keduanya ada.

Default gap maksimum `1.5/12 = 0.125` detik. Batas take/recording, perubahan subject/split/time_source, delta waktu≤0, gap>0.125 s, source frame yang diketahui hilang, atau frame radar tidak layak memutus segmen. Setiap segmen memiliki ID baru. Missing anotasi **tidak** memutus segmen dan tidak mengubah phase window.

Frame radar layak apabila setelah pembersihan terdapat sedikitnya delapan titik finite yang bukan sentinel. Frame dengan 1–7 titik dikeluarkan oleh aturan sensor yang sama pada seluruh split; jumlahnya dilaporkan. Primary window berisi 64 frame layak, tanpa lubang frame. Padding titik tetap diperbolehkan.

## 6. Pembersihan dan pemilihan titik

Urutan operasi wajib:

1. Decode, validasi `[N,4]`, hapus row nonfinite.
2. Hapus hanya sentinel yang konvensinya terverifikasi. Default `sentinel_policy=none`; jangan menganggap row nol sebagai padding tanpa bukti.
3. Jangan crop memakai GT/RGB/bbox, dan jangan normalisasi koordinat per frame dengan pelvis target.
4. Hitung centroid XY semua titik bersih **frame pertama window**: `c=(mean x,mean y,0)`.
5. Kurangkan c dari XYZ semua frame window dan semua posisi target untuk loss fisik. Pengurangan c konstan sepanjang window, sehingga translasi tubuh terhadap waktu tetap ada. Untuk evaluasi global, tambahkan c kembali ke posisi prediksi. Z tidak direcenter.
6. Jika N>150, sampling tanpa pengembalian. Train menggunakan seed dari seed run, epoch, window UID, dan indeks frame. Evaluasi/cache memakai seed tetap dari `sampling/42/<window_uid>/<frame_index>`; tidak bergantung batch order.
7. Jika N<150, isi sisa slot nol dan `point_mask=False`. Jangan mengulang titik untuk mengisi slot.
8. Transform intensity `u=sign(I)*log1p(abs(I)/a)`, dengan `a=max(median_train(abs(I)),1e-6)`. Skala signed mengakomodasi sumber yang bertanda; semantik kanal tetap intensity.
9. XYZ tidak distandardisasi: unit tetap meter pada input dan positional MLP. Intensity ditransform ke `(u−median_train(u))/max(IQR_train(u),1e-6)` lalu clip ke [-10,10]. Padding di-zero-kan kembali setelah transform.

Statistik intensity memakai maksimum satu juta titik train: pilih row dengan hash terkecil dari `(frame_uid,row_index)` secara streaming. Nilai N input mentah tidak menjadi fitur tambahan. Simpan digest daftar sampling statistik dan parameter normalizer; validation/test tidak ikut menghitungnya.

Augmentasi geometri utama `none`. Sampling titik train adalah stochasticity sensor utama. Pilihan ini mencegah augmentasi skala/waktu mengubah arti target kecepatan; eksperimen augmentasi memerlukan transform posisi, waktu, dan label yang konsisten serta ID eksperimen tersendiri.

## 7. Window 64 frame

Untuk segmen dengan F frame layak berurutan, window pertama adalah indeks segmen `[0..63]`. Awal berikutnya `s=stride*k`, hanya jika `s+63<F`. Stride train=8, validation=16, test=16. Reset phase pada setiap segmen. Tidak memakai konteks radar sebelum window, tidak ada padding ujung, dan tidak melintasi recording.

Jumlah window segmen adalah `max(0, 1+floor((F−64)/stride))`. Contoh F=80 menghasilkan tiga window train (0,8,16) dan dua window validation (0,16). Jika satu joint GT hilang di frame 10, daftar window tetap sama.

`window_uid` adalah SHA-256 kanonik dari protocol ID, split hash, recording/segment ID, urutan 64 frame UID, timestamp hash, sampling policy hash. UID tidak memuat label, status GT, atau jawaban. Waktu relatif `τ_t=t_t−t_0`; rentang observasi `t_63−t_0`, bukan `64/fps`.

## 8. Manifest dan pemisahan akses

| Artefak | Field minimum |
|---|---|
| `frames.jsonl` | frame_uid, source_key, subject_id, recording_id, segment_id, source_frame_id, time_s, time_source, split, clean_point_count, sensor_valid, exclusion_reason |
| `windows.jsonl` | window_uid, split, subject_id, recording_id, segment_id, frame_uids[64], source_keys[64], time_s[64], duration_s, time_source, sampling_seed_digest |
| `targets/index.jsonl` | frame_uid, shard, row, annotation_source, coordinate_hash, joint_map_hash |
| `targets/positions_*.npy` | float32 `[F,22,3]`, meter, nilai invalid tersimpan nol dengan mask terpisah |
| `targets/valid_*.npy` | bool `[F,22]`; validity untuk loss/evaluator saja |

`SensorBatch` hanya mengembalikan RPC, mask titik, waktu, centroid c, dan UID opak. `SupervisedBatch` menambahkan target/mask melalui loader training terpisah. Larang `subject_id`, `action_id`, annotation mask, label, atau path yang mengandung label masuk ke `model.forward`. Metadata tetap tersedia di evaluator melalui UID.

Validation/test cache dibuat sekali memakai sampling titik deterministik. Overlap window tidak mengizinkan berbagi fitur temporal Y antarwindow, karena encoder membaca konteks window yang berbeda. Tidak boleh menghasilkan Y satu recording penuh lalu memotongnya untuk menghemat waktu.

## 9. Kondisi data yang menghentikan tahap

Serializer tidak sesuai, transform/unit tidak terbukti, target map tidak cocok, source key ambigu, sumber waktu tidak dapat dipertanggungjawabkan, split leakage, atau hash artefak tidak sesuai menghentikan tahap sebelum training. Anotasi kurang lengkap pada sebagian frame tidak menghentikan seluruh dataset: gunakan mask target, status QA, dan coverage yang ditentukan pada 03. Dataset sumber selalu read-only.
