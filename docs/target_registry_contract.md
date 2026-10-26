# Kontrak registry QA dan penguncian target

`build_qa_registry` membaca target per frame melalui `read(source_key)` yang menghasilkan posisi `[22,3]` dan mask `[22]`. Timestamp dan identitas window berasal dari manifest sensor. Target tidak menentukan pembentukan atau penghapusan window. Tiga record QA tetap disimpan ketika salah satu task berstatus `undefined`; record tersebut dikecualikan dari loss dan skor, serta tetap dihitung dalam coverage.

Pembuatan registry memerlukan attestasi G0, jenis bukti yang eksplisit, serta SHA256 inventory, koordinat, waktu, split, resep target, dan ekspor array. Bukti sintetis selalu ditandai demikian. Metadata registry tidak mengesahkan G1 secara otomatis. Audit label hanya menerima train dan validation; ekspor test tidak menjalankan diagnostik pemilihan ambang.

## Identitas sebelum dan sesudah lock

Identitas berikut tersedia sebelum membaca target test dan dipin oleh `study.lock_test`:

- `plan.source_manifest`: manifest window sensor lengkap yang bersifat tetap.
- `plan.source_inventory` dan `plan.split_lock`: inventory dan split yang telah diaudit.
- `plan.target_provenance`: resep/sertifikat target yang menetapkan joint map, satuan, koordinat, aset referensi, dan bukti ekuivalensi.
- `plan.label_policy`: objek field `LabelPolicy`, atau objek yang memuat field tersebut dalam key `policy`.

`lock.artifacts` menggunakan path absolut sebagai key dan SHA256 file sebagai value. Status lock adalah `locked`, identitas payload harus sah, dan `test_evaluation_authorized` harus bernilai `true`. Builder memanggil `study.validate_test_lock` dan memeriksa bahwa sumber, resep, split, serta policy yang digunakan cocok dengan plan tersebut.

`provenance.target_provenance_hash` merupakan identitas resep target yang tetap. `provenance.target_hash` merupakan identitas metadata ekspor array yang sedang dibaca. Ekspor test dapat mempunyai identitas array berbeda dari ekspor train/validation tanpa mengubah resep. Bila reader menyediakan metadata ekspor, identitas inventory dan resep pada metadata itu juga harus cocok.

Setelah lock, ekspor array dan registry test menghasilkan hash baru. Metadata registry menyimpan hash array, manifest sumber, policy, resep, registry QA, serta file lock yang mengizinkan ekspor. Artefak hasil tersebut dilampirkan sebagai bukti evaluasi; file lock awal tetap tidak berubah.

## Penyelesaian ambiguitas protokol

Dokumen evaluasi meminta hash target/QA dalam lock, sekaligus melarang akses target test sebelum lock. Implementasi menafsirkan identitas target yang wajib tersedia sebelum lock sebagai resep/provenance target dan registry train/validation yang telah digunakan untuk seleksi. Hash byte array dan registry test yang belum terbentuk dicatat sesudah ekspor sebagai artefak turunan yang menunjuk lock. Ini menghindari kebutuhan membaca target test untuk memperoleh hash sebelum keputusan penelitian dikunci. Tidak ada label atau ambang test yang digunakan untuk memilih konfigurasi.

Fixture hanya menguji kontrak perangkat lunak ini. Penerimaan data nyata tetap memerlukan audit M4Human, ekuivalensi target, dan bukti G0/G1 yang terpisah.
