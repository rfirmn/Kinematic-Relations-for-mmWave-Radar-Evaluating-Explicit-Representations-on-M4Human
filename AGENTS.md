# Pedoman implementasi relational64

Authority rancangan: `docs/planning_relasi_kinematik/README.md`, dokumen 00–08,
dan `spesifikasi.yaml`. Proposal di `docs/reference/` hanya sumber kebutuhan
dataset/perangkat; konstanta arsitektur mengikuti paket spesifikasi.

- Repo referensi dan seluruh dataset sumber read-only. Semua hasil baru berada
  dalam repo ini. Jangan mengimpor pipeline, cache, split, atau checkpoint lama.
- Seluruh metadata dan artefak penelitian wajib dibangun ulang dari dataset
  sumber untuk generation aktif. Dilarang membaca metadata/turunan penelitian
  dari folder lama atau arsip untuk meluluskan gate. Semua output tetap di C:
  di dalam repo ini; gunakan meta_root/report_root/checkpoint_root dari runtime.
- Input sensor hanya XYZ+intensity, mask sensor, waktu, dan identitas opaque.
  Intensity bukan Doppler. GT/aktivitas/subjek/bbox tidak masuk forward sensor.
- 64 frame per window, maksimum 150 titik per frame, 22 joint pelvis indeks 0.
  Cabang relasional difusikan menjadi tepat 64 embedding sebelum frozen Qwen.
- Semua statistik/tuning memakai train/validation; test dikunci sebelum evaluasi.
- Catat konflik dan resolution beserta bukti. Jangan meluluskan null provenance.
- Setiap tahap menyimpan JSON pemeriksaan dan hashes. Fixture sintetis hanya
  membuktikan software, bukan koordinat/sinkronisasi M4Human aktual.
- Python wajib `.venv/Scripts/python.exe` atau `uv`. Uji CPU, Windows/CUDA,
  data aktual, dan eksperimen ilmiah dilaporkan terpisah.
- Training penuh merupakan pekerjaan terpisah. Sesi implementasi mengizinkan
  unit tests, pemeriksaan aktual read-only, overfit kecil, profil, dan pilot yang
  seluruh dependensinya telah lulus.
- Training diagnostik bertahap dapat memakai RPC aktual dan calon target/waktu
  provisional untuk menemukan kegagalan implementasi. Gunakan runner dan schema
  diagnostik terpisah, train/validation saja, budget eksplisit, checkpoint
  `diagnostic.pt`, serta flag scientific_acceptance=false. Ini tidak meluluskan
  G0 atau menggantikan artefak training ilmiah; input dan mask tetap diperiksa.
