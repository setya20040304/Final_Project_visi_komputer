# Object Recognition using SIFT

Final Project ini adalah implementasi **Visi Komputer** untuk mendeteksi keberadaan objek tertentu dalam sebuah scene yang kompleks menggunakan algoritma **Scale-Invariant Feature Transform (SIFT)**.

## Deskripsi
Program ini bekerja dengan cara:
1. Mengekstrak fitur keypoints dari gambar referensi (Buku dengan tampilan close up).
2. Mengekstrak fitur dari gambar scene (tmpukan uku-buku).
3. Mencocokkan fitur menggunakan **FLANN Based Matcher**.
4. Menggunakan **Homography** untuk memetakan sudut-sudut objek referensi ke dalam scene.
5. Menggambar **Bounding Box** (kotak hijau) di sekitar objek yang ditemukan.

## Struktur Folder
- `code/`: Berisi source code utama (`main_detection.py`).
- `dataset/`: Berisi gambar input (`reference.jpeg` dan `scene.jpeg`).
- `results/`: Tempat output gambar hasil deteksi disimpan.

## Cara Menjalankan
1. Pastikan Python dan library berikut sudah terinstall:
   pip install opencv-python numpy matplotlib

   lalu ketik:
   python code/main_detection.py

**Ekspektasi Hasil:**
Nanti akan muncul jendela gambar baru.
* **Kiri:** Foto bukumu (referensi).
* **Kanan:** Foto tumpukan buku-buku.
* **Result:** Ada garis-garis penghubung, dan ada **Kotak Hijau** tebal yang membungkus buku di foto kanan. Kalau kotak hijaunya pas/akurat, itu artinya **Object Recognition BERHASIL!**
