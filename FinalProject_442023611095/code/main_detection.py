import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# KONFIGURASI
REF_IMAGE_NAME = 'reference.jpeg'
SCENE_IMAGE_NAME = 'scene.jpeg'

# Setup Path Otomatis
base_dir = os.path.dirname(os.path.abspath(__file__))
ref_path = os.path.join(base_dir, '../dataset', REF_IMAGE_NAME)
scene_path = os.path.join(base_dir, '../dataset', SCENE_IMAGE_NAME)
output_path = os.path.join(base_dir, '../results', 'hasil_deteksi_objek.jpg')

# Muat Citra
print(f"memuat gambar")
img_ref = cv2.imread(ref_path)
img_scene = cv2.imread(scene_path)

if img_ref is None or img_scene is None:
    print("Error:gambar tidak ditemukan!")
    exit()

# Ubah ke Grayscale
gray_ref = cv2.cvtColor(img_ref, cv2.COLOR_BGR2GRAY)
gray_scene = cv2.cvtColor(img_scene, cv2.COLOR_BGR2GRAY)

# Inisialisasi SIFT
print("Mendeteksi Keypoints dengan SIFT.")
sift = cv2.SIFT_create()

# Hitung Keypoints & Descriptors
kp1, des1 = sift.detectAndCompute(gray_ref, None)
kp2, des2 = sift.detectAndCompute(gray_scene, None)

# Pencocokan Fitur (Feature Matching)
# Kita pakai FLANN Matcher biar lebih cepat dan akurat untuk deteksi objek
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)

flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(des1, des2, k=2)

# Filter Matches
good_matches = []
for m, n in matches:
    if m.distance < 0.7 * n.distance:
        good_matches.append(m)

print(f"Jumlah Good Matches ditemukan: {len(good_matches)}")

# Homography & Object Detection
# Kita butuh minimal 10 titik cocok biar kotaknya lebih akurat
MIN_MATCH_COUNT = 10

if len(good_matches) > MIN_MATCH_COUNT:
    # Ambil koordinat titik dari matches yang bagus
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # Hitung Matriks Homografi (M)
    # M adalah rumus transformasi dari gambar referensi ke scene
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    matchesMask = mask.ravel().tolist()

    # Ambil dimensi gambar referensi (lebar & tinggi)
    h, w = gray_ref.shape

    # membentuk kotak (persegi panjang) seukuran gambar referensi
    pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)

    # Transformasikan kotak tadi ke gambar scene menggunakan matriks M
    dst = cv2.perspectiveTransform(pts, M)

    # Gambar kotak hasil deteksi di gambar scene (Warna Hijau Tebal)
    img_scene = cv2.polylines(img_scene, [np.int32(dst)], True, (0, 255, 0), 5, cv2.LINE_AA)
    
    print("Objek ditemukan!")

else:
    print(f"Not enough matches are found - {len(good_matches)}/{MIN_MATCH_COUNT}")
    matchesMask = None

# Visualisasi Akhir
draw_params = dict(matchColor=(0, 255, 0),
                   singlePointColor=None,
                   matchesMask=matchesMask,
                   flags=2)

img_result = cv2.drawMatches(img_ref, kp1, img_scene, kp2, good_matches, None, **draw_params)

# Simpan & Tampilkan
cv2.imwrite(output_path, img_result)
print(f"Hasil sudah disimpan di: {output_path}")

plt.figure(figsize=(15, 8))
plt.imshow(cv2.cvtColor(img_result, cv2.COLOR_BGR2RGB))
plt.title(f'Object Recognition SIFT ({len(good_matches)} Matches)')
plt.axis('off')
plt.show()