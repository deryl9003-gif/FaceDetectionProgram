import cv2

# 1. Load Haar Cascade untuk deteksi wajah
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# 2. Buka webcam (0 adalah kamera default laptop/PC)
cap = cv2.VideoCapture(2)

print("Program deteksi wajah dimulai. Tekan 'q' untuk keluar.")

while True:
    # 3. Baca frame dari webcam
    ret, frame = cap.read()

    if not ret:
        print("Gagal membuka kamera.")
        break

    # 4. Ubah frame warna (BGR) menjadi Grayscale (Abu-abu)
    # OpenCV memerlukan gambar grayscale untuk proses deteksi yang lebih cepat & akurat
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 5. Deteksi wajah pada gambar
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
    )

    # 6. Buat kotak (rectangle) di setiap wajah yang terdeteksi
    for x, y, w, h in faces:
        # Parameter: (gambar, titik_awal, titik_akhir, warna_BGR, ketebalan)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # 7. Tampilkan hasil di jendela baru
    cv2.imshow("Deteksi Wajah - OpenCV", frame)

    # 8. Hentikan program jika tombol 'q' ditekan pada keyboard
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# 9. Lepaskan kamera dan tutup semua jendela
cap.release()
cv2.destroyAllWindows()
