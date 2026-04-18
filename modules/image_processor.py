import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops

class ImageProcessor:
    """
    Menangani semua tugas pemrosesan citra termasuk membaca, mengubah ukuran, 
    masking, dan ekstraksi fitur.
    """
    def __init__(self):
        pass

    def preprocess(self, image_path):
        """
        Membaca citra dari path yang diberikan, mengubah ukurannya, dan menerapkan masking.
        
        Args:
            image_path (str): Path ke file gambar.
            
        Returns:
            tuple: (original_img, gray_img, mask, masked_gray, masked_color)
                   Mengembalikan (None, ...) jika gambar tidak dapat dibaca.
        """
        img = cv2.imread(image_path)
        if img is None:
            return None, None, None, None, None
        
        # Ubah ukuran agar sesuai dengan input model (256x256) dan UI akan menyesuaikan
        img = cv2.resize(img, (256, 256)) 
        
        # Buat mask untuk memisahkan telur dari background
        mask = self.create_mask(img)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Terapkan mask untuk mendapatkan area telur saja
        masked_gray = cv2.bitwise_and(gray, gray, mask=mask)
        masked_color = cv2.bitwise_and(img, img, mask=mask)
        
        return img, gray, mask, masked_gray, masked_color

    def create_mask(self, img):
        """
        Membuat binary mask untuk memisahkan telur dari background menggunakan thresholding HSV.
        """
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        s_channel = hsv[:, :, 1] # Gunakan channel Saturation
        
        # Blur untuk mengurangi noise
        s_blur = cv2.GaussianBlur(s_channel, (5, 5), 0)
        
        # Thresholding Otsu
        _, mask = cv2.threshold(s_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Operasi morfologi untuk membersihkan mask
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
        return mask

    def extract_features(self, img, gray, mask, masked_gray, masked_color):
        """
        Mengekstraksi fitur GLCM (tekstur) dan HSV (warna) dari citra yang sudah di-masking.
        
        Returns:
            list: [contrast, energy, homogeneity, correlation, h_mean, s_mean, v_mean, h_std, s_std, v_std]
        """
        # --- Fitur GLCM (Tekstur) ---
        if np.any(masked_gray > 0):
            glcm = graycomatrix(masked_gray, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4], 256, symmetric=True, normed=True)
            contrast = graycoprops(glcm, 'contrast').mean()
            energy = graycoprops(glcm, 'energy').mean()
            homogeneity = graycoprops(glcm, 'homogeneity').mean()
            correlation = graycoprops(glcm, 'correlation').mean()
        else:
            contrast = energy = homogeneity = correlation = 0

        # --- Fitur HSV (Warna) ---
        if np.any(mask > 0):
            hsv = cv2.cvtColor(masked_color, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(hsv)
            area_telur = mask > 0 # Hanya hitung untuk area telur
            
            h_mean, s_mean, v_mean = h[area_telur].mean(), s[area_telur].mean(), v[area_telur].mean()
            h_std, s_std, v_std = h[area_telur].std(), s[area_telur].std(), v[area_telur].std()
        else:
            h_mean = s_mean = v_mean = h_std = s_std = v_std = 0

        return [contrast, energy, homogeneity, correlation, h_mean, s_mean, v_mean, h_std, s_std, v_std]

    def check_is_object_egg(self, mask):
        """
        Validasi heuristik untuk menentukan apakah mask objek adalah telur.
        Berdasarkan luas area minimal dan bounding box aspect ratio.
        """
        # Ambil kontur untuk dianalisis
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return False, "Terlalu gelap/Tidak ada objek"
            
        # Temukan objek/kontur terbesar di dalam mask
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)
        
        # 1. Cek Area Minimal (Citra 256x256 total 65536 piksel)
        if area < 2000:
            return False, "Terlalu kecil"
            
        # 2. Cek Aspect Ratio
        x, y, w, h = cv2.boundingRect(largest_contour)
        if h == 0:
            return False, "Area tidak valid"
            
        aspect_ratio = float(w) / h
        if aspect_ratio < 0.4 or aspect_ratio > 2.5:
            return False, "Bentuk bukan oval"
            
        # 3. Cek Kepadatan (Solidity)
        # Telur adalah benda berbentuk cembung sempurna (convex).
        # Jika solidity terlalu rendah, berarti bentuknya ada bolong/lekukan (misal postur tubuh manusia)
        hull = cv2.convexHull(largest_contour)
        hull_area = cv2.contourArea(hull)
        if hull_area > 0:
            solidity = float(area) / hull_area
            if solidity < 0.85:
                return False, "Benda memiliki lekukan (Bukan Telur)"
        
        return True, "Telur valid"
