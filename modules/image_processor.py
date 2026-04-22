import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops


class ImageProcessor:
    """Menangani pemrosesan citra: membaca, masking, ekstraksi fitur, dan validasi telur."""

    def __init__(self):
        pass

    # -------------------------------------------------------------------------
    # Preprocessing
    # -------------------------------------------------------------------------

    def preprocess(self, input_data, source_type='file', roi_params=None):
        """
        Membaca citra, mengubah ukuran ke 256x256, dan menerapkan masking.

        Args:
            input_data (str | np.ndarray): Path file gambar atau frame NumPy array.
            source_type (str): 'file' atau 'webcam'.
            roi_params (dict | None): Parameter ROI webcam
                {'cx_pct', 'cy_pct', 'ax_pct', 'ay_pct'}.

        Returns:
            tuple: (img, gray, mask, masked_gray, masked_color)
                   Mengembalikan (None, ...) jika gambar gagal dibaca.
        """
        if isinstance(input_data, str):
            img = cv2.imread(input_data)
        elif isinstance(input_data, np.ndarray):
            img = input_data.copy()
        else:
            return None, None, None, None, None

        if img is None or img.size == 0:
            return None, None, None, None, None

        img = cv2.resize(img, (256, 256))

        if source_type == 'webcam':
            mask = self._create_static_roi_mask(roi_params)
            solid_mask = mask.copy()
        else:
            mask = self.create_mask(img)
            solid_mask = self._create_solid_mask(mask)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        masked_gray = cv2.bitwise_and(gray, gray, mask=solid_mask)
        masked_color = cv2.bitwise_and(img, img, mask=solid_mask)

        return img, gray, mask, masked_gray, masked_color

    def create_mask(self, img):
        """
        Membuat binary mask dengan Otsu Thresholding pada channel Saturation HSV.
        Digunakan untuk input gambar dari file.
        """
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        s_channel = hsv[:, :, 1]
        s_blur = cv2.GaussianBlur(s_channel, (5, 5), 0)

        _, mask = cv2.threshold(s_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
        return mask

    def _create_static_roi_mask(self, roi_params=None):
        """
        Membuat mask oval untuk mode webcam.

        Args:
            roi_params (dict | None): {'cx_pct', 'cy_pct', 'ax_pct', 'ay_pct'}.
                                      Jika None, gunakan nilai default (tengah, 35%x45%).
        """
        mask = np.zeros((256, 256), dtype=np.uint8)
        if roi_params:
            cx = int(256 * roi_params['cx_pct'] / 100)
            cy = int(256 * roi_params['cy_pct'] / 100)
            ax = int(256 * roi_params['ax_pct'] / 100)
            ay = int(256 * roi_params['ay_pct'] / 100)
            center, axes = (cx, cy), (ax, ay)
        else:
            center, axes = (128, 128), (int(256 * 0.35), int(256 * 0.45))
        cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)
        return mask

    def _create_solid_mask(self, mask):
        """Mengonversi mask menjadi solid menggunakan Convex Hull dari kontur terbesar."""
        solid_mask = np.zeros_like(mask)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            hull = cv2.convexHull(largest_contour)
            cv2.drawContours(solid_mask, [hull], -1, 255, thickness=cv2.FILLED)

            # cv2.drawContours(solid_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
        else:
            solid_mask = mask.copy()
        return solid_mask

    # -------------------------------------------------------------------------
    # Ekstraksi Fitur
    # -------------------------------------------------------------------------

    def extract_features(self, img, gray, mask, masked_gray, masked_color):
        """
        Mengekstraksi 10 fitur dari citra yang sudah di-masking.

        Returns:
            list: [contrast, energy, homogeneity, correlation,
                   h_mean, s_mean, v_mean, h_std, s_std, v_std]
        """
        # Fitur GLCM (Tekstur)
        if np.any(masked_gray > 0):
            glcm = graycomatrix(
                masked_gray, [1],
                [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
                256, symmetric=True, normed=True
            )
            contrast = graycoprops(glcm, 'contrast').mean()
            energy = graycoprops(glcm, 'energy').mean()
            homogeneity = graycoprops(glcm, 'homogeneity').mean()
            correlation = graycoprops(glcm, 'correlation').mean()
        else:
            contrast = energy = homogeneity = correlation = 0

        # Fitur HSV (Warna)
        if np.any(mask > 0):
            hsv = cv2.cvtColor(masked_color, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(hsv)
            area = mask > 0
            h_mean, s_mean, v_mean = h[area].mean(), s[area].mean(), v[area].mean()
            h_std, s_std, v_std = h[area].std(), s[area].std(), v[area].std()
        else:
            h_mean = s_mean = v_mean = h_std = s_std = v_std = 0

        return [contrast, energy, homogeneity, correlation,
                h_mean, s_mean, v_mean, h_std, s_std, v_std]

    # -------------------------------------------------------------------------
    # Validasi Telur
    # -------------------------------------------------------------------------

    def check_is_object_egg(self, img, mask):
        """
        Validasi heuristik dua tahap: Uji Geometri lalu Uji Warna.

        Returns:
            tuple: (bool, str) — (valid, alasan)
        """
        is_valid, reason = self._validate_egg_geometry(mask)
        if not is_valid:
            return False, reason

        is_valid, reason = self._validate_egg_color(img, mask)
        if not is_valid:
            return False, reason

        return True, "Telur valid"

    def _validate_egg_geometry(self, mask):
        """Validasi bentuk: luas area, aspect ratio, dan solidity."""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return False, "Terlalu gelap/Tidak ada objek"

        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)

        if area < 2000:
            return False, "Terlalu kecil"
        if area > 52000:
            return False, "Objek terlalu besar / terlalu dekat"

        x, y, w, h = cv2.boundingRect(largest_contour)
        if h == 0:
            return False, "Area tidak valid"

        aspect_ratio = float(w) / h
        if aspect_ratio < 0.4 or aspect_ratio > 2.5:
            return False, "Bentuk bukan oval"

        hull = cv2.convexHull(largest_contour)
        hull_area = cv2.contourArea(hull)
        if hull_area > 0 and float(area) / hull_area < 0.85:
            return False, "Benda memiliki lekukan (Bukan Telur)"

        return True, "Geometri valid"

    def _validate_egg_color(self, img, mask):
        """Validasi warna: tolak warna dingin pekat dan warna neon."""
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, _ = cv2.split(hsv)

        area = mask > 0
        if not np.any(area):
            return True, "Area kosong"

        mean_h = h[area].mean()
        mean_s = s[area].mean()

        # Tolak warna hijau–biru–ungu pekat (bukan warna alami telur puyuh)
        if 70 < mean_h < 180 and mean_s > 60:
            return False, "Warna tidak wajar (Bukan Telur)"

        # Tolak warna terlalu jenuh (plastik, bola, dsb.)
        if mean_s > 150:
            return False, "Warna terlalu pekat (Bukan Telur)"

        return True, "Warna valid"
