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
        Membaca citra ukuran asli dan menerapkan masking.

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

        if source_type == 'webcam':
            # Webcam: pakai ROI + remove background + mask adaptif,
            # supaya yang tersisa benar-benar objek telur.
            roi_mask = self._create_static_roi_mask(img.shape[:2], roi_params)
            img_fg, fg_mask = self.remove_background(img)
            mask_raw = self.create_mask(img_fg)
            mask = cv2.bitwise_and(mask_raw, fg_mask)
            mask = cv2.bitwise_and(mask, roi_mask)
            solid_mask = self._create_solid_mask(mask)
        else:
            # Upload file: hapus background dulu agar objek telur lebih dominan.
            img_fg, fg_mask = self.remove_background(img)
            mask_raw = self.create_mask(img_fg)
            mask = cv2.bitwise_and(mask_raw, fg_mask)
            solid_mask = self._create_solid_mask(mask)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Saat frame diproses (termasuk dari kamera setelah capture), tampilkan
        # hasil objek telur saja dengan mask.
        masked_gray = cv2.bitwise_and(gray, gray, mask=solid_mask)
        masked_color = cv2.bitwise_and(img, img, mask=solid_mask)

        return img, gray, mask, masked_gray, masked_color

    def remove_background(self, img):
        """
        Remove background berbasis GrabCut untuk gambar (upload/webcam).
        Mengembalikan:
            - img_fg: gambar foreground (background dihitamkan)
            - fg_mask: mask biner foreground (0/255)
        """
        h, w = img.shape[:2]
        if h < 20 or w < 20:
            fg_mask = np.ones((h, w), dtype=np.uint8) * 255
            return img.copy(), fg_mask

        # Inisialisasi GrabCut dengan rectangle menyisakan margin tepi.
        gc_mask = np.zeros((h, w), np.uint8)
        bg_model = np.zeros((1, 65), np.float64)
        fg_model = np.zeros((1, 65), np.float64)

        margin_x = max(5, int(w * 0.04))
        margin_y = max(5, int(h * 0.04))
        rect = (margin_x, margin_y, max(1, w - 2 * margin_x), max(1, h - 2 * margin_y))

        try:
            cv2.grabCut(img, gc_mask, rect, bg_model, fg_model, 3, cv2.GC_INIT_WITH_RECT)
            fg_mask = np.where(
                (gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD), 255, 0
            ).astype(np.uint8)
        except Exception:
            # Fallback aman: jangan buang apapun kalau GrabCut gagal.
            fg_mask = np.ones((h, w), dtype=np.uint8) * 255

        # Rapikan mask foreground.
        min_dim = max(1, min(h, w))
        k = max(3, int(min_dim * 0.012))
        if k % 2 == 0:
            k += 1
        kernel = np.ones((k, k), np.uint8)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        img_fg = cv2.bitwise_and(img, img, mask=fg_mask)
        return img_fg, fg_mask

    def create_mask(self, img):
        """
        Membuat binary mask dengan Otsu Thresholding pada channel Saturation HSV.
        Digunakan untuk input gambar dari file.
        """
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        s_channel = hsv[:, :, 1]
        v_channel = hsv[:, :, 2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        s_blur = cv2.GaussianBlur(s_channel, (5, 5), 0)
        v_blur = cv2.GaussianBlur(v_channel, (5, 5), 0)
        gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)

        # Bangun beberapa kandidat mask lalu pilih yang area objek utamanya
        # paling besar. Ini lebih robust untuk telur terang di latar gelap.
        _, mask_s = cv2.threshold(s_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        _, mask_v = cv2.threshold(v_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        _, mask_g = cv2.threshold(gray_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        h, w = img.shape[:2]
        min_dim = max(1, min(h, w))
        k = max(3, int(min_dim * 0.015))
        if k % 2 == 0:
            k += 1
        kernel = np.ones((k, k), np.uint8)
        candidates = []
        for raw in (mask_s, mask_v, mask_g):
            m = cv2.morphologyEx(raw, cv2.MORPH_OPEN, kernel, iterations=1)
            m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel, iterations=3)
            candidates.append(m)

        def largest_area_ratio(candidate):
            contours, _ = cv2.findContours(candidate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                return 0.0
            area = cv2.contourArea(max(contours, key=cv2.contourArea))
            return area / float(h * w)

        best_mask = max(candidates, key=largest_area_ratio)

        # Jika hasil masih terlalu kecil, fallback ke threshold grayscale longgar.
        if largest_area_ratio(best_mask) < 0.02:
            _, fallback = cv2.threshold(gray_blur, 20, 255, cv2.THRESH_BINARY)
            fallback = cv2.morphologyEx(fallback, cv2.MORPH_CLOSE, kernel, iterations=4)
            best_mask = fallback

        # Buang background putih polos (umum pada foto studio produk).
        # Piksel putih biasanya saturasi rendah + value tinggi.
        non_white_bg = np.where((s_channel < 22) & (v_channel > 240), 0, 255).astype(np.uint8)
        best_mask = cv2.bitwise_and(best_mask, non_white_bg)
        best_mask = cv2.morphologyEx(best_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        # Jika hasil terlalu besar (misal tangan + latar ikut ter-segmentasi),
        # lakukan pemisahan agresif lalu pilih kontur yang paling menyerupai oval.
        if largest_area_ratio(best_mask) > 0.70:
            k_big = max(5, int(min_dim * 0.05))
            if k_big % 2 == 0:
                k_big += 1
            big_kernel = np.ones((k_big, k_big), np.uint8)

            refined = cv2.morphologyEx(best_mask, cv2.MORPH_OPEN, big_kernel, iterations=1)
            refined = cv2.morphologyEx(refined, cv2.MORPH_CLOSE, kernel, iterations=2)

            contours, _ = cv2.findContours(refined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                canvas = np.zeros_like(refined)
                best_score = -1.0
                best_contour = None

                for cnt in contours:
                    area = cv2.contourArea(cnt)
                    if area <= 0:
                        continue
                    x, y, bw, bh = cv2.boundingRect(cnt)
                    if bw <= 0 or bh <= 0:
                        continue
                    ar = float(bw) / bh
                    if ar < 0.35 or ar > 2.6:
                        continue

                    hull = cv2.convexHull(cnt)
                    hull_area = cv2.contourArea(hull)
                    solidity = (area / hull_area) if hull_area > 0 else 0.0
                    area_ratio = area / float(h * w)

                    # Skor mengutamakan kontur cukup besar, padat, dan rasio oval.
                    ar_penalty = abs(ar - 0.75)
                    score = (2.0 * area_ratio) + (1.2 * solidity) - (0.25 * ar_penalty)
                    if score > best_score:
                        best_score = score
                        best_contour = cnt

                if best_contour is not None:
                    cv2.drawContours(canvas, [best_contour], -1, 255, thickness=cv2.FILLED)
                    best_mask = canvas

        # Tahap akhir: pilih satu kontur paling "egg-like" dan dekat pusat frame.
        # Ini menstabilkan kasus telur dipegang tangan atau latar bertekstur.
        return self._select_primary_egg_contour(best_mask)

    def _select_primary_egg_contour(self, mask):
        """Pilih satu kontur utama yang paling menyerupai telur."""
        h, w = mask.shape[:2]
        frame_area = float(h * w)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return mask

        cx0, cy0 = w / 2.0, h / 2.0
        diag = np.hypot(w, h)
        best_score = -1e9
        best_contour = None

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area <= 0:
                continue

            area_ratio = area / frame_area
            if area_ratio < 0.01:
                continue
            if area_ratio > 0.92:
                continue

            x, y, bw, bh = cv2.boundingRect(cnt)
            if bw <= 0 or bh <= 0:
                continue

            # Abaikan kontur yang menempel sisi frame (biasanya background).
            border_touch = (x <= 1) or (y <= 1) or ((x + bw) >= (w - 1)) or ((y + bh) >= (h - 1))
            if border_touch:
                continue

            ar = float(bw) / bh
            if ar < 0.28 or ar > 2.8:
                continue

            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            solidity = (area / hull_area) if hull_area > 0 else 0.0

            m = cv2.moments(cnt)
            if m["m00"] == 0:
                continue
            cx = m["m10"] / m["m00"]
            cy = m["m01"] / m["m00"]
            center_dist = np.hypot(cx - cx0, cy - cy0) / diag

            # Score:
            # + area moderat (prefer 6%..55% frame)
            # + bentuk cembung
            # + dekat pusat
            area_pref = 1.0 - min(abs(area_ratio - 0.22) / 0.22, 1.0)
            ar_pref = 1.0 - min(abs(ar - 0.75) / 0.75, 1.0)
            score = (
                (2.0 * area_pref) +
                (1.6 * solidity) +
                (1.0 * ar_pref) -
                (1.8 * center_dist)
            )

            if score > best_score:
                best_score = score
                best_contour = cnt

        if best_contour is None:
            return mask

        selected = np.zeros_like(mask)
        cv2.drawContours(selected, [best_contour], -1, 255, thickness=cv2.FILLED)
        return selected

    def _create_static_roi_mask(self, image_shape, roi_params=None):
        """
        Membuat mask oval untuk mode webcam.

        Args:
            roi_params (dict | None): {'cx_pct', 'cy_pct', 'ax_pct', 'ay_pct'}.
                                      Jika None, gunakan nilai default (tengah, 35%x45%).
        """
        h, w = image_shape
        mask = np.zeros((h, w), dtype=np.uint8)
        if roi_params:
            cx = int(w * roi_params['cx_pct'] / 100)
            cy = int(h * roi_params['cy_pct'] / 100)
            ax = int(w * roi_params['ax_pct'] / 100)
            ay = int(h * roi_params['ay_pct'] / 100)
            center, axes = (cx, cy), (ax, ay)
        else:
            center, axes = (w // 2, h // 2), (int(w * 0.35), int(h * 0.45))
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
    # Validasi Telur Puyuh
    # -------------------------------------------------------------------------

    def check_is_object_egg(self, img, mask, masked_gray=None, source_type='file'):
        """
        Validasi tiga tahap untuk memastikan objek adalah telur puyuh:
            1. Uji Geometri (bentuk oval & ukuran wajar)
            2. Uji Warna   (krem/cokelat hangat khas telur puyuh)
            3. Uji Tekstur (pola bintik/speckle khas telur puyuh)

        Returns:
            tuple: (bool, str) — (valid, alasan)
        """
        # Gunakan mask solid agar evaluasi bentuk lebih stabil terhadap
        # noise/cekungan kecil pada tepi hasil thresholding.
        geometry_mask = self._create_solid_mask(mask)

        is_valid, reason = self._validate_egg_geometry(geometry_mask)
        if not is_valid:
            return False, reason

        # Untuk upload file, validasi warna dibuat lebih longgar agar tidak
        # menolak telur puyuh asli karena variasi pencahayaan/white-balance.
        # Untuk mengurangi false-negative pada kamera (pengaruh lighting/white balance),
        # validasi warna dibuat non-strict.
        strict_color = False
        is_valid, reason = self._validate_egg_color(img, mask, strict_mode=strict_color)
        if not is_valid:
            return False, reason

        if masked_gray is not None:
            is_valid, reason = self._validate_egg_texture(masked_gray, mask)
            if not is_valid:
                return False, reason

        return True, "Telur puyuh valid"

    def _validate_egg_geometry(self, mask):
        """Validasi bentuk: luas relatif, aspect ratio, dan solidity (kontur cembung)."""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return False, "Tidak ada objek terdeteksi"

        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)
        img_area = float(mask.shape[0] * mask.shape[1])
        area_ratio = area / img_area if img_area > 0 else 0.0

        # Gunakan rasio area agar robust untuk ukuran gambar yang tidak seragam.
        if area_ratio < 0.03:
            return False, "Objek terlalu kecil"
        if area_ratio > 0.80:
            return False, "Objek terlalu besar / terlalu dekat"

        x, y, w, h = cv2.boundingRect(largest_contour)
        if h == 0:
            return False, "Area tidak valid"

        aspect_ratio = float(w) / h
        if aspect_ratio < 0.5 or aspect_ratio > 2.0:
            return False, "Bentuk bukan oval"

        hull = cv2.convexHull(largest_contour)
        hull_area = cv2.contourArea(hull)
        # Sedikit dilonggarkan agar telur puyuh asli dengan pola gelap kontras
        # atau kontur hasil segmentasi yang kurang rapi tidak mudah ditolak.
        if hull_area > 0 and float(area) / hull_area < 0.80:
            return False, "Bentuk tidak cembung (bukan telur)"

        return True, "Geometri valid"

    def _validate_egg_color(self, img, mask, strict_mode=True):
        """
        Validasi warna telur puyuh:
            - Cangkang dasar krem/beige hingga cokelat (hue hangat).
            - Saturasi & kecerahan dalam rentang wajar (bukan plastik / neon / pucat ekstrem).
        """
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        area = mask > 0
        if not np.any(area):
            return False, "Area kosong"

        mean_h = float(h[area].mean())
        mean_s = float(s[area].mean())
        mean_v = float(v[area].mean())

        # Tolak warna dingin pekat (hijau / biru / ungu).
        # Pada mode non-strict, ambang dibuat lebih ketat agar tidak mudah false negative.
        sat_cold_threshold = 40 if strict_mode else 85
        if 35 < mean_h < 165 and mean_s > sat_cold_threshold:
            return False, "Warna tidak alami (bukan telur puyuh)"

        # Tolak warna terlalu pekat / neon (plastik, bola, dsb.).
        if mean_s > 160:
            return False, "Warna terlalu pekat (bukan telur puyuh)"

        # Tolak objek terlalu gelap (bayangan / benda hitam).
        if mean_v < 55:
            return False, "Objek terlalu gelap (bukan telur puyuh)"

        # Tolak putih polos / pucat (telur ayam, kertas, dinding).
        if mean_s < 18 and mean_v > 220:
            return False, "Warna terlalu pucat (bukan telur puyuh)"

        # Pastikan hue hangat hanya untuk mode strict (webcam).
        warm_hue = (mean_h <= 35) or (mean_h >= 165)
        if strict_mode and (not warm_hue) and mean_s > 25:
            return False, "Warna tidak khas telur puyuh"

        return True, "Warna valid"

    def _validate_egg_texture(self, masked_gray, mask):
        """
        Validasi tekstur: telur puyuh memiliki bintik gelap (speckle) yang
        membuat variasi intensitas pada cangkangnya. Objek polos (bola, buah
        mulus, kertas) akan ditolak karena variasinya terlalu rendah.
        """
        area = mask > 0
        if not np.any(area):
            return False, "Area kosong"

        pixels = masked_gray[area]
        std_intensity = float(pixels.std())

        if std_intensity < 8.0:
            return False, "Permukaan terlalu polos (bukan telur puyuh)"

        return True, "Tekstur valid"
