import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QFileDialog,
    QTableWidget, QTableWidgetItem, QHeaderView,
    QMessageBox, QFrame, QGraphicsDropShadowEffect,
)
from PyQt5.QtGui import QPixmap, QImage, QColor
from PyQt5.QtCore import Qt, QTimer
from .image_processor import ImageProcessor
from .classifier import EggClassifier


class DraggableROILabel(QLabel):
    """
    QLabel kustom untuk preview kamera.
    ROI oval dapat digeser (drag) dan diubah ukurannya (scroll wheel).
    """

    DEFAULT_CX, DEFAULT_CY = 0.50, 0.50
    DEFAULT_AX, DEFAULT_AY = 0.35, 0.45

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setScaledContents(True)
        self.setStyleSheet("border-radius: 12px;")

        self.roi_cx = self.DEFAULT_CX
        self.roi_cy = self.DEFAULT_CY
        self.roi_ax = self.DEFAULT_AX
        self.roi_ay = self.DEFAULT_AY

        self._interactive = False
        self._dragging = False
        self._drag_start_pos = None
        self._drag_start_roi = None
        self.on_roi_changed = None  # Callback saat ROI berubah (dipakai saat frame beku)

    def set_interactive(self, active):
        """Aktifkan/nonaktifkan interaksi mouse."""
        self._interactive = active
        if active:
            self.setCursor(Qt.OpenHandCursor)
            self.setToolTip("🖱️ Seret untuk memindahkan  •  Scroll untuk zoom in/out")
        else:
            self.setCursor(Qt.ArrowCursor)
            self.setToolTip("")

    def reset_roi(self):
        """Reset ROI ke posisi dan ukuran default."""
        self.roi_cx, self.roi_cy = self.DEFAULT_CX, self.DEFAULT_CY
        self.roi_ax, self.roi_ay = self.DEFAULT_AX, self.DEFAULT_AY

    def get_roi_params(self):
        """Kembalikan ROI sebagai dict persentase untuk ImageProcessor."""
        return {
            'cx_pct': self.roi_cx * 100,
            'cy_pct': self.roi_cy * 100,
            'ax_pct': self.roi_ax * 100,
            'ay_pct': self.roi_ay * 100,
        }

    def mousePressEvent(self, event):
        if self._interactive and event.button() == Qt.LeftButton:
            self._dragging = True
            self._drag_start_pos = event.pos()
            self._drag_start_roi = (self.roi_cx, self.roi_cy)
            self.setCursor(Qt.ClosedHandCursor)
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._interactive and self._dragging and self._drag_start_pos:
            w, h = max(self.width(), 1), max(self.height(), 1)
            dx = (event.x() - self._drag_start_pos.x()) / w
            dy = (event.y() - self._drag_start_pos.y()) / h
            margin_x = self.roi_ax + 0.02
            margin_y = self.roi_ay + 0.02
            self.roi_cx = max(margin_x, min(1.0 - margin_x, self._drag_start_roi[0] + dx))
            self.roi_cy = max(margin_y, min(1.0 - margin_y, self._drag_start_roi[1] + dy))
            if self.on_roi_changed:
                self.on_roi_changed()
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self._interactive:
            self._dragging = False
            self.setCursor(Qt.OpenHandCursor)
        super().mouseReleaseEvent(event)

    def wheelEvent(self, event):
        if self._interactive:
            scale = 1.05 if event.angleDelta().y() > 0 else 0.95
            self.roi_ax = max(0.10, min(0.49, self.roi_ax * scale))
            self.roi_ay = max(0.10, min(0.49, self.roi_ay * scale))
            if self.on_roi_changed:
                self.on_roi_changed()
            event.accept()
        else:
            super().wheelEvent(event)


class EggQualityApp(QMainWindow):
    """Window utama aplikasi. Mengoordinasikan ImageProcessor dan EggClassifier."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Klasifikasi Mutu Telur Puyuh")

        self.image_processor = ImageProcessor()
        self.classifier = EggClassifier()

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.cap = None
        self.camera_active = False
        self.frozen_frame = None
        self.is_paused = False

        self.setFixedSize(1200, 950)

        success, message = self.classifier.load_model()
        if not success:
            QMessageBox.warning(self, "Error", message)

        self.apply_styles()

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)
        self.main_layout.setContentsMargins(50, 30, 50, 30)
        self.main_layout.setSpacing(20)

        self.setup_header()
        self.setup_images_section()
        self.setup_feature_section()
        self.setup_prediction_section()

    # =========================================================================
    # Gaya
    # =========================================================================

    def apply_styles(self):
        """Mendefinisikan stylesheet global untuk aplikasi."""
        self.setStyleSheet("""
            QMainWindow { background-color: #F9FAFB; }
            QWidget {
                font-family: 'Poppins', 'Segoe UI', sans-serif;
                font-size: 14px;
                color: #374151;
            }
            QFrame#ImageCard {
                background-color: #FFFFFF;
                border-radius: 12px;
                border: 1px solid #E5E7EB;
            }
            QFrame#ContentCard {
                background-color: #FFFFFF;
                border-radius: 12px;
                border: 1px solid #E5E7EB;
            }
            QPushButton {
                background-color: #3B82F6;
                color: white;
                border: none;
                border-radius: 8px;
                padding: 10px 20px;
                font-weight: 500;
                font-size: 14px;
            }
            QPushButton:hover { background-color: #2563EB; }
            QTableWidget {
                border: none;
                background-color: transparent;
                gridline-color: #F3F4F6;
                font-size: 14px;
            }
            QHeaderView::section {
                background-color: #FFFFFF;
                padding: 10px;
                border: none;
                border-bottom: 2px solid #E5E7EB;
                font-weight: 600;
                color: #1F2937;
            }
            QTableWidget::item {
                padding: 8px;
                border-bottom: 1px solid #F9FAFB;
            }
            QLabel#Title {
                font-size: 26px;
                font-weight: 600;
                color: #111827;
            }
            QLabel#Subtitle {
                font-size: 14px;
                color: #6B7280;
                font-weight: 400;
            }
            QLabel#ImageTitle {
                font-size: 15px;
                font-weight: 500;
                color: #4B5563;
                margin-top: 10px;
            }
            QLabel#SectionTitle {
                font-size: 16px;
                font-weight: 600;
                color: #374151;
                margin-bottom: 10px;
            }
        """)

    def add_shadow(self, widget):
        """Menambahkan efek drop shadow ke widget."""
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(20)
        shadow.setColor(QColor(0, 0, 0, 15))
        shadow.setOffset(0, 4)
        widget.setGraphicsEffect(shadow)

    # =========================================================================
    # Setup UI
    # =========================================================================

    def setup_header(self):
        """Membangun bagian header: judul dan tombol-tombol aksi."""
        header_layout = QHBoxLayout()

        title_container = QVBoxLayout()
        title_label = QLabel("Klasifikasi Mutu Telur Puyuh")
        title_label.setObjectName("Title")
        subtitle_label = QLabel("Analisis Kualitas Telur Berbasis Citra Digital")
        subtitle_label.setObjectName("Subtitle")
        title_container.addWidget(title_label)
        title_container.addWidget(subtitle_label)

        def make_button(text, slot, width, style=None, visible=True):
            btn = QPushButton(text)
            btn.setCursor(Qt.PointingHandCursor)
            btn.clicked.connect(slot)
            btn.setFixedWidth(width)
            if style:
                btn.setStyleSheet(style)
            if not visible:
                btn.setVisible(False)
            self.add_shadow(btn)
            return btn

        self.btn_select = make_button("+ Pilih Gambar", self.load_image, 160)
        self.btn_camera = make_button("📷 Buka Kamera", self.toggle_camera, 160)
        self.btn_capture = make_button(
            "⚪ Ambil Foto", self.capture_frame, 160,
            style="background-color: #10B981; color: white;", visible=False
        )
        self.btn_reset_roi = make_button(
            "↺ Reset ROI", self._on_reset_roi, 120,
            style="background-color: #6B7280; color: white;", visible=False
        )
        self.btn_pause = make_button(
            "Jeda", self.toggle_pause, 100,
            style="background-color: #6366F1; color: white;", visible=False
        )

        header_layout.addLayout(title_container)
        header_layout.addStretch()
        for btn in [self.btn_reset_roi, self.btn_pause,
                    self.btn_capture, self.btn_camera, self.btn_select]:
            header_layout.addWidget(btn)

        self.main_layout.addLayout(header_layout)

    def setup_images_section(self):
        """Membangun tiga panel tampilan gambar: Asli, GLCM, dan HSV."""
        images_layout = QHBoxLayout()
        images_layout.setSpacing(30)
        images_layout.setAlignment(Qt.AlignCenter)

        self.roi_label = DraggableROILabel()
        self.roi_label.on_roi_changed = self._redraw_frozen_frame

        self.lbl_original = self.create_image_group("Citra Asli", custom_label=self.roi_label)
        self.lbl_glcm = self.create_image_group("Input GLCM")
        self.lbl_hsv = self.create_image_group("Input HSV")

        for group in [self.lbl_original, self.lbl_glcm, self.lbl_hsv]:
            images_layout.addWidget(group['container'])

        self.main_layout.addLayout(images_layout)

    def create_image_group(self, title, custom_label=None):
        """
        Membuat satu panel gambar berbingkai dengan judul di bawahnya.

        Args:
            title (str): Teks judul di bawah gambar.
            custom_label (QLabel | None): Widget label kustom; jika None dibuat QLabel biasa.

        Returns:
            dict: {'container': QWidget, 'label': QLabel}
        """
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.setAlignment(Qt.AlignCenter)

        image_frame = QFrame()
        image_frame.setObjectName("ImageCard")
        image_frame.setFixedSize(300, 300)
        self.add_shadow(image_frame)

        frame_layout = QVBoxLayout(image_frame)
        frame_layout.setContentsMargins(0, 0, 0, 0)

        if custom_label is not None:
            lbl_img = custom_label
        else:
            lbl_img = QLabel()
            lbl_img.setAlignment(Qt.AlignCenter)
            lbl_img.setScaledContents(True)
            lbl_img.setStyleSheet("border-radius: 12px;")

        frame_layout.addWidget(lbl_img)

        lbl_title = QLabel(title)
        lbl_title.setObjectName("ImageTitle")
        lbl_title.setAlignment(Qt.AlignCenter)
        lbl_title.setFixedWidth(300)

        layout.addWidget(image_frame)
        layout.addWidget(lbl_title)

        return {'container': container, 'label': lbl_img}

    def setup_feature_section(self):
        """Membangun dua tabel fitur: GLCM (tekstur) dan HSV (warna)."""
        feature_layout = QHBoxLayout()
        feature_layout.setSpacing(30)

        for label_text, attr, features in [
            ("Fitur GLCM (Tekstur)", "table_glcm",
             ['Contrast', 'Energy', 'Homogeneity', 'Correlation']),
            ("Fitur HSV (Warna)", "table_hsv",
             ['H Mean', 'S Mean', 'V Mean', 'H Std', 'S Std', 'V Std']),
        ]:
            section = QWidget()
            section_layout = QVBoxLayout(section)
            section_layout.setContentsMargins(0, 0, 0, 0)
            lbl = QLabel(label_text)
            lbl.setObjectName("SectionTitle")
            section_layout.addWidget(lbl)
            table_widget = self.create_table(features)
            section_layout.addWidget(table_widget)
            setattr(self, attr, table_widget)
            feature_layout.addWidget(section)

        self.main_layout.addLayout(feature_layout)

    def create_table(self, features):
        """
        Membuat QTableWidget bergaya untuk menampilkan nilai fitur.

        Args:
            features (list[str]): Nama-nama fitur sebagai baris tabel.

        Returns:
            QFrame: Frame yang membungkus tabel.
        """
        ROW_H, HEADER_H = 40, 45

        table_frame = QFrame()
        table_frame.setObjectName("ContentCard")
        self.add_shadow(table_frame)

        layout = QVBoxLayout(table_frame)
        layout.setContentsMargins(0, 0, 0, 0)

        table = QTableWidget()
        table.setRowCount(len(features))
        table.setColumnCount(2)
        table.setHorizontalHeaderLabels(["Fitur", "Nilai"])
        table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        table.horizontalHeader().setFixedHeight(HEADER_H)
        table.verticalHeader().setVisible(False)
        table.setEditTriggers(QTableWidget.NoEditTriggers)
        table.setSelectionMode(QTableWidget.NoSelection)
        table.setFocusPolicy(Qt.NoFocus)
        table.setShowGrid(False)
        table.setFixedHeight(HEADER_H + len(features) * ROW_H + 10)

        for i, name in enumerate(features):
            table.setItem(i, 0, QTableWidgetItem(name))
            val_item = QTableWidgetItem("-")
            val_item.setTextAlignment(Qt.AlignCenter)
            table.setItem(i, 1, val_item)
            table.setRowHeight(i, ROW_H)

        layout.addWidget(table)
        return table_frame

    def setup_prediction_section(self):
        """Membangun panel hasil prediksi di bagian bawah."""
        pred_frame = QFrame()
        pred_frame.setObjectName("ContentCard")
        pred_frame.setStyleSheet("""
            QFrame#ContentCard {
                background-color: #FFFFFF;
                border-radius: 12px;
                border-left: 6px solid #3B82F6;
            }
        """)
        self.add_shadow(pred_frame)

        pred_layout = QHBoxLayout(pred_frame)
        pred_layout.setContentsMargins(30, 25, 30, 25)

        lbl_title = QLabel("Hasil Prediksi:")
        lbl_title.setStyleSheet("font-size: 18px; font-weight: 500; color: #4B5563;")

        self.lbl_prediction = QLabel("-")
        self.lbl_prediction.setStyleSheet(
            "font-size: 24px; font-weight: 600; color: #111827; margin-left: 20px;"
        )

        pred_layout.addWidget(lbl_title)
        pred_layout.addWidget(self.lbl_prediction)
        pred_layout.addStretch()

        self.main_layout.addWidget(pred_frame)

    # =========================================================================
    # Logika Pemrosesan Gambar
    # =========================================================================

    def load_image(self):
        """Membuka dialog file dan memproses gambar yang dipilih."""
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Pilih Gambar Telur", "",
            "Images (*.png *.jpg *.jpeg *.bmp)", options=options
        )
        if file_path:
            self.process_image(file_path, source_type='file')

    def process_image(self, input_data, source_type='file', roi_params=None):
        """
        Alur pemrosesan utama: pra-proses → tampilkan → validasi → ekstraksi → prediksi.

        Args:
            input_data (str | np.ndarray): Path file atau frame NumPy array.
            source_type (str): 'file' atau 'webcam'.
            roi_params (dict | None): Parameter ROI dari DraggableROILabel.
        """
        img, gray, mask, masked_gray, masked_color = self.image_processor.preprocess(
            input_data, source_type, roi_params
        )

        if img is None:
            QMessageBox.warning(self, "Error", "Tidak dapat membaca file gambar.")
            return

        self.display_image(img, self.lbl_original['label'])
        self.display_image(masked_gray, self.lbl_glcm['label'], is_gray=True)
        self.display_image(masked_color, self.lbl_hsv['label'])

        is_egg, reason = self.image_processor.check_is_object_egg(img, mask)
        if not is_egg:
            self._fill_table_zeros()
            self.lbl_prediction.setText(f"Bukan Telur ({reason})")
            self.lbl_prediction.setStyleSheet(
                "font-size: 18px; font-weight: 600; color: #EF4444; margin-left: 20px;"
            )
            return

        features = self.image_processor.extract_features(img, gray, mask, masked_gray, masked_color)
        self._update_feature_tables(features)

        prediction = self.classifier.predict(features)
        self.update_prediction_label(prediction)

    def _fill_table_zeros(self):
        """Mengisi semua sel nilai tabel fitur dengan angka nol."""
        for table_frame, n_rows in [(self.table_glcm, 4), (self.table_hsv, 6)]:
            table = table_frame.findChild(QTableWidget)
            for i in range(n_rows):
                item = QTableWidgetItem("0")
                item.setTextAlignment(Qt.AlignCenter)
                table.setItem(i, 1, item)

    def _update_feature_tables(self, features):
        """Mengisi tabel GLCM (4 fitur) dan HSV (6 fitur) dari daftar fitur."""
        for table_frame, indices in [
            (self.table_glcm, range(4)),
            (self.table_hsv, range(4, 10)),
        ]:
            table = table_frame.findChild(QTableWidget)
            for row, idx in enumerate(indices):
                item = QTableWidgetItem(f"{features[idx]:.4f}")
                item.setTextAlignment(Qt.AlignCenter)
                table.setItem(row, 1, item)

    def update_prediction_label(self, prediction):
        """Memperbarui teks dan warna label hasil prediksi."""
        color_map = {
            'Baik': '#10B981',
            'Sedang': '#F59E0B',
        }
        color = color_map.get(prediction, '#EF4444')
        self.lbl_prediction.setText(prediction)
        self.lbl_prediction.setStyleSheet(
            f"font-size: 24px; font-weight: 600; color: {color}; margin-left: 20px;"
        )

    def display_image(self, img_array, label_widget, is_gray=False):
        """Mengonversi array OpenCV ke QPixmap dan menampilkannya pada QLabel."""
        if is_gray:
            h, w = img_array.shape
            q_img = QImage(img_array.data, w, h, w, QImage.Format_Grayscale8)
        else:
            h, w, ch = img_array.shape
            rgb = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
            q_img = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        label_widget.setPixmap(QPixmap.fromImage(q_img))

    # =========================================================================
    # Kontrol Kamera
    # =========================================================================

    def toggle_camera(self):
        """Membuka atau menutup kamera."""
        if not self.camera_active:
            
            # --- PENGATURAN KAMERA ---
            # Ubah angka 0 menjadi 1 atau 2 jika menggunakan kamera eksternal (misal: Iriun/DroidCam).
            CAMERA_INDEX = 0 
            self.cap = cv2.VideoCapture(CAMERA_INDEX)
            
            if not self.cap.isOpened():
                QMessageBox.warning(self, "Error", "Tidak dapat mendeteksi atau membuka kamera.")
                self.cap = None
                return
            self.camera_active = True
            self.timer.start(30)
            self.btn_camera.setText("⏹️ Tutup Kamera")
            self.btn_camera.setStyleSheet("background-color: #EF4444; color: white;")
            self.btn_capture.setVisible(True)
            self.roi_label.set_interactive(True)
            self.btn_reset_roi.setVisible(True)
            self.btn_pause.setVisible(True)
        else:
            self.stop_camera()

    def stop_camera(self):
        """Menghentikan kamera dan mereset semua state terkait."""
        self.timer.stop()
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.camera_active = False
        self.is_paused = False
        self.frozen_frame = None

        self.btn_camera.setText("📷 Buka Kamera")
        self.btn_camera.setStyleSheet("")
        self.btn_pause.setText("Jeda")
        self.btn_pause.setStyleSheet("background-color: #6366F1; color: white;")

        self.btn_capture.setVisible(False)
        self.btn_reset_roi.setVisible(False)
        self.btn_pause.setVisible(False)

        self.roi_label.set_interactive(False)
        self.lbl_original['label'].clear()

    def toggle_pause(self):
        """Bekukan atau cairkan live feed kamera untuk pengaturan ROI yang presisi."""
        if not self.is_paused:
            self.timer.stop()
            self.is_paused = True
            self.btn_pause.setText("Resume")
            self.btn_pause.setStyleSheet("background-color: #F59E0B; color: white;")
        else:
            self.frozen_frame = None
            self.is_paused = False
            self.timer.start(30)
            self.btn_pause.setText("Jeda")
            self.btn_pause.setStyleSheet("background-color: #6366F1; color: white;")

    def update_frame(self):
        """Membaca frame dari kamera, menggambar ROI, dan menampilkannya."""
        if self.cap is None or not self.cap.isOpened() or self.is_paused:
            return

        ret, frame = self.cap.read()
        if not ret:
            return

        # --- PENGATURAN EFEK CERMIN (MIRROR) ---
        # Gunakan cv2.flip jika ingin tampilan seperti cermin (kiri jadi kanan).
        # Hapus/beri tanda pagar (#) pada baris di bawah jika Anda memfoto benda bersisi tulisan agar tidak terbalik.
        frame = cv2.flip(frame, 1)
        
        h, w = frame.shape[:2]
        min_dim = min(h, w)
        sx, sy = (w - min_dim) // 2, (h - min_dim) // 2
        frame_square = frame[sy:sy + min_dim, sx:sx + min_dim]
        self.frozen_frame = frame_square.copy()

        self._draw_roi_and_show(frame_square)

    def capture_frame(self):
        """Mengambil frame (dari frozen atau live) dan memprosesnya."""
        if not self.camera_active or self.cap is None:
            return

        if self.is_paused and self.frozen_frame is not None:
            frame_square = self.frozen_frame
        else:
            ret, frame = self.cap.read()
            if not ret:
                return
                
            # --- PENGATURAN EFEK CERMIN (MIRROR) ---
            # Pastikan pengaturan di sini sama persis dengan yang ada di fungsi update_frame() di atas.
            frame = cv2.flip(frame, 1)
            
            h, w = frame.shape[:2]
            min_dim = min(h, w)
            sx, sy = (w - min_dim) // 2, (h - min_dim) // 2
            frame_square = frame[sy:sy + min_dim, sx:sx + min_dim]

        roi_params = self.roi_label.get_roi_params()
        self.stop_camera()
        self.process_image(frame_square, source_type='webcam', roi_params=roi_params)

    # =========================================================================
    # Helper Kamera & ROI
    # =========================================================================

    def _draw_roi_and_show(self, frame):
        """Menggambar efek viewfinder (luar gelap, dalam terang) pada panel kamera."""
        display = frame.copy()
        h, w = display.shape[:2]
        cx = int(w * self.roi_label.roi_cx)
        cy = int(h * self.roi_label.roi_cy)
        ax = int(w * self.roi_label.roi_ax)
        ay = int(h * self.roi_label.roi_ay)

        # 1. Buat versi gelap dari frame asli (Opacity 40%)
        dark_frame = cv2.addWeighted(display, 0.5, np.zeros_like(display), 0, 0)

        # 2. Buat mask untuk area dalam oval
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.ellipse(mask, (cx, cy), (ax, ay), 0, 0, 360, 255, -1)

        # 3. Gabungkan: dalam oval = terang, luar oval = gelap
        display = np.where(mask[:, :, np.newaxis] == 255, display, dark_frame)

        # 4. Gambar garis tepi oval (putih sedikit kehijauan)
        cv2.ellipse(display, (cx, cy), (ax, ay), 0, 0, 360, (200, 255, 200), 2)

        self.display_image(display, self.lbl_original['label'], is_gray=False)

    def _redraw_frozen_frame(self):
        """Gambar ulang frozen frame dengan ROI terkini (callback saat drag/scroll di mode jeda)."""
        if self.is_paused and self.frozen_frame is not None:
            self._draw_roi_and_show(self.frozen_frame)

    def _on_reset_roi(self):
        """Reset ROI ke default dan perbarui tampilan jika sedang dalam mode jeda."""
        self.roi_label.reset_roi()
        self._redraw_frozen_frame()
