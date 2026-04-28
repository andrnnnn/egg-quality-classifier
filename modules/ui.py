import cv2
from collections import Counter
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QFileDialog,
    QTableWidget, QTableWidgetItem, QHeaderView,
    QMessageBox, QFrame, QGraphicsDropShadowEffect,
    QDialog, QSlider, QInputDialog,
)
from PyQt5.QtGui import QPixmap, QImage, QColor, QPainter, QPen
from PyQt5.QtCore import Qt, QTimer, QRect, QPoint
from .image_processor import ImageProcessor
from .classifier import EggClassifier


class ImagePreviewLabel(QLabel):
    """Label preview dengan dukungan seleksi crop via drag mouse."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("background-color: #111827; border-radius: 8px;")
        self.setMinimumSize(520, 340)
        self._img_bgr = None
        self._display_rect = QRect()
        self._pixmap = None
        self._dragging = False
        self._start = QPoint()
        self._end = QPoint()

    def set_image(self, img_bgr):
        self._img_bgr = img_bgr.copy()
        self._start = QPoint()
        self._end = QPoint()
        self._update_pixmap()

    def _update_pixmap(self):
        if self._img_bgr is None:
            self.clear()
            self._display_rect = QRect()
            self._pixmap = None
            return
        h, w, ch = self._img_bgr.shape
        rgb = cv2.cvtColor(self._img_bgr, cv2.COLOR_BGR2RGB)
        q_img = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        self._pixmap = QPixmap.fromImage(q_img)
        scaled = self._pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        x = (self.width() - scaled.width()) // 2
        y = (self.height() - scaled.height()) // 2
        self._display_rect = QRect(x, y, scaled.width(), scaled.height())
        super().setPixmap(scaled)
        self.update()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_pixmap()

    def mousePressEvent(self, event):
        if self._display_rect.contains(event.pos()) and event.button() == Qt.LeftButton:
            self._dragging = True
            self._start = event.pos()
            self._end = event.pos()
            self.update()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._dragging:
            self._end = event.pos()
            self.update()
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self._dragging and event.button() == Qt.LeftButton:
            self._end = event.pos()
            self._dragging = False
            self.update()
        super().mouseReleaseEvent(event)

    def paintEvent(self, event):
        super().paintEvent(event)
        if self._start.isNull() or self._end.isNull():
            return
        rect = self.get_selection_rect()
        if rect.isNull() or rect.width() < 4 or rect.height() < 4:
            return
        qp = QPainter(self)
        qp.setPen(QPen(QColor(59, 130, 246), 2, Qt.DashLine))
        qp.drawRect(rect)
        qp.end()

    def get_selection_rect(self):
        if self._start.isNull() or self._end.isNull():
            return QRect()
        left = max(min(self._start.x(), self._end.x()), self._display_rect.left())
        right = min(max(self._start.x(), self._end.x()), self._display_rect.right())
        top = max(min(self._start.y(), self._end.y()), self._display_rect.top())
        bottom = min(max(self._start.y(), self._end.y()), self._display_rect.bottom())
        return QRect(QPoint(left, top), QPoint(right, bottom)).normalized()

    def get_selection_in_image_coords(self):
        if self._img_bgr is None or self._display_rect.isNull():
            return None
        rect = self.get_selection_rect()
        if rect.isNull() or rect.width() < 6 or rect.height() < 6:
            return None
        ih, iw = self._img_bgr.shape[:2]
        sx = iw / float(self._display_rect.width())
        sy = ih / float(self._display_rect.height())
        x1 = int((rect.left() - self._display_rect.left()) * sx)
        y1 = int((rect.top() - self._display_rect.top()) * sy)
        x2 = int((rect.right() - self._display_rect.left()) * sx)
        y2 = int((rect.bottom() - self._display_rect.top()) * sy)
        x1 = max(0, min(iw - 1, x1))
        y1 = max(0, min(ih - 1, y1))
        x2 = max(0, min(iw, x2))
        y2 = max(0, min(ih, y2))
        if x2 - x1 < 8 or y2 - y1 < 8:
            return None
        return x1, y1, x2, y2


class UploadPreviewDialog(QDialog):
    """Dialog preview upload: crop, flip, dan resize."""

    def __init__(self, img_bgr, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Preview Gambar Upload")
        self.setModal(True)
        self.setMinimumSize(760, 560)
        self.original_img = img_bgr.copy()
        self.edited_img = img_bgr.copy()

        layout = QVBoxLayout(self)
        self.preview = ImagePreviewLabel()
        self.preview.set_image(self.edited_img)
        layout.addWidget(self.preview)

        tools = QHBoxLayout()
        self.btn_flip_h = QPushButton("Flip Horizontal")
        self.btn_flip_v = QPushButton("Flip Vertical")
        self.btn_crop = QPushButton("Crop Selection")
        self.btn_reset = QPushButton("Reset")
        tools.addWidget(self.btn_flip_h)
        tools.addWidget(self.btn_flip_v)
        tools.addWidget(self.btn_crop)
        tools.addWidget(self.btn_reset)
        layout.addLayout(tools)

        resize_row = QHBoxLayout()
        self.lbl_resize = QLabel("Resize: 100%")
        self.slider_resize = QSlider(Qt.Horizontal)
        self.slider_resize.setMinimum(40)
        self.slider_resize.setMaximum(160)
        self.slider_resize.setValue(100)
        resize_row.addWidget(self.lbl_resize)
        resize_row.addWidget(self.slider_resize)
        layout.addLayout(resize_row)

        actions = QHBoxLayout()
        actions.addStretch()
        self.btn_cancel = QPushButton("Batal")
        self.btn_ok = QPushButton("Gunakan Gambar Ini")
        actions.addWidget(self.btn_cancel)
        actions.addWidget(self.btn_ok)
        layout.addLayout(actions)

        self.btn_flip_h.clicked.connect(self.flip_horizontal)
        self.btn_flip_v.clicked.connect(self.flip_vertical)
        self.btn_crop.clicked.connect(self.crop_selection)
        self.btn_reset.clicked.connect(self.reset_image)
        self.slider_resize.valueChanged.connect(self.on_resize_change)
        self.btn_cancel.clicked.connect(self.reject)
        self.btn_ok.clicked.connect(self.accept)

    def flip_horizontal(self):
        self.edited_img = cv2.flip(self.edited_img, 1)
        self.preview.set_image(self.edited_img)

    def flip_vertical(self):
        self.edited_img = cv2.flip(self.edited_img, 0)
        self.preview.set_image(self.edited_img)

    def crop_selection(self):
        coords = self.preview.get_selection_in_image_coords()
        if not coords:
            QMessageBox.information(self, "Info", "Pilih area crop dengan drag mouse di preview.")
            return
        x1, y1, x2, y2 = coords
        cropped = self.edited_img[y1:y2, x1:x2]
        if cropped.size == 0:
            QMessageBox.warning(self, "Error", "Area crop tidak valid.")
            return
        self.edited_img = cropped
        self.preview.set_image(self.edited_img)

    def reset_image(self):
        self.edited_img = self.original_img.copy()
        self.slider_resize.setValue(100)
        self.preview.set_image(self.edited_img)

    def on_resize_change(self, value):
        self.lbl_resize.setText(f"Resize: {value}%")

    def get_result_image(self):
        scale = self.slider_resize.value() / 100.0
        if abs(scale - 1.0) < 1e-9:
            return self.edited_img
        h, w = self.edited_img.shape[:2]
        new_w = max(32, int(w * scale))
        new_h = max(32, int(h * scale))
        return cv2.resize(self.edited_img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)


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
        self.selected_camera_index = 0

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
            img = cv2.imread(file_path)
            if img is None or img.size == 0:
                QMessageBox.warning(self, "Error", "Tidak dapat membaca file gambar.")
                return
            dialog = UploadPreviewDialog(img, self)
            if dialog.exec_() == QDialog.Accepted:
                edited_img = dialog.get_result_image()
                self.process_image(edited_img, source_type='file')

    def process_image(self, input_data, source_type='file', roi_params=None, prediction_override=None):
        """
        Alur pemrosesan utama: pra-proses → tampilkan → validasi → ekstraksi → prediksi.

        Bila objek bukan telur puyuh, prediksi dihentikan dan label menampilkan
        "Bukan Telur Puyuh" beserta alasannya. Bila valid, fitur diekstraksi dan
        diskor menggunakan model SVM + scaler yang telah dimuat.

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

        is_egg, reason = self.image_processor.check_is_object_egg(
            img, mask, masked_gray, source_type=source_type
        )
        if not is_egg:
            self._fill_table_zeros()
            self.update_prediction_label("Bukan Telur Puyuh", reason=reason)
            return

        if self.classifier.model is None or self.classifier.scaler is None:
            self._fill_table_zeros()
            self.update_prediction_label("Model belum dimuat")
            return

        features = self.image_processor.extract_features(img, gray, mask, masked_gray, masked_color)
        self._update_feature_tables(features)

        prediction = prediction_override if prediction_override is not None else self.classifier.predict(features)
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

    def update_prediction_label(self, prediction, reason=None):
        """
        Memperbarui teks dan warna label hasil prediksi.

        Args:
            prediction (str): Hasil model ('Baik', 'Sedang', 'Buruk') atau
                              status non-klasifikasi seperti 'Bukan Telur Puyuh'.
            reason (str | None): Alasan tambahan, ditampilkan di bawah label.
        """
        color_map = {
            'Baik': '#10B981',
            'Sedang': '#F59E0B',
            'Buruk': '#EF4444',
        }
        color = color_map.get(prediction, '#EF4444')

        if reason:
            text = f"{prediction}  —  {reason}"
            font_size = 18
        else:
            text = str(prediction)
            font_size = 24

        self.lbl_prediction.setText(text)
        self.lbl_prediction.setStyleSheet(
            f"font-size: {font_size}px; font-weight: 600; "
            f"color: {color}; margin-left: 20px;"
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
            selected = self._select_camera_source()
            if selected is None:
                return

            self.selected_camera_index = selected
            self.cap = cv2.VideoCapture(self.selected_camera_index)
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

    def _select_camera_source(self):
        """
        Menyediakan pilihan kamera (internal/USB) dari device yang terdeteksi.
        Return:
            int | None: index kamera yang dipilih, atau None jika batal/tidak ada kamera.
        """
        available = []
        for idx in range(5):
            cap = cv2.VideoCapture(idx)
            if cap is not None and cap.isOpened():
                available.append(idx)
                cap.release()
            elif cap is not None:
                cap.release()

        if not available:
            QMessageBox.warning(self, "Error", "Tidak ada kamera yang terdeteksi.")
            return None

        # Nama pilihan dibuat sederhana sesuai permintaan dua sumber kamera.
        labels = []
        for i, idx in enumerate(available):
            if i == 0:
                labels.append(f"Kamera Laptop (index {idx})")
            elif i == 1:
                labels.append(f"USB Webcam (index {idx})")
            else:
                labels.append(f"Kamera Lain {i + 1} (index {idx})")

        if len(labels) == 1:
            return available[0]

        choice, ok = QInputDialog.getItem(
            self,
            "Pilih Sumber Kamera",
            "Pilih kamera yang ingin digunakan:",
            labels,
            0,
            False,
        )
        if not ok:
            return None

        selected_pos = labels.index(choice)
        return available[selected_pos]

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
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            min_dim = min(h, w)
            sx, sy = (w - min_dim) // 2, (h - min_dim) // 2
            frame_square = frame[sy:sy + min_dim, sx:sx + min_dim]

        roi_params = self.roi_label.get_roi_params()

        # Stabilkan prediksi webcam dengan voting beberapa frame.
        final_prediction = self._predict_webcam_stable(frame_square, roi_params, n_frames=3)

        self.stop_camera()
        self.process_image(
            frame_square,
            source_type='webcam',
            roi_params=roi_params,
            prediction_override=final_prediction
        )

    def _predict_webcam_stable(self, first_frame_square, roi_params, n_frames=3):
        """
        Mengambil beberapa frame webcam dan melakukan majority voting agar
        prediksi tidak mudah berubah karena noise/lighting sesaat.
        """
        frames = [first_frame_square]

        while len(frames) < n_frames and self.cap is not None and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            min_dim = min(h, w)
            sx, sy = (w - min_dim) // 2, (h - min_dim) // 2
            frame_square = frame[sy:sy + min_dim, sx:sx + min_dim]
            frames.append(frame_square)

        votes = []
        for frame_square in frames:
            img, gray, mask, masked_gray, masked_color = self.image_processor.preprocess(
                frame_square, source_type='webcam', roi_params=roi_params
            )
            if img is None:
                continue

            is_egg, _ = self.image_processor.check_is_object_egg(
                img, mask, masked_gray, source_type='webcam'
            )
            if not is_egg:
                continue

            features = self.image_processor.extract_features(img, gray, mask, masked_gray, masked_color)
            pred = self.classifier.predict(features)
            if pred in ("Baik", "Sedang", "Buruk"):
                votes.append(pred)
                # Early stop: jika 2 vote sudah sama, mayoritas tidak akan berubah.
                if len(votes) >= 2 and votes[-1] == votes[-2]:
                    break

        if not votes:
            return None

        counts = Counter(votes)
        top_count = max(counts.values())
        tied = [label for label, c in counts.items() if c == top_count]
        priority = {"Baik": 0, "Sedang": 1, "Buruk": 2}
        tied.sort(key=lambda x: priority.get(x, 99))
        return tied[0]

    # =========================================================================
    # Helper Kamera & ROI
    # =========================================================================

    def _draw_roi_and_show(self, frame):
        """Menggambar elips ROI pada frame dan menampilkannya di panel kamera."""
        display = frame.copy()
        h, w = display.shape[:2]
        cx = int(w * self.roi_label.roi_cx)
        cy = int(h * self.roi_label.roi_cy)
        ax = int(w * self.roi_label.roi_ax)
        ay = int(h * self.roi_label.roi_ay)
        cv2.ellipse(display, (cx, cy), (ax, ay), 0, 0, 360, (0, 255, 0), 2)
        self.display_image(display, self.lbl_original['label'], is_gray=False)

    def _redraw_frozen_frame(self):
        """Gambar ulang frozen frame dengan ROI terkini (callback saat drag/scroll di mode jeda)."""
        if self.is_paused and self.frozen_frame is not None:
            self._draw_roi_and_show(self.frozen_frame)

    def _on_reset_roi(self):
        """Reset ROI ke default dan perbarui tampilan jika sedang dalam mode jeda."""
        self.roi_label.reset_roi()
        self._redraw_frozen_frame()
