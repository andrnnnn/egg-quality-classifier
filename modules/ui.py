import cv2
from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QPushButton, QFileDialog, 
                             QTableWidget, QTableWidgetItem, QHeaderView, QMessageBox, 
                             QFrame, QGraphicsDropShadowEffect)
from PyQt5.QtGui import QPixmap, QImage, QColor
from PyQt5.QtCore import Qt
from .image_processor import ImageProcessor
from .classifier import EggClassifier

class EggQualityApp(QMainWindow):
    """
    Window Aplikasi Utama.
    Mengelola UI dan mengoordinasikan ImageProcessor dan EggClassifier.
    """
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Klasifikasi Mutu Telur Puyuh")
        
        # Inisialisasi kelas helper
        self.image_processor = ImageProcessor()
        self.classifier = EggClassifier()
        
        # Pengaturan Window
        self.setFixedSize(1200, 950) 
        
        # Muat Model dan Scaler
        success, message = self.classifier.load_model()
        if not success:
            QMessageBox.warning(self, "Error", message)

        # Terapkan Gaya dan Atur UI
        self.apply_styles()
        
        # Kontainer Layout Utama
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)
        self.main_layout.setContentsMargins(50, 30, 50, 30)
        self.main_layout.setSpacing(20)

        # Atur Bagian UI
        self.setup_header()             # 1. Header
        self.setup_images_section()     # 2. Tampilan Gambar
        self.setup_feature_section()    # 3. Tabel Fitur
        self.setup_prediction_section() # 4. Hasil Prediksi

    def apply_styles(self):
        """Mendefinisikan stylesheet CSS untuk aplikasi."""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #F9FAFB;
            }
            QWidget {
                font-family: 'Poppins', 'Segoe UI', sans-serif;
                font-size: 14px;
                color: #374151;
            }
            /* Card untuk Gambar SAJA */
            QFrame#ImageCard {
                background-color: #FFFFFF;
                border-radius: 12px;
                border: 1px solid #E5E7EB;
            }
            /* Card untuk Tabel dan Prediksi */
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
            QPushButton:hover {
                background-color: #2563EB;
            }
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
        """Menambahkan efek bayangan (drop shadow) ke widget."""
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(20)
        shadow.setColor(QColor(0, 0, 0, 15))
        shadow.setOffset(0, 4)
        widget.setGraphicsEffect(shadow)

    def setup_header(self):
        """Mengatur header atas dengan judul dan tombol 'Pilih Gambar'."""
        header_layout = QHBoxLayout()
        
        title_container = QVBoxLayout()
        title_label = QLabel("Klasifikasi Mutu Telur Puyuh")
        title_label.setObjectName("Title")
        subtitle_label = QLabel("Analisis Kualitas Telur Berbasis Citra Digital")
        subtitle_label.setObjectName("Subtitle")
        
        title_container.addWidget(title_label)
        title_container.addWidget(subtitle_label)
        
        self.btn_select = QPushButton("+ Pilih Gambar")
        self.btn_select.setCursor(Qt.PointingHandCursor)
        self.btn_select.clicked.connect(self.load_image)
        self.btn_select.setFixedWidth(160)
        self.add_shadow(self.btn_select)

        header_layout.addLayout(title_container)
        header_layout.addStretch()
        header_layout.addWidget(self.btn_select)
        
        self.main_layout.addLayout(header_layout)

    def setup_images_section(self):
        """Mengatur area untuk menampilkan citra Asli, GLCM (Masked Gray), dan HSV (Masked Color)."""
        images_layout = QHBoxLayout()
        images_layout.setSpacing(30)
        images_layout.setAlignment(Qt.AlignCenter) 

        self.lbl_original = self.create_image_group("Citra Asli")
        self.lbl_glcm = self.create_image_group("Input GLCM")
        self.lbl_hsv = self.create_image_group("Input HSV")

        images_layout.addWidget(self.lbl_original['container'])
        images_layout.addWidget(self.lbl_glcm['container'])
        images_layout.addWidget(self.lbl_hsv['container'])
        
        self.main_layout.addLayout(images_layout)

    def create_image_group(self, title):
        """Helper untuk membuat kontainer gambar berbingkai dengan judul."""
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
        """Mengatur tabel untuk menampilkan nilai fitur GLCM dan HSV."""
        feature_layout = QHBoxLayout()
        feature_layout.setSpacing(30)
        
        # --- Tabel GLCM ---
        glcm_container = QWidget()
        glcm_layout = QVBoxLayout(glcm_container)
        glcm_layout.setContentsMargins(0, 0, 0, 0)
        
        lbl_glcm = QLabel("Fitur GLCM (Tekstur)")
        lbl_glcm.setObjectName("SectionTitle")
        glcm_layout.addWidget(lbl_glcm)
        
        self.table_glcm = self.create_table(['Contrast', 'Energy', 'Homogeneity', 'Correlation'])
        glcm_layout.addWidget(self.table_glcm)
        
        feature_layout.addWidget(glcm_container)
        
        # --- Tabel HSV ---
        hsv_container = QWidget()
        hsv_layout = QVBoxLayout(hsv_container)
        hsv_layout.setContentsMargins(0, 0, 0, 0)
        
        lbl_hsv = QLabel("Fitur HSV (Warna)")
        lbl_hsv.setObjectName("SectionTitle")
        hsv_layout.addWidget(lbl_hsv)
        
        self.table_hsv = self.create_table(['H Mean', 'S Mean', 'V Mean', 'H Std', 'S Std', 'V Std'])
        hsv_layout.addWidget(self.table_hsv)
        
        feature_layout.addWidget(hsv_container)
        
        self.main_layout.addLayout(feature_layout)

    def create_table(self, features):
        """Helper untuk membuat QTableWidget yang bergaya."""
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
        table.verticalHeader().setVisible(False)
        table.setEditTriggers(QTableWidget.NoEditTriggers)
        table.setSelectionMode(QTableWidget.NoSelection)
        table.setFocusPolicy(Qt.NoFocus)
        table.setShowGrid(False)
        
        # Hitung Tinggi
        header_height = 45
        row_height = 40
        total_height = header_height + (len(features) * row_height) + 10
        
        table.horizontalHeader().setFixedHeight(header_height)
        table.setFixedHeight(total_height)
        
        for i, f in enumerate(features):
            table.setItem(i, 0, QTableWidgetItem(f))
            val_item = QTableWidgetItem("-")
            val_item.setTextAlignment(Qt.AlignCenter)
            table.setItem(i, 1, val_item)
            table.setRowHeight(i, row_height)
            
        layout.addWidget(table)
        return table_frame 

    def setup_prediction_section(self):
        """Mengatur area tampilan hasil prediksi."""
        pred_frame = QFrame()
        pred_frame.setObjectName("ContentCard")
        pred_frame.setStyleSheet("""
            QFrame#ContentCard {
                background-color: #FFFFFF;
                border-radius: 12px;
                border-left: 6px solid #3B82F6; /* Aksen Biru */
            }
        """)
        self.add_shadow(pred_frame)
        
        pred_layout = QHBoxLayout(pred_frame)
        pred_layout.setContentsMargins(30, 25, 30, 25)
        
        lbl_title = QLabel("Hasil Prediksi:")
        lbl_title.setStyleSheet("font-size: 18px; font-weight: 500; color: #4B5563;")
        
        self.lbl_prediction = QLabel("-")
        self.lbl_prediction.setStyleSheet("font-size: 24px; font-weight: 600; color: #111827; margin-left: 20px;")
        
        pred_layout.addWidget(lbl_title)
        pred_layout.addWidget(self.lbl_prediction)
        pred_layout.addStretch()
        
        self.main_layout.addWidget(pred_frame)

    def load_image(self):
        """Membuka dialog file untuk memilih gambar."""
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, "Pilih Gambar Telur", "", 
                                                 "Images (*.png *.jpg *.jpeg *.bmp)", options=options)
        if file_path:
            self.process_image(file_path)

    def process_image(self, file_path):
        """
        Alur pemrosesan utama:
        1. Pra-pemrosesan citra
        2. Tampilkan gambar
        3. Validasi Objek Telur
        4. Ekstraksi fitur
        5. Perbarui tabel
        6. Prediksi dan perbarui hasil
        """
        # 1. Pra-pemrosesan Citra (Delegasi ke ImageProcessor)
        img, gray, mask, masked_gray, masked_color = self.image_processor.preprocess(file_path)
        
        if img is None:
            QMessageBox.warning(self, "Error", "Tidak dapat membaca file gambar.")
            return

        # 2. Tampilkan Gambar
        self.display_image(img, self.lbl_original['label'])
        self.display_image(masked_gray, self.lbl_glcm['label'], is_gray=True)
        self.display_image(masked_color, self.lbl_hsv['label'])

        # 3. Validasi Apakah Objek Adalah Telur
        is_egg, reason = self.image_processor.check_is_object_egg(mask)
        if not is_egg:
            # Kosongkan tampilan fitur (set 0)
            table_glcm_widget = self.table_glcm.findChild(QTableWidget)
            for i in range(4):
                val_item = QTableWidgetItem("0")
                val_item.setTextAlignment(Qt.AlignCenter)
                table_glcm_widget.setItem(i, 1, val_item)
                
            table_hsv_widget = self.table_hsv.findChild(QTableWidget)
            for i in range(6):
                val_item = QTableWidgetItem("0")
                val_item.setTextAlignment(Qt.AlignCenter)
                table_hsv_widget.setItem(i, 1, val_item)
                
            # Tampilkan peringatan di label prediksi
            self.lbl_prediction.setText(f"Bukan Telur ({reason})")
            self.lbl_prediction.setStyleSheet("font-size: 18px; font-weight: 600; color: #EF4444; margin-left: 20px;")
            return

        # 4. Ekstraksi Fitur (Delegasi ke ImageProcessor)
        features = self.image_processor.extract_features(img, gray, mask, masked_gray, masked_color)
        
        # 5. Perbarui Tabel
        # Fitur GLCM (4 Pertama)
        table_glcm_widget = self.table_glcm.findChild(QTableWidget)
        for i in range(4):
            val_item = QTableWidgetItem(f"{features[i]:.4f}")
            val_item.setTextAlignment(Qt.AlignCenter)
            table_glcm_widget.setItem(i, 1, val_item)
            
        # Fitur HSV (6 Berikutnya)
        table_hsv_widget = self.table_hsv.findChild(QTableWidget)
        for i in range(6):
            val_item = QTableWidgetItem(f"{features[4+i]:.4f}")
            val_item.setTextAlignment(Qt.AlignCenter)
            table_hsv_widget.setItem(i, 1, val_item)

        # 5. Prediksi (Delegasi ke EggClassifier)
        prediction = self.classifier.predict(features)
        self.update_prediction_label(prediction)

    def update_prediction_label(self, prediction):
        """Memperbarui teks dan warna label prediksi berdasarkan hasil."""
        self.lbl_prediction.setText(prediction)
        
        if prediction == 'Baik':
            self.lbl_prediction.setStyleSheet("font-size: 24px; font-weight: 600; color: #10B981; margin-left: 20px;")
        elif prediction == 'Sedang':
            self.lbl_prediction.setStyleSheet("font-size: 24px; font-weight: 600; color: #F59E0B; margin-left: 20px;")
        else:
            self.lbl_prediction.setStyleSheet("font-size: 24px; font-weight: 600; color: #EF4444; margin-left: 20px;")

    def display_image(self, img_array, label_widget, is_gray=False):
        """Mengonversi citra CV2 ke QPixmap dan menampilkannya pada QLabel."""
        if is_gray:
            h, w = img_array.shape
            bytes_per_line = w
            q_img = QImage(img_array.data, w, h, bytes_per_line, QImage.Format_Grayscale8)
        else:
            h, w, ch = img_array.shape
            bytes_per_line = ch * w
            rgb_img = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
            q_img = QImage(rgb_img.data, w, h, bytes_per_line, QImage.Format_RGB888)
            
        pixmap = QPixmap.fromImage(q_img)
        label_widget.setPixmap(pixmap)
