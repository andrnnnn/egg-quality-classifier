import os
import joblib
import warnings

class EggClassifier:
    """
    Menangani aspek machine learning: memuat model dan melakukan prediksi.
    """
    def __init__(self):
        self.model = None
        self.scaler = None

    def load_model(self, model_path='models/model_svm_linear.pkl', scaler_path='models/scaler_data.pkl'):
        """
        Memuat model SVM yang sudah dilatih dan scaler dari disk.
        """
        try:
            if os.path.exists(model_path) and os.path.exists(scaler_path):
                self.model = joblib.load(model_path)
                self.scaler = joblib.load(scaler_path)
                return True, "Model berhasil dimuat"
            else:
                return False, f"File model tidak ditemukan di: {model_path} atau {scaler_path}"
        except Exception as e:
            return False, str(e)

    def predict(self, features):
        """
        Memprediksi kualitas telur berdasarkan fitur yang diekstraksi.
        """
        if self.model and self.scaler:
            try:
                # Skala fitur sebelum prediksi
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore",
                        message="X does not have valid feature names, but StandardScaler was fitted with feature names",
                        category=UserWarning
                    )
                    features_scaled = self.scaler.transform([features])
                prediction = self.model.predict(features_scaled)[0]
                return self._calibrate_prediction(prediction, features)
            except Exception as e:
                print(f"Error prediksi: {e}")
                return "Error"
        return "Error"

    def _calibrate_prediction(self, prediction, features):
        """
        Kalibrasi ringan berbasis domain setelah output model.
        Menjaga alur utama model+scaler, namun menurunkan false-positive
        "Baik" pada telur dengan pola bercak/variasi warna yang cukup tinggi.
        """
        if prediction != "Baik" or not features or len(features) < 10:
            return prediction

        contrast = float(features[0])
        h_std = float(features[7])
        s_std = float(features[8])
        v_std = float(features[9])

        # Turunkan ke "Sedang" hanya jika indikator "bercak kasar" muncul
        # secara bersamaan (bukan hanya satu indikator tinggi).
        severe_count = 0
        if contrast >= 165.0:
            severe_count += 1
        if h_std >= 70.0:
            severe_count += 1
        if v_std >= 55.0:
            severe_count += 1
        if s_std >= 36.0:
            severe_count += 1

        # Minimal dua sinyal kuat baru diturunkan.
        if severe_count >= 2:
            return "Sedang"

        return prediction
