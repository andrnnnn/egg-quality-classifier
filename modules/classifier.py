import os
import joblib

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
                features_scaled = self.scaler.transform([features])
                prediction = self.model.predict(features_scaled)[0]
                return prediction
            except Exception as e:
                print(f"Error prediksi: {e}")
                return "Error"
        return "Error"
