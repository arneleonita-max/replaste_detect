import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import io
import tensorflow as tf
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
from google.cloud import storage
import tempfile
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Configure Google Cloud Storage
BUCKET_NAME = os.getenv("BUCKET_NAME", "replaste-h5")  # Ganti dengan nama bucket Anda
MODEL_PATH = os.getenv("MODEL_PATH", "model1.h5")  # Path ke model di bucket

def download_model_from_gcs():
    """Download model dari Google Cloud Storage"""
    try:
        logger.info(f"Downloading model from gs://{BUCKET_NAME}/{MODEL_PATH}")
        storage_client = storage.Client()
        bucket = storage_client.bucket(BUCKET_NAME)
        blob = bucket.blob(MODEL_PATH)
        
        # Download ke temporary file
        _, temp_local_filename = tempfile.mkstemp()
        blob.download_to_filename(temp_local_filename)
        logger.info("Model downloaded successfully")
        
        return temp_local_filename
    except Exception as e:
        logger.error(f"Error downloading model: {str(e)}")
        raise

# Load model dari GCS
model_path = download_model_from_gcs()
model = tf.keras.models.load_model(model_path)
os.remove(model_path)  # Hapus temporary file

class_names = ['HDPE', 'LDPE', 'PET', 'PP', 'PS', 'PVC']

class PlasticInfo:
    def __init__(self, name, description, recycling_time, uses, recycling_symbol, environmental_impact, recycling_tips):
        self.name = name
        self.description = description
        self.recycling_time = recycling_time
        self.uses = uses
        self.recycling_symbol = recycling_symbol
        self.environmental_impact = environmental_impact
        self.recycling_tips = recycling_tips

    def to_dict(self):
        return {
            "name": self.name,
            "description": self.description,
            "recycling_time": self.recycling_time,
            "uses": self.uses,
            "recycling_symbol": self.recycling_symbol,
            "environmental_impact": self.environmental_impact,
            "recycling_tips": self.recycling_tips
        }

plastic_info = {
    "PET": PlasticInfo(
        name="Polyethylene Terephthalate (PET)",
        description="PET adalah plastik yang sering digunakan untuk botol minuman dan wadah makanan karena sifatnya yang kuat dan ringan.",
        recycling_time="20-500 tahun",
        uses=["Botol minuman", "Wadah makanan", "Serat tekstil", "Kemasan makanan"],
        recycling_symbol="1",
        environmental_impact="PET dapat mencemari lingkungan jika tidak didaur ulang dengan benar, namun merupakan salah satu jenis plastik yang paling mudah didaur ulang.",
        recycling_tips=["Bilas botol sebelum didaur ulang", "Lepaskan tutup botol", "Tekan botol untuk menghemat ruang"]
    ),
    "HDPE": PlasticInfo(
        name="High-Density Polyethylene (HDPE)",
        description="HDPE adalah plastik yang keras dan tahan terhadap berbagai zat kimia, sering digunakan untuk botol susu dan wadah produk pembersih.",
        recycling_time="30-500 tahun",
        uses=["Botol susu", "Wadah produk pembersih", "Kantong belanja", "Pipa"],
        recycling_symbol="2",
        environmental_impact="HDPE lebih tahan lama di lingkungan dibanding PET, namun juga dapat didaur ulang dengan mudah.",
        recycling_tips=["Kosongkan dan bersihkan wadah", "Lepaskan label jika memungkinkan", "Pisahkan tutup jika berbeda jenis plastik"]
    ),
    "PVC": PlasticInfo(
        name="Polyvinyl Chloride (PVC)",
        description="PVC memiliki ketahanan tinggi terhadap kelembaban dan bahan kimia, sering digunakan dalam pipa dan kemasan medis.",
        recycling_time="50-500 tahun",
        uses=["Pipa air", "Kemasan medis", "Vinil lantai", "Frame jendela"],
        recycling_symbol="3",
        environmental_impact="PVC dapat melepaskan zat berbahaya saat dibakar dan sulit didaur ulang, membutuhkan penanganan khusus.",
        recycling_tips=["Cari fasilitas daur ulang khusus", "Jangan dibakar", "Pisahkan dari plastik lain"]
    ),
    "LDPE": PlasticInfo(
        name="Low-Density Polyethylene (LDPE)",
        description="LDPE adalah plastik fleksibel yang sering digunakan untuk kantong plastik dan film pembungkus.",
        recycling_time="10-100 tahun",
        uses=["Kantong plastik", "Film pembungkus", "Lapisan karton minuman", "Botol yang bisa diremas"],
        recycling_symbol="4",
        environmental_impact="LDPE dapat mencemari laut dan tanah, namun memiliki masa degradasi yang lebih singkat dibanding plastik lain.",
        recycling_tips=["Bersihkan dari kontaminan", "Kumpulkan dalam jumlah besar", "Pastikan dalam kondisi kering"]
    ),
    "PP": PlasticInfo(
        name="Polypropylene (PP)",
        description="PP adalah plastik yang tahan panas dan banyak digunakan dalam wadah makanan, sedotan, dan produk medis.",
        recycling_time="20-30 tahun",
        uses=["Wadah makanan", "Sedotan", "Alat medis", "Komponen otomotif"],
        recycling_symbol="5",
        environmental_impact="PP relatif aman untuk makanan dan minuman panas, namun tetap berkontribusi pada pencemaran lingkungan jika tidak didaur ulang.",
        recycling_tips=["Bersihkan sisa makanan", "Pastikan kering sebelum didaur ulang", "Pisahkan tutup jika berbeda jenis"]
    ),
    "PS": PlasticInfo(
        name="Polystyrene (PS)",
        description="PS adalah plastik yang umum ditemukan dalam bentuk styrofoam dan digunakan untuk kemasan sekali pakai serta isolasi.",
        recycling_time="50-500 tahun",
        uses=["Kemasan styrofoam", "Cangkir sekali pakai", "Isolasi", "Peralatan makan sekali pakai"],
        recycling_symbol="6",
        environmental_impact="PS sangat sulit terurai dan dapat mencemari lingkungan dalam waktu sangat lama, terutama di lautan.",
        recycling_tips=["Hindari penggunaan jika memungkinkan", "Gunakan alternatif ramah lingkungan", "Bersihkan sebelum didaur ulang"]
    )
}

def preprocess_image(image, target_size):
    """
    Resize image to target size and preprocess it for the model.
    """
    try:
        img = image.resize(target_size)
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    except Exception as e:
        logger.error(f"Error preprocessing image: {str(e)}")
        raise

@app.route("/predict", methods=["POST"])
def predict():
    """
    Endpoint utama untuk prediksi jenis plastik
    """
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400
        
        file = request.files["file"]
        
        try:
            image = Image.open(file.stream).convert("RGB")
        except Exception as e:
            logger.error(f"Error opening image: {str(e)}")
            return jsonify({"error": "Invalid image format"}), 400
        
        target_size = model.input_shape[1:3]
        img_array = preprocess_image(image, target_size)

        predictions = model.predict(img_array)
        predicted_index = np.argmax(predictions)
        predicted_class = class_names[predicted_index]
        confidence = np.max(predictions) * 100

        plastic_detail = plastic_info[predicted_class].to_dict()

        response = {
            "predicted_class": predicted_class,
            "confidence": f"{confidence:.2f}%",
            "plastic_info": plastic_detail
        }

        logger.info(f"Successful prediction: {predicted_class} with confidence {confidence:.2f}%")
        return jsonify(response)

    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500
    
@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint for Cloud Run"""
    return jsonify({"status": "healthy"}), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))