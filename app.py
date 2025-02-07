from flask import Flask, render_template, request, jsonify, Response
from tensorflow.keras.models import load_model
import numpy as np
import os
from werkzeug.utils import secure_filename
from PIL import Image
from groq import Groq
from dotenv import load_dotenv
import json
import base64
from io import BytesIO

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
groq_client = Groq(api_key=GROQ_API_KEY)

# Initialize Flask app
app = Flask(__name__)

# Load the model
MODEL_PATH = "./model/sequential-model2.keras"
model = load_model(MODEL_PATH)

# Constants
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}
IMAGE_SIZE = (256, 256)
class_labels = ["Benign Cases", "Malignant Cases", "Normal Cases"]

# System prompt for chat - now stored in a global variable
SYSTEM_PROMPT = {
    "role": "system",
    "content": (
        "Nama anda adalah ParuAI. Anda adalah seorang dokter spesialis kanker paru-paru yang sangat berpengetahuan dan penuh empati. "
        "Keahlian utama Anda adalah mengklasifikasikan kondisi paru-paru menjadi malignant (ganas), benign (jinak), atau normal berdasarkan "
        "gejala, hasil diagnosis, dan riwayat pasien. Anda memiliki pemahaman mendalam tentang kanker paru-paru, termasuk faktor risiko, "
        "gejala, tahapan, pengobatan, dan metode diagnosis seperti pencitraan, biopsi, serta pengujian molekuler.\n\n"
        "Respon Anda harus:\n"
        "1. Jelas dan mudah dipahami, menghindari penggunaan istilah teknis yang rumit kecuali diperlukan.\n"
        "2. Empatik dan berpusat pada pasien, dengan memperhatikan aspek emosional saat membahas kondisi kesehatan.\n"
        "3. Relevan dengan kanker paru-paru dan kesehatan pernapasan terkait ketika pengguna bertanya tentang topik tersebut.\n\n"
        "Jika pertanyaan tidak terkait dengan kanker paru-paru atau keahlian Anda, jawab dengan sopan dan gunakan pengetahuan medis umum "
        "Anda jika memungkinkan. Selalu usahakan untuk mengarahkan kembali percakapan ke spesialisasi Anda.\n\n"
        "Informasi tambahan: \n"
        "1. Aplikasi prediksi/klasifikasi jenis tumor kanker paru-paru berdasarkan gambar CT Scan pada situs ParuAI menggunakan teknologi "
        "AI Deep Learning Convolutional Neural Network (CNN) yang dirancang dan dikembangkan oleh Belgis Anggita. ParuAI memiliki performa "
        "luar biasa dengan akurasi mencapai 99-100% dan tingkat loss hanya 3%. Model ini dilatih menggunakan data yang terpercaya, yaitu "
        "dataset dari Iraq-Oncology Teaching Hospital/National Center for Cancer Diseases (IQ-OTH/NCCD) yang dikumpulkan pada tahun 2019. "
        "Dataset tersebut diperoleh dari publikasi Mendeley Data, dengan ucapan terima kasih kepada para penulisnya: AL-Huseiny, Muayed "
        "dan alyasriy, Hamdalla (Referensi: alyasriy, Hamdalla; AL-Huseiny, Muayed (2021), “The IQ-OTHNCCD lung cancer dataset”, Mendeley "
        "Data, V2, doi: 10.17632/bhmdr45bh2.2.)\n"
        "2. Cara Penggunaan Aplikasi: Pengguna dapat mengunggah gambar CT Scan paru-paru pasien ke aplikasi (bisa juga menggunakan contoh yang tertera),"
        "lalu ParuAI akan memberikan klasifikasi jenis tumor kanker paru-paru berdasarkan gambar tersebut. ParuAI akan memberikan hasil klasifikasi dalam bentuk benign, "
        "malignant, atau normal.\n"
        "Selalu menggunakan bahasa Indonesia"
    ),
}

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Parse the incoming JSON payload
        data = request.json
        if not data or "file_name" not in data or "file_data" not in data:
            return jsonify({"error": "Invalid payload"}), 400

        file_name = data["file_name"]
        file_data = data["file_data"]

        # Decode the Base64 string to binary data
        try:
            image_data = base64.b64decode(file_data)
            image = Image.open(BytesIO(image_data)).convert("L")  # Convert to grayscale
        except Exception as e:
            return jsonify(
                {"error": f"Failed to decode and process Base64 image: {str(e)}"}
            ), 400

        # Preprocess the image for model input
        try:
            image = image.resize(IMAGE_SIZE)
            img_array = np.array(image)
            img_array = np.expand_dims(
                img_array, axis=-1
            )  # Add channel dimension for grayscale
            img_array = img_array / 255.0  # Normalize pixel values
            img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        except Exception as e:
            return jsonify({"error": f"Failed to preprocess image: {str(e)}"}), 500

        # Perform the prediction
        try:
            predictions = model.predict(img_array)
            predicted_class = np.argmax(predictions[0])
            predicted_label = class_labels[predicted_class]
        except Exception as e:
            return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

        # Return the results
        return jsonify(
            {
                "label": predicted_label,
                "confidence": f"{np.max(predictions[0]) * 100:.2f}%",  
                "filename": file_name,
                "file_data" : file_data
            }
        ), 200

    except Exception as e:
        return jsonify({"error": f"An unexpected error occurred: {str(e)}"}), 500


@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    user_message = data.get("message")

    if not user_message:
        return jsonify({"error": "No message provided"}), 400

    try:
        # Create a new conversation array for each request
        conversation = [SYSTEM_PROMPT, {"role": "user", "content": user_message}]

        completion = groq_client.chat.completions.create(
            model="llama3-70b-8192",
            messages=conversation,
            temperature=1,
            max_tokens=1024,
            top_p=1,
            stream=False,
        )

        response = completion.choices[0].message.content
        return jsonify({"response": response}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
