from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
from utils_extract import extract_face_from_one_video
from pathlib import Path
import os
import shutil
import numpy as np
from PIL import Image
import pandas as pd
from datetime import datetime
import tensorflow as tf
from keras import layers, models
from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input
from reportlab.pdfgen import canvas

# --- Config ---
UPLOAD_FOLDER = 'static/uploads'
FACE_FOLDER = 'static/faces'
MODEL_WEIGHTS = 'model/face.t5'

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50 MB
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(FACE_FOLDER, exist_ok=True)

model = None

def build_model():
    vggface = VGGFace(include_top=False, input_shape=(224, 224, 3), pooling='avg')
    inputs = layers.Input(shape=(10, 224, 224, 3), name='Input')
    x = layers.TimeDistributed(layers.Rescaling(1./255.))(inputs)
    x = layers.TimeDistributed(vggface)(x)
    x = layers.LSTM(128, return_sequences=True)(x)
    x = layers.LSTM(64)(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(1024)(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(5, activation='sigmoid')(x)
    return models.Model(inputs, outputs)

def load_model():
    global model
    if model is None:
        print("[INFO] Loading model...")
        model = build_model()
        model.load_weights(MODEL_WEIGHTS)
        print("[INFO] Model loaded successfully.")
    return model

def generate_description(prediction):
    description = []

    if prediction["Openness"] >= 50:
        description.append("terbuka terhadap pengalaman baru dan imajinatif")
    else:
        description.append("lebih konvensional dan realistis")

    if prediction["Conscientiousness"] >= 50:
        description.append("terorganisir dan bertanggung jawab")
    else:
        description.append("lebih santai dan spontan")

    if prediction["Extraversion"] >= 50:
        description.append("aktif dan mudah bergaul")
    else:
        description.append("lebih pendiam dan menikmati waktu sendiri")

    if prediction["Agreeableness"] >= 50:
        description.append("mudah percaya dan kooperatif")
    else:
        description.append("lebih kritis dan mandiri")

    if prediction["Neuroticism"] >= 50:
        description.append("cenderung emosional dan sensitif terhadap stres")
    else:
        description.append("tenang dan stabil secara emosi")

    result = "Dalam prediksi kepribadian ini, kamu " + ", ".join(description[:-1])
    result += ", " + description[-1] + "."
    return result

def save_prediction_to_pdf(prediction, username, video_name, output_path, description):
    c = canvas.Canvas(output_path)
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, 800, "Hasil Prediksi Kepribadian")
    c.setFont("Helvetica", 12)
    c.drawString(50, 780, f"Nama: {username}")
    c.drawString(50, 765, f"Video: {video_name}")
    c.drawString(50, 750, f"Tanggal: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    y = 720
    for trait, score in prediction.items():
        c.drawString(60, y, f"{trait}: {score}%")
        y -= 20
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y - 20, "Ringkasan Kepribadian:")
    c.setFont("Helvetica", 11)

    # Potong kalimat jika terlalu panjang untuk 1 baris
    text_lines = []
    words = description.split()
    line = ""
    for word in words:
        if len(line + word) <= 90:
            line += word + " "
        else:
            text_lines.append(line.strip())
            line = word + " "
    if line:
        text_lines.append(line.strip())

    y -= 40
    for line in text_lines:
        c.drawString(60, y, line)
        y -= 18
    c.save()

def predict_ocean_from_faces(face_dir):
    model = load_model()
    face_files = sorted(os.listdir(face_dir))[:10]
    if len(face_files) < 10:
        raise ValueError("Tidak cukup wajah terdeteksi, minimal 10 frame dibutuhkan.")

    faces = []
    for file in face_files:
        img = Image.open(os.path.join(face_dir, file)).resize((224, 224)).convert("RGB")
        img = np.array(img).astype('float32')
        faces.append(img)

    faces = np.array(faces)
    faces = preprocess_input(faces)
    faces = np.expand_dims(faces, axis=0)

    prediction = model.predict(faces)[0]
    traits = ["Openness", "Conscientiousness", "Extraversion", "Agreeableness", "Neuroticism"]
    result = {trait: round(score * 100, 2) for trait, score in zip(traits, prediction)}
    return result

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    username = ""
    trait_labels = {}
    video_path = None

    if request.method == 'POST':
        username = request.form.get('username', '').strip().replace(" ", "_")
        if not username:
            prediction = {"Error": "Nama pengguna wajib diisi."}
            return render_template("index.html", prediction=prediction, username=username)

        shutil.rmtree(FACE_FOLDER, ignore_errors=True)
        os.makedirs(FACE_FOLDER, exist_ok=True)

        file = request.files.get('video')
        if not file or file.filename == '':
            prediction = {"Error": "File video tidak ditemukan."}
            return render_template("index.html", prediction=prediction, username=username)

        filename = secure_filename(f"{username}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.webm")
        video_full_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(video_full_path)

        extract_face_from_one_video(video_full_path, FACE_FOLDER, num_images=10)
        face_path = os.path.join(FACE_FOLDER, Path(filename).stem)

        if not os.path.exists(face_path):
            prediction = {"Error": "Gagal ekstrak wajah dari video."}
        else:
            try:
                prediction = predict_ocean_from_faces(face_path)
                description = generate_description(prediction)

                csv_path = f'static/result_{username}.csv'
                row = {"User": username, "Video": filename, "Waktu": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "Deskripsi": description}
                row.update(prediction)

                if os.path.exists(csv_path):
                    df = pd.read_csv(csv_path)
                    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
                else:
                    df = pd.DataFrame([row])
                df.to_csv(csv_path, index=False)

                pdf_path = f'static/result_{username}.pdf'
                description = generate_description(prediction)
                save_prediction_to_pdf(prediction, username, filename, pdf_path, description)


            except Exception as e:
                print(f"[ERROR] {e}")
                prediction = {"Error": str(e)}

        # Pastikan trait_labels dibuat aman
        trait_labels = {}
        if prediction and isinstance(prediction, dict):
            for trait, score in prediction.items():
                if trait in ["Openness", "Conscientiousness", "Extraversion", "Agreeableness", "Neuroticism"]:
                    try:
                        nilai = float(score)
                        trait_labels[trait] = "Tinggi" if nilai >= 50 else "Rendah"
                    except ValueError:
                        trait_labels[trait] = "Invalid"

        video_path = '/' + video_full_path.replace('\\', '/')

    return render_template("index.html",
                           prediction=prediction,
                           username=username,
                           labels=trait_labels,
                           video_path=video_path,
                           description=description if 'description' in locals() else None)

if __name__ == "__main__":
    print("🔥 Menjalankan Flask app di http://localhost:5000")
    app.run(debug=True)