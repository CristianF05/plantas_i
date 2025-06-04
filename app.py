from flask import Flask, request, render_template_string
import cv2
import numpy as np
import joblib

app = Flask(__name__)

# Cargar modelo y nombres de clases
clf = joblib.load("svm_model.pkl")
class_names = joblib.load("class_names.pkl")

@app.route("/", methods=["GET", "POST"])
def upload_predict():
    result = ""
    if request.method == "POST":
        file = request.files.get("image")
        if file:
            # Leer imagen en memoria
            file_bytes = np.frombuffer(file.read(), np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

            # Preprocesar igual que en entrenamiento
            img = cv2.resize(img, (64, 64))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img_flat = img.flatten().reshape(1, -1)

            # Predecir
            pred = clf.predict(img_flat)
            result = f"Predicción: {class_names[pred[0]]}"

    # HTML simple
    return render_template_string("""
        <h2>Sube una imagen para clasificar</h2>
        <form method="POST" enctype="multipart/form-data">
          <input type="file" name="image" required>
          <input type="submit" value="Clasificar">
        </form>
        <p>{{result}}</p>
    """, result=result)

if __name__ == "__main__":
    app.run(debug=True)
