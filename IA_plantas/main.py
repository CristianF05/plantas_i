from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from typing import List
import numpy as np
import joblib
import cv2
import pandas as pd
import os
from entrenar_modelo import entrenar_modelo
import threading
app = FastAPI()

origins = ["http://localhost:3000"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/imagenes", StaticFiles(directory="Imagenes_plantas"), name="imagenes")

# Cargar modelo, clases y datos
model = joblib.load("svm_model.pkl")
class_names = joblib.load("class_names.pkl")
df_plantas = pd.read_csv("plantas_medicinales_peru.csv")

CONFIDENCE_THRESHOLD = 0.16

def preparar_imagen(file_bytes: bytes):
    nparr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Imagen inválida o corrupta")
    img = cv2.resize(img, (64, 64))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img.flatten().reshape(1, -1)
    return img

def obtener_info_planta(nombre_clase: str, base_url: str):
    info = df_plantas.loc[df_plantas['etiqueta'] == nombre_clase]
    if info.empty:
        raise HTTPException(status_code=404, detail="La planta no está en la información disponible.")
    info_dict = info.iloc[0].to_dict()
    info_dict.pop('etiqueta', None)

    imagen_url_relativa = info_dict.get('imagen_url')
    if imagen_url_relativa:
        if not imagen_url_relativa.startswith("/imagenes"):
            imagen_url_relativa = "/imagenes" + ("/" if not imagen_url_relativa.startswith("/") else "") + imagen_url_relativa
        info_dict['imagen_url'] = base_url.rstrip("/") + imagen_url_relativa
    else:
        info_dict['imagen_url'] = None

    return info_dict

@app.post("/predecir")
async def predecir(request: Request, file: UploadFile = File(...)):
    try:
        file_bytes = await file.read()
        img = preparar_imagen(file_bytes)

        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(img)[0]
            pred = int(np.argmax(proba))
            confidence = float(proba[pred])

            if confidence < CONFIDENCE_THRESHOLD:
                raise HTTPException(status_code=404, detail="La planta no está registrada o la confianza es muy baja.")
        else:
            pred = int(model.predict(img)[0])
            confidence = 1.0

        nombre_clase = class_names[pred]

        info_extra = obtener_info_planta(nombre_clase, str(request.base_url))
        info_extra['confianza'] = round(confidence * 100, 2)

        return {
            "prediccion": nombre_clase,
            **info_extra
        }

    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        print(f"Error en /predecir: {e}")
        raise HTTPException(status_code=500, detail="Error interno del servidor")

@app.post("/agregar_planta")
async def agregar_planta(
    nombre_archivo: str = Form(...),
    etiqueta: str = Form(...),
    nombre_comun: str = Form(...),
    nombre_cientifico: str = Form(...),
    categoria: str = Form(...),
    uso_medicinal_detallado: str = Form(...),
    descripcion: str = Form(...),
    regiones: str = Form(...),
    es_medicinal: str = Form(...),
    imagenes: List[UploadFile] = File(...)
):
    try:
        # 📌 Calcular el nuevo ID a partir del CSV
        try:
            df_existente = pd.read_csv("plantas_medicinales_peru.csv")
            if not df_existente.empty:
                nuevo_id = int(df_existente['id'].max()) + 1
            else:
                nuevo_id = 1
        except FileNotFoundError:
            nuevo_id = 1

        # 📁 Guardar imágenes
        ruta_carpeta = os.path.join("Imagenes_plantas", nombre_archivo)
        os.makedirs(ruta_carpeta, exist_ok=True)

        nombre_imagen_principal = None

        for idx, img in enumerate(imagenes):
            contenido = await img.read()
            ruta_guardado = os.path.join(ruta_carpeta, img.filename)
            with open(ruta_guardado, "wb") as f:
                f.write(contenido)
            if idx == 0:
                nombre_imagen_principal = img.filename

        imagen_url = f"/imagenes/{nombre_archivo}/{nombre_imagen_principal}" if nombre_imagen_principal else None

        global df_plantas

        # Validar que no se repita la etiqueta
        if (df_plantas['etiqueta'] == etiqueta).any():
            raise HTTPException(status_code=400, detail="Ya existe una planta con la misma etiqueta")

        # 🧾 Crear nuevo registro
        nuevo_registro = {
            "id": nuevo_id,
            "etiqueta": etiqueta,
            "nombre_comun": nombre_comun,
            "nombre_cientifico": nombre_cientifico,
            "categoria": categoria,
            "uso_medicinal_detallado": uso_medicinal_detallado,
            "descripcion": descripcion,
            "regiones": regiones,
            "es_medicinal": es_medicinal,
            "imagen_url": imagen_url
        }

        df_plantas = pd.concat([df_plantas, pd.DataFrame([nuevo_registro])], ignore_index=True)
        df_plantas.to_csv("plantas_medicinales_peru.csv", index=False)

        # Reentrenar modelo en segundo plano
        threading.Thread(target=entrenar_modelo).start()

        return {"msg": "Datos recibidos y guardados correctamente", "id_asignado": nuevo_id}

    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        print(f"Error en /agregar_planta: {e}")
        raise HTTPException(status_code=500, detail="Error interno del servidor")
