import os
import cv2
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import joblib

def entrenar_modelo():
    def load_images_from_folder(folder_path, image_size=(64, 64)):
        images = []
        labels = []
        classes = os.listdir(folder_path)
        for label, class_name in enumerate(classes):
            class_folder = os.path.join(folder_path, class_name)
            for filename in os.listdir(class_folder):
                img_path = os.path.join(class_folder, filename)
                img = cv2.imread(img_path)
                if img is not None:
                    img = cv2.resize(img, image_size)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    images.append(img.flatten())
                    labels.append(label)
        return np.array(images), np.array(labels), classes

    # Cargar datos
    X, y, class_names = load_images_from_folder("Imagenes_plantas")

    # Separar entrenamiento y prueba (opcional)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Entrenar modelo con probability=True
    clf = SVC(kernel='rbf', probability=True)
    clf.fit(X_train, y_train)

    # Guardar modelo y clases
    joblib.dump(clf, "svm_model.pkl")
    joblib.dump(class_names, "class_names.pkl")

    print("Modelo entrenado y guardado con soporte para probabilidades.")
