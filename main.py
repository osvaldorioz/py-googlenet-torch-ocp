from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel
import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import googlenet 
import json
import shutil
import os

matplotlib.use('Agg')  # Usar backend no interactivo
app = FastAPI()

# Directorio del PVC montado en OpenShift
UPLOAD_FOLDER = "/opt/app-root/src/data/"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def preprocess_image(image_path):
    # Cargar la imagen
    image = Image.open(image_path).convert("RGB")

    # Definir las transformaciones
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Aplicar transformaciones
    tensor = transform(image).unsqueeze(0)  # Añadir dimensión batch

    # Guardar el tensor en el PVC
    tensor_path = os.path.join(UPLOAD_FOLDER, "preprocessed_tensor.pt")
    torch.save(tensor, tensor_path)

def load_imagenet_labels():
    labels_path = os.path.join(UPLOAD_FOLDER, "imagenet_classes.txt")
    try:
        with open(labels_path, "r") as f:
            return [line.strip() for line in f.readlines()]
    except FileNotFoundError:
        print(f"Advertencia: {labels_path} no encontrado. Usando etiquetas simuladas.")
        return [f"simulated_class_{i}" for i in range(1000)]

@app.post("/pre-procesar-imagen")
async def preprocesar(file: UploadFile = File(...)):
    file_location = os.path.join(UPLOAD_FOLDER, file.filename)
    with open(file_location, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    preprocess_image(file_location)
    return {"filename": file.filename, "message": "Imagen procesada exitosamente"}

@app.post("/clasificador")
def calculo():
    output_file = os.path.join(UPLOAD_FOLDER, 'googlenet_results.png')
    
    model_path = os.path.join(UPLOAD_FOLDER, "googlenet.pt")
    tensor_path = os.path.join(UPLOAD_FOLDER, "preprocessed_tensor.pt")
    class_names = load_imagenet_labels()

    service = googlenet.GoogLeNetService("clasificador")
    
    try:
        results = service.classify(model_path, tensor_path, class_names)
        
        print("Top-5 predicciones:")
        for label, prob in results:
            print(f"{label}: {prob:.4f}")
        
        labels = [label for label, prob in results]
        probs = [prob for label, prob in results]
        
        # Usar un colormap para las barras
        colors = cm.viridis([i/len(probs) for i in range(len(probs))])
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(labels, probs, color=colors, edgecolor='black')
        plt.title("Top-5 Predicciones de GoogLeNet", fontsize=14, pad=15)
        plt.xlabel("Clase", fontsize=12)
        plt.ylabel("Probabilidad", fontsize=12)
        plt.xticks(rotation=45, ha="right", fontsize=10)
        plt.grid(True, axis='y', linestyle='--', alpha=0.7)
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2.0, yval, f"{yval:.4f}", va='bottom', fontsize=10)
        plt.tight_layout()
        plt.savefig(output_file)
        plt.close()
        
    except Exception as e:
        print(f"Error: {e}")
        return {"error": str(e)}
    
    j1 = {
        "Grafica generada": output_file
    }
    jj = json.dumps(str(j1))

    return jj

@app.get("/googlenet-graph")
def getGraph():
    output_file = os.path.join(UPLOAD_FOLDER, 'googlenet_results.png')
    return FileResponse(output_file, media_type="image/png", filename=output_file)
