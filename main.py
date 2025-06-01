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

    # Guardar el tensor
    torch.save(tensor, "preprocessed_tensor.pt")

def load_imagenet_labels():
    try:
        with open("imagenet_classes.txt", "r") as f:
            return [line.strip() for line in f.readlines()]
    except FileNotFoundError:
        print("Advertencia: imagenet_classes.txt no encontrado. Usando etiquetas simuladas.")
        return [f"simulated_class_{i}" for i in range(1000)]
    

UPLOAD_FOLDER = ""
#os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.post("/pre-procesar-imagen")
async def preprocesar(file: UploadFile = File(...)):
    file_location = f"{UPLOAD_FOLDER}{file.filename}"
    with open(file_location, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    preprocess_image(file.filename)
    return {"filename": file.filename, "message": "Imagen procesada exitosamente"}

@app.post("/clasificador")
def calculo():
    output_file = 'googlenet_results.png'
    
    #model_path = "/opt/app-root/src/models/googlenet.pt"
    model_path = "googlenet.pt"
    tensor_path = "preprocessed_tensor.pt"
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
    
    j1 = {
        "Grafica generada": output_file
    }
    jj = json.dumps(str(j1))

    return jj

@app.get("/googlenet-graph")
def getGraph():
    output_file = 'googlenet_results.png'
    return FileResponse(output_file, media_type="image/png", filename=output_file)