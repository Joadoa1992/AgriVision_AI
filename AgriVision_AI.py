import torch
from ultralytics import YOLO
import cv2
import glob
import matplotlib.pyplot as plt
import os


def train_classification_model():
    # Setup
    HOME = os.getcwd()
    # Stien til selve hovedmappen der indeholder 'train' og 'valid'
    DATASET_PATH = os.path.join(HOME, "New Plant Diseases Dataset(Augmented)")

    print(f"🏠 Arbejdsmappe: {HOME}")

    # Hardware
    device = 0 if torch.cuda.is_available() else "cpu"

    # Træn YOLOv11-Classification
    model = YOLO("yolo11m-cls.pt")

    model.train(
        data=DATASET_PATH,  # klassificering
        epochs=10,  # Starter med 10, planter er ofte hurtige at lære
        imgsz=224,  # Billede størrelse
        batch=32,
        device=device, # Sætter hardware
        workers=0
    )

    # Køre test på de lokale test-billeder
    TEST_PATH = os.path.join(HOME, "test", "test")

    print(f"🔍 Prøver at genkende planter i: {TEST_PATH}")
    results = model.predict(source=TEST_PATH, save=True)

    # RESULTATER
    # gemmer klassificeringsresultater
    last_predict_dir = results[0].save_dir

    # Henter et par eksempler
    image_paths = glob.glob(os.path.join(last_predict_dir, "*.jpg"))

    for image_path in image_paths[:3]:
        image = cv2.imread(image_path)
        if image is not None:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            plt.figure(figsize=(8, 5))
            plt.imshow(image)
            plt.title("AI'ens gæt ses i toppen af billedet")
            plt.axis('off')
            plt.show()


if __name__ == '__main__':
    train_classification_model()