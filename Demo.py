import torch
from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
import os


def run_demo():
    # 1. Stien til din model
    model_path = "runs/classify/train/weights/last.pt"

    if not os.path.exists(model_path):
        print(f"❌ Fejl: Kunne ikke finde modellen på {model_path}")
        return

    model = YOLO(model_path)

    # 2. Vælg et billede
    image_to_test = "test/test/AppleCedarRust1.JPG"

    if not os.path.exists(image_to_test):
        print("❌ Fejl: Test-billedet blev ikke fundet.")
        return

    # 3. Kør AI-analysen
    results = model.predict(source=image_to_test, conf=0.25)

    for r in results:
        # Her henter vi sandsynlighederne korrekt
        # top5 og top5conf er allerede lister i de nyere YOLO-versioner
        top5_indices = r.probs.top5
        top5_conf = r.probs.top5conf
        class_names = r.names

        print("\n--- AI'ens Top 3 Gæt ---")
        for i in range(3):
            name = class_names[top5_indices[i]]
            conf = float(top5_conf[i]) * 100
            print(f"{i + 1}. {name}: {conf:.2f}%")

        # Vis billedet
        img = cv2.imread(image_to_test)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        plt.figure(figsize=(10, 6))
        plt.imshow(img)
        best_guess = class_names[r.probs.top1]
        best_conf = float(r.probs.top1conf) * 100
        plt.title(f"AI Gæt: {best_guess} ({best_conf:.1f}%)")
        plt.axis('off')
        plt.show()


if __name__ == '__main__':
    run_demo()