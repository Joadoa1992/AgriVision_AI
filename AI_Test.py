from ultralytics import YOLO
import os

# 1. Load den model du lige har trænet (selvom den blev afbrudt)
# Tjek om stien passer - den lander typisk i 'train', 'train2' osv.
model_path = "runs/classify/train/weights/last.pt"

if os.path.exists(model_path):
    model = YOLO(model_path)

    # 2. Prøv at gætte på test-billederne
    results = model.predict(source="test/test", save=True, conf=0.2)
    print(f"✅ Test færdig! Se resultaterne i: {results[0].save_dir}")
else:
    print("❌ Kunne ikke finde last.pt. Du skal måske køre træningen i 2 minutter igen.")