from ultralytics import YOLO

def validate():
    # Load din model fra mappen i billede_4.png
    model = YOLO("runs/classify/train/weights/best.pt")

    # Kør validering
    # Vi sætter workers=0 for at være helt sikre på at undgå multiprocessing-fejl på Windows
    model.val(workers=0)

if __name__ == '__main__':
    validate()