from ultralytics import YOLO

def main():
   # Load model
   model = YOLO("yolov8n.pt")

   # Train
   model.train(data="datasets/pvz/pvztrain.yaml", epochs=250,patience=150)

   # Validate
   model.val()

if __name__ == "__main__":
   main()
