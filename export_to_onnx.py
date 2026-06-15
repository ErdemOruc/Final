# pyrefly: ignore [missing-import]
from ultralytics import YOLO

def export_model():
    model_path = r'C:\Users\Erdem\Desktop\Final\YOLO\runs\detect\WeightForLeafOrFruit-2\weights\best.pt'
    
    print(f"Loading model: {model_path}...")
    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"Error! Model not found. Please ensure your training is complete. Details: {e}")
        return

    model.export(format='onnx', half=True)
    
    print("\n✅ Conversion Successful!")

if __name__ == '__main__':
    export_model()
