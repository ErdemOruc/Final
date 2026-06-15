# pyrefly: ignore [missing-import]
from ultralytics import YOLO
import tensorflow as tf

def LoadModel():
    try:
        yolo_model = YOLO(r'YOLO\runs\detect\WeightForLeafOrFruit-2\weights\best.pt')
        keras_LeafDisease  = tf.keras.models.load_model(r'KERAS\LeafDisease\LeafDiseaseWeight.keras')
        keras_OldDamaged   = tf.keras.models.load_model(r'KERAS\OldDamaged\OldDamagedWeight.keras')
        keras_RipeUnripe   = tf.keras.models.load_model(r'KERAS\RipeUnripe\RipeUnripeWeight.keras')
        print("Models loaded successfully.")
        return yolo_model, keras_LeafDisease, keras_OldDamaged, keras_RipeUnripe
    except Exception as e:
        print(f"ERROR loading models: {e}")
        return None, None, None, None
