import tensorflow as tf
from LoadModels import LoadModel

print("Loading models...")
_, keras_LeafDisease, keras_OldDamaged, keras_RipeUnripe = LoadModel()

def inspect(model, name):
    print(f"\n--- Inspecting {name} ---")
    if model is None:
        print("Model is None")
        return
    print("Input shape:", model.input_shape)
    print("Output shape:", model.output_shape)

inspect(keras_LeafDisease, "LeafDisease")
inspect(keras_OldDamaged, "OldDamaged")
inspect(keras_RipeUnripe, "RipeUnripe")
