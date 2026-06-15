import os 
import tensorflow as tf                                   
from tensorflow.keras.optimizers import Adam               
from tensorflow.keras.callbacks import EarlyStopping               

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_DIR = os.path.join(BASE_DIR, "old_damaged_data", "train")
VAL_DIR = os.path.join(BASE_DIR, "old_damaged_data", "val")
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "OldDamagedWeight.keras")

train_data = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    labels='inferred',
    label_mode='categorical',
    image_size=(256, 256),
    batch_size=32)

train_data = train_data.map(lambda x, y: (x / 255.0, y))

val_data = tf.keras.preprocessing.image_dataset_from_directory(
    VAL_DIR,
    labels='inferred',
    label_mode='categorical',
    image_size=(256, 256),
    batch_size=32)

val_data = val_data.map(lambda x, y: (x / 255.0, y))

if not os.path.exists(MODEL_SAVE_PATH):
    print(f"ERROR: {MODEL_SAVE_PATH} not found! You should run OldDamagedModel.py first.")
    exit(1)

model = tf.keras.models.load_model(MODEL_SAVE_PATH)


conv_base = None
for layer in model.layers:
    if layer.name.startswith("densenet"):
        conv_base = layer
        break

if conv_base is None:
    print("ERROR: Model's DenseNet121 layer is not found!")
    exit(1)

conv_base.trainable = True

for layer in conv_base.layers[:-50]:
    layer.trainable = False

model.compile(optimizer=Adam(learning_rate=1e-5), loss='categorical_crossentropy', metrics=['accuracy'])

history_fine = model.fit(
    train_data, 
    epochs=30, 
    validation_data=val_data, 
    callbacks=[EarlyStopping(patience=6, restore_best_weights=True)]
)

evaluation = model.evaluate(val_data)

print("Validation Loss:", evaluation[0])
print("Validation Accuracy:", evaluation[1])

model.save(MODEL_SAVE_PATH)
