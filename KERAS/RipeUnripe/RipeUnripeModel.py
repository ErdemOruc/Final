import os 
import tensorflow as tf                                                           
from tensorflow.keras.models import Sequential, Model      
from tensorflow.keras.optimizers import Adam               
from tensorflow.keras.callbacks import EarlyStopping               
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization  
# pyrefly: ignore [missing-import]
from tensorflow.keras.applications import DenseNet121

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_DIR = os.path.join(BASE_DIR, "ripe_unripe_data", "train")
TEST_DIR = os.path.join(BASE_DIR, "ripe_unripe_data", "test")
VAL_DIR = os.path.join(BASE_DIR, "ripe_unripe_data", "val")
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "RipeUnripeWeight.keras")

train_data = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    labels='inferred',
    label_mode='categorical',
    image_size=(256, 256),
    batch_size=32)

train_data = train_data.map(lambda x, y: (x / 255.0, y))

test_data = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    labels='inferred',
    label_mode='categorical',
    image_size=(256, 256),
    batch_size=32)

test_data = test_data.map(lambda x, y: (x / 255.0, y))

val_data = tf.keras.preprocessing.image_dataset_from_directory(
    VAL_DIR,
    labels='inferred',
    label_mode='categorical',
    image_size=(256, 256),
    batch_size=32)

val_data = val_data.map(lambda x, y: (x / 255.0, y))

conv_base = DenseNet121(
    weights='imagenet',
    include_top = False,
    input_shape=(256,256,3),
    pooling='avg'
)

conv_base.trainable = False

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal_and_vertical"),
    tf.keras.layers.RandomRotation(0.2),
    tf.keras.layers.RandomZoom(0.2),
    tf.keras.layers.RandomBrightness(0.2),
])

model = Sequential()
model.add(data_augmentation)
model.add(conv_base)
model.add(BatchNormalization())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.35))
model.add(BatchNormalization())
model.add(Dense(120, activation='relu'))
model.add(Dense(2, activation='softmax'))

model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

early_stop = EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)

history = model.fit(train_data, epochs=100, validation_data=val_data, callbacks=[early_stop])

test_evaluation = model.evaluate(test_data)

print("Test Loss:", test_evaluation[0])
print("Test Accuracy:", test_evaluation[1])

evaluation = model.evaluate(val_data)

print("Validation Loss:", evaluation[0])
print("Validation Accuracy:", evaluation[1])

model.save(MODEL_SAVE_PATH)
print("Model saved successfully!")