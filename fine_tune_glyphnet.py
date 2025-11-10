# fine_tune_glyphnet.py
# Fine-tune an existing glyph classifier and rebalance classes automatically.
from pathlib import Path
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from sklearn.utils import class_weight
import os

# CONFIG - adjust if you want
MODEL_IN = "notebook/safe_glyphnet_best.keras"      # existing model
MODEL_OUT = "notebook/safe_glyphnet_rebalanced.keras"
DATA_DIR = "dataset/Train"                          # must contain subfolders Corrected Normal Reversal
BATCH = 64
EPOCHS = 6                                         # small number to start; increase if you have time
LR = 1e-4

p = Path(DATA_DIR)
if not p.exists():
    raise SystemExit(f"DATA_DIR not found: {DATA_DIR} - run from project root")

print("Loading base model:", MODEL_IN)
model = load_model(MODEL_IN)

# infer target size
ishape = model.input_shape
if ishape and len(ishape) >= 3 and ishape[1] and ishape[2]:
    IMG_H, IMG_W = int(ishape[1]), int(ishape[2])
else:
    IMG_H = IMG_W = 96

print("Using image size:", IMG_H, "x", IMG_W)

# generator with augmentation
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=12,
    width_shift_range=0.12,
    height_shift_range=0.12,
    shear_range=0.08,
    zoom_range=0.12,
    brightness_range=(0.8,1.2),
    fill_mode='nearest',
    validation_split=0.12
)

train_flow = datagen.flow_from_directory(
    DATA_DIR,
    target_size=(IMG_H, IMG_W),
    color_mode='grayscale',
    batch_size=BATCH,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

val_flow = datagen.flow_from_directory(
    DATA_DIR,
    target_size=(IMG_H, IMG_W),
    color_mode='grayscale',
    batch_size=BATCH,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

# compute class weights from files on disk to rebalance
labels = train_flow.classes  # numeric labels for each sample in training
if len(labels) == 0:
    raise SystemExit("No training images found in DATA_DIR.")
cw = class_weight.compute_class_weight('balanced', classes=np.unique(labels), y=labels)
class_weights = {i: float(cw[i]) for i in range(len(cw))}
print("Computed class weights:", class_weights)

# callbacks
ckpt = ModelCheckpoint(MODEL_OUT, monitor='val_loss', save_best_only=True, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=1)
early = EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True, verbose=1)

# compile & train
model.compile(optimizer=Adam(LR), loss='categorical_crossentropy', metrics=['accuracy'])
print("Starting fine-tune for", EPOCHS, "epochs ...")
history = model.fit(
    train_flow,
    validation_data=val_flow,
    epochs=EPOCHS,
    callbacks=[ckpt, reduce_lr, early],
    class_weight=class_weights
)

# save final (checkpoint already saved best)
if not Path(MODEL_OUT).exists():
    model.save(MODEL_OUT)
print("Done. Fine-tuned model saved to:", MODEL_OUT)

