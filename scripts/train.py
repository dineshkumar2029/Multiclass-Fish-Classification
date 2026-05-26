import os
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    Dropout,
    GlobalAveragePooling2D
)

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.applications import (
    MobileNet,
    VGG16,
    ResNet50,
    InceptionV3,
    EfficientNetB0
)

# =========================
# CREATE FOLDERS
# =========================

os.makedirs("models", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

# =========================
# DATA PREPROCESSING
# =========================

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 3

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(
    rescale=1./255
)

test_datagen = ImageDataGenerator(
    rescale=1./255
)

# =========================
# LOAD DATASET
# =========================

train_data = train_datagen.flow_from_directory(
    'data/train',
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

val_data = val_datagen.flow_from_directory(
    'data/val',
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

test_data = test_datagen.flow_from_directory(
    'data/test',
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

# =========================
# CNN MODEL FROM SCRATCH
# =========================

print("\nTraining CNN Model...\n")

cnn_model = Sequential([

    Conv2D(32, (3,3), activation='relu', input_shape=(224,224,3)),
    MaxPooling2D(2,2),

    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Flatten(),

    Dense(128, activation='relu'),
    Dropout(0.5),

    Dense(train_data.num_classes, activation='softmax')

])

cnn_model.compile(
    optimizer=Adam(),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

cnn_history = cnn_model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS
)

cnn_model.save("models/cnn_model.h5")

cnn_loss, cnn_acc = cnn_model.evaluate(test_data)

print(f"\nCNN Accuracy: {cnn_acc*100:.2f}%")

# =========================
# TRANSFER LEARNING FUNCTION
# =========================

def train_transfer_model(base_model_class, model_name):

    print(f"\nTraining {model_name}...\n")

    base_model = base_model_class(
        weights='imagenet',
        include_top=False,
        input_shape=(224,224,3)
    )

    base_model.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)

    predictions = Dense(
        train_data.num_classes,
        activation='softmax'
    )(x)

    model = Model(
        inputs=base_model.input,
        outputs=predictions
    )

    model.compile(
        optimizer=Adam(),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=EPOCHS
    )

    model.save(f"models/{model_name}.h5")

    loss, accuracy = model.evaluate(test_data)

    print(f"{model_name} Accuracy: {accuracy*100:.2f}%")

    return model, accuracy

# =========================
# TRAIN PRETRAINED MODELS
# =========================

mobilenet_model, mobilenet_acc = train_transfer_model(
    MobileNet,
    "mobilenet_model"
)

vgg16_model, vgg16_acc = train_transfer_model(
    VGG16,
    "vgg16_model"
)

resnet50_model, resnet50_acc = train_transfer_model(
    ResNet50,
    "resnet50_model"
)

inceptionv3_model, inceptionv3_acc = train_transfer_model(
    InceptionV3,
    "inceptionv3_model"
)

efficientnetb0_model, efficientnetb0_acc = train_transfer_model(
    EfficientNetB0,
    "efficientnetb0_model"
)

# =========================
# BEST MODEL SELECTION
# =========================

model_accuracies = {
    "CNN": cnn_acc,
    "MobileNet": mobilenet_acc,
    "VGG16": vgg16_acc,
    "ResNet50": resnet50_acc,
    "InceptionV3": inceptionv3_acc,
    "EfficientNetB0": efficientnetb0_acc
}

best_model_name = max(
    model_accuracies,
    key=model_accuracies.get
)

best_accuracy = model_accuracies[best_model_name]

print(f"\nBest Model: {best_model_name}")
print(f"Best Accuracy: {best_accuracy*100:.2f}%")

# =========================
# SAVE BEST MODEL
# =========================

if best_model_name == "CNN":
    cnn_model.save("best_model.h5")

elif best_model_name == "MobileNet":
    mobilenet_model.save("best_model.h5")

elif best_model_name == "VGG16":
    vgg16_model.save("best_model.h5")

elif best_model_name == "ResNet50":
    resnet50_model.save("best_model.h5")

elif best_model_name == "InceptionV3":
    inceptionv3_model.save("best_model.h5")

elif best_model_name == "EfficientNetB0":
    efficientnetb0_model.save("best_model.h5")

# =========================
# ACCURACY COMPARISON GRAPH
# =========================

models = [
    'CNN',
    'MobileNet',
    'VGG16',
    'ResNet50',
    'InceptionV3',
    'EfficientNetB0'
]

accuracies = [
    cnn_acc * 100,
    mobilenet_acc * 100,
    vgg16_acc * 100,
    resnet50_acc * 100,
    inceptionv3_acc * 100,
    efficientnetb0_acc * 100
]

plt.figure(figsize=(12,6))

bars = plt.bar(models, accuracies)

plt.xlabel("Models")
plt.ylabel("Accuracy")
plt.title("Model Accuracy Comparison")

for bar, acc in zip(bars, accuracies):
    plt.text(
        bar.get_x() + bar.get_width()/2,
        bar.get_height(),
        f"{acc:.2f}%",
        ha='center',
        va='bottom'
    )

plt.xticks(rotation=15)

plt.tight_layout()

plt.savefig("outputs/model_comparison.png")

plt.show()

# =========================
# FINAL SUMMARY
# =========================

print("\n==============================")
print("MODEL ACCURACY SUMMARY")
print("==============================")

for model_name, accuracy in model_accuracies.items():
    print(f"{model_name}: {accuracy*100:.2f}%")

print("\nBest model saved as best_model.h5")
print("Training completed successfully!")