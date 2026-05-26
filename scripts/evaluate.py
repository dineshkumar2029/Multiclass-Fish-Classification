import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    classification_report,
    confusion_matrix
)

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# =========================
# LOAD MODEL
# =========================

model = load_model("best_model.h5")

# =========================
# LOAD TEST DATA
# =========================

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

test_datagen = ImageDataGenerator(
    rescale=1./255
)

test_data = test_datagen.flow_from_directory(
    'data/test',
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

# =========================
# PREDICTIONS
# =========================

predictions = model.predict(test_data)

predicted_classes = np.argmax(predictions, axis=1)

true_classes = test_data.classes

class_labels = list(test_data.class_indices.keys())

# =========================
# CLASSIFICATION REPORT
# =========================

report = classification_report(
    true_classes,
    predicted_classes,
    target_names=class_labels
)

print("\nClassification Report:\n")
print(report)

# =========================
# CONFUSION MATRIX
# =========================

cm = confusion_matrix(
    true_classes,
    predicted_classes
)

plt.figure(figsize=(12,10))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=class_labels,
    yticklabels=class_labels
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")

plt.xticks(rotation=90)
plt.yticks(rotation=0)

plt.tight_layout()

plt.savefig("outputs/confusion_matrix.png")

plt.show()