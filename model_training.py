from tensorflow.keras.applications import VGG19
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, AdamW
from tensorflow.keras import backend as K
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import tensorflow as tf
import numpy as np
import os
import math

IMG_SIZE = (224, 224)
BATCH_SIZE = 32


def load_dataset_files(directory):
    """Loads file paths and labels from the directory"""
    class_dirs = [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]
    classes = {class_name: i for i, class_name in enumerate(class_dirs)}

    file_paths = []
    labels = []

    for class_name in class_dirs:
        class_dir = os.path.join(directory, class_name)
        class_idx = classes[class_name]

        files = [os.path.join(class_dir, f) for f in os.listdir(class_dir)
                 if os.path.isfile(os.path.join(class_dir, f)) and
                 f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        file_paths.extend(files)
        labels.extend([class_idx] * len(files))

    return file_paths, labels, classes


def process_image(file_path, label):
    img = tf.io.read_file(file_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMG_SIZE)
    img = tf.cast(img, tf.float32) / 255.0

    label = tf.cast(label, tf.float32)
    return img, label



def new_augment_image(image, label):
    # Horizontal flip
    image = tf.image.random_flip_left_right(image)

    # Brightness
    image = tf.image.random_brightness(image, max_delta=0.15)

    # Contrast
    image = tf.image.random_contrast(image, 0.85, 1.15)

    # Saturation
    image = tf.image.random_saturation(image, 0.9, 1.1)

    # Hue
    image = tf.image.random_hue(image, max_delta=0.05)

    # Random Gaussian blur (only 30% of the time)
    if tf.random.uniform(()) > 0.7:
        # Use tf.nn.gaussian_blur via tf.image. stateless_random_brightness
        image = tf.image.stateless_random_brightness(
            image,
            max_delta=0.0,
            seed=[42, 42]
        )

    # Trim values to stay within [0, 1]
    image = tf.clip_by_value(image, 0.0, 1.0)

    return image, label


def create_test_dataset(test_dir, batch_size=BATCH_SIZE):
    file_paths, labels, classes = load_dataset_files(test_dir)

    print(f"Classes in the test set: {classes}")
    print(f"Number of test images: {len(file_paths)}")

    # Create test dataset
    test_ds = tf.data.Dataset.from_tensor_slices((file_paths, labels))
    test_ds = test_ds.map(process_image, num_parallel_calls=tf.data.AUTOTUNE)
    test_ds = test_ds.batch(batch_size, drop_remainder=True)
    test_ds = test_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

    # Calculate the number of steps
    test_steps = len(file_paths) // batch_size + (1 if len(file_paths) % batch_size != 0 else 0)

    return test_ds, test_steps, classes


def create_balanced_train_dataset(train_dir, batch_size=BATCH_SIZE, val_split=0.2):
    """Create a balanced training dataset with oversampling for train and a separate validation"""
    file_paths, labels, classes = load_dataset_files(train_dir)

    df = pd.DataFrame({'file_path': file_paths, 'label': labels})

    # Split into train/val BEFORE oversampling
    train_df, val_df = train_test_split(
        df,
        test_size=val_split,
        stratify=df['label'],
        random_state=42
    )

    print(f"Classes: {classes}")
    print(f"Total number of images: {len(file_paths)}")
    print(f"Number of training images: {len(train_df)}")
    print(f"Number of validation images: {len(val_df)}")

    # Oversampling only on the training part
    class_counts = train_df['label'].value_counts().to_dict()

    minority_label = min(class_counts, key=class_counts.get)
    majority_label = max(class_counts, key=class_counts.get)

    minority_df = train_df[train_df['label'] == minority_label]
    majority_df = train_df[train_df['label'] == majority_label]

    # Amount of "minority" after oversampling: e.g. 50-70% of the majority
    target_minority_count = int(len(majority_df) * 1.0)
    oversampled_minority_df = minority_df.sample(target_minority_count, replace=True, random_state=42)

    balanced_df = pd.concat([majority_df, oversampled_minority_df])
    balanced_df = balanced_df.sample(frac=1.0, random_state=42).reset_index(drop=True)

    print(f"After oversampling: {balanced_df['label'].value_counts().to_dict()}")

    # ---- TRAIN DATASET ----
    train_ds = tf.data.Dataset.from_tensor_slices((balanced_df['file_path'].values, balanced_df['label'].values))
    train_ds = train_ds.map(process_image, num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.map(new_augment_image, num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.shuffle(buffer_size=len(balanced_df))
    train_ds = train_ds.batch(batch_size)
    train_ds = train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    train_ds = train_ds.repeat()

    steps_per_epoch = math.ceil(len(balanced_df) / batch_size)

    # ---- VALIDATION DATASET ----
    val_ds = tf.data.Dataset.from_tensor_slices((val_df['file_path'].values, val_df['label'].values))
    val_ds = val_ds.map(process_image, num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.batch(batch_size)
    val_ds = val_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    val_ds = val_ds.repeat()

    val_steps = math.ceil(len(val_df) / batch_size)

    return train_ds, steps_per_epoch, val_ds, val_steps, classes


# For testing
test_ds, test_steps, test_classes = create_test_dataset(
    test_dir='/kaggle/input/faces-artefact-recognition/trainee_dataset/test',
    batch_size=BATCH_SIZE
)

# Balanced for training
balanced_train_ds, balanced_steps, val_ds, val_steps, _ = create_balanced_train_dataset(
    train_dir='/kaggle/input/faces-artefact-recognition/trainee_dataset/train',
    batch_size=BATCH_SIZE,
    val_split=0.2

)

def micro_f1(y_true, y_pred):
    y_pred = tf.squeeze(y_pred, axis=-1)
    y_pred = tf.cast(y_pred > 0.5, tf.float32)  # Binarization

    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    tp = tf.reduce_sum(tf.cast(y_true * y_pred, tf.float32))  # True Positives
    fp = tf.reduce_sum(tf.cast((1 - y_true) * y_pred, tf.float32))  # False Positives
    fn = tf.reduce_sum(tf.cast(y_true * (1 - y_pred), tf.float32))  # False Negatives

    precision = tp / (tp + fp + tf.keras.backend.epsilon())  # Add a small number to avoid /0
    recall = tp / (tp + fn + tf.keras.backend.epsilon())

    f1 = 2 * (precision * recall) / (precision + recall + tf.keras.backend.epsilon())

    return f1


callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        'best_model.keras',
        monitor="val_micro_f1",
        mode="max",
        save_best_only=True,
        verbose=1
    )
]

def focal_loss_binary(alpha=0.25, gamma=2.0):
    def loss(y_true, y_pred):
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)

        loss_pos = -alpha * K.pow(1. - y_pred, gamma) * y_true * K.log(y_pred)
        loss_neg = -(1 - alpha) * K.pow(y_pred, gamma) * (1 - y_true) * K.log(1. - y_pred)

        return K.mean(loss_pos + loss_neg)
    return loss


base_model = VGG19(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = True

# Build the new model
x = base_model.output
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dense(1024)(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Activation('relu')(x)
x = tf.keras.layers.Dropout(0.3)(x)
x = tf.keras.layers.Dense(512)(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Activation('relu')(x)
x = tf.keras.layers.Dropout(0.2)(x)
predictions = tf.keras.layers.Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=predictions)


lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=1e-5,
        decay_steps=400,         # approximately every epoch (81 batches = ~1 epoch)
        decay_rate=0.85,          # smooth drop
        staircase=True           # discretely, in steps
)

model.compile(
    optimizer=Adam(learning_rate=lr_schedule),
    loss=focal_loss_binary(),
    metrics=['accuracy', micro_f1]
)

for images, labels in balanced_train_ds.take(1):
    print("Images batch shape:", images.shape)
    print("Labels batch shape:", labels.shape)
    print("Labels dtype:", labels.dtype)
    print("Unique labels in batch:", tf.unique(tf.reshape(labels, [-1]))[0].numpy())
    print("Min/max pixel values:", tf.reduce_min(images).numpy(), tf.reduce_max(images).numpy())

class_weights = {
    0: 8.0,
    1: 1
}

history = model.fit(
    balanced_train_ds,
    validation_data=val_ds,
    epochs=20,
    batch_size=BATCH_SIZE,
    class_weight=class_weights,
    steps_per_epoch=balanced_steps,
    validation_steps=val_steps,
    callbacks=callbacks
)

model = tf.keras.models.load_model("best_model.keras", compile=False)


def evaluate_model_on_test(model, test_ds, test_steps, class_names=None):
    y_true = []
    y_pred = []

    for images, labels in test_ds.take(test_steps):
        preds = model.predict(images, verbose=0)

        # If the model returns probabilities
        if preds.shape[-1] > 1:
            pred_classes = np.argmax(preds, axis=1)
        else:
            pred_classes = (preds > 0.5).astype(int).flatten()

        y_pred.extend(pred_classes)
        y_true.extend(labels.numpy().astype(int))

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # ---- Metrics ----
    print("\n📊 Classification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))

    # ---- Confusion Matrix ----
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.show()

    return {
        'y_true': y_true,
        'y_pred': y_pred,
        'confusion_matrix': cm,
        'report': classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    }


results = evaluate_model_on_test(model, test_ds, test_steps, test_classes)
