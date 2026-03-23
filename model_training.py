import argparse
import math
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from tensorflow.keras import backend as K
from tensorflow.keras.applications import VGG19
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

IMG_SIZE = (224, 224)


def load_dataset_files(directory):
    """Loads file paths and labels from a directory where each subdirectory is a class."""
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


def augment_image(image, label):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=0.15)
    image = tf.image.random_contrast(image, 0.85, 1.15)
    image = tf.image.random_saturation(image, 0.9, 1.1)
    image = tf.image.random_hue(image, max_delta=0.05)
    image = tf.clip_by_value(image, 0.0, 1.0)
    return image, label


def create_test_dataset(test_dir, batch_size):
    file_paths, labels, classes = load_dataset_files(test_dir)

    print(f"Classes in test set: {classes}")
    print(f"Number of test images: {len(file_paths)}")

    test_ds = tf.data.Dataset.from_tensor_slices((file_paths, labels))
    test_ds = test_ds.map(process_image, num_parallel_calls=tf.data.AUTOTUNE)
    test_ds = test_ds.batch(batch_size, drop_remainder=True)
    test_ds = test_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

    test_steps = len(file_paths) // batch_size + (1 if len(file_paths) % batch_size != 0 else 0)

    return test_ds, test_steps, classes


def create_balanced_train_dataset(train_dir, batch_size, val_split=0.2):
    """Creates a balanced training dataset with oversampling and a separate validation split."""
    file_paths, labels, classes = load_dataset_files(train_dir)

    df = pd.DataFrame({'file_path': file_paths, 'label': labels})

    # Split into train/val BEFORE oversampling to avoid data leakage
    train_df, val_df = train_test_split(
        df,
        test_size=val_split,
        stratify=df['label'],
        random_state=42
    )

    print(f"Classes: {classes}")
    print(f"Total images: {len(file_paths)}")
    print(f"Training images: {len(train_df)}")
    print(f"Validation images: {len(val_df)}")

    class_counts = train_df['label'].value_counts().to_dict()
    minority_label = min(class_counts, key=class_counts.get)
    majority_label = max(class_counts, key=class_counts.get)

    minority_df = train_df[train_df['label'] == minority_label]
    majority_df = train_df[train_df['label'] == majority_label]

    target_minority_count = int(len(majority_df) * 1.0)
    oversampled_minority_df = minority_df.sample(target_minority_count, replace=True, random_state=42)

    balanced_df = pd.concat([majority_df, oversampled_minority_df])
    balanced_df = balanced_df.sample(frac=1.0, random_state=42).reset_index(drop=True)

    print(f"After oversampling: {balanced_df['label'].value_counts().to_dict()}")

    train_ds = tf.data.Dataset.from_tensor_slices((balanced_df['file_path'].values, balanced_df['label'].values))
    train_ds = train_ds.map(process_image, num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.map(augment_image, num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.shuffle(buffer_size=len(balanced_df))
    train_ds = train_ds.batch(batch_size)
    train_ds = train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    train_ds = train_ds.repeat()

    steps_per_epoch = math.ceil(len(balanced_df) / batch_size)

    val_ds = tf.data.Dataset.from_tensor_slices((val_df['file_path'].values, val_df['label'].values))
    val_ds = val_ds.map(process_image, num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.batch(batch_size)
    val_ds = val_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    val_ds = val_ds.repeat()

    val_steps = math.ceil(len(val_df) / batch_size)

    return train_ds, steps_per_epoch, val_ds, val_steps, classes


def micro_f1(y_true, y_pred):
    y_pred = tf.squeeze(y_pred, axis=-1)
    y_pred = tf.cast(y_pred > 0.5, tf.float32)
    y_true = tf.cast(y_true, tf.float32)

    tp = tf.reduce_sum(y_true * y_pred)
    fp = tf.reduce_sum((1 - y_true) * y_pred)
    fn = tf.reduce_sum(y_true * (1 - y_pred))

    precision = tp / (tp + fp + K.epsilon())
    recall = tp / (tp + fn + K.epsilon())

    return 2 * (precision * recall) / (precision + recall + K.epsilon())


def focal_loss_binary(alpha=0.25, gamma=2.0):
    def loss(y_true, y_pred):
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)

        loss_pos = -alpha * K.pow(1. - y_pred, gamma) * y_true * K.log(y_pred)
        loss_neg = -(1 - alpha) * K.pow(y_pred, gamma) * (1 - y_true) * K.log(1. - y_pred)

        return K.mean(loss_pos + loss_neg)
    return loss


def build_model(learning_rate=1e-5, decay_steps=400, decay_rate=0.85):
    base_model = VGG19(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = True

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
        initial_learning_rate=learning_rate,
        decay_steps=decay_steps,
        decay_rate=decay_rate,
        staircase=True
    )

    model.compile(
        optimizer=Adam(learning_rate=lr_schedule),
        loss=focal_loss_binary(),
        metrics=['accuracy', micro_f1]
    )

    return model


def evaluate_model(model, test_ds, test_steps, class_names=None):
    y_true = []
    y_pred = []

    for images, labels in test_ds.take(test_steps):
        preds = model.predict(images, verbose=0)

        if preds.shape[-1] > 1:
            pred_classes = np.argmax(preds, axis=1)
        else:
            pred_classes = (preds > 0.5).astype(int).flatten()

        y_pred.extend(pred_classes)
        y_true.extend(labels.numpy().astype(int))

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))

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


def parse_args():
    parser = argparse.ArgumentParser(description="Train VGG-19 artifact classifier")
    parser.add_argument('--train-dir', type=str, required=True, help="Path to training data directory")
    parser.add_argument('--test-dir', type=str, required=True, help="Path to test data directory")
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--learning-rate', type=float, default=1e-5)
    parser.add_argument('--val-split', type=float, default=0.2)
    parser.add_argument('--output-model', type=str, default='best_model.keras', help="Path to save the best model")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    test_ds, test_steps, test_classes = create_test_dataset(args.test_dir, args.batch_size)

    balanced_train_ds, balanced_steps, val_ds, val_steps, _ = create_balanced_train_dataset(
        args.train_dir, args.batch_size, args.val_split
    )

    model = build_model(learning_rate=args.learning_rate)

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            args.output_model,
            monitor="val_micro_f1",
            mode="max",
            save_best_only=True,
            verbose=1
        )
    ]

    class_weights = {0: 8.0, 1: 1}

    model.fit(
        balanced_train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        batch_size=args.batch_size,
        class_weight=class_weights,
        steps_per_epoch=balanced_steps,
        validation_steps=val_steps,
        callbacks=callbacks
    )

    model = tf.keras.models.load_model(args.output_model, compile=False)
    evaluate_model(model, test_ds, test_steps, test_classes)
