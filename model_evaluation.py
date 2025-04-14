from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import numpy as np
import os

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


# For testing
test_ds, test_steps, test_classes = create_test_dataset(
    test_dir='/kaggle/input/faces-artefact-recognition/trainee_dataset/test',
    batch_size=BATCH_SIZE
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


def visualize_model_predictions(model, test_dataset, classes, num_images=9):
    # Invert class dictionary to get class names by indexes
    class_names = {v: k for k, v in classes.items()}

    # Get the first batch_size of images and their labels
    images, labels = next(iter(test_dataset))

    if num_images > len(images):
        num_images = len(images)

    predictions = model.predict(images[:num_images])
    predicted_classes = np.argmax(predictions, axis=1)

    # Create a grid to display images
    rows = int(np.ceil(num_images / 3))
    cols = min(3, num_images)

    plt.figure(figsize=(12, 4 * rows))

    for i in range(num_images):
        plt.subplot(rows, cols, i + 1)

        # Display image
        img = images[i].numpy()
        plt.imshow(img)

        # Get real and predicted class labels
        true_label = int(labels[i].numpy())
        pred_label = predicted_classes[i]

        true_class_name = class_names[true_label]
        pred_class_name = class_names[pred_label]

        color = 'green' if true_label == pred_label else 'red'

        plt.title(f"Real: {true_class_name}\nPredicted: {pred_class_name}",
                  color=color)

        plt.axis('off')

    plt.tight_layout()
    plt.show()


visualize_model_predictions(
    model=model,
    test_dataset=test_ds,
    classes=test_classes,
    num_images=9
)
