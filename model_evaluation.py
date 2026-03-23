import argparse
import json
import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc

IMG_SIZE = (224, 224)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


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


def create_test_dataset(test_dir, batch_size):
    file_paths, labels, classes = load_dataset_files(test_dir)

    logger.info(f"Classes in test set: {classes}")
    logger.info(f"Number of test images: {len(file_paths)}")

    test_ds = tf.data.Dataset.from_tensor_slices((file_paths, labels))
    test_ds = test_ds.map(process_image, num_parallel_calls=tf.data.AUTOTUNE)
    test_ds = test_ds.batch(batch_size, drop_remainder=True)
    test_ds = test_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

    test_steps = len(file_paths) // batch_size + (1 if len(file_paths) % batch_size != 0 else 0)

    return test_ds, test_steps, classes, file_paths


def evaluate_model(model, test_ds, test_steps):
    y_true = []
    y_pred = []
    y_scores = []

    for images, labels in test_ds.take(test_steps):
        preds = model.predict(images, verbose=0)

        if preds.shape[-1] > 1:
            pred_classes = np.argmax(preds, axis=1)
            scores = preds[:, 1]
        else:
            pred_classes = (preds > 0.5).astype(int).flatten()
            scores = preds.flatten()

        y_pred.extend(pred_classes)
        y_scores.extend(scores)
        y_true.extend(labels.numpy().astype(int))

    return np.array(y_true), np.array(y_pred), np.array(y_scores)


def save_confusion_matrix(y_true, y_pred, class_names, output_dir):
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()

    path = os.path.join(output_dir, 'confusion_matrix.png')
    plt.savefig(path, dpi=150)
    plt.close()
    logger.info(f"Saved confusion matrix to {path}")


def save_roc_curve(y_true, y_scores, output_dir):
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.tight_layout()

    path = os.path.join(output_dir, 'roc_curve.png')
    plt.savefig(path, dpi=150)
    plt.close()
    logger.info(f"Saved ROC curve to {path} (AUC = {roc_auc:.4f})")

    return roc_auc


def save_sample_predictions(model, test_ds, class_names, output_dir, num_images=12):
    images, labels = next(iter(test_ds))
    num_images = min(num_images, len(images))

    predictions = model.predict(images[:num_images], verbose=0)

    if predictions.shape[-1] > 1:
        pred_classes = np.argmax(predictions, axis=1)
    else:
        pred_classes = (predictions > 0.5).astype(int).flatten()

    class_name_map = {v: k for k, v in class_names.items()}
    rows = int(np.ceil(num_images / 4))

    plt.figure(figsize=(16, 4 * rows))
    for i in range(num_images):
        plt.subplot(rows, 4, i + 1)
        plt.imshow(images[i].numpy())

        true_label = int(labels[i].numpy())
        pred_label = pred_classes[i]
        color = 'green' if true_label == pred_label else 'red'

        plt.title(f"True: {class_name_map[true_label]}\nPred: {class_name_map[pred_label]}", color=color)
        plt.axis('off')

    plt.tight_layout()

    path = os.path.join(output_dir, 'sample_predictions.png')
    plt.savefig(path, dpi=150)
    plt.close()
    logger.info(f"Saved sample predictions to {path}")


def save_metrics_report(y_true, y_pred, class_names, roc_auc, output_dir):
    report_dict = classification_report(y_true, y_pred, target_names=list(class_names.keys()), output_dict=True)
    report_dict['roc_auc'] = roc_auc

    path = os.path.join(output_dir, 'metrics.json')
    with open(path, 'w') as f:
        json.dump(report_dict, f, indent=2)
    logger.info(f"Saved metrics to {path}")

    report_text = classification_report(y_true, y_pred, target_names=list(class_names.keys()))
    logger.info("Classification Report:\n" + report_text)

    return report_dict


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate trained artifact classifier")
    parser.add_argument('--test-dir', type=str, required=True, help="Path to test data directory")
    parser.add_argument('--model-path', type=str, required=True, help="Path to saved .keras model")
    parser.add_argument('--output-dir', type=str, default='eval_results', help="Directory to save evaluation results")
    parser.add_argument('--batch-size', type=int, default=32)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    logger.info(f"Results will be saved to {args.output_dir}")

    logger.info(f"Loading model from {args.model_path}")
    model = tf.keras.models.load_model(args.model_path, compile=False)

    test_ds, test_steps, test_classes, _ = create_test_dataset(args.test_dir, args.batch_size)

    logger.info("Running evaluation...")
    y_true, y_pred, y_scores = evaluate_model(model, test_ds, test_steps)

    save_confusion_matrix(y_true, y_pred, list(test_classes.keys()), args.output_dir)
    roc_auc = save_roc_curve(y_true, y_scores, args.output_dir)
    save_sample_predictions(model, test_ds, test_classes, args.output_dir)
    report = save_metrics_report(y_true, y_pred, test_classes, roc_auc, args.output_dir)

    logger.info(f"Evaluation complete. Accuracy: {report['accuracy']:.4f}, AUC: {roc_auc:.4f}")
