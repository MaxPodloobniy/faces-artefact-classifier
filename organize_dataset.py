import os
import shutil


def organize_images_into_subfolders(base_path):
    for split in ['train', 'test']:
        split_path = os.path.join(base_path, split)
        if not os.path.isdir(split_path):
            print(f"Directory '{split_path}' not found. Skipping...")
            continue

        files = [f for f in os.listdir(split_path) if f.endswith('.png')]
        print(f"Found {len(files)} images in '{split_path}'")

        for fname in files:
            try:
                label = fname.split('_')[-1].split('.')[0]  # "image_00002_1.png" → "1"
                label_dir = os.path.join(split_path, label)
                os.makedirs(label_dir, exist_ok=True)

                src_path = os.path.join(split_path, fname)
                dst_path = os.path.join(label_dir, fname)

                shutil.move(src_path, dst_path)
            except Exception as e:
                print(f"Error processing file '{fname}': {e}")

    print("✅ Done organizing files into class subfolders.")


# Використання:
organize_images_into_subfolders('/Users/maxim/Downloads/trainee_dataset')
# Наприклад:
# organize_images_into_subfolders('/Users/maxim/PycharmProjects/Faces_Artifact_Classifier/data')
