import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import kagglehub  # Import KaggleHub for dataset download
import shutil
# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Define constants
IMG_SIZE = 224  # Standard input size for many pre-trained models
BATCH_SIZE = 32

# Define global dataset path
DATASET_PATH = "E:\Soul_Ai\image_classification_project\intel_image_dataset"

def download_dataset():
    """Download Intel Image Classification dataset using KaggleHub and move it to project directory."""
    global DATASET_PATH  # Ensure it's accessible throughout the script
    print("📥 Downloading Intel Image Classification dataset...")

    try:
        downloaded_path = kagglehub.dataset_download("puneet6060/intel-image-classification")
        print(f"✅ Dataset downloaded at: {downloaded_path}")

        # Move dataset to project folder if needed
        if not os.path.exists(DATASET_PATH):
            shutil.move(downloaded_path, DATASET_PATH)
        else:
            print("✅ Dataset already exists in project folder, skipping move.")

        print(f"📂 Dataset moved to: {DATASET_PATH}")

    except Exception as e:
        print(f"❌ Error downloading dataset: {str(e)}")
        print("Make sure `kagglehub` is installed: pip install kagglehub")


def load_and_explore_data(dataset_path):
    """Load and perform exploratory data analysis on the dataset."""
    train_dir = os.path.join(dataset_path, "seg_train", "seg_train")

    if not os.path.exists(train_dir):
        print(f"❌ Error: Training data not found at {train_dir}. Please check dataset path.")
        return [], {}

    class_names = os.listdir(train_dir)

    # Count images per class
    class_counts = {class_name: len(os.listdir(os.path.join(train_dir, class_name))) for class_name in class_names}

    # Plot class distribution
    plt.figure(figsize=(10, 6))
    plt.bar(class_counts.keys(), class_counts.values(), color="skyblue")
    plt.title('Class Distribution in Training Set')
    plt.xlabel('Class')
    plt.ylabel('Number of Images')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    return class_names, class_counts

def create_data_generators():
    """Create train, validation, and test data generators with augmentation."""
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=0.2  # 20% for validation
    )

    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        os.path.join(DATASET_PATH, "seg_train", "seg_train"),
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training'
    )

    validation_generator = train_datagen.flow_from_directory(
        os.path.join(DATASET_PATH, "seg_train", "seg_train"),
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation'
    )

    test_generator = test_datagen.flow_from_directory(
        os.path.join(DATASET_PATH, "seg_test", "seg_test"),
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )

    return train_generator, validation_generator, test_generator

def visualize_augmentations(train_generator):
    """Visualize the effect of data augmentation on sample images."""
    x_batch, y_batch = next(train_generator)

    plt.figure(figsize=(12, 8))
    for i in range(min(8, len(x_batch))):
        plt.subplot(2, 4, i+1)
        plt.imshow(x_batch[i])
        plt.axis('off')

    plt.suptitle('Sample Augmented Images')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Download the dataset
    download_dataset()
    
    # Explore the dataset
    class_names, class_counts = load_and_explore_data(DATASET_PATH)
    print(f"🗂 Classes: {class_names}")
    print(f"📊 Class distribution: {class_counts}")

    # Create data generators
    train_gen, val_gen, test_gen = create_data_generators()
    print(f"🔄 Training batches: {len(train_gen)}")
    print(f"✅ Validation batches: {len(val_gen)}")
    print(f"🧪 Test batches: {len(test_gen)}")

    # Visualize augmentations
    visualize_augmentations(train_gen)

    print("✅ Data preprocessing completed successfully!")
