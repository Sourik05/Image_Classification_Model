import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Import from the preprocessing script
from preprocessing import create_data_generators

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Constants
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 30
DATASET_PATH = "path_to_intel_image_dataset"  # Update with actual path
MODEL_PATH = "models/intel_classifier.h5"

def build_model(num_classes):
    """Build a transfer learning model using MobileNetV2."""
    # Load the pre-trained model without the classification layer
    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )
    
    # Freeze the base model layers
    for layer in base_model.layers:
        layer.trainable = False
    
    # Add custom classification layers
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.3)(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    
    # Create the full model
    model = Model(inputs=base_model.input, outputs=predictions)
    
    # Compile the model
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model, base_model

def train_model(model, train_generator, validation_generator, epochs=EPOCHS):
    """Train the model with callbacks for early stopping and learning rate scheduling."""
    # Define callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=3,
            min_lr=1e-6
        ),
        ModelCheckpoint(
            MODEL_PATH,
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]
    
    # Train the model
    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=validation_generator,
        callbacks=callbacks
    )
    
    return history

def fine_tune_model(model, base_model, train_generator, validation_generator, epochs=10):
    """Fine-tune the model by unfreezing some of the base model layers."""
    # Unfreeze the last few layers of the base model
    for layer in base_model.layers[-30:]:
        layer.trainable = True
    
    # Recompile the model with a lower learning rate
    model.compile(
        optimizer=Adam(learning_rate=1e-5),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Define callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=3,
            min_lr=1e-7
        ),
        ModelCheckpoint(
            MODEL_PATH.replace('.h5', '_fine_tuned.h5'),
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]
    
    # Fine-tune the model
    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=validation_generator,
        callbacks=callbacks
    )
    
    return history

def plot_training_history(history, fine_tune_history=None):
    """Plot the training and validation accuracy/loss."""
    plt.figure(figsize=(12, 4))
    
    # Accuracy plot
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    
    if fine_tune_history:
        # Adjust epochs for fine-tuning plots
        ft_epochs = range(len(history.history['accuracy']), 
                         len(history.history['accuracy']) + len(fine_tune_history.history['accuracy']))
        plt.plot(ft_epochs, fine_tune_history.history['accuracy'])
        plt.plot(ft_epochs, fine_tune_history.history['val_accuracy'])
        plt.axvline(x=len(history.history['accuracy'])-1, color='r', linestyle='--')
    
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation', 'Train (fine-tuned)', 'Validation (fine-tuned)'] if fine_tune_history else ['Train', 'Validation'], loc='lower right')
    
    # Loss plot
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    
    if fine_tune_history:
        plt.plot(ft_epochs, fine_tune_history.history['loss'])
        plt.plot(ft_epochs, fine_tune_history.history['val_loss'])
        plt.axvline(x=len(history.history['loss'])-1, color='r', linestyle='--')
    
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation', 'Train (fine-tuned)', 'Validation (fine-tuned)'] if fine_tune_history else ['Train', 'Validation'], loc='upper right')
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()

def evaluate_model(model, test_generator, class_names):
    """Evaluate the model and generate metrics."""
    # Get predictions
    test_generator.reset()
    y_pred_probs = model.predict(test_generator)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    # Get true labels
    y_true = test_generator.classes
    
    # Calculate accuracy
    test_accuracy = np.sum(y_pred == y_true) / len(y_true)
    print(f"Test accuracy: {test_accuracy:.4f}")
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.show()
    
    return test_accuracy, y_pred, y_true

def save_model_for_deployment(model, model_path='models/intel_classifier_deployment'):
    """Save the model in TensorFlow SavedModel format for deployment."""
    if not os.path.exists('models'):
        os.makedirs('models')
    
    # Save in SavedModel format
    model.save(model_path)
    print(f"Model saved for deployment at: {model_path}")
    
    # Save class mapping information
    class_indices = train_generator.class_indices
    class_names = list(class_indices.keys())
    np.save(model_path + '/class_names.npy', class_names)
    
    return model_path

if __name__ == "__main__":
    # Create data generators
    train_generator, validation_generator, test_generator = create_data_generators()
    
    # Get number of classes
    num_classes = len(train_generator.class_indices)
    class_names = list(train_generator.class_indices.keys())
    print(f"Training with {num_classes} classes: {class_names}")
    
    # Build the model
    model, base_model = build_model(num_classes)
    print("Model built successfully!")
    
    # Train the model
    print("Starting initial training phase...")
    history = train_model(model, train_generator, validation_generator)
    
    # Fine-tune the model
    print("Starting fine-tuning phase...")
    fine_tune_history = fine_tune_model(model, base_model, train_generator, validation_generator)
    
    # Plot training history
    plot_training_history(history, fine_tune_history)
    
    # Evaluate the model
    print("Evaluating model performance...")
    test_accuracy, y_pred, y_true = evaluate_model(model, test_generator, class_names)
    
    # Save the model for deployment
    model_path = save_model_for_deployment(model)
    
    print("Model training and evaluation completed successfully!")