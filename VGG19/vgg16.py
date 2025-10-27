import os
import numpy as np
import keras
from keras import layers, models, optimizers
from keras.utils import image_dataset_from_directory
import matplotlib.pyplot as plt

# Configuration
DATASET_PATH = r"C:\vscode\concave"
WEIGHTS_PATH = r"C:\vscode\deepLeviathon\vgg16_weights_tf_dim_ordering_tf_kernels.h5"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.0001

def create_vgg16_model(num_classes, weights_path=None):
    """
    Create VGG16 model architecture for transfer learning
    """
    # VGG16 architecture
    model = models.Sequential([
        # Input layer
        layers.Input(shape=(224, 224, 3)),
        # Block 1
        layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1'),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2'),
        layers.MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool'),
        
        # Block 2
        layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1'),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2'),
        layers.MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool'),
        
        # Block 3
        layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1'),
        layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2'),
        layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3'),
        layers.MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool'),
        
        # Block 4
        layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1'),
        layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2'),
        layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3'),
        layers.MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool'),
        
        # Block 5
        layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1'),
        layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2'),
        layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3'),
        layers.MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool'),
        
        # Flatten
        layers.Flatten(name='flatten'),
        
        # New classification head
        layers.Dense(256, activation='relu', name='fc1'),
        layers.Dropout(0.5, name='dropout1'),
        layers.Dense(128, activation='relu', name='fc2'),
        layers.Dropout(0.5, name='dropout2'),
        layers.Dense(num_classes, activation='softmax', name='predictions')
    ])
    
    # Load pre-trained weights if provided
    if weights_path and os.path.exists(weights_path):
        print(f"Loading pre-trained weights from {weights_path}")
        try:
            model.load_weights(weights_path, by_name=True, skip_mismatch=True)
            print("✓ Weights loaded successfully!")
        except Exception as e:
            print(f"Warning: Could not load weights - {e}")
            print("Proceeding with random initialization...")
    
    return model

def freeze_layers(model, num_layers_to_freeze):
    """
    Freeze the first N layers for transfer learning
    """
    for layer in model.layers[:num_layers_to_freeze]:
        layer.trainable = False
    
    print(f"\nFrozen first {num_layers_to_freeze} layers")
    trainable_count = sum([1 for layer in model.layers if layer.trainable])
    print(f"Trainable layers: {trainable_count}/{len(model.layers)}")

def load_dataset(dataset_path, img_size, batch_size):
    """
    Load training and validation datasets
    """
    print(f"\nLoading dataset from: {dataset_path}")
    
    # Load training data
    train_ds = image_dataset_from_directory(
        dataset_path,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=img_size,
        batch_size=batch_size,
        label_mode='categorical'
    )
    
    # Load validation data
    val_ds = image_dataset_from_directory(
        dataset_path,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=img_size,
        batch_size=batch_size,
        label_mode='categorical'
    )
    
    # Get class names
    class_names = train_ds.class_names
    print(f"Found {len(class_names)} classes: {class_names}")
    
    # Optimize performance
    AUTOTUNE = keras.config.backend()
    train_ds = train_ds.prefetch(buffer_size=32)
    val_ds = val_ds.prefetch(buffer_size=32)
    
    return train_ds, val_ds, class_names

def plot_training_history(history):
    """
    Plot training and validation accuracy/loss
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot accuracy
    ax1.plot(history.history['accuracy'], label='Training Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True)
    
    # Plot loss
    ax2.plot(history.history['loss'], label='Training Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=150, bbox_inches='tight')
    print("\n✓ Training history saved as 'training_history.png'")
    plt.show()

def main():
    print("=" * 60)
    print("VGG16 Transfer Learning with Keras 3")
    print("=" * 60)
    
    # Load dataset
    train_ds, val_ds, class_names = load_dataset(DATASET_PATH, IMG_SIZE, BATCH_SIZE)
    num_classes = len(class_names)
    
    # Create model
    print(f"\nBuilding VGG16 model for {num_classes} classes...")
    model = create_vgg16_model(num_classes, WEIGHTS_PATH)
    
    # Freeze convolutional layers (first 19 layers are conv + pool)
    freeze_layers(model, num_layers_to_freeze=19)
    
    # Compile model
    print("\nCompiling model...")
    model.compile(
        optimizer=optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Model summary
    print("\nModel Summary:")
    model.summary()
    
    # Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            'best_model.keras',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]
    
    # Train model
    print("\n" + "=" * 60)
    print("Starting Training...")
    print("=" * 60)
    
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate model
    print("\n" + "=" * 60)
    print("Final Evaluation")
    print("=" * 60)
    
    train_loss, train_acc = model.evaluate(train_ds, verbose=0)
    val_loss, val_acc = model.evaluate(val_ds, verbose=0)
    
    print(f"\nTraining Accuracy: {train_acc:.4f}")
    print(f"Validation Accuracy: {val_acc:.4f}")
    print(f"\nTraining Loss: {train_loss:.4f}")
    print(f"Validation Loss: {val_loss:.4f}")
    
    # Plot training history
    plot_training_history(history)
    
    # Save final model
    model.save('final_model.keras')
    print("\n✓ Final model saved as 'final_model.keras'")
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()