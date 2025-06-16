# =============================================================================
# STEP 1 - Import Libraries
# =============================================================================
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetV2B2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# =============================================================================
# STEP 2 - Path Setup
# =============================================================================
base_path = "Dataset"  # adjust if needed
train_path = os.path.join(base_path, 'train')
val_path = os.path.join(base_path, 'validation')
test_path = os.path.join(base_path, 'test')

# =============================================================================
# STEP 3 - Image Parameters and Augmentation
# =============================================================================
img_size = (260, 260)
batch_size = 16

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_path,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical'
)

val_generator = val_test_datagen.flow_from_directory(
    val_path,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical'
)

test_generator = val_test_datagen.flow_from_directory(
    test_path,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

num_classes = train_generator.num_classes

# =============================================================================
# STEP 4 - Model Creation (EfficientNetV2B2)
# =============================================================================
base_model = EfficientNetV2B2(
    include_top=False,
    input_shape=img_size + (3,),
    weights='imagenet'
)
base_model.trainable = False  # Freeze base

x = GlobalAveragePooling2D()(base_model.output)
x = Dropout(0.3)(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.3)(x)
outputs = Dense(num_classes, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=outputs)

# =============================================================================
# STEP 5 - Compile Model
# =============================================================================
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=3e-4),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# =============================================================================
# STEP 6 - Callbacks
# =============================================================================
callbacks = [
    EarlyStopping(monitor='val_accuracy', patience=7, restore_best_weights=True),
    ModelCheckpoint('best_currency_model.keras', save_best_only=True),
    ReduceLROnPlateau(monitor='val_accuracy', factor=0.3, patience=3, verbose=1)
]

# =============================================================================
# STEP 7 - Train the Model
# =============================================================================
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=50,
    callbacks=callbacks
)

# =============================================================================
# STEP 8 - Evaluate the Model
# =============================================================================
loss, acc = model.evaluate(test_generator)
print(f"Test accuracy: {acc:.4f}")

# =============================================================================
# STEP 9 - Save Final Model
# =============================================================================
model.save('final_currency_model.keras')
