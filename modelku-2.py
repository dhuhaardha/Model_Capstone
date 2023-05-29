import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Flatten, Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

IMAGE_SIZE = 224
NUM_CLASSES = 6
BATCH_SIZE = 32 #32                             , 22                                     , 22                                     , 22
# BATCH_SIZE2 = 19 #26 Loss: 0.60 Accuracy: 68.75%, 16 Loss: 0.82 Accuracy: 81.25% epoch:50, 16 Loss: 1.01 Accuracy: 81.25% epoch:60, 16 Loss: 6.32 Accuracy: 62.50% epoch:50
EPOCHS = 50

file_dir = 'data'
train_dir = os.path.join(file_dir, 'train')
valid_dir = os.path.join(file_dir, 'validation')

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=60, #20, 30, 40, from 50 to 60
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
)

test_datagen = ImageDataGenerator(rescale=1./255)


train_generator = train_datagen.flow_from_directory(
    'data/train',
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
)

validation_generator = test_datagen.flow_from_directory(
    'data/validation',
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
)

base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3))

for layer in base_model.layers:
    layer.trainable = False
    
    
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x) #plus flatten, hapus FLatten
x = Dropout(0.5)(x)
predictions = Dense(NUM_CLASSES, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)


model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])


# create call back
class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if (logs.get('accuracy')>0.97):
      print("\n sudah mencapai 97% accuracy so cancelling training!")
      self.model.stop_training = True

callbacks = myCallback()


hist = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=validation_generator,
    callbacks = callbacks
)

class_names = ["Abrasions", "Bruises", "Burns", "Laceration", "Stab_wound"]

# Evaluate the model
evaluation = model.evaluate(validation_generator)
print("Loss: {:.2f}".format(evaluation[0]))
print("Accuracy: {:.2f}%".format(evaluation[1] * 100))

# Test the model with new image data
test_image_path = 'laseration (16).jpg'
test_image = tf.keras.utils.load_img(test_image_path, target_size=(IMAGE_SIZE, IMAGE_SIZE))
test_image = tf.keras.utils.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
test_image = test_image / 255.0  # Normalize pixel values

predictions = model.predict(test_image)
class_index = np.argmax(predictions[0])
class_label = class_names[class_index]
print("Predicted class:", class_label)

# Plot accuracy
plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

# Plot loss
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['train', 'val'], loc='upper right')
plt.show()

from tensorflow.keras.models import save_model
save_model(model, "modelku-8.h5")



