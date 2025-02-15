

import tensorflow as tf
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import DenseNet169

import pandas as pd   


train_path = 'dataset/train'
test_path = 'dataset/test'
valid_path = 'dataset/val'





batch_size = 16
img_height = 224
img_width = 224





from tensorflow.keras.preprocessing.image import ImageDataGenerator
# Create ImageDataGenerator for data augmentation on the training set
image_gen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True
)

# Create ImageDataGenerator for normalization on the test and validation sets
test_data_gen = ImageDataGenerator(rescale=1./255)





# Load and preprocess the training set
train = image_gen.flow_from_directory(
    train_path,
    target_size=(img_height, img_width),
    color_mode='rgb',  # DenseNet requires 3 channels (RGB)
    class_mode='binary',
    batch_size=batch_size
)

# Load and preprocess the test set
test = test_data_gen.flow_from_directory(
    test_path,
    target_size=(img_height, img_width),
    color_mode='rgb',
    shuffle=False,
    class_mode='binary',
    batch_size=batch_size
)

# Load and preprocess the validation set
valid = test_data_gen.flow_from_directory(
    valid_path,
    target_size=(img_height, img_width),
    color_mode='rgb',
    class_mode='binary',
    batch_size=batch_size
)





base_model = DenseNet169(
    weights='imagenet',
    include_top=False,
    input_shape=(img_height, img_width, 3)
)





x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
predictions = Dense(1, activation='sigmoid')(x)





model = Model(inputs=base_model.input, outputs=predictions)





for layer in base_model.layers:
    layer.trainable = False





model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])








model.fit(train, epochs=5, validation_data=valid)

model.save("model_densenet.h5")

pd.DataFrame(model.history.history).plot()


test_accu = model.evaluate(test)
print('The testing accuracy is :',test_accu[1]*100, '%')










