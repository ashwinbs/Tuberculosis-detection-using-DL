

from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam






train_path = 'dataset/train'
test_path = 'dataset/test'
valid_path = 'dataset/val'


batch_size = 16
img_height = 299
img_width = 299





train_data_gen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    # Add more augmentation options if needed
)

test_data_gen = ImageDataGenerator(rescale=1./255)





train = train_data_gen.flow_from_directory(
    train_path,
    target_size=(img_height, img_width),
    color_mode='rgb',
    class_mode='binary',
    batch_size=batch_size
)





test = test_data_gen.flow_from_directory(
    test_path,
    target_size=(img_height, img_width),
    color_mode='rgb',
    shuffle=False,
    class_mode='binary',
    batch_size=batch_size
)





valid = test_data_gen.flow_from_directory(
    valid_path,
    target_size=(img_height, img_width),
    color_mode='rgb',
    class_mode='binary',
    batch_size=batch_size
)





base_model = InceptionV3(
    weights='imagenet',
    include_top=False,
    input_shape=(img_height, img_width, 3)  # Ensure input_shape matches the image dimensions and channels
)

model = Sequential()
model.add(base_model)
model.add(GlobalAveragePooling2D())
model.add(Dense(128, activation='relu'))
model.add(Dense(1, activation='sigmoid'))





base_model.trainable = False





model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])





model.fit(train, epochs=5, validation_data=valid)



model.save("model_InceptionV3.h5")
import pandas as pd

pd.DataFrame(model.history.history).plot()


test_accu = model.evaluate(test)
print('The testing accuracy is :',test_accu[1]*100, '%')



