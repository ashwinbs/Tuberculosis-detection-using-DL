# import splitfolders
import matplotlib.pyplot as plt #For Visualization
import numpy as np              #For handling arrays
import pandas as pd             # For handling data
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Conv2D,Flatten,MaxPooling2D
from tensorflow.keras.callbacks import EarlyStopping,ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import os





train_path = 'dataset-TB/dataset/train'
test_path = 'dataset-TB/dataset/test'
valid_path = 'dataset-TB/dataset/val'




#The batch refers to the number of training examples utilized in one #iteration
batch_size = 16 
#The dimension of the images we are going to define is 500x500 img_height = 500
img_height = 500
img_width = 500






image_gen = ImageDataGenerator(
                                rescale = 1./255,
                                #shear_range = 0,
                                #zoom_range = 0,
                                horizontal_flip = True,          
                               )
# Create Image Data Generator for Test/Validation Set
test_data_gen = ImageDataGenerator(rescale = 1./255)


train = image_gen.flow_from_directory(
      train_path,
      target_size=(img_height, img_width),
      color_mode='grayscale',
      class_mode='binary',
      batch_size=batch_size
      )

test = test_data_gen.flow_from_directory(
      test_path,
      target_size=(img_height, img_width),
      color_mode='grayscale',
      shuffle=True, 
      class_mode='binary',
      batch_size=batch_size
      )

valid = test_data_gen.flow_from_directory(
      valid_path,
      target_size=(img_height, img_width),
      color_mode='grayscale',
      class_mode='binary', 
      batch_size=batch_size
      )





cnn = Sequential()
cnn.add(Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 1)))
cnn.add(MaxPooling2D((2, 2)))
cnn.add(Conv2D(64, (3, 3), activation='relu'))
cnn.add(MaxPooling2D((2, 2)))
cnn.add(Conv2D(128, (3, 3), activation='relu'))
cnn.add(MaxPooling2D((2, 2)))
cnn.add(Flatten())
cnn.add(Dense(128, activation='relu'))
cnn.add(Dense(1, activation='sigmoid'))

early = EarlyStopping(monitor= "val_loss", mode= "min", patience= 3)
learning_rate_reduction = ReduceLROnPlateau(monitor='val_loss', patience = 2, verbose=1,factor=0.3, min_lr=0.000001)
callbacks_list = [ early, learning_rate_reduction]

# Assuming 'train.classes' is the target class values
class_values = train.classes

# Convert class_values to a pandas Series
class_series = pd.Series(class_values)

# Calculate class frequencies
class_counts = class_series.value_counts()

# Calculate class weights
total_samples = len(class_values)
class_weights = total_samples / (len(class_counts) * class_counts)

cw = class_weights.to_dict()

cnn.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
cnn.fit(train, epochs=5, validation_data=valid, class_weight=cw, callbacks=callbacks_list)


# cnn.save("C:\\Users\\Dell\\Desktop\\TBclassfication\\TB\\model\\model.h5")
cnn.save("model_cnn_new.h5")

pd.DataFrame(cnn.history.history).plot()


test_accu = cnn.evaluate(test)
print('The testing accuracy is :',test_accu[1]*100, '%')




