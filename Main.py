'''
Train a computer vision model to classify regions of phase microscopy images of
scratch wounds into one of three categories: non-wound, wound, or wound edge.
'''

###############################################################################
# Imports
from Functions import *

import numpy as np

from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential, Model
from keras.layers import Dropout, Flatten, Dense, Conv2D, BatchNormalization
from keras.layers import AveragePooling2D, MaxPooling2D
from keras.callbacks import ModelCheckpoint
from keras.utils import to_categorical


###############################################################################
# Parameters
## Directory containing the tif files.
directory = 'C:/Users/...'

'''
The number of frames in the tif between each interval in the outlines. For
example, if you had _0m followed by _30m outline files but had imaged every 5
minutes to make the tif file, then frames_per_interval would be 6.
'''
frames_per_interval = 6

'''
The number of tif files specified by files_per_chunk is loaded into memory at a
time, then trained for the number of iterations specified by epochs on batches
of the size specified by batch_size. This is repeated the number of times
specified by num_outer_epochs.
'''
files_per_chunk = 100
epochs = 10
batch_size = 128
num_outer_epochs = 10

'''
The data is randomly partitioned into training, validation, and testing data
sets based on the proportions specified below.
'''
train_proportion = 0.8
validation_proportion = 0.1
test_proportion = 0.1

# Set random seed.
np.random.seed(8)


###############################################################################
# Perform initial processing.
# Clear the Metadata and Cache folders.
clear_metadata_folder()
clear_cache_folder()

# Create data files from images.
create_training_data()

# Map files
FileMap = map_files(directory, frames_per_interval)
save_file_map(FileMap)


################################################################################
# Get metadata for the files in the Cache directory.

# Load an example input file.
example_FileID = (FileMap[FileMap.FileType == 'txt']).FileID.iloc[0]
example_X = load_X(example_FileID)
shape_X = example_X.shape
input_dim = shape_X[1:]

example_Y = load_Y(example_FileID)
example_Y = to_categorical(example_Y)

# Determine input dimensions
img_width, img_height = example_X.shape[1:]

# Get list of successfully generated X files.
CacheMap = map_Cache()


###############################################################################
# Specify the indices of the training, validation, and test sets.

file_numbers = np.array(list(set(CacheMap.FileNumber))).astype(int)
max_index = file_numbers.max()
indices = np.arange(max_index)
np.random.shuffle(indices)

train_max_index = int(train_proportion*max_index)
train_indices = indices[0:train_max_index]

validation_min_index = train_max_index + 1
validation_max_index = train_max_index + int(validation_proportion*max_index)
validation_indices = indices[validation_min_index:validation_max_index]

test_min_index = validation_max_index + 1
test_max_index = validation_max_index + int(test_proportion*max_index)
test_indices = indices[test_min_index:test_max_index]

# Store in FileMap information about which data set each file belongs to.
FileMap['train'] = False
FileMap['validation'] = False
FileMap['test'] = False

for i in train_indices:
    FileMap['train'].iloc[i] = True
for i in validation_indices:
    FileMap['validation'].iloc[i] = True
for i in test_indices:
    FileMap['test'].iloc[i] = True

save_file_map(FileMap)

###############################################################################
# Create the model.
model=Sequential()
model.add(AveragePooling2D(pool_size=(2),
                           strides=(1),
                           data_format='channels_last',
                           padding='valid',
                           name='pool1',
                           input_shape=(input_dim[0],input_dim[1],1)))
model.add(MaxPooling2D(pool_size=(2),
                       strides=(2),
                       data_format='channels_last',
                       padding='valid',
                       name='pool2',
                       input_shape=(input_dim[0],input_dim[1],1)))
model.add(BatchNormalization())
model.add(Conv2D(filters=1024,
                 kernel_size=(3),
                 strides=(3),
                 activation='relu',
                 data_format='channels_last',
                 name='conv2d_1',
                 kernel_initializer='random_normal',
                 use_bias=False))
model.add(BatchNormalization())
model.add(Conv2D(filters=512,
                 kernel_size=(2),
                 strides=(2),
                 activation='relu',
                 data_format='channels_last',
                 name='conv2d_2',
                 kernel_initializer='random_normal',
                 use_bias=False))
model.add(BatchNormalization())
model.add(Flatten(name='flatten'))
model.add(Dense(1024,
                activation='relu',
                name='dense_1',
                kernel_initializer='random_normal'))
model.add(BatchNormalization())
model.add(Dense(1024,
                activation='relu',
                name='dense_2',
                kernel_initializer='random_normal'))
model.add(BatchNormalization())
model.add(Dense(1024,
                activation='relu',
                name='dense_3',
                kernel_initializer='random_normal'))
model.add(BatchNormalization())
model.add(Dense(3,
                activation='softmax',
                name='softmax',
                kernel_initializer='random_normal'))

# Compile the model
model.compile(loss = "categorical_crossentropy",
              optimizer = optimizers.SGD(),
              metrics=["accuracy"])


###############################################################################
# Create data augmentation generators.
datagen = ImageDataGenerator(
horizontal_flip = True,
vertical_flip = True)


###############################################################################
# Train the model.

# Train on the files in random order.
np.random.shuffle(train_indices)

for outer_epoch in range(num_outer_epochs):
    files_loaded = 0
    for chunk in range(int(len(train_indices)/files_per_chunk)):
        # Determine which files should be loaded into this chunk.
        start = 0
        print('Outer epoch '+str(outer_epoch+1))
        print('Chunk number '+str(chunk+1))
        # Ensure the upper index < the number of training indices.
        upper_bound = start+files_per_chunk
        if upper_bound > len(train_indices):
            upper_bound = len(train_indices)
        train_indices_chunk = train_indices[start:upper_bound]
        start = start+files_per_chunk+1

        # Load the data.
        Xl = []
        Yl = []
        for tr_idx in train_indices_chunk:
            try:
                X = load_X(tr_idx)
                X = X/255
                X = X.reshape(X.shape[0],X.shape[1],X.shape[2],1)
                Y = load_Y(tr_idx)
                Xl.append(X)
                Yl.append(Y)
            except:
                pass
            files_loaded += 1
            if (files_loaded % 20) == 0:
                print(str(files_loaded)+' files loaded.')

        Xl=np.concatenate(Xl)
        Yl=np.concatenate(Yl)

        # Randomize indices so data from individual files is not contiguous.
        new_indices = np.random.choice(Yl.shape[0], Yl.shape[0], replace=False)
        Xl = Xl[new_indices]
        Yl = Yl[new_indices]

        # Specify the data augmentation generator.
        datagen.fit(X)

        # Specify how the model should be saved.
        checkpoint = ModelCheckpoint("Model/model.h5",
                                     monitor='val_acc',
                                     save_weights_only=False,
                                     mode='auto',
                                     period=1)

        # Fit and save the model.
        history = model.fit_generator(
                datagen.flow(Xl, to_categorical(Yl),
                             batch_size=batch_size),
                             verbose=1,
                             steps_per_epoch=int(Yl.shape[0]/batch_size)+1,
                             epochs=epochs,
                             callbacks=[checkpoint],
                             max_queue_size=4)




###############################################################################
# Evaluate the model on a single file.
scores = model.evaluate(example_X, example_Y)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))



probs = model.predict(example_X.reshape(example_X.shape[0],
                                        example_X.shape[1],
                                        example_X.shape[2],
                                        1))
predictions = probs > 0.5


###############################################################################
# To load the model in another session use the below code.

#from keras.models import load_model
#model = load_model('Model/model.h5')
