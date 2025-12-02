import keras.models as KM
import keras.layers as KL
import tensorflow as tf
import numpy as np
import os
from matplotlib import pyplot as plt
from PIL import Image

PATH_TO_IMAGES = "./preprocessed_imgs/"

def get_num_files(a_dir):
    count = 0

    for path in os.listdir(a_dir):
      if os.path.isfile(os.path.join(a_dir, path)):
        count += 1
    return count-1

def LoadData (path1):
    """
    Looks for relevant filenames in the shared path
    Returns 2 lists for original and masked files respectively
    
    """
    # Read the images folder like a list
    len = get_num_files(path1)
    # Make a list for images and masks filenames
    img = []
    for i in range(len):
        img.append(path1 + str(i) + '.jpg')

    return img

def PreprocessData(img, target_shape_img, path1):

    # Pull the relevant dimensions for image and mask
    m = len(img)                     # number of images
    i_h,i_w = target_shape_img   # pull height, width, and channels of image
    
    # Define X and Y as number of images along with shape of one image
    X = np.zeros((m,i_h,i_w), dtype=np.float32)
    
    # Resize images and masks
    for file in img:
        # convert image into an array of desired shape (3 channels)
        # index = img.index(file)
        # path = os.path.join(path1, file)
        single_img = Image.open(file).convert('L')
        single_img = np.asarray(single_img)
        single_img = single_img/255
        X[img.index(file)] = single_img
        
    return X

img_list = LoadData(PATH_TO_IMAGES)
 
X = PreprocessData(img_list, [200,2000], PATH_TO_IMAGES)

Y = np.loadtxt("latLong.csv",
                 delimiter=",", dtype=str)
Y=Y.astype(float)

test_split = 0.1
split_index = int(np.floor(X.shape[0]*test_split))
X_test = X[0:split_index]
# X_train = X[split_index:]
Y_test = Y[0:split_index]
# Y_train = Y[split_index:]

# X_train = X_train[:, :, :, np.newaxis] 
X_test = X_test[:, :, :, np.newaxis]
print(X_test.shape)

def create_model():
    inputs = KL.Input(shape=(200, 2000, 1))
    c = KL.Conv2D(32, (3, 3), padding="same", activation=tf.nn.relu)(inputs)
    m = KL.MaxPool2D((2, 2), (2, 2))(c)
    d = KL.Dropout(0.5)(m)
    c = KL.Conv2D(64, (3, 3), padding="same", activation=tf.nn.relu)(d)
    m = KL.MaxPool2D((2, 2), (2, 2))(c)
    d = KL.Dropout(0.5)(m)
    c = KL.Conv2D(64, (3, 3), padding="same", activation=tf.nn.relu)(d)
    m = KL.MaxPool2D((2, 2), (2, 2))(c)
    d = KL.Dropout(0.5)(m)
    c = KL.Conv2D(64, (3, 3), padding="same", activation=tf.nn.relu)(d)
    m = KL.MaxPool2D((5, 5), (5, 5))(c)
    d = KL.Dropout(0.5)(m)
    c = KL.Conv2D(128, (3, 3), padding="same", activation=tf.nn.relu)(d)
    f = KL.Flatten()(c)
    d = KL.Dense(100, activation=tf.nn.relu)(f)
    outputs = KL.Dense(2, activation=tf.keras.activations.linear)(d)
    model = KM.Model(inputs, outputs)
    model.summary()
    return model


# Create a new model instance
model = create_model()

model.compile(optimizer="adam",
                loss="mean_squared_error",
                metrics=["accuracy"])

# Load the previously saved weights
checkpoint_path = "checkpoints/training_1/cp-{epoch:05d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# latest = tf.train.latest_checkpoint(checkpoint_dir)
# model.load_weights(latest)
model.load_weights("checkpoints/training_1/cp-00030.ckpt")

# Re-evaluate the model
# loss, acc = model.evaluate(X_test, Y_test, verbose=2)

latlong = model.predict(tf.expand_dims(X_test[0], 0))

''' test_accs = []
test_losses = []

for i in range(19):
    checkpoint_path = "checkpoints/training_1/cp-{epoch:05d}.ckpt".format(epoch=i*5+5) #epoch:05d pads the numbers
    model.load_weights(checkpoint_path)

    test_loss, test_acc = model.evaluate(X_test, Y_test)
    test_losses.append(test_loss)
    test_accs.append(test_acc)
    # print("Model loss and accuracy at {} epochs: {}, {}".format(i*5+5, test_loss, test_acc)) '''
# print("Restored model, accuracy: {:5.2f}%".format(100 * acc))