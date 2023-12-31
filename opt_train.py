import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D, MaxPooling2D, Dropout, Flatten, Dense, Input
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
import os
import random
import numpy as np
import pickle

# Set TensorFlow log level
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Set mixed precision policy
dtype = "float32"
mixed_precision = False

if mixed_precision:
    dtype = "float16"
    tf.keras.mixed_precision.experimental.set_policy('mixed_float16')

class CModel(tf.Module):
    def __init__(self, i_width, i_height):
        super(CModel, self).__init__()
        self.width = i_width
        self.height = i_height
        self.opt = tf.keras.optimizers.Adam(learning_rate=1e-4)
        self.var_map = {}
        self.op_map = {}
        self.init_var()

    # Initialize model parameters
    def init_var(self):
        # Create the base layers of the model
        z_size = 3  # RGB
        num_of_types = 2  # 0: no mask, 1: mask

        baseModel = MobileNetV2(weights="imagenet", include_top=False, input_tensor=Input(shape=(self.width, self.height, z_size)))

        headModel = baseModel.output
        headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
        headModel = Flatten(name="flatten")(headModel)
        headModel = Dense(64, activation="relu")(headModel)
        headModel = Dropout(0.25)(headModel)
        headModel = Dense(32, activation="relu")(headModel)
        # headModel = Dropout(0.25)(headModel)
        # headModel = Dense(16, activation="relu")(headModel)
        # headModel = Dropout(0.25)(headModel)
        # headModel = Dense(16, activation="relu")(headModel)
        headModel = Dropout(0.5)(headModel)  # Dropout neurons to avoid overfitting
        headModel = Dense(num_of_types, activation="softmax")(headModel)

        model = Model(inputs=baseModel.input, outputs=headModel)

        for layer in baseModel.layers:
            layer.trainable = False

        self.var_map['model'] = model

    # Define the prediction function using TensorFlow function signature
    @tf.function(input_signature=[tf.TensorSpec([None, 224, 224, 3], tf.float32)])
    def predict(self, i_x):
        # Prediction process
        self.op_map['tf_input_reshape'] = i_x
        self.op_map['prediction'] = self.var_map['model'](self.op_map['tf_input_reshape'])
        return self.op_map['prediction']

    # Compute the loss function
    def get_loss(self, y, prediction):
        return tf.reduce_mean(-tf.reduce_sum(tf.constant(y, dtype=tf.float32) * tf.math.log(prediction), axis=[1]))

    # Test the model
    def test(self, testY, testX):
        _testX = TF_Process.to_tensor(testX)
        prediction = self.predict(_testX)
        prediction = tf.math.argmax(prediction, 1)
        z = np.argmax(testY, axis=1) - prediction.numpy()
        _succ = np.sum(z == 0)
        total = len(testY)
        return _succ / total

    # Train the model
    def train(self, y, x):
        with tf.GradientTape() as tape:
            x = TF_Process.to_tensor(x)
            prediction = self.predict(x)
            L = self.get_loss(y, prediction)
            g = tape.gradient(L, self.var_map['model'].trainable_variables)
        result = zip(g, self.var_map['model'].trainable_variables)
        self.opt.apply_gradients(result)
        return L

class TF_Process():
    @staticmethod
    def Image_to_Array(pic, width=224, height=224):
        image = load_img(pic, target_size=(width, height))
        image = img_to_array(image)
        image = preprocess_input(image)
        return image

    @staticmethod
    def to_tensor(i_x):
        channel = 3
        x = tf.constant(i_x, shape=[np.shape(i_x)[0], np.shape(i_x)[1], np.shape(i_x)[2], channel], dtype=tf.float32)
        return x

class Data_Process():
    @staticmethod
    def load_pictures(picdir, width_size, height_size):
        x_data = []
        y_labels = []

        for dirPath, _, fileNames in os.walk(picdir):
            for f in fileNames:
                ans = int(dirPath[len(picdir):])
                PicDir = os.path.join(dirPath, f)
                NArray_2D = TF_Process.Image_to_Array(PicDir, width_size, height_size)

                y_labels.append(ans)
                x_data.append(NArray_2D)

        return y_labels, x_data

    @staticmethod
    def clean_data(i_y, i_x):
        x_data = np.array(i_x, dtype="float32")
        y_labels = np.array(i_y)
        lb = LabelBinarizer()
        y_labels3 = to_categorical(y_labels, 2)

        return y_labels3, x_data

    @staticmethod
    def save_sample(i_y, i_x, i_filename):
        with open(i_filename, 'wb') as f:
            pickle.dump([i_y, i_x], f)

    @staticmethod
    def load_sample(i_filename):
        with open(i_filename, 'rb') as f:
            r = pickle.load(f)
        y_labels = r[0]
        x_data = r[1]
        return y_labels, x_data

    @staticmethod
    def get_sample(i_y, i_x, i_sample_size):
        if len(i_y) < i_sample_size:
            i_sample_size = len(i_y)

        x_batch = []
        y_batch = []

        sample = random.sample(range(len(i_y)), i_sample_size)
        x_batch = [i_x[i] for i in sample]
        y_batch = [i_y[i] for i in sample]

        return y_batch, x_batch

def main():
    # Read images and process data
    picdir = "./TrainData/"
    width_size = 224
    height_size = 224

    y_labels, x_data = Data_Process.load_pictures(picdir, width_size, height_size)
    Data_Process.save_sample(y_labels, x_data, 'pic_mask_db_224_224.p')
    y_labels, x_data = Data_Process.load_sample('pic_mask_db_224_224.p')

    y_c_labels, x_c_data = Data_Process.clean_data(y_labels, x_data)
    (trainX, testX, trainY, testY) = train_test_split(x_c_data, y_c_labels, test_size=0.1, stratify=y_c_labels, random_state=42)

    # Define training parameters
    Training_times = 10
    steps_in_one_training_time = 20

    # Create a model instance using an image data generator
    aug = ImageDataGenerator(
        rotation_range=20,
        zoom_range=0.15,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        horizontal_flip=True,
        fill_mode="nearest")

    # Create an instance of the CModel
    model = CModel(width_size, height_size)

    # Perform multiple training loops
    for e in range(Training_times):
        i = 0
        # Perform multiple steps within each training time
        for x_batch, y_batch in aug.flow(trainX, trainY, batch_size=32):
            i += 1
            if i >= steps_in_one_training_time:
                break

            # Train the model
            print('Epoch: ', e, " step ", i, flush=True, end='')
            model.train(y_batch, x_batch)

            # Test and display accuracy
            accuracy = model.test(testY, testX)
            print(", accuracy, {}".format(accuracy), flush=True)

            # Clear the session to release GPU memory
            tf.keras.backend.clear_session()

    print(" training finished ")

    # Display the final testing accuracy
    _sample_y, _sample_x = Data_Process.get_sample(testY, testX, 128)
    accuracy = model.test(_sample_y, _sample_x)
    print("final testing: accuracy {}".format(accuracy), flush=True)

    # Save the trained model
    tf.saved_model.save(model, "masked_module")

if __name__ == '__main__':
    main()
