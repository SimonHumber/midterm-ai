import numpy as np  # linear algebra
import struct
from array import array
from os.path import join
import random
import matplotlib.pyplot as plt
import tensorflow as tf


#
# MNIST Data Loader Class
#
class MnistDataloader(object):
    def __init__(
        self,
        training_images_filepath,
        training_labels_filepath,
        test_images_filepath,
        test_labels_filepath,
    ):
        self.training_images_filepath = training_images_filepath
        self.training_labels_filepath = training_labels_filepath
        self.test_images_filepath = test_images_filepath
        self.test_labels_filepath = test_labels_filepath

    def read_images_labels(self, images_filepath, labels_filepath):
        labels = []
        with open(labels_filepath, "rb") as file:
            magic, size = struct.unpack(">II", file.read(8))
            if magic != 2049:
                raise ValueError(
                    "Magic number mismatch, expected 2049, got {}".format(magic)
                )
            labels = array("B", file.read())

        with open(images_filepath, "rb") as file:
            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
            if magic != 2051:
                raise ValueError(
                    "Magic number mismatch, expected 2051, got {}".format(magic)
                )
            image_data = array("B", file.read())
        images = []
        for i in range(size):
            images.append([0] * rows * cols)
        for i in range(size):
            img = np.array(image_data[i * rows * cols : (i + 1) * rows * cols])
            img = img.reshape(28, 28)
            images[i][:] = img

        return images, labels

    def load_data(self):
        x_train, y_train = self.read_images_labels(
            self.training_images_filepath, self.training_labels_filepath
        )
        x_test, y_test = self.read_images_labels(
            self.test_images_filepath, self.test_labels_filepath
        )
        return (x_train, y_train), (x_test, y_test)


#
# Helper function to show a list of images with their relating titles
#
def show_images(images, title_texts):
    cols = 5
    rows = int(len(images) / cols) + 1
    plt.figure(figsize=(30, 20))
    plt.subplots_adjust(wspace=1.5, hspace=0.6)
    index = 1
    for x in zip(images, title_texts):
        image = x[0]
        title_text = x[1]
        plt.subplot(rows, cols, index)
        plt.imshow(image, cmap=plt.cm.gray)
        if title_text != "":
            plt.title(title_text, fontsize=15)
        index += 1
    plt.show()


def train_mnist_cnn(x_train, y_train, x_test, y_test, epochs=5, batch_size=64):
    # Normalize images
    x_train = np.array(x_train).astype("float32") / 255.0
    x_test = np.array(x_test).astype("float32") / 255.0

    # Add channel dimension for grayscale images
    if len(x_train.shape) == 3:
        x_train = np.expand_dims(x_train, -1)  # shape: (N, 28, 28, 1)
        x_test = np.expand_dims(x_test, -1)

    y_train = np.array(y_train)
    y_test = np.array(y_test)

    # Define CNN model
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Conv2D(
                32, (3, 3), activation="relu", input_shape=(28, 28, 1)
            ),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(10, activation="softmax"),
        ]
    )

    model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )

    print(model.summary())

    # Train model
    model.fit(
        x_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.1,
        verbose=2,
    )

    # Evaluate model
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
    print(f"\nTest accuracy: {test_acc:.4f}")

    return model


if __name__ == "__main__":
    #
    # Set file paths based on added MNIST Datasets
    #
    input_path = "/Users/simontran/Humber/Semester5/AI/Midterm/mnist-dataset"
    training_images_filepath = join(
        input_path, "train-images-idx3-ubyte/train-images-idx3-ubyte"
    )
    training_labels_filepath = join(
        input_path, "train-labels-idx1-ubyte/train-labels-idx1-ubyte"
    )
    test_images_filepath = join(
        input_path, "t10k-images-idx3-ubyte/t10k-images-idx3-ubyte"
    )
    test_labels_filepath = join(
        input_path, "t10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte"
    )

    # Load dataset
    mnist_dataloader = MnistDataloader(
        training_images_filepath,
        training_labels_filepath,
        test_images_filepath,
        test_labels_filepath,
    )
    (x_train, y_train), (x_test, y_test) = mnist_dataloader.load_data()

    # Optional: show some random images
    images_2_show = []
    titles_2_show = []
    for i in range(0, 10):
        r = random.randint(1, 60000)
        images_2_show.append(x_train[r])
        titles_2_show.append("training image [" + str(r) + "] = " + str(y_train[r]))

    for i in range(0, 5):
        r = random.randint(1, 10000)
        images_2_show.append(x_test[r])
        titles_2_show.append("test image [" + str(r) + "] = " + str(y_test[r]))

    # Uncomment to show images
    # show_images(images_2_show, titles_2_show)

    # Train CNN model and print results
    model = train_mnist_cnn(x_train, y_train, x_test, y_test, epochs=5, batch_size=64)
