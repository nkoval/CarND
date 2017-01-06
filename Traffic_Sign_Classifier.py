import pickle
import numpy as np
import csv
import math
from random import randint
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

training_file = 'train.p'
testing_file = 'test.p'
labels = []

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)
with open('signnames.csv', mode='r') as f:
    next(f)  # skip the first line
    reader = csv.reader(f)
    for row in reader:
        labels.append(row[1])

X_train, y_train = train['features'], train['labels']
X_test, y_test = test['features'], test['labels']


n_train = len(X_train)

n_test = len(X_test)

image_shape = X_train[0].shape

n_classes = y_train.max()

print("Number of training examples =", n_train)
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)


def plot_images(train_data, train_labels, labels, plots_number=6):
    # Plots random images from the dataset
    # Create figure with sub-plots.

    y_axes = 3
    x_axes = math.ceil(plots_number / 3.0)

    fig, axes = plt.subplots(x_axes, y_axes, figsize=(10, 10))

    # Adjust vertical spacing.
    fig.subplots_adjust(hspace=0.1, wspace=0.1)

    y = 0
    x = 0
    for i in range(0, plots_number):

        ax = axes[x][y]
        rnd_index = randint(0, len(train_data))
        ax.imshow(train_data[rnd_index] / 255.0, interpolation='sinc', cmap='gray')
        try:
            ax.set_xlabel(labels[train_labels[rnd_index]])
        except:
            print('Sign Without Label')
        y += 1
        if y == 3:
            y = 0
            x += 1

    # Remove ticks from all the plots.
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])

    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()


def process_images(images):
    processed = rgb_to_grayscale(images)
    processed = [normalize(image) for image in processed]
    return np.asarray(processed)


def rgb_to_grayscale(images_array):
    images_array = np.dot(images_array[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)
    return np.expand_dims(images_array, axis=4)


def normalize(image_data, max=255, deviation=0.5):
    return image_data / max - deviation


X_train = process_images(X_train)
X_test = process_images(X_test)


# Get validation set from the train set
from sklearn.model_selection import train_test_split

X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=0.2)

print("Number of validating examples =", len(X_validation))


color_channels = 1
n_classes = 42
# Params

EPOCHS = 100
BATCH_SIZE = 128
rate = 0.001
layer_width = {
    'layer_1': 32,
    'layer_2': 64,
    'layer_3': 128,
    'fully_connected_1': 512
}

mu = 0
sigma = 0.1


def add_conv_layer(input, filter_size, layer_width):
    filter = tf.Variable(tf.truncated_normal(filter_size, mean=mu, stddev=sigma))
    bias = tf.Variable(tf.zeros(layer_width))
    conv = tf.nn.conv2d(input, filter, [1, 1, 1, 1], "VALID") + bias
    conv = tf.nn.relu(conv)
    conv = tf.nn.max_pool(conv, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    return conv


def model(x):
    conv_1 = add_conv_layer(x, (5, 5, color_channels, layer_width['layer_1']), layer_width['layer_1'])
    conv_2 = add_conv_layer(conv_1, (5, 5, layer_width['layer_1'], layer_width['layer_2']), layer_width['layer_2'])
    conv_3 = add_conv_layer(conv_2, (5, 5, layer_width['layer_2'], layer_width['layer_3']), layer_width['layer_3'])

    flatten = tf.contrib.layers.flatten(conv_3)

    fc1_W = tf.Variable(
        tf.truncated_normal((layer_width['layer_3'], layer_width['fully_connected_1']), mean=mu, stddev=sigma))
    fc1_b = tf.Variable(tf.zeros(layer_width['fully_connected_1']))

    fc_1 = tf.matmul(flatten, fc1_W) + fc1_b
    fc_1 = tf.nn.relu(fc_1)

    logits_W = tf.Variable(tf.truncated_normal((layer_width['fully_connected_1'], n_classes), mean=mu, stddev=sigma))
    logits_b = tf.Variable(tf.zeros(n_classes))
    logits = tf.matmul(fc_1, logits_W) + logits_b

    return logits


import tensorflow as tf

x = tf.placeholder(tf.float32, (None, 32, 32, color_channels))
y = tf.placeholder(tf.int32, (None))
one_hot_y = tf.one_hot(y, n_classes)

logits = model(x)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=one_hot_y))
optimizer = tf.train.AdamOptimizer(learning_rate=rate).minimize(cost)

correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset + BATCH_SIZE], y_data[offset:offset + BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples


# In[ ]:

saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    num_examples = len(X_train)

    print("Training...")
    print()
    for i in range(EPOCHS):
        X_train, y_train = shuffle(X_train, y_train)
        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = X_train[offset:end], y_train[offset:end]
            sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})

        validation_accuracy = evaluate(X_validation, y_validation)
        print("EPOCH {} ...".format(i + 1))
        print("Validation Accuracy = {:.3f}".format(validation_accuracy))
        print()

    print("Test Accuracy = {:.3f}".format(evaluate(X_test, y_test)))

    save_path = saver.save(sess, 'trained-model.ckpt')

