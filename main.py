import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

def scale(image, label):
    image = tf.cast(image, tf.float32)
    image /= 255.
    return image, label

mnist_dataset, mnist_info = tfds.load(name='mnist', with_info=True, as_supervised=True)
mnist_train, mnist_test = mnist_dataset['train'], mnist_dataset['test']
number_of_validation_samples = 0.1 * mnist_info.splits['train'].num_examples
number_of_validation_samples = tf.cast(number_of_validation_samples, tf.int64)
number_of_test_samples = mnist_info.splits['test'].num_examples
number_of_test_samples = tf.cast(number_of_test_samples, tf.int64)

scaled_train_and_validation_data = mnist_train.map(scale)
scaled_test_data = mnist_test.map(scale)

BUFFER_SIZE = 10000
shuffled_train_and_validation_data = scaled_train_and_validation_data.shuffle(BUFFER_SIZE)
validation_data = shuffled_train_and_validation_data.take(number_of_validation_samples)
train_data = shuffled_train_and_validation_data.skip(number_of_validation_samples)

BATCH_SIZE = 100
train_data = train_data.batch(BATCH_SIZE)
validation_data = validation_data.batch(number_of_validation_samples)
test_data = scaled_test_data.batch(number_of_test_samples)