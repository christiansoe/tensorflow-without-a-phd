# encoding: UTF-8
# Copyright 2016 Google.com
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import tensorflow as tf
import tensorflowvisu
import mnistdata
import math
print("Tensorflow version " + tf.__version__)
tf.set_random_seed(0)

# neural network with 1 layer of 10 softmax neurons
#
# · · · · · · · · · ·       (input data, flattened pixels)       X [batch, 784]        # 784 = 28 * 28
# \x/x\x/x\x/x\x/x\x/    -- fully connected layer (softmax)      W [784, 10]     b[10]
#   · · · · · · · ·                                              Y [batch, 10]

# The model is:
#
# Y = softmax( X * W + b)
#              X: matrix for 100 grayscale images of 28x28 pixels, flattened (there are 100 images in a mini-batch)
#              W: weight matrix with 784 lines and 10 columns
#              b: bias vector with 10 dimensions
#              +: add with broadcasting: adds the vector to each line of the matrix (numpy)
#              softmax(matrix) applies softmax on each line
#              softmax(line) applies an exp to each value then divides by the norm of the resulting line
#              Y: output matrix with 100 lines and 10 columns

# Download images and labels into mnist.test (10K images+labels) and mnist.train (60K images+labels)
mnist = mnistdata.read_data_sets("data", one_hot=True, reshape=False)

# weights W[784, 10]   784=28*28
W = tf.Variable(tf.zeros([784, 10]))
# biases b[10]
B = tf.Variable(tf.zeros([10]))

def model(X):
    # flatten the images into a single line of pixels
    # -1 in the shape definition means "the only possible dimension that will preserve the number of elements"
    XX = tf.reshape(X, [-1, 784])

    # The model
    Y = tf.nn.softmax(tf.matmul(XX, W) + B)
    return Y

# loss function: cross-entropy = - sum( Y_i * log(Yi) )
#                           Y: the computed output vector
#                           Y_: the desired output vector

def loss(Y_, Y):
    # cross-entropy
    # log takes the log of each element, * multiplies the tensors element by element
    # reduce_mean will add all the components in the tensor
    # so here we end up with the total cross-entropy for all images in the batch
    cross_entropy = -tf.reduce_mean(Y_ * tf.math.log(Y)) * 1000.0  # normalized for batches of 100 images,
                                                            # *10 because  "mean" included an unwanted division by 10
    return cross_entropy

# training, learning rate = 0.005
optimizer = tf.optimizers.Adam(0.005)

datavis = tensorflowvisu.MnistDataVis()

# You can call this function in a loop to train the model, 100 images at a time
def training_step_internal(i, update_test_data, update_train_data):

    # training on batches of 100 images with 100 labels
    batch_X, batch_Y = mnist.train.next_batch(100)
    predY = model(batch_X)
    cross_ent_loss = loss(batch_Y, predY)

    # compute training values for visualisation
    if update_train_data:
        # accuracy of the trained model, between 0 (worst) and 1 (best)
        correct_prediction = tf.equal(tf.argmax(predY, 1), tf.argmax(batch_Y, 1))
        a = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        c = cross_ent_loss
        im = tensorflowvisu.tf_format_mnist_images(batch_X, batch_Y, predY)
        w = tf.reshape(W, [-1])
        b = tf.reshape(B, [-1])
        
        datavis.append_training_curves_data(i, a, c)
        datavis.append_data_histograms(i, w, b)
        datavis.update_image1(im)
        print(str(i) + ": accuracy:" + str(a) + " loss: " + str(c))

    # compute test values for visualisation
    if update_test_data:
        predY = model(mnist.test.images)
        
        a = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        c = loss(mnist.test.labels, predY)
        im = tensorflowvisu.tf_format_mnist_images(mnist.test.images, mnist.test.labels, predY, 1000, lines=25)  # 1000 images on 25 lines
        datavis.append_test_curves_data(i, a, c)
        datavis.update_image2(im)
        print(str(i) + ": ********* epoch " + str(i*100//mnist.train.images.shape[0]+1) + " ********* test accuracy:" + str(a) + " test loss: " + str(c))

    return cross_ent_loss

# You can call this function in a loop to train the model, 100 images at a time
def training_step(i, update_test_data, update_train_data):
    with tf.GradientTape() as tape:
        current_loss = training_step_internal(i, update_test_data, update_train_data)
    grads = tape.gradient( current_loss , [W, B] )
    optimizer.apply_gradients( zip( grads , [W, B] ) )
    print( tf.reduce_mean( current_loss ) )


datavis.animate(training_step, iterations=2000+1, train_data_update_freq=10, test_data_update_freq=50, more_tests_at_start=True)

# to save the animation as a movie, add save_movie=True as an argument to datavis.animate
# to disable the visualisation use the following line instead of the datavis.animate line
# for i in range(2000+1): training_step(i, i % 50 == 0, i % 10 == 0)

print("max test accuracy: " + str(datavis.get_max_test_accuracy()))

# final max test accuracy = 0.9268 (10K iterations). Accuracy should peak above 0.92 in the first 2000 iterations.
