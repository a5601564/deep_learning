# -*- coding: utf-8 -*-
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

"""
MNIST数据集，每一张图片包含28X28个像素点
"""
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
print(mnist)

sess=tf.InteractiveSession()
x=tf.placeholder(tf.float32,[None,784])

W=tf.Variable(tf.zeros([784,10]))
b=tf.Variable(tf.zeros([10]))

y=tf.nn.softmax(tf.matmul(x,W)+b)

loss_func_y = tf.placeholder(tf.float32,[None,10])

cross_entroy = tf.reduce_mean(-tf.reduce_sum(loss_func_y*tf.log(y),reduction_indices=[1]) )

#优化器

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entroy)


tf.global_variables_initializer().run()

for i in range(1000):
    batch_xs,batch_ys = mnist.train.next_batch(100)
    train_step.run({x:batch_xs,loss_func_y:batch_ys} )


#prediction

correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(loss_func_y,1))

sess.run(correct_prediction)

accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

print(accuracy.eval({x:mnist.test.images,loss_func_y:mnist.test.labels}))