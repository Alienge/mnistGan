import datetime
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
from tensorflow.examples.tutorials.mnist import input_data
import cv2
mnist = input_data.read_data_sets('../../Mnist_data')

#sample_image = mnist.train.next_batch(1)
#print(sample_image[0].shape)
#sample_image = sample_image.reshape([28,28])
#plt.imshow(sample_image,cmap='Greys')
#plt.show()
def _bias_variable(name,shape,initializer):
    var = tf.get_variable(name, shape, initializer=initializer, dtype=tf.float32)
    return var
def _weight_variable(name,shape,std):
    return _bias_variable(name, shape, initializer=tf.truncated_normal_initializer(stddev=std,dtype=tf.float32)
                          )
def _batch_norm(x,name,reuse=None):
    with tf.variable_scope(name_or_scope=name,reuse=reuse):
        batch,rows,cols,channels = [ i.value for i in x.get_shape()]
        var_shape=[channels]
        mu,sigma_sq = tf.nn.moments(x,[1,2],keep_dims=True)
        shift = tf.get_variable(name='shift',shape=var_shape,initializer=tf.constant_initializer(0))
        scale = tf.get_variable(name='sacle',shape=var_shape,initializer=tf.constant_initializer(1))
        epsilon = 1e-3
        normalized = (x-mu)/(sigma_sq+epsilon)**(.5)
        return normalized*scale+shift

def discriminator(images,reuse = None):
    with tf.variable_scope("d1",reuse=reuse):
        w = _weight_variable('weights',[5,5,1,32],std=0.02)
        b = _bias_variable('bias',[32],tf.constant_initializer(0.1))
        d1 = tf.nn.conv2d(images,w,strides=[1,1,1,1],padding='SAME')
        d1 = tf.nn.relu(d1+b,name='d1')
        d1 = tf.nn.avg_pool(d1,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME',name='pool1')

    with tf.variable_scope('d2',reuse=reuse):
        w = _weight_variable('weights',[5,5,32,64],std=0.1)
        b = _bias_variable('bias',[64],tf.constant_initializer(0))
        d2 = tf.nn.conv2d(d1,w,[1,1,1,1],padding="SAME")
        d2 = tf.nn.relu(d2+b,name="d2")
        d2 = tf.nn.avg_pool(d2,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME',name='pool2')
    with tf.variable_scope('d3',reuse=reuse):
        d3 = tf.reshape(d2,[-1,7*7*64])
        w = _weight_variable('weights',[7*7*64,1024],std=0.01)
        b = _bias_variable('bias',[1024],tf.constant_initializer(0))
        d3 = tf.matmul(d3,w)+b
        d3 = tf.nn.relu(d3,name='d3')
    with tf.variable_scope('d4',reuse=reuse):
        w = _weight_variable('weights',[1024,1],std=0.02)
        b = _bias_variable('bias',[1],tf.constant_initializer(0))
        d4 = tf.matmul(d3,w)+b
        return d4

def generator(z,batch_size,z_dim,reuse=None):
    with tf.variable_scope('g1',reuse=reuse) as scope:
        w = _weight_variable('weights',[z_dim,3136],std=0.02)
        b = _bias_variable('bias',[3136],tf.constant_initializer(0))
        g1 = tf.matmul(z,w)+b
        g1 = tf.reshape(g1,[-1,56,56,1])
        g1 = _batch_norm(g1,name=scope.name,reuse=reuse)
        g1 = tf.nn.relu(g1,name='g1')
    with tf.variable_scope('g2',reuse=reuse) as scope:
        w = _weight_variable('weights',shape=[3,3,1,z_dim/2],std=0.02)
        b = _bias_variable('bias',[z_dim/2],tf.constant_initializer(0.1))
        g2 = tf.nn.conv2d(g1,w,[1,2,2,1],padding='SAME')
        g2 = g2+b
        g2 = _batch_norm(g2,name=scope.name,reuse=reuse)
        g2=tf.nn.relu(g2)
        g2 = tf.image.resize_images(g2,[56,56])
    with tf.variable_scope('g3',reuse=reuse) as scope:
        w = _weight_variable("weights",[3,3,z_dim/2,z_dim/4],std=0.02)
        b = _bias_variable('bias',[z_dim/4],tf.constant_initializer(0.1))
        g3 = tf.nn.conv2d(g2,w,[1,2,2,1],padding='SAME')
        g3 = g3+b
        g3 = _batch_norm(g3,scope.name,reuse)
        g3 = tf.nn.relu(g3)
        g3 = tf.image.resize_images(g3,[56,56])

    with tf.variable_scope('g4',reuse=reuse) as scope:
        w = _weight_variable('weights',[1,1,z_dim/4,1],std=0.02)
        b = _bias_variable('bias',[1],tf.constant_initializer(0.1))
        g4 = tf.nn.conv2d(g3,w,[1,2,2,1],padding='SAME')
        g4 = g4+b
        g4 = tf.sigmoid(g4)

        return g4

batch_size = 50
z_dimensions = 100
z_placeholder = tf.placeholder(tf.float32,[None,z_dimensions],name='z_placeholder')
x_placeholder = tf.placeholder(tf.float32,[None,28,28,1],name='x_placeholder')

Gz = generator(z_placeholder,batch_size,z_dimensions)
Dx = discriminator(x_placeholder)
Dg = discriminator(Gz,reuse=True)

d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=Dx,labels=tf.ones_like(Dx)))
d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=Dg,labels=tf.zeros_like(Dg)))
d_loss = d_loss_fake+d_loss_real

g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=Dg,labels=tf.ones_like(Dg)))

tvars = tf.trainable_variables()
d_vars = [var for var in tvars if 'd' in var.name]
g_vars = [var for var in tvars if 'g' in var.name]

d_trainer_real = tf.train.AdamOptimizer(0.003).minimize(d_loss_real,var_list=d_vars)
d_trainer_fake = tf.train.AdamOptimizer(0.003).minimize(d_loss_fake,var_list=d_vars)
d_trainer = tf.train.AdamOptimizer(0.003).minimize(d_loss,var_list=d_vars)
g_traner = tf.train.AdamOptimizer(0.001).minimize(g_loss,var_list=g_vars)

import pprint
pp = pprint.PrettyPrinter()
pp.pprint(d_vars)
print("#"*20)
pp.pprint(g_vars)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    #预训练分辨器
    for i in range(500):
        print(".",end='')
        z_batch = np.random.normal(0,1,size=[batch_size,z_dimensions])
        real_image_batch = mnist.train.next_batch(batch_size)[0].reshape([batch_size,28,28,1])
        _,_,dLossReal,dLossFake = sess.run([d_trainer_real,d_trainer_fake,d_loss_real,d_loss_real],
                                           {x_placeholder:real_image_batch,z_placeholder:z_batch} )
        if(i%100==0):
            print("\rdLossReal:",dLossReal,'dLossFake',dLossFake)
    #交替训练
    for i in range(100000):
        print('.',end="")
        real_image_batch=mnist.train.next_batch(batch_size)[0].reshape([batch_size,28,28,1])
        z_batch = np.random.normal(0, 1, size=[batch_size, z_dimensions])
        _,dLossReal,dLossFake = sess.run([d_trainer,d_loss_real,d_loss_fake],
                                         feed_dict={x_placeholder:real_image_batch,z_placeholder:z_batch})
        #训练generate
        z_bacth = np.random.normal(0,1,size=[batch_size,z_dimensions])
        _ = sess.run(g_traner,feed_dict={z_placeholder:z_batch})

        if i %100 == 0:
            #每100次，输出生成图像
            print("\rIteration:",i,'at',datetime.datetime.now())
            z_bacth = np.random.normal(0,1,size=[1,z_dimensions])
            generated_images = generator(z_placeholder,1,z_dimensions,reuse=True)
            images = sess.run(generated_images,{z_placeholder:z_batch})
            images_real = images[0].reshape([28,28])*255
            #im1 = abs(images_real)

            im1 = np.uint8(images_real)
            #images_real = images.astype(np.uint8)
            im = Image.fromarray(images_real)
            im = im.convert('L')
            im.save('./images/%d.jpg'%i)


            im = images[0].reshape([1,28,28,1])
            reuslt = discriminator(x_placeholder,reuse=True)
            estimate = sess.run(reuslt,{x_placeholder:im})
            print("EStimate:",np.squeeze(estimate))