import tensorflow as tf
import numpy as np
import os
from PIL import Image

slim = tf.contrib.slim

IMAGE_WIDTH = 256
IMAGE_HEIGHT = 256
'''
path = 'data/train/dog.1.jpg'
img = Image.open(path)
if 'dog' in path :
	print('yes')
img.show()
'''
def preprocess(img, W=128, H=128):
	return img.resize((W, H))

def read_data(path, batch_size=50) :
	
	image_list = []
	for filename in os.listdir(path):
		image_list.append(path + '/' + filename)
	
	return image_list

def get_batch(image_list, epoch, batch_size=50) :
	images = []
	labels = []
	offset = batch_size * (epoch - 1)
	for i in range(offset, offset + batch_size) :
		filename = image_list[i]

		img = Image.open(filename)
		img = np.asarray(preprocess(img, IMAGE_WIDTH, IMAGE_HEIGHT), dtype=np.float32) 
		img /= 255
		if 'cat' in filename :
			label = 0
		else :
			label = 1
		images.append(img)
		labels.append(label)

	return images, labels

def CatNet(inputs, isTraining=True, scope='CatNet'):

	with slim.arg_scope( [slim.convolution2d, slim.fully_connected],
                                weights_initializer=slim.initializers.xavier_initializer(),
                                biases_initializer=slim.init_ops.zeros_initializer(),
                                activation_fn=tf.nn.relu):
		with tf.variable_scope(scope) as sc :
			
			net = slim.convolution2d(inputs, 32, [3, 3], stride=1, scope='conv1')
			print(net)
			net = slim.max_pool2d(net, [2, 2], stride=1, padding='SAME', scope='pool1')
			print(net)
			net = slim.convolution2d(net, 32, [3, 3], stride=2, scope='conv2')
			print(net)
			net = slim.max_pool2d(net, [2, 2], stride=2, padding='SAME', scope='pool2')
			print(net)
			net = slim.convolution2d(net, 64, [3, 3], stride=1, scope='conv3')
			print(net)
			net = slim.max_pool2d(net, [2, 2], stride=2, padding='SAME', scope='pool3')
			print(net)
			net = slim.flatten(net, scope='flatten')
			print(net)
			net = slim.fully_connected(net, 64, scope='fc1')
			print(net)
			net = slim.fully_connected(net, 2, activation_fn=None, scope='fc2')
			print(net)

	return net

#image_list = read_data('data/train/')
#images, labels = get_batch(image_list, 2)

def main():

	image_list = read_data('data/train/') 
	x = tf.placeholder(tf.float32, shape=(None, IMAGE_WIDTH, IMAGE_WIDTH, 3))
	y_ = tf.placeholder(tf.int32, shape=(None,))
	keep_prob = tf.placeholder(tf.float32)
	one_hot_labels = tf.one_hot(y_, 2)
	one_hot_labels = tf.one_hot(y_, 2)
	print(one_hot_labels)
	output = CatNet(x, isTraining=True)
	'''loss = tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=one_hot_labels)
	print('output', output)
	print('loss', loss)
	cross_entropy = tf.reduce_mean(loss)
	l2_losses = [0.001 * tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'weights' in v.name]
	reduced_loss = cross_entropy + tf.add_n(l2_losses)

	train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(reduced_loss)

	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	sess = tf.Session(config=config)
	init = tf.global_variables_initializer()
	sess.run(init)	
	correct_preds = tf.equal(tf.argmax(output, 1), tf.argmax(one_hot_labels, 1))
	validate = tf.reduce_mean(tf.cast(correct_preds, tf.float32))
	'''
	cross_entropy = tf.reduce_mean(
	tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_labels, logits=output))
	train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
	tf.summary.scalar('Cross entropy', cross_entropy)
	#inference
	correct_prediction = tf.equal(tf.argmax(output, 1), tf.argmax(one_hot_labels, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	tf.summary.scalar('Train accuracy', accuracy)
	merged = tf.summary.merge_all()
	with tf.Session() as sess:
	  sess.run(tf.global_variables_initializer())
	  writer = tf.summary.FileWriter('summary/train', sess.graph)
	  for i in range(500):
	  	images, labels = get_batch(image_list, i, batch_size=50)
	  	summary, ce, _ = sess.run([merged, cross_entropy, train_step], feed_dict={x: images, y_: labels, keep_prob: 0.5})
	  	print('Step {}, Loss {}'.format(i, ce))
	  	if i % 10 == 0:
	  		summary, train_accuracy = sess.run([merged, accuracy], feed_dict={x: images, y_: labels, keep_prob: 1.0})
	  		print('step %d, training accuracy %g' % (i, train_accuracy))
	  	writer.add_summary(summary, i)

	'''
	for epoch in range(1, 500) :
		images, labels = get_batch(image_list, epoch, batch_size=50)
		feed_dict={x: images, y_: labels}
		loss, _ = sess.run([reduced_loss, train_op], feed_dict)

		print('Epoch: {}, loss: {}'.format(epoch, loss))
		if epoch % 5 == 0 :
			print('Training Accuracy: ', sess.run(validate, feed_dict))
    
	sess.close()
	'''
main()