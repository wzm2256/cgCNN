import tensorflow as tf
import numpy as np
import keras
import Model
import PIL.Image as Image
import argparse
import matplotlib.pyplot as plt
import time
import util
import os

parser = argparse.ArgumentParser(description='Dynamic texture synthesis')
parser.add_argument('video', type=str, help='video folder')
parser.add_argument('--Iter', type=int, default=1000, required=False, help='Number of iterations to run.')
parser.add_argument('--layer_D', type=int, default=6, help='number of deep layers')
parser.add_argument('--Adam', type=int, default=0, help='use Adam|Rmsprop')
parser.add_argument('--mean', type=int, default=1, help='use mean|Gram')
parser.add_argument('--inner', type=int, default=10, help='number of Langevin steps in each iteration')


args = parser.parse_args()

Scale = 1e12
im_dir = './Image/'

Out_name = os.path.basename(args.video)

train_data = util.getTrainingData('Image/' + args.video, num_frames=12, image_size=128)

print(train_data.shape)

train_data = train_data / 255. 
train_data = train_data.astype(np.float32)

_, time_l, img_nrows, img_ncols, _ = train_data.shape

AVE = np.average(train_data, axis=(0, 1, 2, 3))
image_processed = train_data - AVE

ref = tf.constant(image_processed, dtype=tf.float32)
x = tf.Variable(np.zeros((2, time_l, img_nrows, img_ncols, 3)), dtype=tf.float32)

model = Model.model_3d(0)


out, h_dict, var_list, _ = model.run(x)
out_ref, h_dict_ref, _, _ = model.run(ref)

h_list_ref = list(h_dict_ref.values())


layer_name_ = ['relu1', 'relu2', 'block1_pool', 'relu3', 'relu4', 'block2_pool']
layer_name = layer_name_[0:args.layer_D]
layer_name_sum =[]


def gram_matrix(feature_maps):
	"""Computes the Gram matrix for a set of feature maps."""
	batch_size, time, height, width, channels = tf.unstack(tf.shape(feature_maps))
	denominator = tf.to_float(time * height * width)
	feature_maps = tf.reshape(feature_maps, tf.stack([batch_size, time * height * width, channels]))
	matrix = tf.matmul(feature_maps, feature_maps, adjoint_a=True)
	return matrix / denominator


def gram_loss(feature1, reference):
	F1 = gram_matrix(feature1)
	F2 = gram_matrix(reference)
	loss = tf.reduce_mean((F1 - F2) ** 2)
	return loss

def mean_loss(feature1, reference):
	m1 = tf.reduce_mean(feature1, axis=(1,2,3))
	m2 = tf.reduce_mean(reference, axis=(1,2,3))
	loss = tf.reduce_mean(tf.square(m1 - m2))
	return loss

LOSS_LIST = []
for layer in layer_name:
	if args.mean == 0:
		print(layer)
		tmp_loss = gram_loss(h_dict_ref[layer], h_dict[layer])
	else:
		print(layer)
		tmp_loss = mean_loss(h_dict_ref[layer], h_dict[layer])
	LOSS_LIST.append(tmp_loss)
	
Loss = tf.add_n(LOSS_LIST)
Loss += tf.reduce_sum(x ** 2) * 0.00000001
Loss *= Scale


if args.Adam == 0:
	op_I = tf.train.RMSPropOptimizer(learning_rate=0.001).minimize(Loss, var_list=x)
	op_w = tf.train.RMSPropOptimizer(learning_rate=0.001).minimize(-Loss, var_list=var_list)
else:
	op_I = tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.5, beta2=0.5).minimize(Loss, var_list=x)
	op_w = tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.5, beta2=0.5).minimize(-Loss, var_list=var_list)


sess = tf.Session()
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()


start_time = time.time()
for i in range(args.Iter+1):

	for _ in range(args.inner):
		sess.run(op_I)

	sess.run(op_w)
	out = sess.run([x, Loss] + LOSS_LIST)
	current_time = time.time()
	
	print(i, out[1], out[2:], 'already used %ds' % (current_time - start_time))
	if i % 100 == 0:
		tmp = (np.clip(out[0] + AVE, 0, 1) * 255.).astype(np.uint8)
		util.saveSampleVideo(tmp, 'Produce/' + Out_name  + '_IsMean_' + str(args.mean) + '_IsAdam_' + str(args.Adam), global_step=i)