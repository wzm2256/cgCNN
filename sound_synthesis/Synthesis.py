import tensorflow as tf
import numpy as np
import keras
import Model
import PIL.Image as Image
import argparse
import matplotlib.pyplot as plt
import time
import librosa
import os

parser = argparse.ArgumentParser(description='Sound texture synthesis')
parser.add_argument('image1', type=str, help='audio name')
parser.add_argument('--Iter', type=int, default=3000, required=False, help='number of iterations to run.')
parser.add_argument('--layer_D', type=int, default=4, help='number of deep layers')
parser.add_argument('--Adam', type=int, default=0, help='use Adam|Rmsprop')
parser.add_argument('--mean', type=int, default=0, help='use mean|Gram')
parser.add_argument('--Model', type=int, default=1, help='which model 1|2')
parser.add_argument('--inner', type=int, default=10, help='number of Langevin steps in each iteration')
parser.add_argument('--Gau', type=float, default=0.00000001, help='Gaussian penalty')
parser.add_argument('--sn', type=float, default=0, help='Fourier norm penalty')


args = parser.parse_args()

Scale = 1e12
im_dir = './Sound/'

image_path1 = im_dir + args.image1

I, fs = librosa.load(image_path1)
if len(I.shape) == 2:
	I = I[:,0]

assert len(I > 50000)
I = I[:50000]
I_ori1 = I.copy()
M = np.max(I)
m = np.min(I)

I = (I - m) / (M - m) - 0.5

I_ori = (I + 0.5)  * (M - m) + m

if not os.path.isfile('Out/Ori' + args.image1):
	librosa.output.write_wav('Out/Ori' + args.image1, I_ori, fs)

img_nrows = 50000

ref = tf.expand_dims(tf.expand_dims(tf.constant(I, dtype=tf.float32), axis=0), axis=-1)
x = tf.Variable(np.random.randn(2, img_nrows, 1)*0.1, dtype=tf.float32)


if args.Model == 1:
	model = Model.Sound_1(0)
elif args.Model == 2:
	model = Model.Sound_2(0)
else:
	raise NotImplementedError

out, h_dict, var_list, _ = model.run(x)
out_ref, h_dict_ref, _, _ = model.run(ref)

h_list_ref = list(h_dict_ref.values())

layer_name_ = ['relu1', 'relu2', 'relu3', 'relu4', 'relu5', 'relu6', 'block3_pool', 'relu7', 'relu8', 'block4_pool', 'relu9', 'relu10', 'block5_pool']
layer_name = layer_name_[0:args.layer_D]


def gram_matrix(feature_maps):
	"""Computes the Gram matrix for a set of feature maps."""
	batch_size, height, channels = tf.unstack(tf.shape(feature_maps))
	denominator = tf.to_float(height)
	feature_maps = tf.reshape(feature_maps, tf.stack([batch_size, height, channels]))
	matrix = tf.matmul(feature_maps, feature_maps, adjoint_a=True)
	return matrix / denominator


def gram_loss(feature1, reference):
	# batch = feature1.get_shape[0].value
	F1 = gram_matrix(feature1)
	F2 = gram_matrix(reference)
	loss = tf.reduce_mean((F1 - F2) ** 2)
	return loss

def mean_loss(feature1, reference):
	m1 = tf.reduce_mean(feature1, axis=1)
	m2 = tf.reduce_mean(reference, axis=1)
	loss = tf.reduce_mean(tf.square(m1 - m2))
	return loss

def spectral(x):
	y = tf.transpose(x, [0,2,1])
	y = tf.cast(y, tf.complex64)
	y_hat = tf.spectral.fft(y)
	z = tf.transpose(y_hat, [0,2,1])
	return z

def sn_loss(x, y):
	xx = spectral(x)
	yy = spectral(y)
	loss = tf.reduce_mean(tf.square(tf.abs(xx) - tf.abs(yy)))
	return loss


LOSS_LIST = []
for layer in layer_name:
	if args.mean == 0:
		tmp_loss = gram_loss(h_dict_ref[layer], h_dict[layer])
	else:
		tmp_loss = mean_loss(h_dict_ref[layer], h_dict[layer])
	LOSS_LIST.append(tmp_loss)
Loss = tf.add_n(LOSS_LIST)

Loss += sn_loss(x, ref) * args.sn
Loss += tf.reduce_sum(x ** 2) * args.Gau
Loss *= Scale


if args.Adam == 0:
	op_I = tf.train.RMSPropOptimizer(learning_rate=0.001).minimize(Loss, var_list=x)
	op_w = tf.train.RMSPropOptimizer(learning_rate=0.001).minimize(-Loss, var_list=var_list)
else:
	op_I = tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.5, beta2=0.5).minimize(Loss, var_list=x)
	op_w = tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.5, beta2=0.5).minimize(-Loss, var_list=var_list)


sess = tf.Session()
sess.run(tf.global_variables_initializer())

start_time = time.time()
sample = x.get_shape()[0].value

for i in range(args.Iter+1):
	for _ in range(args.inner):
		sess.run(op_I)

	sess.run(op_w)

	out = sess.run([x, Loss] + LOSS_LIST)
	current_time = time.time()
	print(i, out[1], out[2:], 'already used %ds' % (current_time - start_time))


	if i % 1000 == 0:
		for j in range(sample):
			tmp = np.clip(out[0][j], -1, 1)
			tmp = (tmp + 0.5) * (M - m) + m
			librosa.output.write_wav('Out/' + args.image1 + '_Model_' + str(args.Model) + '_depth_' + str(args.layer_D)  + '_IsMean_' + str(args.mean) + '_Adam_' + str(args.Adam) + '_Fou_' + str(args.sn) + '_Gau_' + str(args.Gau) + '_inner_' + str(args.inner) + '_' + str(j) + '_' + str(i) +  '_' + '.wav', tmp, fs)