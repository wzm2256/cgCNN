import tensorflow as tf
import numpy as np
import keras
import Model
import PIL.Image as Image
import argparse
import matplotlib.pyplot as plt
import time
import util
import Generator
import os

parser = argparse.ArgumentParser(description='Dynamic texture expansion. Training phase.')
parser.add_argument('video', type=str, help='video folder')
parser.add_argument('--Iter', type=int, default=3000, required=False, help='number of iterations to run.')
parser.add_argument('--layer_D', type=int, default=6, help='number of deep layers')
parser.add_argument('--Adam', type=int, default=0, help='use Adam|Rmsprop')
parser.add_argument('--mean', type=int, default=0, help='use mean|Gram')
parser.add_argument('--inner', type=int, default=10, help='number of Langevin steps in each iteration')
parser.add_argument('--Gau', type=float, default=0.00000001, help='Gaussian penalty')
parser.add_argument('--pad_type', type=str, default='zero', help='pad type')
parser.add_argument('--normal', action='store_true', help='use normal or uniform noise')
parser.add_argument('--diversity', type=str, default='No', help='diversity penalty in which layer, relu7|No')
parser.add_argument('--d_weight', type=int, default=0, help='diversity penalty 0|1e2')

args = parser.parse_args()

Out_name = os.path.basename(args.video)

Scale = 1e12
im_dir = './Image/'


train_data = util.getTrainingData('Image/' + args.video, num_frames=12, image_size=128)

print(train_data.shape)


train_data = train_data / 255. 
train_data = train_data.astype(np.float32)

_, time_l, img_nrows, img_ncols, _ = train_data.shape

AVE = np.average(train_data, axis=(0, 1, 2, 3))

image_processed = train_data - AVE

ref = tf.constant(image_processed, dtype=tf.float32)

print('Synthesizing ' + args.video)


x_out_ = Generator.pyramid_tf((12,128), 3, 2, is_training=True, pad_type=args.pad_type, is_normal=args.normal)

x_out = (x_out_ + 1.0) / 2.0
x = x_out - tf.constant(AVE, dtype=tf.float32)

t_vars = tf.trainable_variables()
g_vars = [var for var in t_vars if 'generator' in var.name]

model = Model.model1(0)

out, h_dict, var_list, _ = model.run(x)
out_ref, h_dict_ref, _, _ = model.run(ref)

h_list_ref = list(h_dict_ref.values())

layer_name_ = ['relu1', 'relu2', 'block1_pool', 'relu3', 'relu4', 'block2_pool', 'relu5', 'relu6', 'block3_pool']
layer_name = layer_name_[0:args.layer_D]

######################################################
# diversity_name = 'relu6'
diversity_name = args.diversity
def diversity_loss(feature):
	t = feature.shape[1].value
	h = feature.shape[2].value
	w = feature.shape[3].value
	c = feature.shape[4].value
	print([h,w,c,(h*w*c)])
	Permute_f = tf.gather(feature, tf.constant([1,0]))
	# Permute_f = tf.gather(feature, tf.random_shuffle(tf.range(tf.shape(feature)[0])))
	# Permute_f = tf.random_shuffle(feature)
	loss = -tf.reduce_sum(tf.square(Permute_f - feature)) / (h * w * c * t)
	return loss

if diversity_name != 'No':
	d_loss = diversity_loss(h_dict[diversity_name]) * Scale / args.d_weight
else:
	d_loss = tf.constant(0, dtype=tf.float32)
######################################

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
Loss += tf.reduce_sum(x ** 2) * args.Gau
Loss *= Scale

if args.Adam == 0:
	op_I = tf.train.RMSPropOptimizer(learning_rate=0.001).minimize(Loss + d_loss, var_list=g_vars)
	op_w = tf.train.RMSPropOptimizer(learning_rate=0.001).minimize(-Loss, var_list=var_list)
else:
	op_I = tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.5, beta2=0.5).minimize(Loss + d_loss, var_list=g_vars)
	op_w = tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.5, beta2=0.5).minimize(-Loss, var_list=var_list)


G_vars = tf.global_variables()
G_g_vars = [var for var in G_vars if 'generator' in var.name and 'RMSProp' not in var.name and 'Adam' not in var.name]


save_name = Out_name + '_depth_' + str(args.layer_D) + \
			'_inner_' + str(args.inner) + '_IsMean_' + str(args.mean) + '_Adam_' + str(args.Adam) + '_padtype_' + args.pad_type + \
			'_normal_' + str(args.normal) + '_Gau_' + str(args.Gau) + '_diversity_' + args.diversity + '_d_weight_' + str(args.d_weight)

Saver = tf.train.Saver(var_list=G_g_vars, max_to_keep=10)
checkpoint_dir = './Saved_model/' + save_name + '/'

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
	if i % 500 == 0:
		s = Saver.save(sess, checkpoint_dir, global_step=i, write_meta_graph=False)
		tmp = (np.clip(out[0] + AVE, 0, 1) * 255.).astype(np.uint8)
		util.saveSampleVideo(tmp, 'Out/' + save_name, global_step=i)
