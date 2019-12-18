import tensorflow as tf
import numpy as np
import keras
import Model
import PIL.Image as Image
import argparse
import matplotlib.pyplot as plt
import time
import Generator
import os

parser = argparse.ArgumentParser(description='Image texture expansion. Training phase.')
parser.add_argument('image1', type=str, help='image name')
parser.add_argument('--Iter', type=int, default=5000, required=False, help='number of iterations to run.')
parser.add_argument('--layer_S', type=int, default=3, help='number of shallow layers')
parser.add_argument('--layer_D', type=int, default=9, help='number of deep layers')
parser.add_argument('--Adam', type=int, default=1, help='use Adam|Rmsprop')
parser.add_argument('--mean', type=int, default=0, help='use mean|Gram')
parser.add_argument('--inner', type=int, default=10, help='number of Langevin steps in each iteration')
parser.add_argument('--im_dir', type=str, default='./Image/', help='image folder')
parser.add_argument('--normal', action='store_true', help='use normal or uniform noise')
parser.add_argument('--Gau', type=float, default=0.0, help='Gaussian penalty')
parser.add_argument('--Fou', type=float, default=0.0, help='Fourier norm penalty')
parser.add_argument('--diversity', type=str, default='No', help='diversity penalty in which layer, relu7|No')
parser.add_argument('--d_weight', type=int, default=0, help='diversity penalty 0|1e2')

args = parser.parse_args()

Scale = 1e12

image_path1 = args.im_dir + args.image1
I = Image.open(image_path1).resize((256,256))
loaded_image_array1 = np.array(I, dtype=np.float) / 255

img_nrows, img_ncols, _ = loaded_image_array1.shape

AVE = np.average(loaded_image_array1, axis=(0, 1))
image_processed = loaded_image_array1 - AVE

ref = tf.expand_dims(tf.constant(image_processed, dtype=tf.float32), axis=0)

pad_type = 'zero'
sn = False
noise_depth = 3
samples = 3

x_out_ = Generator.pyramid_tf20(256, noise_depth, samples, sn=sn, pad_type=pad_type, is_training=True, is_normal=args.normal)
x_out = (x_out_ + 1.0) / 2.0
x = x_out - tf.expand_dims(tf.constant(AVE, dtype=tf.float32), axis=0)


t_vars = tf.trainable_variables()
g_vars = [var for var in t_vars if 'generator' in var.name]


model = Model.model1(args.layer_S)


out, h_dict, var_list, _ = model.run(x)
out_ref, h_dict_ref, _, _ = model.run(ref)

h_list_ref = list(h_dict_ref.values())


layer_name_ = ['relu1', 'relu2', 'block1_pool', 'relu3', 'relu4', 'block2_pool', 'relu5', 'relu6', 'block3_pool', 'relu7', 'relu8', 'block4_pool', 'relu9', 'relu10', 'block5_pool']
layer_name = layer_name_[0:args.layer_D]

layer_c = ['composite1', 'composite2', 'composite3', 'composite4', 'composite5']
layer_name += layer_c[0: args.layer_S]


##################################
# diversity_name = 'relu6'
diversity_name = args.diversity
def diversity_loss(feature):
	h = feature.shape[1].value
	w = feature.shape[2].value
	c = feature.shape[3].value
	print([h,w,c,(h*w*c)])
	Permute_f = tf.gather(feature, tf.random_shuffle(tf.range(tf.shape(feature)[0])))
	# Permute_f = tf.random_shuffle(feature)
	loss = -tf.reduce_sum(tf.square(Permute_f - feature)) / (h * w * c)
	return loss

if diversity_name != 'No':
	d_loss = diversity_loss(h_dict[diversity_name]) * Scale / args.d_weight
else:
	d_loss = tf.constant(0, dtype=tf.float32)
#################################


def gram_matrix(feature_maps):
	"""Computes the Gram matrix for a set of feature maps."""
	batch_size, height, width, channels = tf.unstack(tf.shape(feature_maps))
	denominator = tf.to_float(height * width)
	feature_maps = tf.reshape(feature_maps, tf.stack([batch_size, height * width, channels]))
	matrix = tf.matmul(feature_maps, feature_maps, adjoint_a=True)
	return matrix / denominator

def gram_loss(feature1, reference):
	# batch = feature1.get_shape[0].value
	F1 = gram_matrix(feature1)
	F2 = gram_matrix(reference)
	loss = tf.reduce_mean((F1 - F2) ** 2)
	return loss
	
def mean_loss(feature1, reference):
	m1 = tf.reduce_mean(feature1, axis=(1,2))
	m2 = tf.reduce_mean(reference, axis=(1,2))
	loss = tf.reduce_mean(tf.square(m1 - m2))
	return loss

def spectral_trans(x):
	# only spatial
	x_reshape = tf.transpose(x, (0, 3, 1, 2))
	x_reshape = tf.cast(x_reshape, dtype=tf.complex64)
	tmp = tf.spectral.fft2d(x_reshape)
	tmp = tf.transpose(tmp, (0, 2, 3, 1))

	return tmp

def RP_loss(style, combination):
	style_spe = spectral_trans(style)
	comb_spe = spectral_trans(combination)

	loss1 = tf.reduce_mean(tf.square(tf.abs(style_spe) - tf.abs(comb_spe)))
	return loss1


LOSS_LIST = []
for layer in layer_name:
	if args.mean == 0:
		tmp_loss = gram_loss(h_dict_ref[layer], h_dict[layer])
	else:
		tmp_loss = mean_loss(h_dict_ref[layer], h_dict[layer])
	LOSS_LIST.append(tmp_loss)

Loss_spectral = RP_loss(x, ref) * args.Fou
LOSS_LIST.append(Loss_spectral)

Loss = tf.add_n(LOSS_LIST)

Loss += tf.reduce_sum(x ** 2) * args.Gau

Loss *= Scale

if args.Adam == 0:
	op_I = tf.train.RMSPropOptimizer(learning_rate=0.001).minimize(Loss, var_list=g_vars)
	op_w = tf.train.RMSPropOptimizer(learning_rate=0.001).minimize(-Loss, var_list=var_list)
elif args.Adam == 1:
	op_I = tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.5, beta2=0.5).minimize(Loss + d_loss, var_list=g_vars)
	op_w = tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.5, beta2=0.5).minimize(-Loss, var_list=var_list)

    
G_vars = tf.global_variables()
G_g_vars = [var for var in G_vars if 'generator' in var.name and 'RMSProp' not in var.name and 'Adam' not in var.name]


save_name = args.image1 + '_layer_S_' + str(args.layer_S) + '_layer_D_' + str(args.layer_D) + '_inner_' + str(args.inner) + '_IsMean_' + str(args.mean) + '_Adam_' + str(args.Adam) + '_normal_' + str(args.normal) + '_Gau_' + str(args.Gau) + '_Fou_' + str(args.Fou) + '_diversity_' + diversity_name + '_d_weight_' + str(args.d_weight)

saver = tf.train.Saver(var_list=G_g_vars, max_to_keep=20)
checkpoint_dir = './Saved_model/' + save_name + '/'

if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)


sess = tf.Session()
sess.run(tf.global_variables_initializer())

start_time = time.time()
sample = x.get_shape()[0].value


for i in range(args.Iter+1):
	for _ in range(args.inner):
		sess.run(op_I)

	sess.run(op_w)
	out = sess.run([x, Loss] + LOSS_LIST + [d_loss / Scale])
	current_time = time.time()

	print(i, out[1], out[2:], 'already used %ds' % (current_time - start_time))

	if i % 200 == 0 or i % args.Iter == 0:
		s = saver.save(sess, checkpoint_dir, global_step=i, write_meta_graph=False)
		print(s)
		current_time = time.time()

		for j in range(sample):
			tmp = (np.clip(out[0][j] + AVE, 0, 1) * 255.).astype(np.uint8)
			plt.imsave('Produce/' + save_name + '_' + str(j) + '_' + str(i) + '_' + '.jpg', tmp)
