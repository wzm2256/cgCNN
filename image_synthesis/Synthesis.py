import tensorflow as tf
import numpy as np
import keras
import Model
import PIL.Image as Image
import argparse
import matplotlib.pyplot as plt
import time

parser = argparse.ArgumentParser(description='image texture synthesis')
parser.add_argument('image1', type=str, help='image name')
parser.add_argument('--Iter', type=int, default=5000, required=False, help='number of iterations to run.')
parser.add_argument('--layer_S', type=int, default=3, help='number of shallow layers')
parser.add_argument('--layer_D', type=int, default=9, help='number of deep layers')
parser.add_argument('--Adam', type=int, default=0, help='use Adam|Rmsprop')
parser.add_argument('--mean', type=int, default=0, help='use mean|Gram')
parser.add_argument('--inner', type=int, default=10, help='number of Langevin steps in each iteration')
parser.add_argument('--Gau', type=float, default=0.000001, help='Gaussian penalty.')
parser.add_argument('--save_weights', type=int, default=0, help='save weights')

args = parser.parse_args()


Scale = 1e12

im_dir = './Image/'

# load image
image_path1 = im_dir+args.image1
I = Image.open(image_path1).resize((256,256))
loaded_image_array1 = np.array(I, dtype=np.float) / 255

img_nrows, img_ncols, _ = loaded_image_array1.shape

AVE = np.average(loaded_image_array1, axis=(0, 1))
image_processed = loaded_image_array1 - AVE

ref = tf.expand_dims(tf.constant(image_processed, dtype=tf.float32), axis=0)

x = tf.Variable(np.random.randn(3, img_nrows, img_ncols, 3), dtype=tf.float32)

# model construction
model = Model.model1(args.layer_S)

out, h_dict, var_list, _ = model.run(x)
out_ref, h_dict_ref, _, _ = model.run(ref)


layer_name_ = ['relu1', 'relu2', 'block1_pool', 'relu3', 'relu4', 'block2_pool', 'relu5', 'relu6', 'block3_pool', 'relu7', 'relu8', 'block4_pool', 'relu9', 'relu10', 'block5_pool']
layer_name = layer_name_[0:args.layer_D]
layer_c = ['composite1', 'composite2', 'composite3', 'composite4', 'composite5']
layer_name += layer_c[0: args.layer_S]


def gram_matrix(feature_maps):
	"""Computes the Gram matrix for a set of feature maps."""
	batch_size, height, width, channels = tf.unstack(tf.shape(feature_maps))
	denominator = tf.to_float(height * width)
	feature_maps = tf.reshape(feature_maps, tf.stack([batch_size, height * width, channels]))
	matrix = tf.matmul(feature_maps, feature_maps, adjoint_a=True)
	return matrix / denominator


def gram_loss(feature1, reference):
	F1 = gram_matrix(feature1)
	F2 = gram_matrix(reference)
	loss = tf.reduce_mean((F1 - F2) ** 2)
	return loss

	
def mean_loss(feature1, reference):
	m1 = tf.reduce_mean(feature1, axis=(1,2))
	m2 = tf.reduce_mean(reference, axis=(1,2))
	loss = tf.reduce_mean(tf.square(m1 - m2))
	return loss


LOSS_LIST = []
for layer in layer_name:
	if args.mean == 0:
		tmp_loss = gram_loss(h_dict_ref[layer], h_dict[layer])
	else:
		tmp_loss = mean_loss(h_dict_ref[layer], h_dict[layer])
	LOSS_LIST.append(tmp_loss)
Loss = tf.add_n(LOSS_LIST)
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



G_vars = tf.global_variables()
G_g_vars = [var for var in G_vars if 'RMSProp' not in var.name and 'Adam' not in var.name]

saver = tf.train.Saver(var_list=G_g_vars, max_to_keep=5)

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
		if args.save_weights == 1:
			s = saver.save(sess, 'SAVE_WEIGHT/' + args.image1 + '/', global_step=i, write_meta_graph=False)

		for j in range(sample):
			tmp = (np.clip(out[0][j] + AVE, 0, 1) * 255.).astype(np.uint8)
			plt.imsave('Produce/' + args.image1 + '_' + 'inner_' + str(args.inner) + '_' +  '_layer_S_' + str(args.layer_S) + '_layer_D_' + str(args.layer_D) + '_IsMean_' + str(args.mean) + '_Adam_' + str(args.Adam) + '_' + str(j) + '_' + str(i) + '_' + '.jpg', tmp)
