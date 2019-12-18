import tensorflow as tf
import numpy as np
import keras
import Model
import PIL.Image as Image
import argparse
import matplotlib.pyplot as plt
import time
import util

parser = argparse.ArgumentParser(description='Dynamic texture inpainting.')

parser.add_argument('--image', type=str, default='VIDEO/grass_3', help='original image sequence')
parser.add_argument('--mask', type=str, default='mask2.png', help='mask')
parser.add_argument('--mean', type=int, default=1, help='use mean or Gram')
parser.add_argument('--inner', type=int, default=10, help='number of Langevin steps in each iteration')
parser.add_argument('--tv', type=float, default=0.000000000, help='total variation penalty')
parser.add_argument('--fou', type=float, default=0.00001, help='Fourier norm penalty')
parser.add_argument('--Iter', type=int, default=1000, help='Number of iterations')
args = parser.parse_args()


Scale = 1e12

im_dir = './Image/'

save_name = args.image + '_' + args.mask + '_mean_' + str(args.mean) + '_inner_' + str(args.inner) + '_tv_' + str(args.tv) + '_fou_' + str(args.fou)

# load image
image_path1 = im_dir + args.image
train_data = util.getTrainingData(image_path1, num_frames=12, image_size=128)


loaded_image_array1 = np.asarray(train_data, dtype=np.float32) / 255
_, num_frame, img_nrows, img_ncols, _ = loaded_image_array1.shape

# load mask
mask_path1 = im_dir + args.mask
I_m = Image.open(mask_path1).convert('L').resize((128,128))
I_m_np = np.asarray(I_m, dtype=np.float32) / 255
loaded_mask_array1 = (I_m_np > 0.5).astype(np.float32)
loaded_mask_array1 = np.expand_dims(np.expand_dims(np.expand_dims(loaded_mask_array1, axis=-1), axis=0), axis=0)
mask_tf = tf.constant(loaded_mask_array1, dtype=tf.float32)
_, _, img_nrows_m, img_ncols_m, _ = loaded_mask_array1.shape

# prepare masked image
masked_image_array = loaded_image_array1 * loaded_mask_array1

AVE = np.sum(masked_image_array, axis=(0,1,2,3), keepdims=True) / (np.sum(loaded_mask_array1) * num_frame)

Init = masked_image_array + (1 - loaded_mask_array1) * AVE


util.saveSampleVideo((masked_image_array * 255.).astype(np.uint8), 'Inpainted/masked_' + save_name, global_step=-1)

def bound_box(mask, border=6):

	mask_1 = 1 - np.squeeze(mask)
	assert mask_1.ndim == 2

	flatten_0 = np.sum(mask_1, axis=1) > 5
	flatten_1 = np.sum(mask_1, axis=0) > 5

	min_0 = np.min(np.nonzero(flatten_0)) - border
	max_0 = np.max(np.nonzero(flatten_0)) + border

	min_1 = np.min(np.nonzero(flatten_1)) - border
	max_1 = np.max(np.nonzero(flatten_1)) + border

	bound_box = np.zeros_like(mask_1)
	bound_box[min_0:max_0+1, min_1:max_1+1] = 1

	return min_0, max_0, min_1, max_1

image_processed = Init - AVE

ref = tf.Variable(image_processed, dtype=tf.float32)

min_0, max_0, min_1, max_1 = bound_box(loaded_mask_array1, border=2)

Var_patch = ref[0:1, :, min_0:max_0, min_1:max_1, : ]

#water_4
# Ref_patch = ref[0:1, :, min_0:max_0, min_1+30:max_1+30, :]
# or
# Ref_patch = ref[0:1, :, min_0:max_0, max_1:2*max_1-min_1, :]
# or
# Ref_patch = ref[0:1, :, max_0:2*max_0-min_0, min_1:max_1, :]

# sea_2
# Ref_patch = ref[0:1, :, min_0:max_0, max_1:2*max_1-min_1, :]

# grass_3
Ref_patch = ref[0:1, :, min_0+30:max_0+30, min_1:max_1, :] # The template is assigned to be the patch below the corrupted region.



Patches = tf.concat([Var_patch, Ref_patch], axis=0)


model = Model.small_1()

_, h_dict, var_list = model.run(Patches)

h_list_ref = list(h_dict.values())


layer_name = ['relu1', 'relu2', 'block1_pool', 'relu3']


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


def spectral_loss(feature1, ref):
	f1_trans = tf.transpose(feature1, (0,1,4,2,3))
	f2_trans = tf.transpose(ref, (0,1,4,2,3))

	m1 = tf.abs(tf.fft2d(tf.cast(f1_trans, tf.complex64)))
	m2 = tf.abs(tf.fft2d(tf.cast(f2_trans, tf.complex64)))

	loss = tf.reduce_mean(tf.square(m1 - m2))
	return loss


LOSS_LIST = []
for layer in layer_name:
	if args.mean == 0:
		tmp_loss = gram_loss(h_dict[layer][0:1], h_dict[layer][1:])
	else:
		tmp_loss = mean_loss(h_dict[layer][0:1], h_dict[layer][1:])
	LOSS_LIST.append(tmp_loss)



tv_loss =  tf.reduce_sum(tf.image.total_variation(ref[0]) * args.tv)

fou_loss = spectral_loss(Patches[0:1], Patches[1:]) * args.fou

LOSS_LIST.append(tv_loss)
LOSS_LIST.append(fou_loss)

Loss = tf.add_n(LOSS_LIST)

Loss *= Scale


optimizer = tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.5, beta2=0.5)
gvs = optimizer.compute_gradients(Loss, var_list=ref)
capped_gvs = [((1 - mask_tf) * grad, var) for grad, var in gvs]
op_I = optimizer.apply_gradients(capped_gvs)

op_w = tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.5, beta2=0.5).minimize(-Loss, var_list=var_list)


sess = tf.Session()
sess.run(tf.global_variables_initializer())

G_vars = tf.global_variables()
G_g_vars = [var for var in G_vars if 'RMSProp' not in var.name and 'Adam' not in var.name and 'output' not in var.name]

start_time = time.time()


for i in range(args.Iter+1):

	for _ in range(args.inner):
		sess.run(op_I)

	sess.run(op_w)

	current_time = time.time()
	out = sess.run([ref, Loss] + LOSS_LIST)
	print(i, out[1], out[2:], 'already used %ds' % (current_time - start_time))

	if i % 100 == 0 and i < 1000 or i % 1000 == 0:
		tmp = (np.clip(out[0][0] + AVE, 0, 1) * 255.).astype(np.uint8)
		util.saveSampleVideo(tmp, 'Inpainted/' + save_name, global_step=i)
