import tensorflow as tf
import numpy as np
import keras
import Model
import PIL.Image as Image
import argparse
import matplotlib.pyplot as plt
import time


parser = argparse.ArgumentParser(description='Neural style transfer with Keras.')

parser.add_argument('--image', type=str, default='022.jpg', help='original image')
parser.add_argument('--mask', type=str, default='mask2.png', help='mask')
parser.add_argument('--mean', type=int, default=1, help='use mean or Gram')
parser.add_argument('--inner', type=int, default=10, help='Number of Langevin steps in each iteration')
parser.add_argument('--tv', type=float, default=0.00000000, help='total variation penalty')
parser.add_argument('--fou', type=float, default=0.0001, help='Fourier norm penalty')
parser.add_argument('--Iter', type=int, default=1000, help='Number of iterations')

args = parser.parse_args()

if args.mean == 0:
	raise NotImplementedError

Scale = 1e12

im_dir = './Image/'

save_name = args.image + '_' + args.mask + '_mean_' + str(args.mean) + '_inner_' + str(args.inner) + '_tv_' + str(args.tv) + '_fou_' + str(args.fou)

# load image
image_path1 = im_dir + args.image
I = Image.open(image_path1).convert('RGB').resize((256,256))
loaded_image_array1 = np.asarray(I, dtype=np.float32) / 255
img_nrows, img_ncols, _ = loaded_image_array1.shape

# load mask
mask_path1 = im_dir + args.mask
I_m = Image.open(mask_path1).convert('L').resize((256,256))
I_m_np = np.asarray(I_m, dtype=np.float32) / 255
loaded_mask_array1 = (I_m_np > 0.5).astype(np.float32)
loaded_mask_array1 = np.expand_dims(loaded_mask_array1, axis=-1)
mask_tf = tf.constant(loaded_mask_array1, dtype=tf.float32)
img_nrows_m, img_ncols_m, _ = loaded_mask_array1.shape

# prepare mased image
masked_image_array = loaded_image_array1 * loaded_mask_array1

AVE = np.sum(masked_image_array, axis=(0,1), keepdims=True) / np.sum(loaded_mask_array1)


# fill corrputed image with average color
Init = masked_image_array + (1-loaded_mask_array1) * AVE

plt.imsave('Inpainted/masked_' + args.image, masked_image_array)


def bound_box(mask, border=8):

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

	y_0 = (min_0/mask_1.shape[0]).astype(np.float32)
	y_l = ((max_0 - min_0)/mask_1.shape[0]).astype(np.float32)
	x_0 = (min_1/mask_1.shape[1]).astype(np.float32)
	x_l = ((max_1-min_0)/mask_1.shape[1]).astype(np.float32)
	return y_0, y_l, x_0, x_l 

def List_bound(bound_size, ex_bound=None, n_stride=63):
	y_l_b, x_l_b = bound_size
	if ex_bound != None:
		y_0_e, y_l_e, x_0_e, x_l_e = ex_bound
	else:
		y_0_e, y_l_e, x_0_e, x_l_e = [100, 1 , 100, 1]

	n = 0
	boxes = np.zeros((n_stride * n_stride, 4))
	for y_i in np.linspace(0, 1-y_l_b, n_stride):
		for x_i in np.linspace(0, 1-x_l_b, n_stride):
			if not (y_i > y_0_e - y_l_b and y_i < y_0_e + y_l_e and x_i > x_0_e - x_l_b and x_i < x_0_e + x_l_e):
				boxes[n,0] = y_i
				boxes[n,1] = x_i
				boxes[n,2] = y_i + y_l_b
				boxes[n,3] = x_i + x_l_b
				n += 1
	
	Boxes = boxes[0:n]

	return Boxes




image_processed = Init - AVE

image_tf = tf.Variable(image_processed, dtype=tf.float32)
ref = tf.expand_dims(image_tf, axis=0)

y_0,y_l,x_0,x_l = bound_box(loaded_mask_array1)


Large_Boxes = List_bound([y_l, x_l], ex_bound=[y_0,y_l,x_0,x_l], n_stride=16)
np.random.shuffle(Large_Boxes)


bound_np = np.array([[y_0, x_0, y_0 + y_l, x_0 + x_l]])
bound_tf = tf.constant(bound_np, dtype=tf.float32)

box_x = tf.Variable(Large_Boxes[0:1].astype(np.float32))

Init_Box = tf.concat([bound_tf, box_x], axis=0)


box_ind = tf.constant(np.zeros(2).astype(np.int32))

crop_size_np = np.array([y_l * img_nrows, x_l * img_ncols]).astype(np.float32).astype(np.int32)
crop_size = tf.constant(crop_size_np, dtype=tf.int32)

# prepare all patches
Patches = tf.image.crop_and_resize(ref, Init_Box, box_ind=box_ind, crop_size=crop_size)


def Select_Patch(model, ref, boxes, ref_box):

	boxes_tf = tf.constant(boxes, dtype=tf.float32)
	ref_box_tf = tf.constant(ref_box, dtype=tf.float32)

	boxes_all = tf.concat([ref_box_tf, boxes_tf], axis=0)

	box_ind = tf.constant(np.zeros(boxes.shape[0] + 1).astype(np.int32))

	y_l = boxes[0, 2] - boxes[0, 0]
	x_l = boxes[0, 3] - boxes[0, 1]

	crop_size_np = np.array([y_l * img_nrows, x_l * img_ncols]).astype(np.float32).astype(np.int32)
	crop_size = tf.constant(crop_size_np, dtype=tf.int32)

	Patches = tf.image.crop_and_resize(ref, boxes_all, box_ind=box_ind, crop_size=crop_size)

	_, h_dict, _ = model.run(Patches)


	LOSS_LIST = []
	for layer in layer_name:
		tmp_loss = mean_loss_batch(h_dict[layer][0:1], h_dict[layer][1:])
		LOSS_LIST.append(tmp_loss)

	
	fou_loss = spectral_loss_batch(Patches[0:1], Patches[1:]) * args.fou
	LOSS_LIST.append(fou_loss)

	Loss = tf.add_n(LOSS_LIST)
	Loss *= Scale

	small_index = tf.argmin(Loss)

	return boxes_tf[small_index: small_index + 1], Loss, small_index



model = Model.small_1()

_, h_dict, var_list = model.run(Patches)

h_list_ref = list(h_dict.values())


layer_name = ['relu1', 'relu2', 'block1_pool', 'relu3']


# def gram_matrix(feature_maps):
# 	"""Computes the Gram matrix for a set of feature maps."""
# 	batch_size, height, width, channels = tf.unstack(tf.shape(feature_maps))
# 	denominator = tf.to_float(height * width)
# 	feature_maps = tf.reshape(feature_maps, tf.stack([batch_size, height * width, channels]))
# 	matrix = tf.matmul(feature_maps, feature_maps, adjoint_a=True)
# 	return matrix / denominator


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

def mean_loss_batch(feature1, reference):
	m1 = tf.reduce_mean(feature1, axis=(1,2))
	m2 = tf.reduce_mean(reference, axis=(1,2))
	loss = tf.reduce_mean(tf.square(m1 - m2), axis=-1)
	return loss


def spectral_loss(feature1, ref):
	f1_trans = tf.transpose(feature1, (0,3,1,2))
	f2_trans = tf.transpose(ref, (0,3,1,2))

	m1 = tf.abs(tf.fft2d(tf.cast(f1_trans, tf.complex64)))
	m2 = tf.abs(tf.fft2d(tf.cast(f2_trans, tf.complex64)))

	loss = tf.reduce_mean(tf.square(m1 - m2))
	return loss

def spectral_loss_batch(feature1, ref):
	f1_trans = tf.transpose(feature1, (0,3,1,2))
	f2_trans = tf.transpose(ref, (0,3,1,2))

	m1 = tf.abs(tf.fft2d(tf.cast(f1_trans, tf.complex64)))
	m2 = tf.abs(tf.fft2d(tf.cast(f2_trans, tf.complex64)))

	loss = tf.reduce_mean(tf.square(m1 - m2), axis=(1,2,3))
	return loss


LOSS_LIST = []
for layer in layer_name:
	if args.mean == 0:
		tmp_loss = gram_loss(h_dict[layer][0:1], h_dict[layer][1:])
	else:
		tmp_loss = mean_loss(h_dict[layer][0:1], h_dict[layer][1:])
	LOSS_LIST.append(tmp_loss)



tv_loss =  tf.reduce_sum(tf.image.total_variation(ref) * args.tv)

fou_loss = spectral_loss(Patches[0:1], Patches[1:]) * args.fou

LOSS_LIST.append(tv_loss)
LOSS_LIST.append(fou_loss)

Loss = tf.add_n(LOSS_LIST)

Loss *= Scale


optimizer = tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.5, beta2=0.5)
gvs = optimizer.compute_gradients(Loss, var_list=image_tf)
capped_gvs = [((1 - mask_tf) * grad, var) for grad, var in gvs]
op_I = optimizer.apply_gradients(capped_gvs)

op_w = tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.5, beta2=0.5).minimize(-Loss, var_list=var_list)

sess = tf.Session()

sess.run(tf.global_variables_initializer())


G_vars = tf.global_variables()
G_g_vars = [var for var in G_vars if 'RMSProp' not in var.name and 'Adam' not in var.name and 'output' not in var.name]

start_time = time.time()



for i in range(args.Iter+1):

	if i % 100 == 0:
		tmp, debug_loss, debug_argmin = Select_Patch(model, ref, Large_Boxes, bound_np)
		assign_box1 = tf.assign(box_x, tmp)
		sess.run(assign_box1)

	for _ in range(args.inner):
		sess.run(op_I)

	sess.run(op_w)

	current_time = time.time()
	out = sess.run([ref, Loss] + LOSS_LIST)

	print(i, out[1], out[2:], 'already used %ds' % (current_time - start_time))

	if i % 100 == 0 and i < 1000 or i % 1000 == 0:

		tmp = (np.clip(out[0][0] + AVE, 0, 1) * 255.).astype(np.uint8)
		plt.imsave('Inpainted/' + save_name + str(i) + '_' + '.jpg', tmp)