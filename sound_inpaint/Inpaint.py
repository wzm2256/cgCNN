import tensorflow as tf
import numpy as np
import keras
import Model
import argparse
import matplotlib.pyplot as plt
import time
import librosa
import os

parser = argparse.ArgumentParser(description='Sound inpainting')

parser.add_argument('--image', type=str, help='audio name')
parser.add_argument('--mean', type=int, default=1, help='use mean or Gram')
parser.add_argument('--inner', type=int, default=10, help='Number of Langevin steps in each iteration')
parser.add_argument('--fou', type=float, default=1, help='Fourier norm penalty')
parser.add_argument('--Iter', type=int, default=1000, help='Number of iterations')

args = parser.parse_args()


if args.mean == 0:
	raise NotImplementedError

Scale = 1e12

im_dir = './Sound/'

save_name = args.image + '_' + '_mean_' + str(args.mean) + '_inner_' + str(args.inner) + '_fou_' + str(args.fou)

################################################# Load audio
image_path1 = im_dir + args.image

I, fs = librosa.load(image_path1)

if len(I.shape) == 2:
	I = I[:,0]

img_nrows = 50000
assert len(I > img_nrows)
I = I[:img_nrows]
I_ori1 = I.copy()
M = np.max(I)
m = np.min(I)

I = (I - m) / (M - m) - 0.5

I_ori = (I + 0.5)  * (M - m) + m

if not os.path.isfile('Inpainted/' + args.image):
	librosa.output.write_wav('Inpainted/Ori' + args.image, I_ori, fs)

plt.ylim(m - 0.1 * np.abs(m), M + 0.1 * np.abs(M))
plt.plot(I_ori)
plt.savefig('Inpainted/Ori_' + args.image + '.jpg')
plt.close()


################# construct mask   #####################
loaded_mask_array1 = np.ones_like(I)
masked_start = 20000
masked_end   = 30000

loaded_mask_array1[masked_start:masked_end]=0
mask_tf = tf.expand_dims(tf.expand_dims(tf.constant(loaded_mask_array1, dtype=tf.float32), axis=0), axis=-1)

####################################### masked audio

masked_I =  I * loaded_mask_array1

ori_masked = (masked_I + 0.5)  * (M - m) + m


librosa.output.write_wav('Inpainted/masked_' + args.image, ori_masked, fs)


plt.ylim(m - 0.1 * np.abs(m), M + 0.1 * np.abs(M))
plt.plot(ori_masked)
plt.savefig('Inpainted/masked_' + args.image + '.jpg')
plt.close()

################################### prepare all patches

def List_bound(bound_size, ex_bound=None, n_stride=63):
	x_l_b = bound_size
	if ex_bound != None:
		x_0_e, x_l_e = ex_bound
	else:
		x_0_e, x_l_e = [100, 1]

	n = 0
	boxes = np.zeros((n_stride, 4))

	for x_i in np.linspace(0, 1-x_l_b, n_stride):
		if not (x_i > x_0_e - x_l_b and x_i < x_0_e + x_l_e):
			boxes[n,0] = 0
			boxes[n,1] = x_i
			boxes[n,2] = 1
			boxes[n,3] = x_i + x_l_b
			n += 1
	
	Boxes = boxes[0:n]

	return Boxes


I_np = np.expand_dims(np.expand_dims(masked_I, axis=0), axis=-1)
ref = tf.Variable(I_np, dtype=tf.float32)
ref_crop = tf.expand_dims(ref, axis=0)

y_0 = 0.
y_l = 1.
border = 1000
x_0 = (masked_start - border) / img_nrows
x_l = (masked_end - masked_start + border * 2) / img_nrows



Large_Boxes = List_bound(x_l, ex_bound=[x_0,x_l], n_stride=1280)
np.random.shuffle(Large_Boxes)



bound_np = np.array([[y_0, x_0, y_0 + y_l, x_0 + x_l]])
bound_tf = tf.constant(bound_np, dtype=tf.float32)

box_x = tf.Variable(Large_Boxes[0:1].astype(np.float32))

Init_Box = tf.concat([bound_tf, box_x], axis=0)


box_ind = tf.constant(np.zeros(2).astype(np.int32))

crop_size_np = np.array([1, x_l * img_nrows]).astype(np.float32).astype(np.int32)
crop_size = tf.constant(crop_size_np, dtype=tf.int32)


print(ref.shape)
Patches_ = tf.image.crop_and_resize(ref_crop, Init_Box, box_ind=box_ind, crop_size=crop_size)
Patches = Patches_[:, 0, :, :]
print(Patches.shape)

###########################################  construct model


def Select_Patch(model, ref, boxes, ref_box):

	boxes_tf = tf.constant(boxes, dtype=tf.float32)
	ref_box_tf = tf.constant(ref_box, dtype=tf.float32)

	boxes_all = tf.concat([ref_box_tf, boxes_tf], axis=0)

	box_ind = tf.constant(np.zeros(boxes.shape[0] + 1).astype(np.int32))

	y_l = boxes[0, 2] - boxes[0, 0]
	x_l = boxes[0, 3] - boxes[0, 1]

	crop_size_np = np.array([1, x_l * img_nrows]).astype(np.float32).astype(np.int32)
	crop_size = tf.constant(crop_size_np, dtype=tf.int32)

	Patches_ = tf.image.crop_and_resize(ref, boxes_all, box_ind=box_ind, crop_size=crop_size)
	Patches = Patches_[:, 0, :, :]

	_, h_dict, _ = model.run(Patches)


	LOSS_LIST = []
	for layer in layer_name:
		tmp_loss = mean_loss_batch(h_dict[layer][0:1], h_dict[layer][1:])
		LOSS_LIST.append(tmp_loss)

	Loss = tf.add_n(LOSS_LIST)
	Loss *= Scale

	small_index = tf.argmin(Loss)

	return boxes_tf[small_index: small_index + 1], Loss, small_index


model = Model.Sound_inpaint()

_, h_dict, var_list = model.run(Patches)

h_list_ref = list(h_dict.values()) 




layer_name = ['relu1', 'relu2', 'relu3']


def gram_matrix(feature_maps):
	"""Computes the Gram matrix for a set of feature maps."""
	batch_size, height, channels = tf.unstack(tf.shape(feature_maps))
	denominator = tf.to_float(height)
	feature_maps = tf.reshape(feature_maps, tf.stack([batch_size, height, channels]))
	matrix = tf.matmul(feature_maps, feature_maps, adjoint_a=True)
	return matrix / denominator


# def gram_loss(feature1, reference):
# 	# batch = feature1.get_shape[0].value
# 	F1 = gram_matrix(feature1)
# 	F2 = gram_matrix(reference)
# 	loss = tf.reduce_mean((F1 - F2) ** 2)
# 	return loss

def mean_loss(feature1, reference):
	m1 = tf.reduce_mean(feature1, axis=1)
	m2 = tf.reduce_mean(reference, axis=1)
	loss = tf.reduce_mean(tf.square(m1 - m2))
	return loss

def mean_loss_batch(feature1, reference):
	m1 = tf.reduce_mean(feature1, axis=1)
	m2 = tf.reduce_mean(reference, axis=1)
	loss = tf.reduce_mean(tf.square(m1 - m2), axis=-1)
	return loss


def spectral_loss(feature1, ref):
	f1_trans = tf.transpose(feature1, (0,2,1))
	f2_trans = tf.transpose(ref, (0,2,1))

	m1 = tf.abs(tf.fft(tf.cast(f1_trans, tf.complex64)))
	m2 = tf.abs(tf.fft(tf.cast(f2_trans, tf.complex64)))

	loss = tf.reduce_mean(tf.square(m1 - m2))
	return loss


LOSS_LIST = []
for layer in layer_name:
	if args.mean == 0:
		raise NotImplementedError
		# tmp_loss = gram_loss(h_dict[layer][0:1], h_dict[layer][1:])
	else:
		tmp_loss = mean_loss(h_dict[layer][0:1], h_dict[layer][1:])
	LOSS_LIST.append(tmp_loss)




fou_loss = spectral_loss(Patches[0:1], Patches[1:]) * args.fou

LOSS_LIST.append(fou_loss)

Loss = tf.add_n(LOSS_LIST)

Loss *= Scale



optimizer = tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.5, beta2=0.5)
gvs = optimizer.compute_gradients(Loss, var_list=ref)
capped_gvs = [((1 - mask_tf) * grad, var) for grad, var in gvs]
op_I = optimizer.apply_gradients(capped_gvs)

op_w = tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.5, beta2=0.5).minimize(-Loss, var_list=var_list)


# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True

sess = tf.Session()
# sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())

G_vars = tf.global_variables()
G_g_vars = [var for var in G_vars if 'RMSProp' not in var.name and 'Adam' not in var.name and 'output' not in var.name]

start_time = time.time()


for i in range(args.Iter+1):
	if i % 100 == 1:
		tmp, debug_loss, debug_argmin = Select_Patch(model, ref_crop, Large_Boxes, bound_np)
		assign_box1 = tf.assign(box_x, tmp)
		sess.run(assign_box1)

	for _ in range(args.inner):
		sess.run(op_I)

	sess.run(op_w)

	current_time = time.time()
	out = sess.run([ref, Loss] + LOSS_LIST)
	print(i, out[1], out[2:], 'already used %ds' % (current_time - start_time))

	if i % 100 == 0 and i < 1000 or i % 1000 == 0:
		tmp = np.clip(out[0][0], -1, 1)
		tmp = (tmp + 0.5) * (M - m) + m
		librosa.output.write_wav('Inpainted/' + save_name + str(i) + '_.wav', tmp, fs)
		plt.ylim(m - 0.1 * np.abs(m), M + 0.1 * np.abs(M))
		plt.plot(tmp)
		plt.savefig('Inpainted/' + save_name + '_' + str(i) + '_.jpg')
		plt.close()
