import tensorflow as tf
import numpy as np
import keras
import Model
import PIL.Image as Image
import argparse
import matplotlib.pyplot as plt
import time
import Generator


parser = argparse.ArgumentParser(description='Image texture expansion. Test phase.')
parser.add_argument('image1', type=str, help='image name')
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

parser.add_argument('--step', type=int, default=5000, help='which model to use')
parser.add_argument('--size', type=int, default=None, help='output size')

args = parser.parse_args()

pad_type = 'zero'
sn = False
samples = 1

noise_depth = 3
x_out_ = Generator.pyramid_tf20(args.size, noise_depth, samples, sn=sn, pad_type=pad_type, is_training=False, is_normal=args.normal)

x_out = (x_out_ + 1.0) / 2.0



saver = tf.train.Saver()
sess = tf.Session()

save_name = args.image1 + '_layer_S_' + str(args.layer_S) + '_layer_D_' + str(args.layer_D) + \
	'_inner_' + str(args.inner) + '_IsMean_' + str(args.mean) + '_Adam_' + str(args.Adam) + \
	'_normal_' + str(args.normal) + '_Gau_' + str(args.Gau) + '_Fou_' + str(args.Fou) + \
	'_diversity_' + args.diversity + '_d_weight_' + str(args.d_weight)



checkpoint_dir = './Saved_model/' + save_name + '/' + '-' + str(args.step)
saver.restore(sess, checkpoint_dir)

tmp = sess.run(x_out)


for i in range(tmp.shape[0]):
	tmp = np.clip(tmp[i], 0., 1.)
	tmp = tmp[64: tmp.shape[0]-64, 64:tmp.shape[1]-64, :]
	plt.imsave('Produce_Generator/' + save_name + '_' + str(i) + '_step_' + str(args.step) + '_.jpg', tmp)
