import tensorflow as tf
import numpy as np
import keras
import Model
import PIL.Image as Image
import argparse
import matplotlib.pyplot as plt
import time
import Generator
import librosa
import Generator

parser = argparse.ArgumentParser(description='Sound texture expansion. Test phase')
parser.add_argument('image1', type=str, help='audio name')
parser.add_argument('--Iter', type=int, default=1000, required=False, help='number of iterations to run.')
parser.add_argument('--layer_D', type=int, default=4, help='number of deep layers')
parser.add_argument('--Adam', type=int, default=0, help='use Adam|Rmsprop')
parser.add_argument('--mean', type=int, default=0, help='use mean|Gram')
parser.add_argument('--Model', type=int, default=2, help='which model')
parser.add_argument('--inner', type=int, default=10, help='number of Langevin steps in each iteration')
parser.add_argument('--Gau', type=float, default=0.0, help='Gaussian penalty')
parser.add_argument('--sn', type=float, default=0, help='Fourier norm penalty')
parser.add_argument('--diversity', type=str, default='No', help='diversity penalty in which layer')
parser.add_argument('--d_weight', type=float, default=0, help='diversity penalty')
parser.add_argument('--step', type=int, default=0, help='which model to use')

args = parser.parse_args()

img_nrows = 24576

x = Generator.pyramid_tf(img_nrows * 5, 8, 1, is_training=False)


saver = tf.train.Saver()
sess = tf.Session()

save_name = args.image1 + '_depth_' + str(args.layer_D) + \
			'_inner_' + str(args.inner) + '_IsMean_' + str(args.mean) + '_Adam_' + str(args.Adam) + \
			'_Gau_' + str(args.Gau) + '_diversity_' + args.diversity + '_d_weight_' + str(args.d_weight)

checkpoint_dir = './Saved_model/' + save_name + '/' + '-' + str(args.step)
saver.restore(sess, checkpoint_dir)

out = sess.run(x)

arg_dir = './Saved_model/' + save_name + '/'
m, M, fs = np.load(arg_dir + 'args.npy')
fs = int(fs)

tmp = np.clip(out[0], -1, 1)

tmp = (tmp + 0.5) * (M - m) + m

librosa.output.write_wav('Produce_Generator/' + save_name + '.wav', tmp, fs)

