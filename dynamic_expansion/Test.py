import tensorflow as tf
import numpy as np
import keras
import Model
import PIL.Image as Image
import argparse
import matplotlib.pyplot as plt
import time
import Generator
import util
import os

parser = argparse.ArgumentParser(description='Dynamic texture expansion. Test phase.')
parser.add_argument('video', type=str, help='video folder')
parser.add_argument('--size_t', type=int, default=None, help='output temporal size')
parser.add_argument('--size_s', type=int, default=None, help='output spatial size')
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
parser.add_argument('--step', type=int, default=5000, help='which model to use')

args = parser.parse_args()

Out_name = os.path.basename(args.video)

x_out_ = Generator.pyramid_tf((args.size_t,args.size_s), 3, 1, is_training=False, pad_type=args.pad_type, is_normal=args.normal)
x_out = (x_out_ + 1.0) / 2.0

saver = tf.train.Saver()
sess = tf.Session()

save_name = Out_name + '_depth_' + str(args.layer_D) + \
			'_inner_' + str(args.inner) + '_IsMean_' + str(args.mean) + '_Adam_' + str(args.Adam) + '_padtype_' + args.pad_type + \
			'_normal_' + str(args.normal) + '_Gau_' + str(args.Gau) + '_diversity_' + args.diversity + '_d_weight_' + str(args.d_weight)

checkpoint_dir = './Saved_model/' + save_name + '/' + '-' + str(args.step)
saver.restore(sess, checkpoint_dir)

tmp = sess.run(x_out)

tmp = (np.clip(tmp, 0., 1.) * 255.).astype(np.uint8)
tmp = tmp[:, 2:tmp.shape[1]-2, 32:tmp.shape[2]-32, 32:tmp.shape[3]-32,  :]
util.saveSampleVideo(tmp, 'Produce_Generator/' + save_name, global_step=args.step)
