import time
from ops import *
from tensorflow.contrib.data import prefetch_to_device, shuffle_and_repeat, map_and_batch
import tensorflow as tf


def res_block(inputres, scope="resnet", sn=False, pad_type='zero'):
    with tf.variable_scope(scope):
        ch = inputres.shape[-1].value
        out_res = conv(inputres, channels=ch, kernel=3, stride=1, pad=1, sn=sn, scope='res1', pad_type=pad_type)
        out_res = conv(out_res,  channels=ch, kernel=3, stride=1, pad=1, sn=sn, scope='res2', pad_type=pad_type)

        return tf.nn.relu(out_res + inputres)


def conv1(x, channels, scope='Conv3',pad_type='zero', sn=False, is_training=True):

    with tf.variable_scope(scope, reuse=False):
        x = conv(x, channels=channels, kernel=3, stride=1, pad=1, sn=sn, scope='1st_conv', pad_type=pad_type)
        x = batch_norm(x, is_training, scope='1st_bn')
        x = relu(x)
    return x

def pyramid_tf20(size, noise_depth, batch, is_training=True, pad_type='zero', sn=False, is_normal=False):
    # extra two res block
    ratio_list = [32, 16, 8, 4, 2, 1]
    num_filter = [8, 8, 8, 8, 8, 8]
    if is_normal:
        noise_list = [tf.random_normal((batch, int(size//i), int(size//i), noise_depth)) for i in ratio_list]
    else:
        noise_list = [tf.random_uniform((batch, int(size//i), int(size//i), noise_depth)) for i in ratio_list]
    with tf.variable_scope('generator', reuse=False):
        for i in range(len(ratio_list)):
            seq = conv1(noise_list[i], num_filter[i], is_training=is_training, pad_type=pad_type, sn=sn, scope='Conv3_seq_' + str(i))
            if i == 0:
                seq = up_sample(seq, scale_factor=2)
                cur = seq
            else:
                cur_tmp = cur
                seq = batch_norm(seq, is_training, scope='bn_seq_' + str(i))
                cur_tmp = batch_norm(cur_tmp, is_training, scope='bn_cur_tmp_' + str(i))
                cur = tf.concat([cur_tmp, seq], -1)
                
                cur = conv1(cur, cur.shape[-1].value, is_training=is_training, pad_type=pad_type, sn=sn, scope='Conv3_cur_' + str(i))
    
                if i == len(ratio_list) - 1:
                    cur = res_block(cur, scope="res1", sn=sn, pad_type=pad_type)
                    cur = res_block(cur, scope="res2", sn=sn, pad_type=pad_type)
                    cur = conv(cur, channels=3, kernel=3, stride=1, pad=1, sn=sn, scope='G_conv_logit', pad_type=pad_type)
                    cur = tanh(cur)
                else:
                    cur = up_sample(cur, scale_factor=2)
        return cur
