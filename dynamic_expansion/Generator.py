import tensorflow as tf
from keras.layers import Conv3D, Activation, UpSampling3D, Conv3DTranspose
import tensorflow.contrib as tf_contrib

def batch_norm(x, is_training=True, scope='batch_norm'):
    return tf_contrib.layers.batch_norm(x,
                                        decay=0.9, epsilon=1e-05,
                                        center=True, scale=True, updates_collections=None,
                                        is_training=is_training, scope=scope)

def conv3d_pad(x, channels, filters=(3,3,3), strides=1, pad=1, scope='conv3d', pad_type='zero'):
    #pad and conv
    # 3,3,3
    with tf.variable_scope(scope, reuse=False):
        if pad_type == 'zero' :
            x = tf.pad(x, [[0, 0], [pad, pad], [pad, pad], [pad, pad], [0, 0]])
        elif pad_type == 'reflect' :
            x = tf.pad(x, [[0, 0], [pad, pad], [pad, pad], [pad, pad], [0, 0]], mode='REFLECT')
        else:
            raise NotImplementedError
        out = Conv3D(channels, filters, strides=strides, name='conv')(x)
        return out

def conv1_pad(x, channels, scope='Conv1', pad_type='zero', is_training=True):
    # 3,3,3
    with tf.variable_scope(scope, reuse=False):
        x = conv3d_pad(x, channels, filters=(3,3,3), strides=1, pad=1, pad_type='zero', scope='conv1_conv')
        x = batch_norm(x, is_training=is_training, scope='1st_bn')
        x = Activation("relu", name='relu')(x)
    return x

def res_block_pad(inputres, scope="resnet", pad_type='zero'):
    with tf.variable_scope(scope):
        ch = inputres.shape[-1].value
        out_res = conv3d_pad(inputres, channels=ch, filters=(3,3,3), strides=1, pad=1, scope='res1', pad_type=pad_type)
        out_res = conv3d_pad(out_res,  channels=ch, filters=(3,3,3), strides=1, pad=1, scope='res2', pad_type=pad_type)
        return tf.nn.relu(out_res + inputres)

def pyramid_tf(size, noise_depth, batch, is_training=True, pad_type='zero', sn=False, is_normal=False):
    ratio_list_s = [16, 8, 4, 2, 1]
    ratio_list_t = [1, 1, 1, 1, 1]
    num_filter = [8, 8, 8, 8, 8]
    if is_normal:
        noise_list = [tf.random_normal((batch, int(size[0]//ratio_list_t[i]), int(size[1]//ratio_list_s[i]), 
                                                int(size[1]//ratio_list_s[i]), noise_depth)) for i in range(len(ratio_list_s))]
    else:
        noise_list = [tf.random_uniform((batch, int(size[0]//ratio_list_t[i]), int(size[1]//ratio_list_s[i]), 
                                                int(size[1]//ratio_list_s[i]), noise_depth)) for i in range(len(ratio_list_s))]

    with tf.variable_scope('generator', reuse=False):
        for i in range(len(ratio_list_s)):
            seq = conv1_pad(noise_list[i], num_filter[i], is_training=is_training, pad_type=pad_type, scope='Conv3_seq_' + str(i))
            if i == 0:
                seq = UpSampling3D(size=(1,2,2))(seq)
                cur = seq
            else:
                cur_tmp = cur
                seq = batch_norm(seq, is_training=is_training, scope='bn_seq_' + str(i))
                cur_tmp = batch_norm(cur_tmp, is_training=is_training, scope='bn_cur_tmp_' + str(i))
                cur = tf.concat([cur_tmp, seq], -1)
                
                cur = conv1_pad(cur, cur.shape[-1].value, is_training=is_training, pad_type=pad_type, scope='Conv3_cur_' + str(i))
    
                if i == len(ratio_list_s) - 1:
                    cur = res_block_pad(cur, scope="res1",  pad_type=pad_type)
                    cur = res_block_pad(cur, scope="res2",  pad_type=pad_type)
                    cur = conv3d_pad(cur, channels=3, filters=(1,1,1), strides=1, pad=0, scope='G_conv_logit', pad_type=pad_type)
                    cur = tf.tanh(cur)
                else:
                    cur = UpSampling3D(size=(1,2,2))(cur)
        return cur