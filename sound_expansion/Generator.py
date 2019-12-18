import tensorflow as tf
import tensorflow.contrib as tf_contrib
from keras.layers import Conv1D, Activation, UpSampling1D

def batch_norm(x, is_training=True, scope='batch_norm'):
	return tf_contrib.layers.batch_norm(x,
										decay=0.9, epsilon=1e-05,
										center=True, scale=True, updates_collections=None,
										is_training=is_training, scope=scope)

def Cubic_Upsampling1D(x, scale=2):
	batch, length, channel = x.shape.as_list()
	x_e = tf.expand_dims(x, axis=-1)
	x_e_h = tf.image.resize_bicubic(x_e, (scale * length, channel))
	return x_e_h[:,:,:,0]

def conv(x, channels, scope='Conv1', kernel_size=25, strides=3, dilation_rate=1, pad_type='zero', is_training=True):
	# kernel 25
	# stride 3
	with tf.variable_scope(scope, reuse=False):
		x = Conv1D(filters=channels, kernel_size=kernel_size, strides=strides, dilation_rate=dilation_rate, padding='same', name='conv1_conv')(x)
		# x = BatchNormalization(name='1st_bn')(x, training=is_training)
		x = batch_norm(x, is_training, scope='1st_bn')
		x = Activation("relu", name='relu')(x)
	return x

def pyramid_tf(size, noise_depth, batch, is_training=True, pad_type='zero', sn=False, is_normal=False):

	ratio_list_t = [1024, 256, 64, 16, 4, 1]
	num_filter = [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8]

	if is_normal:
		noise_list = [tf.random_normal((batch, int(size//ratio_list_t[i]),
												noise_depth)) for i in range(len(ratio_list_t))]
	else:
		noise_list = [tf.random_uniform((batch, int(size//ratio_list_t[i]),
												noise_depth)) for i in range(len(ratio_list_t))]

	with tf.variable_scope('generator', reuse=False):
		for i in range(len(ratio_list_t)):
			seq = conv(noise_list[i], num_filter[i], kernel_size=7, strides=1, dilation_rate=1, is_training=is_training, pad_type=pad_type, scope='Conv3_seq_' + str(i))

			if i == 0:
				seq = Cubic_Upsampling1D(seq, scale=4)
				cur = seq
			else:
				seq = batch_norm(seq, is_training, scope='bn_seq_' + str(i))
				cur = batch_norm(cur, is_training, scope='bn_cur_' + str(i))

				cur = tf.concat([cur, seq], -1)
				cur = conv(cur, cur.shape[-1].value, kernel_size=7, dilation_rate=1, strides=1, is_training=is_training, pad_type=pad_type, scope='Conv3_cur_' + str(i))
				if i == len(ratio_list_t) - 1:
					cur = Conv1D(filters=1, kernel_size=7, strides=1, dilation_rate=1, padding='same', name='G_conv_logit')(cur)
				else:
					cur = UpSampling1D(size=4)(cur)
		return cur
