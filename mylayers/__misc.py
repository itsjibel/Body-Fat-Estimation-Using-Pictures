import tensorflow as tf
import keras
import keras.backend as K

def resize_images(*args, **kwargs):
    return tf.image.resize_images(*args, **kwargs)

class DeformableDeConv(keras.layers.Layer):
	def __init__(self, kernel_size, stride, filter_num, *args, **kwargs):
		self.stride = stride
		self.filter_num = filter_num
		self.kernel_size =kernel_size
		super(DeformableDeConv, self).__init__(*args,**kwargs)

	def build(self, input_shape):
		# Create a trainable weight variable for this layer.
		in_filters = self.filter_num
		out_filters = self.filter_num
		self.kernel = self.add_weight(name='kernel',
									  shape=[self.kernel_size, self.kernel_size, out_filters, in_filters],
									  initializer='uniform',
									  trainable=True)

		super(DeformableDeConv, self).build(input_shape)

	def call(self, inputs, **kwargs):
		source, target = inputs
		target_shape = K.shape(target)
		return tf.nn.conv2d_transpose(source, 
									self.kernel, 
									output_shape=target_shape, 
									strides=self.stride, 
									padding='SAME', 
									data_format='NHWC')
	def get_config(self):
		config = {'kernel_size': self.kernel_size, 'stride': self.stride, 'filter_num': self.filter_num}
		base_config = super(DeformableDeConv, self).get_config()
		return dict(list(base_config.items()) + list(config.items()))

class UpsampleLike(keras.layers.Layer):
    def call(self, inputs, **kwargs):
        source, target = inputs
        target_shape = K.shape(target)
        return resize_images(source, (target_shape[1], target_shape[2]))

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0],) + input_shape[1][1:3] + (input_shape[0][-1],)

class ScalingLayer(keras.layers.Layer):
    def call(self, inputs, **kwargs):
        source, target = inputs
        target_shape = K.shape(target)
        return resize_images(source, (target_shape[1], target_shape[2]))

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0],) + input_shape[1][1:3] + (input_shape[0][-1],)
