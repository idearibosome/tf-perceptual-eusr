import math
import os

import numpy as np
import tensorflow as tf

from models.base_model import BaseModel

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string('eusr_model_scales', '2,4,8', 'Supported scales of the model. Use the \',\' character to specify multiple scales (e.g., 2,4,8). This parameter is involved in constructing the multi-scale structure of the model.')
tf.flags.DEFINE_integer('eusr_conv_features', 64, 'The number of convolutional features.')
tf.flags.DEFINE_integer('eusr_shared_blocks', 32, 'The number of local residual blocks (LRBs) in the shared feature extraction part.')
tf.flags.DEFINE_integer('eusr_upscale_blocks', 1, 'The number of local residual blocks (LRBs) in the enhanced upscaling modules (EUMs).')
tf.flags.DEFINE_string('eusr_rgb_mean', '114.4,111.5,103.0', 'Mean R, G, and B values of the training images (e.g., 114.4,111.5,103.0).')

tf.flags.DEFINE_float('eusr_learning_rate', 1e-4, 'Initial learning rate.')
tf.flags.DEFINE_float('eusr_learning_rate_decay', 0.5, 'Learning rate decay factor.')
tf.flags.DEFINE_integer('eusr_learning_rate_decay_steps', 200000, 'The number of training steps to perform learning rate decay.')

def create_model():
  return EUSR()

class EUSR(BaseModel):
  def __init__(self):
    super().__init__()
  

  def prepare(self, is_training, global_step=0):
    # config. parameters
    self.global_step = global_step

    self.scale_list = list(map(lambda x: int(x), FLAGS.scales.split(',')))
    self.model_scale_list = list(map(lambda x: int(x), FLAGS.eusr_model_scales.split(',')))
    print(self.model_scale_list)
    for scale in self.scale_list:
      if (not scale in self.model_scale_list):
        raise ValueError('Unsupported scale is provided.')
    for scale in self.model_scale_list:
      if (scale & (scale - 1)) != 0:
        raise ValueError('Unsupported scale is provided.')

    self.num_conv_features = FLAGS.eusr_conv_features
    self.num_shared_blocks = FLAGS.eusr_shared_blocks
    self.num_upscale_blocks = FLAGS.eusr_upscale_blocks

    num_expected_residual_blocks = 0
    num_expected_residual_blocks += (2 * len(self.scale_list)) # scale-specific local residual blocks
    num_expected_residual_blocks += self.num_shared_blocks # shared residual module
    for scale in self.scale_list:
      num_expected_residual_blocks += (int(math.log(scale, 2)) * 4 * self.num_upscale_blocks) # enhanced upscaling modules
    self.num_expected_residual_blocks = num_expected_residual_blocks

    self.shift_mean_list = list(map(lambda x: float(x), FLAGS.eusr_rgb_mean.split(',')))

    if (is_training):
      self.initial_learning_rate = FLAGS.eusr_learning_rate
      self.learning_rate_decay = FLAGS.eusr_learning_rate_decay
      self.learning_rate_decay_steps = FLAGS.eusr_learning_rate_decay_steps


    # tensorflow graph
    self.tf_graph = tf.Graph()
    with self.tf_graph.as_default():

      self.tf_input = tf.placeholder(tf.float32, [None, None, None, 3], name=BaseModel.TF_INPUT_NAME)
      self.tf_scale = tf.placeholder(tf.float32, [], name=BaseModel.TF_INPUT_SCALE_NAME)
      
      self.tf_output = self._generator(input_list=self.tf_input, scale=self.tf_scale)
      
      if (is_training):
        input_summary = tf.cast(self.tf_input, tf.uint8)
        tf.summary.image('input', input_summary)
        output_summary = tf.cast(self.tf_output, tf.uint8)
        tf.summary.image('output', output_summary)
        
        self.tf_truth = tf.placeholder(tf.float32, [None, None, None, 3])
        truth_summary = tf.cast(self.tf_truth, tf.uint8)
        tf.summary.image('truth', truth_summary)

        self.tf_global_step = tf.placeholder(tf.int64, [])
        self.tf_train_op, self.tf_loss = self._optimize(output_list=self.tf_output, truth_list=self.tf_truth, global_step=self.tf_global_step)

        for key, loss in self.loss_dict.items():
          tf.summary.scalar(('loss/%s' % (key)), loss)

        self.tf_saver = tf.train.Saver(max_to_keep=FLAGS.save_max_keep)
        self.tf_summary_op = tf.summary.merge_all()
      
      self.tf_init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())


    # tensorflow session
    self.tf_session = tf.Session(config=tf.ConfigProto(
        log_device_placement=False,
        allow_soft_placement=True
    ), graph=self.tf_graph)
    self.tf_session.run(self.tf_init_op)
      
  
  def save(self, base_path):
    save_path = os.path.join(base_path, 'model.ckpt')
    self.tf_saver.save(sess=self.tf_session, save_path=save_path, global_step=self.global_step)


  def restore(self, ckpt_path, target=None):
    # TODO
    raise NotImplementedError
  

  def get_session(self):
    return self.tf_session
  

  def get_next_train_scale(self):
    scale = self.scale_list[np.random.randint(len(self.scale_list))]
    return scale


  def train_step(self, input_list, scale, truth_list, with_summary=False):
    feed_dict = {}
    feed_dict[self.tf_input] = input_list
    feed_dict[self.tf_scale] = scale
    feed_dict[self.tf_truth] = truth_list
    feed_dict[self.tf_global_step] = self.global_step

    summary = None

    if (with_summary):
      _, loss, summary = self.tf_session.run([self.tf_train_op, self.tf_loss, self.tf_summary_op], feed_dict=feed_dict)
    else:
      _, loss = self.tf_session.run([self.tf_train_op, self.tf_loss], feed_dict=feed_dict)

    self.global_step += 1

    return loss, summary


  def upscale(self, input_list, scale):
    # TODO
    raise NotImplementedError
  
  
  def _mean_shift(self, image_list):
    image_list = image_list - self.shift_mean_list
    return image_list
  
  def _mean_inverse_shift(self, image_list):
    image_list = image_list + self.shift_mean_list
    return image_list
  

  def _conv2d(self, x, num_features, kernel_size, strides=(1, 1)):
    return tf.layers.conv2d(x, filters=num_features, kernel_size=kernel_size, strides=strides, padding='same')
  
  def _conv2d_for_residual_block(self, x, num_features, kernel_size, strides=(1, 1)):
    kernel_initializer = tf.contrib.layers.variance_scaling_initializer(factor=1e-4/self.num_expected_residual_blocks, mode='FAN_IN', uniform=False)
    return tf.layers.conv2d(x, filters=num_features, kernel_size=kernel_size, strides=strides, padding='same', kernel_initializer=kernel_initializer)
  

  def _local_residual_block(self, x, num_features, kernel_size, weight=1.0):
    res = self._conv2d_for_residual_block(x, num_features=num_features, kernel_size=kernel_size)
    res = tf.nn.relu(res)
    res = self._conv2d_for_residual_block(res, num_features=num_features, kernel_size=kernel_size)
    res *= weight
    x = x + res
    return x
  
  def _residual_module(self, x, num_features, num_blocks):
    res = x
    for _ in range(num_blocks):
      res = self._local_residual_block(res, num_features=num_features, kernel_size=(3, 3))
    res = self._conv2d(res, num_features=num_features, kernel_size=(3, 3))
    x = x + res
    return x
  
  def _enhanced_upscaling_module(self, x, scale):
    if (scale & (scale - 1)) != 0:
      raise NotImplementedError
    
    for module_index in range(int(math.log(scale, 2))):
      with tf.variable_scope('m%d' % (module_index)):
        x_list = []
        for pixel_index in range(4):
          with tf.variable_scope('px%d' % (pixel_index)):
            x = self._residual_module(x, num_features=self.num_conv_features, num_blocks=self.num_upscale_blocks)
            x_list.append(x)
        x = tf.concat(x_list, axis=3)
        x = tf.depth_to_space(x, 2)
    
    return x

  
  def _scale_specific_processing(self, x, scale):
    with tf.variable_scope('x%d' % (scale)):
      x = self._local_residual_block(x, num_features=self.num_conv_features, kernel_size=(5, 5))
      x = self._local_residual_block(x, num_features=self.num_conv_features, kernel_size=(5, 5))
    return x
  
  def _scale_specific_upsampling(self, x, scale):
    with tf.variable_scope('x%d' % (scale)):
      x = self._enhanced_upscaling_module(x, scale=scale)
      x = self._conv2d(x, num_features=3, kernel_size=(3, 3))
    return x


  def _generator(self, input_list, scale):
    with tf.variable_scope('generator'):
      # pre-process
      input_list = tf.cast(input_list, tf.float32)
      input_list = self._mean_shift(input_list)
      x = input_list

      # first convolutional layer
      with tf.variable_scope('first_conv'):
        x = self._conv2d(x, num_features=self.num_conv_features, kernel_size=(3, 3))

      # scale-specific local residual blocks
      with tf.variable_scope('initial_blocks'):
        pred_fn_pairs = []
        if (2 in self.model_scale_list):
          pred_fn_pairs.append((tf.equal(scale, 2), lambda: self._scale_specific_processing(x, scale=2)))
        if (4 in self.model_scale_list):
          pred_fn_pairs.append((tf.equal(scale, 4), lambda: self._scale_specific_processing(x, scale=4)))
        if (8 in self.model_scale_list):
          pred_fn_pairs.append((tf.equal(scale, 8), lambda: self._scale_specific_processing(x, scale=8)))
        x = tf.case(pred_fn_pairs, exclusive=True)
      
      # shared residual module
      with tf.variable_scope('shared'):
        x = self._residual_module(x, num_features=self.num_conv_features, num_blocks=self.num_shared_blocks)

      # scale-specific upsampling
      with tf.variable_scope('upscaling'):
        pred_fn_pairs = []
        if (2 in self.model_scale_list):
          pred_fn_pairs.append((tf.equal(scale, 2), lambda: self._scale_specific_upsampling(x, scale=2)))
        if (4 in self.model_scale_list):
          pred_fn_pairs.append((tf.equal(scale, 4), lambda: self._scale_specific_upsampling(x, scale=4)))
        if (8 in self.model_scale_list):
          pred_fn_pairs.append((tf.equal(scale, 8), lambda: self._scale_specific_upsampling(x, scale=8)))
        x = tf.case(pred_fn_pairs, exclusive=True)
      
      # post-process
      output_list = x
      output_list = self._mean_inverse_shift(output_list)
    
    return output_list
  
  def _optimize(self, output_list, truth_list, global_step):
    loss_l1 = tf.reduce_mean(tf.losses.absolute_difference(output_list, truth_list))
    self.loss_dict['recon_l1'] = loss_l1

    loss = 1.0 * loss_l1
    self.loss_dict['final'] = loss

    learning_rate = tf.train.exponential_decay(self.initial_learning_rate, global_step=global_step, decay_steps=self.learning_rate_decay_steps, decay_rate=self.learning_rate_decay, staircase=True)
    tf.summary.scalar('learning_rate', learning_rate)

    optimizer = tf.train.AdamOptimizer(learning_rate)
    train_op = optimizer.minimize(loss)
    
    return train_op, loss





