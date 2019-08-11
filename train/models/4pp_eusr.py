import math
import os

import numpy as np
import tensorflow as tf

from models.base_model import BaseModel

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_boolean('eusr_disable_multipass', False, 'Specify this option to disable multi-pass upscaling.')
tf.flags.DEFINE_integer('eusr_conv_features', 64, 'The number of convolutional features.')
tf.flags.DEFINE_integer('eusr_shared_blocks', 32, 'The number of local residual blocks (LRBs) in the shared feature extraction part.')
tf.flags.DEFINE_integer('eusr_upscale_blocks', 1, 'The number of local residual blocks (LRBs) in the enhanced upscaling modules (EUMs).')
tf.flags.DEFINE_string('eusr_rgb_mean', '114.4,111.5,103.0', 'Mean R, G, and B values of the training images (e.g., 114.4,111.5,103.0).')

tf.flags.DEFINE_string('eusr_aesthetic_nima_path', 'ava.pb', 'Path of the frozen aesthetic score predictor model.')
tf.flags.DEFINE_string('eusr_subjective_nima_path', 'tid2013.pb', 'Path of the frozen subjective score predictor model.')

tf.flags.DEFINE_float('eusr_weight_lr', 0.05, 'Weight of the reconstruction loss (l_r).')
tf.flags.DEFINE_float('eusr_weight_lg', 0.1, 'Weight of the adversarial loss (l_g).')
tf.flags.DEFINE_float('eusr_weight_las', 0.01, 'Weight of the aesthetic score loss (l_as).')
tf.flags.DEFINE_float('eusr_weight_lar', 0.1, 'Weight of the aesthetic representation loss (l_ar).')
tf.flags.DEFINE_float('eusr_weight_lss', 0.01, 'Weight of the subjective score loss (l_ss).')
tf.flags.DEFINE_float('eusr_weight_lsr', 0.1, 'Weight of the subjective representation loss (l_sr).')

tf.flags.DEFINE_float('eusr_learning_rate', 1e-5, 'Initial learning rate (generator).')
tf.flags.DEFINE_float('eusr_learning_rate_discriminator', 2e-5, 'Initial learning rate (discriminator).')
tf.flags.DEFINE_float('eusr_learning_rate_decay', 0.5, 'Learning rate decay factor.')
tf.flags.DEFINE_integer('eusr_learning_rate_decay_steps', 1000000, 'The number of training steps to perform learning rate decay.')

def create_model():
  return PerceptualEUSR()

class PerceptualEUSR(BaseModel):
  def __init__(self):
    super().__init__()
  

  def prepare(self, is_training, global_step=0):
    # config. parameters
    self.global_step = global_step

    self.scale_list = list(map(lambda x: int(x), FLAGS.scales.split(',')))
    if (is_training):
      for scale in self.scale_list:
        if (not scale in [4]): # only x4 is supported
          raise ValueError('Unsupported scale is provided.')
    else:
      for scale in self.scale_list:
        if (not scale in [2, 4, 8]): # only x2, x4, and x8 are supported
          raise ValueError('Unsupported scale is provided.')

    self.multipass_upscaling = (not FLAGS.eusr_disable_multipass)
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
      self.initial_learning_rate_discriminator = FLAGS.eusr_learning_rate_discriminator
      self.learning_rate_decay = FLAGS.eusr_learning_rate_decay
      self.learning_rate_decay_steps = FLAGS.eusr_learning_rate_decay_steps

      self.aesthetic_nima_path = FLAGS.eusr_aesthetic_nima_path
      self.subjective_nima_path = FLAGS.eusr_subjective_nima_path

      self.loss_weight_r = FLAGS.eusr_weight_lr
      self.loss_weight_g = FLAGS.eusr_weight_lg
      self.loss_weight_as = FLAGS.eusr_weight_las
      self.loss_weight_ar = FLAGS.eusr_weight_lar
      self.loss_weight_ss = FLAGS.eusr_weight_lss
      self.loss_weight_sr = FLAGS.eusr_weight_lsr


    # tensorflow graph
    self.tf_graph = tf.Graph()
    with self.tf_graph.as_default():

      self.tf_input = tf.placeholder(tf.float32, [None, None, None, 3], name=BaseModel.TF_INPUT_NAME)
      self.tf_scale = tf.placeholder(tf.float32, [], name=BaseModel.TF_INPUT_SCALE_NAME)

      # generator > 2x
      self.tf_output_2x = self._generator(input_list=self.tf_input, scale=2, reuse=False)
      self.tf_output_2x = self._generator(input_list=self.tf_output_2x, scale=2, reuse=True)

      # generator > 4x
      self.tf_output_4x = self._generator(input_list=self.tf_input, scale=4, reuse=True)

      # generator > 8x
      self.tf_output_8x = self._generator(input_list=self.tf_input, scale=8, reuse=True)
      self.tf_output_8x = tf.image.resize_images(self.tf_output_8x, size=[tf.shape(self.tf_input)[1] * 4, tf.shape(self.tf_input)[2] * 4], method=tf.image.ResizeMethod.BICUBIC, align_corners=False)

      # output based on tf_scale placeholder
      output_fn_pairs = []
      output_fn_pairs.append((tf.equal(self.tf_scale, 2), lambda: tf.identity(self.tf_output_2x)))
      output_fn_pairs.append((tf.equal(self.tf_scale, 4), lambda: tf.identity(self.tf_output_4x)))
      output_fn_pairs.append((tf.equal(self.tf_scale, 8), lambda: tf.identity(self.tf_output_8x)))
      self.tf_output = tf.case(output_fn_pairs, exclusive=True)
      
      if (is_training):
        input_summary = tf.cast(tf.clip_by_value(self.tf_input, 0.0, 255.0), tf.uint8)
        tf.summary.image('input', input_summary)

        if (self.multipass_upscaling):
          output_summary = tf.cast(tf.clip_by_value(self.tf_output_2x, 0.0, 255.0), tf.uint8)
          tf.summary.image('output_2x', output_summary)
          output_summary = tf.cast(tf.clip_by_value(self.tf_output_4x, 0.0, 255.0), tf.uint8)
          tf.summary.image('output_4x', output_summary)
          output_summary = tf.cast(tf.clip_by_value(self.tf_output_8x, 0.0, 255.0), tf.uint8)
          tf.summary.image('output_8x', output_summary)
        else:
          output_summary = tf.cast(tf.clip_by_value(self.tf_output, 0.0, 255.0), tf.uint8)
          tf.summary.image('output', output_summary)
        
        self.tf_truth = tf.placeholder(tf.float32, [None, None, None, 3])
        truth_summary = tf.cast(tf.clip_by_value(self.tf_truth, 0.0, 255.0), tf.uint8)
        tf.summary.image('truth', truth_summary)

        self.tf_global_step = tf.placeholder(tf.int64, [])

        combined_output_list = []
        if (self.multipass_upscaling):
          combined_output_list.append(self.tf_output_2x)
          combined_output_list.append(self.tf_output_4x)
          combined_output_list.append(self.tf_output_8x)
        else:
          combined_output_list.append(self.tf_output)

        self.tf_train_op, self.tf_loss = self._optimize(output_list=combined_output_list, truth_list=self.tf_truth, global_step=self.tf_global_step)

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
    with self.tf_graph.as_default():
      if (target == 'generator'):
        train_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
        restorer = tf.train.Saver(var_list=train_variables)
      else:
        restorer = tf.train.Saver()
      restorer.restore(sess=self.tf_session, save_path=ckpt_path)
  

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
    feed_dict = {}
    feed_dict[self.tf_input] = input_list
    feed_dict[self.tf_scale] = scale

    output_list = self.tf_session.run(self.tf_output, feed_dict=feed_dict)

    return output_list
  
  
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


  def _generator(self, input_list, scale, reuse=False):
    with tf.variable_scope('generator', reuse=reuse):
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
        pred_fn_pairs.append((tf.equal(scale, 2), lambda: self._scale_specific_processing(x, scale=2)))
        pred_fn_pairs.append((tf.equal(scale, 4), lambda: self._scale_specific_processing(x, scale=4)))
        pred_fn_pairs.append((tf.equal(scale, 8), lambda: self._scale_specific_processing(x, scale=8)))
        x = tf.case(pred_fn_pairs, exclusive=True)
      
      # shared residual module
      with tf.variable_scope('shared'):
        x = self._residual_module(x, num_features=self.num_conv_features, num_blocks=self.num_shared_blocks)

      # scale-specific upsampling
      with tf.variable_scope('upscaling'):
        pred_fn_pairs = []
        pred_fn_pairs.append((tf.equal(scale, 2), lambda: self._scale_specific_upsampling(x, scale=2)))
        pred_fn_pairs.append((tf.equal(scale, 4), lambda: self._scale_specific_upsampling(x, scale=4)))
        pred_fn_pairs.append((tf.equal(scale, 8), lambda: self._scale_specific_upsampling(x, scale=8)))
        x = tf.case(pred_fn_pairs, exclusive=True)
      
      # post-process
      output_list = x
      output_list = self._mean_inverse_shift(output_list)
    
    return output_list
  

  def _discriminator(self, input_list, reuse=False):
    with tf.variable_scope('discriminator', reuse=reuse):
      x = input_list

      with tf.variable_scope('conv1'):
        x = self._conv2d(x, num_features=32, kernel_size=(3, 3), strides=(1, 1))
        x = tf.nn.leaky_relu(x)

      with tf.variable_scope('conv2'):
        x = self._conv2d(x, num_features=32, kernel_size=(3, 3), strides=(2, 2))
        x = tf.nn.leaky_relu(x)

      with tf.variable_scope('conv3'):
        x = self._conv2d(x, num_features=64, kernel_size=(3, 3), strides=(1, 1))
        x = tf.nn.leaky_relu(x)

      with tf.variable_scope('conv4'):
        x = self._conv2d(x, num_features=64, kernel_size=(3, 3), strides=(2, 2))
        x = tf.nn.leaky_relu(x)

      with tf.variable_scope('conv5'):
        x = self._conv2d(x, num_features=128, kernel_size=(3, 3), strides=(1, 1))
        x = tf.nn.leaky_relu(x)

      with tf.variable_scope('conv6'):
        x = self._conv2d(x, num_features=128, kernel_size=(3, 3), strides=(2, 2))
        x = tf.nn.leaky_relu(x)

      with tf.variable_scope('conv7'):
        x = self._conv2d(x, num_features=256, kernel_size=(3, 3), strides=(1, 1))
        x = tf.nn.leaky_relu(x)

      with tf.variable_scope('conv8'):
        x = self._conv2d(x, num_features=256, kernel_size=(3, 3), strides=(2, 2))
        x = tf.nn.leaky_relu(x)

      with tf.variable_scope('conv9'):
        x = self._conv2d(x, num_features=512, kernel_size=(3, 3), strides=(1, 1))
        x = tf.nn.leaky_relu(x)

      with tf.variable_scope('conv10'):
        x = self._conv2d(x, num_features=512, kernel_size=(3, 3), strides=(2, 2))
        x = tf.nn.leaky_relu(x)
      
      with tf.variable_scope('fc'):
        x = tf.layers.flatten(x)
        x = tf.reshape(x, [-1, 512])  # ensure the last dimension
        x = tf.layers.dense(x, 1024)
        x = tf.nn.leaky_relu(x)
        
      with tf.variable_scope('final'):
        output = tf.layers.dense(x, 1)
    
    return output


  def _optimize(self, output_list, truth_list, global_step):

    # employ discriminator
    truth_adversarial = self._discriminator(truth_list, reuse=False)
    output_adversarial_list = []
    for each_list in output_list:
      output_adversarial_list.append(self._discriminator(each_list, reuse=True))


    # employ score predictors
    nima_input = [x for x in output_list]
    nima_input.append(truth_list)
    nima_input = tf.concat(nima_input, axis=0)
    nima_input = (tf.cast(nima_input, tf.float32) - 127.5) / 127.5


    # score predictors > aesthetic
    with tf.gfile.GFile(self.aesthetic_nima_path, 'rb') as f:
      nima_graph_def = tf.GraphDef()
      nima_graph_def.ParseFromString(f.read())
    aesthetic_scores, aesthetic_features = tf.import_graph_def(nima_graph_def, name='aesthetic_nima', input_map={'input_1:0': nima_input}, return_elements=['output_scores:0', 'output_features:0'])

    aesthetic_scores = tf.multiply(aesthetic_scores, tf.lin_space(1.0, 10.0, 10))
    aesthetic_scores = 10.0 - tf.reduce_sum(aesthetic_scores, axis=1)

    aesthetic_score_list = tf.split(aesthetic_scores, num_or_size_splits=len(output_list)+1, axis=0)
    truth_aesthetic_scores = aesthetic_score_list[-1]
    output_aesthetic_scores_list = aesthetic_score_list[:-1]
    aesthetic_feature_list = tf.split(aesthetic_features, num_or_size_splits=len(output_list)+1, axis=0)
    truth_aesthetic_features = aesthetic_feature_list[-1]
    output_aesthetic_features_list = aesthetic_feature_list[:-1]


    # score predictors > subjective
    with tf.gfile.GFile(self.subjective_nima_path, 'rb') as f:
      nima_graph_def = tf.GraphDef()
      nima_graph_def.ParseFromString(f.read())
    subjective_scores, subjective_features = tf.import_graph_def(nima_graph_def, name='subjective_nima', input_map={'input_1:0': nima_input}, return_elements=['output_scores:0', 'output_features:0'])

    subjective_scores = tf.multiply(subjective_scores, tf.lin_space(1.0, 10.0, 10))
    subjective_scores = 10.0 - tf.reduce_sum(subjective_scores, axis=1)

    subjective_score_list = tf.split(subjective_scores, num_or_size_splits=len(output_list)+1, axis=0)
    truth_subjective_scores = subjective_score_list[-1]
    output_subjective_scores_list = subjective_score_list[:-1]
    subjective_feature_list = tf.split(subjective_features, num_or_size_splits=len(output_list)+1, axis=0)
    truth_subjective_features = subjective_feature_list[-1]
    output_subjective_features_list = subjective_feature_list[:-1]
    
    
    # reconstruction loss (l_r)
    loss_r = 0.0
    for each_list in output_list:
      loss_r += tf.reduce_mean(tf.losses.absolute_difference(each_list, truth_list))
    loss_r /= len(output_list)
    self.loss_dict['recon'] = loss_r


    # adversarial loss
    loss_g = 0.0
    for each_list in output_adversarial_list:
      loss_g += tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=each_list, labels=tf.ones_like(each_list)))
    loss_g /= len(output_list)
    self.loss_dict['adv'] = loss_g


    # aesthetic score loss (l_as)
    loss_as = 0.0
    for each_list in output_aesthetic_scores_list:
      loss_as += tf.reduce_mean(tf.maximum(0.0, each_list - (truth_aesthetic_scores * 0.8)))
    loss_as /= len(output_list)
    self.loss_dict['aesthetic_score'] = loss_as


    # aesthetic representation loss (l_ar)
    loss_ar = 0.0
    for each_list in output_aesthetic_features_list:
      loss_ar += tf.reduce_mean(tf.losses.mean_squared_error(each_list, truth_aesthetic_features))
    loss_ar /= len(output_list)
    self.loss_dict['aesthetic_rep'] = loss_ar


    # subjective score loss (l_ss)
    loss_ss = 0.0
    for each_list in output_subjective_scores_list:
      loss_ss += tf.reduce_mean(tf.maximum(0.0, each_list - (truth_subjective_scores * 0.8)))
    loss_ss /= len(output_list)
    self.loss_dict['subjective_score'] = loss_ss


    # subjective representation loss (l_sr)
    loss_sr = 0.0
    for each_list in output_subjective_features_list:
      loss_sr += tf.reduce_mean(tf.losses.mean_squared_error(each_list, truth_subjective_features))
    loss_sr /= len(output_list)
    self.loss_dict['subjective_rep'] = loss_sr


    # final loss (l)
    loss = 0.0
    loss += (self.loss_weight_r * loss_r)
    loss += (self.loss_weight_g * loss_g)
    loss += (self.loss_weight_as * loss_as)
    loss += (self.loss_weight_ar * loss_ar)
    loss += (self.loss_weight_ss * loss_ss)
    loss += (self.loss_weight_sr * loss_sr)
    self.loss_dict['final'] = loss


    # discriminator loss
    loss_discriminator = 0.0
    for each_list in output_adversarial_list:
      loss_discriminator += tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=each_list, labels=tf.zeros_like(each_list)))
    loss_discriminator /= len(output_list)
    loss_discriminator += tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=truth_adversarial, labels=tf.ones_like(truth_adversarial)))
    self.loss_dict['discriminator'] = loss_discriminator


    # optimize (generator)
    g_learning_rate = tf.train.exponential_decay(self.initial_learning_rate, global_step=global_step, decay_steps=self.learning_rate_decay_steps, decay_rate=self.learning_rate_decay, staircase=True)
    tf.summary.scalar('learning_rate', g_learning_rate)

    g_optimizer = tf.train.AdamOptimizer(g_learning_rate)
    g_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
    g_train_op = g_optimizer.minimize(loss, var_list=g_variables)


    # optimize (discriminator)
    d_learning_rate = tf.train.exponential_decay(self.initial_learning_rate_discriminator, global_step=global_step, decay_steps=self.learning_rate_decay_steps, decay_rate=self.learning_rate_decay, staircase=True)
    tf.summary.scalar('learning_rate_discriminator', d_learning_rate)

    d_optimizer = tf.train.AdamOptimizer(d_learning_rate)
    d_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')
    d_train_op = d_optimizer.minimize(loss_discriminator, var_list=d_variables)


    # finalize
    train_op = tf.group([g_train_op, d_train_op])

    
    return train_op, loss





