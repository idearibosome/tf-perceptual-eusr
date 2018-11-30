import os

import numpy as np
import tensorflow as tf
import scipy.stats

from dataloaders.base_loader import BaseLoader

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string('tid2013_score_path', 'tid2013_name_mos_std.txt', 'Path of the TID2013 score data (name, mean, and standard deviation).')
tf.flags.DEFINE_string('tid2013_image_path', None, 'Path of the distorted TID2013 images.')
tf.flags.DEFINE_string('tid2013_image_extension', 'bmp', 'Extension of the distorted TID2013 images (either bmp or png).')
tf.flags.DEFINE_string('tid2013_train_range', '360,3000', 'Range of indices of the training images.')
tf.flags.DEFINE_string('tid2013_validate_range', '0,360', 'Range of indices of the validation images.')
tf.flags.DEFINE_integer('tid2013_num_threads', 4, 'The number of threads to retrieve image (i.e., num_parallel_calls in tf.data.Dataset.map).')

def create_loader():
  return TID2013Loader()

class TID2013Loader(BaseLoader):

  def __init__(self):
    super().__init__()


  def prepare(self):
    tf.logging.info('data: preparing')

    FLAGS.tid2013_image_extension = FLAGS.tid2013_image_extension.lower()
    
    self.train_ranges = list(map(lambda x: int(x), FLAGS.tid2013_train_range.split(',')))
    self.validate_ranges = list(map(lambda x: int(x), FLAGS.tid2013_validate_range.split(',')))


    # image path dict
    self.image_path_dict = {}
    for root, _, files in os.walk(FLAGS.tid2013_image_path):
      for filename in files:
        if (filename.lower().endswith('.' + FLAGS.tid2013_image_extension)):
          image_name = filename.split('.')[0]
          image_path = os.path.join(root, filename)
          self.image_path_dict[image_name] = image_path
    

    # load scores
    self.image_path_list = []
    self.score_list = []
    with open(FLAGS.tid2013_score_path, mode='r') as f:
      lines = f.readlines()
      for i, line in enumerate(lines):
        token = line.split()
        image_name = token[0]
        mos_mean = float(token[1])
        mos_std = float(token[2]) + 1e-10 # prevent DIV/0

        values = np.zeros(10, dtype='float32')
        for center in range(0, 10):
          value_start = scipy.stats.norm.cdf(center-0.5, loc=mos_mean, scale=mos_std)
          value_end = scipy.stats.norm.cdf(center+0.5, loc=mos_mean, scale=mos_std)
          values[center] = value_end - value_start
        values /= values.sum()

        if (image_name in self.image_path_dict):
          file_path = self.image_path_dict[image_name]
          self.image_path_list.append(file_path)
          self.score_list.append(values)
    self.image_path_list = np.array(self.image_path_list)
    self.score_list = np.array(self.score_list, dtype='float32')

    self.train_ranges[0] = min(self.train_ranges[0], self.score_list.shape[0])
    self.train_ranges[0] = max(self.train_ranges[0], -self.score_list.shape[0])
    if (self.train_ranges[0] == 0):
      self.train_ranges[0] = None
    self.train_ranges[1] = min(self.train_ranges[1], self.score_list.shape[0])
    self.train_ranges[1] = max(self.train_ranges[1], -self.score_list.shape[0])
    if (self.train_ranges[1] == 0):
      self.train_ranges[1] = None
    self.validate_ranges[0] = min(self.validate_ranges[0], self.score_list.shape[0])
    self.validate_ranges[0] = max(self.validate_ranges[0], -self.score_list.shape[0])
    if (self.validate_ranges[0] == 0):
      self.validate_ranges[0] = None
    self.validate_ranges[1] = min(self.validate_ranges[1], self.score_list.shape[0])
    self.validate_ranges[1] = max(self.validate_ranges[1], -self.score_list.shape[0])
    if (self.validate_ranges[1] == 0):
      self.validate_ranges[1] = None
    

    # divide train/validate set
    self.train_image_path_list = self.image_path_list[self.train_ranges[0]:self.train_ranges[1]]
    self.train_score_list = self.score_list[self.train_ranges[0]:self.train_ranges[1]]
    self.validate_image_path_list = self.image_path_list[self.validate_ranges[0]:self.validate_ranges[1]]
    self.validate_score_list = self.score_list[self.validate_ranges[0]:self.validate_ranges[1]]


    tf.logging.info('data: prepared (%d for training, %d for validation)' % (self.train_score_list.shape[0], self.validate_score_list.shape[0]))

  
  def get_num_training_data(self):
    return self.train_score_list.shape[0]
  

  def get_num_validation_data(self):
    return self.validate_score_list.shape[0]
  

  def generator(self, is_training, batch_size, patch_size):

    def _local_get_patch(filename, score):
      image = self._get_patch(is_training=is_training, filename=filename, patch_size=patch_size)
      return image, score
    
    with tf.Graph().as_default():
      if (is_training):
        dataset = tf.data.Dataset().from_tensor_slices((self.train_image_path_list, self.train_score_list))
      else:
        dataset = tf.data.Dataset().from_tensor_slices((self.validate_image_path_list, self.validate_score_list))

      if (FLAGS.tid2013_num_threads > 0):
        dataset = dataset.map(_local_get_patch, num_parallel_calls=FLAGS.tid2013_num_threads)
      else:
        dataset = dataset.map(_local_get_patch)
      
      dataset = dataset.batch(batch_size).repeat()
      if (is_training):
        dataset = dataset.shuffle(buffer_size=64)
      
      sess = tf.Session(config=tf.ConfigProto(
        device_count={'GPU': 0}
      ))
      
      iterator = dataset.make_initializable_iterator()
      next_element = iterator.get_next()
      sess.run(iterator.initializer)

      while True:
        try:
          image_list, score_list = sess.run(next_element)
          yield (image_list, score_list)
        except tf.errors.OutOfRangeError:
          break
  
  
  def _get_patch(self, is_training, filename, patch_size):
    image = tf.read_file(filename)
    if (FLAGS.tid2013_image_extension == 'bmp'):
      image = tf.image.decode_bmp(image, channels=3)
    elif (FLAGS.tid2013_image_extension == 'png'):
      image = tf.image.decode_png(image, channels=3)
    
    if (is_training):
      image = tf.random_crop(image, size=(patch_size, patch_size, 3))
      image = tf.image.random_flip_left_right(image)
      image = tf.image.random_flip_up_down(image)
    else:
      image = tf.image.resize_image_with_crop_or_pad(image, patch_size, patch_size)
    
    image = (tf.cast(image, tf.float32) - 127.5) / 127.5

    return image