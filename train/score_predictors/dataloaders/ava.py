import os

import numpy as np
import tensorflow as tf
import scipy.stats

from dataloaders.base_loader import BaseLoader

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string('ava_dataset_path', 'AVA.txt', 'Path of the AVA score data (AVA.txt).')
tf.flags.DEFINE_string('ava_image_path', None, 'Path of the AVA images. Name of each image should be an image ID.')
tf.flags.DEFINE_string('ava_train_range', '0,-5000', 'Range of indices of the training images.')
tf.flags.DEFINE_string('ava_validate_range', '-5000,0', 'Range of indices of the validation images.')
tf.flags.DEFINE_integer('ava_num_threads', 4, 'The number of threads to retrieve image (i.e., num_parallel_calls in tf.data.Dataset.map).')
tf.flags.DEFINE_boolean('ava_validate_images', False, 'Specify this to check validity of the images. This takes a long time since it requires loading all images.')

def create_loader():
  return AVALoader()

class AVALoader(BaseLoader):

  def __init__(self):
    super().__init__()


  def prepare(self):
    tf.logging.info('data: preparing')
    
    self.train_ranges = list(map(lambda x: int(x), FLAGS.ava_train_range.split(',')))
    self.validate_ranges = list(map(lambda x: int(x), FLAGS.ava_validate_range.split(',')))


    # image path dict
    self.image_path_dict = {}
    for root, _, files in os.walk(FLAGS.ava_image_path):
      for filename in files:
        if (filename.lower().endswith('.jpg')):
          image_name = filename.split('.')[0]
          image_path = os.path.join(root, filename)
          self.image_path_dict[image_name] = image_path
    

    # validate images
    if (FLAGS.ava_validate_images):
      tf.logging.info('data: validating (this may take a while...)')

      read_session, read_input, read_output = self._get_decode_jpeg_session()

      validated_image_path_dict = {}
      for (image_name, image_path) in self.image_path_dict.items():
        try:
          image = read_session.run(read_output, feed_dict={read_input: image_path})
          if (image is None):
            raise ValueError
          if (len(image.shape) != 3):
            raise ValueError
          if (image.shape[0] <= 0 or image.shape[1] <= 0 or image.shape[2] != 3):
            raise ValueError
          
          validated_image_path_dict[image_name] = image_path

        except:
          tf.logging.info('data: invalid image %s (ignored)' % (image_name))
      
      self.image_path_dict = validated_image_path_dict
    

    # load scores
    self.image_path_list = []
    self.score_list = []
    with open(FLAGS.ava_dataset_path, mode='r') as f:
      lines = f.readlines()
      for i, line in enumerate(lines):
        token = line.split()
        image_name = token[1]

        values = np.array(token[2:12], dtype='float32')
        values /= values.sum()

        if (image_name in self.image_path_dict):
          file_path = self.image_path_dict[image_name]
          self.image_path_list.append(file_path)
          self.score_list.append(values)
        else:
          tf.logging.info('data: image %s not found or invalid (ignored)' % (image_name))
    
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

      if (FLAGS.ava_num_threads > 0):
        dataset = dataset.map(_local_get_patch, num_parallel_calls=FLAGS.ava_num_threads)
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
    image = tf.image.decode_jpeg(image, channels=3)
    
    if (is_training):
      image_height = tf.shape(image)[0]
      image_width = tf.shape(image)[1]
      new_image_size = tf.minimum(image_height, image_width)
      image = tf.random_crop(image, size=(new_image_size, new_image_size, 3))

      resize_size = tf.maximum(new_image_size, patch_size)
      image = tf.image.resize_images(image, (resize_size, resize_size))
      
      image = tf.random_crop(image, size=(patch_size, patch_size, 3))
      image = tf.image.random_flip_left_right(image)
      image = tf.image.random_flip_up_down(image)
    else:
      image_height = tf.shape(image)[0]
      image_width = tf.shape(image)[1]
      new_image_size = tf.minimum(image_height, image_width)
      image = tf.image.resize_image_with_crop_or_pad(image, new_image_size, new_image_size)

      resize_size = tf.maximum(new_image_size, patch_size)
      image = tf.image.resize_images(image, (resize_size, resize_size))

      image = tf.image.resize_image_with_crop_or_pad(image, patch_size, patch_size)
    
    image = (tf.cast(image, tf.float32) - 127.5) / 127.5

    return image
  

  def _get_decode_jpeg_session(self):
    tf_graph = tf.Graph()
    with tf_graph.as_default():
      tf_filename = tf.placeholder(tf.string, [])
      tf_image = tf.read_file(tf_filename)
      tf_image = tf.image.decode_jpeg(tf_image, channels=3)
    
      tf_init = tf.global_variables_initializer()
      tf_session = tf.Session(config=tf.ConfigProto(device_count={'GPU': 0}))
      tf_session.run(tf_init)
    
    return tf_session, tf_filename, tf_image
