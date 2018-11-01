import os

import numpy as np
import tensorflow as tf

FLAGS = tf.flags.FLAGS

def create_loader():
  return BaseLoader()

class BaseLoader():

  def __init__(self):
    pass

  def prepare(self):
    """
    Prepare the data loader.
    """
    raise NotImplementedError
  
  def get_num_training_data(self):
    """
    Get the number of training data.
    Returns:
      The number of training data.
    """
    raise NotImplementedError
  
  def get_num_validation_data(self):
    """
    Get the number of validation data.
    Returns:
      The number of validation data.
    """
    raise NotImplementedError
  
  def generator(self, is_training, batch_size, patch_size):
    """
    Create a generator that returns a batch of training/validation image patches.
    Args:
      is_training: A boolean that specifies whether the generator is for training or not.
      batch_size: The number of image patches.
      patch_size: Size of the image patches.
    Returns:
      input_list: A list of input image patches.
      truth_list: A list of ground-truth image patches.
    """
    raise NotImplementedError