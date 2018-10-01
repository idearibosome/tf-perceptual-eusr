import tensorflow as tf

FLAGS = tf.flags.FLAGS

def create_model():
  return BaseModel()

class BaseModel:

  TF_INPUT_NAME = 'sr_input'
  TF_INPUT_SCALE_NAME = 'sr_input_scale'

  def __init__(self):
    self.global_step = 0

    self.loss_dict = {}

  def prepare(self, is_training, global_step=0):
    """
    Prepare the model to be used. This function should be called before calling any other functions.
    Args:
      is_training: A boolean that specifies whether the model is for training or not.
      global_step: Initial global step. Specify this to resume the training.
    """
    raise NotImplementedError
  
  def save(self, base_path):
    """
    Save the current trained model.
    Args:
      base_path: Path of the checkpoint directory to be saved.
    """
    raise NotImplementedError
  
  def restore(self, ckpt_path, target=None):
    """
    Restore parameters of the model.
    Args:
      ckpt_path: Path of the checkpoint file to be restored.
      target: (Optional) Target of the restoration.
    """
    raise NotImplementedError

  def get_session(self):
    """
    Get main session of the model.
    Returns:
      The main tf.Session.
    """
    raise NotImplementedError
  
  def get_next_train_scale(self):
    """
    Get next image scale for training.
    Returns:
      A scale value.
    """
    raise NotImplementedError

  def train_step(self, input_list, scale, truth_list, with_summary=False):
    """
    Perform a training step.
    Args:
      input_list: List of the input images.
      scale: Scale to be super-resolved.
      truth_list: List of the ground-truth images. Should be the same shape as input_list.
      with_summary: Retrieve serialized summary data.
    Returns:
      loss: A representative loss value of the current training step.
      summary: Serialized summary data. None if with_summary=False.
    """
    raise NotImplementedError
  
  def upscale(self, input_list, scale):
    """
    Upscale the input images without training.
    Args:
      input_list: List of the input images.
      scale: Scale to be super-resolved.
    """
    raise NotImplementedError