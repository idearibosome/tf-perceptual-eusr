import tensorflow as tf

FLAGS = tf.flags.FLAGS

def create_model():
  return BaseModel()

class BaseModel:

  MODEL_OUTPUT_SCORES_NAME = 'output_scores'
  MODEL_OUTPUT_FEATURES_NAME = 'output_features'

  def __init__(self):
    pass

  def prepare(self, is_training, input_size):
    """
    Prepare the model to be used. This function should be called before calling any other functions.
    Args:
      is_training: A boolean that specifies whether the model is for training or not.
      input_size: Size of the input images.
    """
    raise NotImplementedError
  
  def restore(self, restore_path):
    """
    Restore parameters of the model.
    Args:
      restore_path: Path of the weight file to be restored.
    """
    raise NotImplementedError
  
  def freeze(self, freeze_path):
    """
    Freeze the model.
    Args:
      freeze_path: Path of the freezed model file to be saved.
    """
    raise NotImplementedError

  def train(self, train_generator, train_steps, train_epochs, validate_generator, validate_steps, save_path):
    """
    Perform the training.
    Args:
      train_generator: A generator for training.
      train_steps: The number of training steps for each epoch.
      train_epochs: The number of training epochs.
      validate_generator: A generator for validation.
      validate_steps: The number of validation steps.
      save_path: The path (with filename) of the weights to be saved.
    """
    raise NotImplementedError