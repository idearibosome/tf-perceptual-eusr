import argparse
import importlib
import json
import os
import time

import tensorflow as tf

import dataloaders
import models

FLAGS = tf.flags.FLAGS

DEFAULT_DATALOADER = 'ava'
DEFAULT_MODEL = 'mobilenetv2'

if __name__ == '__main__':
  tf.flags.DEFINE_integer('input_patch_size', 192, 'Size of each input image patch.')

  tf.flags.DEFINE_integer('batch_size', 128, 'Size of the batches for each training step.')
  tf.flags.DEFINE_integer('epochs', 5, 'The number of total epochs.')
  tf.flags.DEFINE_float('learning_rate', 1e-3, 'Learning rate.')

  tf.flags.DEFINE_string('dataloader', DEFAULT_DATALOADER, 'Name of the data loader.')
  tf.flags.DEFINE_string('model', DEFAULT_MODEL, 'Name of the model.')
  tf.flags.DEFINE_string('cuda_device', '0', 'CUDA device index to be used in training. This parameter may be set to the environment variable \'CUDA_VISIBLE_DEVICES\'. Specify it as -1 to disable GPUs.')

  tf.flags.DEFINE_string('train_path', './train/', 'Base path of the trained model to be saved.')
  tf.flags.DEFINE_string('weight_filename', 'score_predictor.h5', 'File name of the trained model to be saved.')
  tf.flags.DEFINE_string('restore_path', None, 'Model path to be restored. Specify this to resume the training or use pre-trained parameters.')

  # parse data loader and model first and import them
  pre_parser = argparse.ArgumentParser(add_help=False)
  pre_parser.add_argument('--dataloader', default=DEFAULT_DATALOADER)
  pre_parser.add_argument('--model', default=DEFAULT_MODEL)
  pre_parsed = pre_parser.parse_known_args()[0]
  if (pre_parsed.dataloader is not None):
    DATALOADER_MODULE = importlib.import_module('dataloaders.' + pre_parsed.dataloader)
  if (pre_parsed.model is not None):
    MODEL_MODULE = importlib.import_module('models.' + pre_parsed.model)


def main(unused_argv):
  # initialize
  os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.cuda_device
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.gfile.MakeDirs(FLAGS.train_path)

  # data loader
  dataloader = DATALOADER_MODULE.create_loader()
  dataloader.prepare()

  # model
  model = MODEL_MODULE.create_model()
  model.prepare(is_training=True, input_size=FLAGS.input_patch_size)

  # model > restore
  if (FLAGS.restore_path is not None):
    model.restore(restore_path=FLAGS.restore_path)
    tf.logging.info('restored the model')
  
  # save arguments
  arguments_path = os.path.join(FLAGS.train_path, 'arguments.json')
  with open(arguments_path, 'w') as f:
    f.write(json.dumps(FLAGS.flag_values_dict(), sort_keys=True, indent=2))
  
  # train
  save_path = os.path.join(FLAGS.train_path, FLAGS.weight_filename)
  model.train(
    train_generator=dataloader.generator(is_training=True, batch_size=FLAGS.batch_size, patch_size=FLAGS.input_patch_size),
    train_steps=(dataloader.get_num_training_data() // FLAGS.batch_size),
    train_epochs=FLAGS.epochs,
    validate_generator=dataloader.generator(is_training=False, batch_size=FLAGS.batch_size, patch_size=FLAGS.input_patch_size),
    validate_steps=(dataloader.get_num_validation_data() // FLAGS.batch_size),
    save_path=save_path
  )
    
  # finalize
  tf.logging.info('finished')


if __name__ == '__main__':
  tf.app.run()