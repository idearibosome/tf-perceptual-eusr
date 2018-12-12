import argparse
import importlib
import json
import os
import time

import tensorflow as tf

import dataloaders
import models

FLAGS = tf.flags.FLAGS

DEFAULT_MODEL = 'mobilenetv2'

if __name__ == '__main__':
  tf.flags.DEFINE_string('model', DEFAULT_MODEL, 'Name of the model.')
  tf.flags.DEFINE_integer('input_patch_size', 192, 'Size of each input image patch.')
  tf.flags.DEFINE_string('restore_path', 'score_predictor.h5', 'Model path to be restored.')
  tf.flags.DEFINE_string('output_path', 'score_predictor_freezed.pb', 'Model path to be saved.')

  # parse data loader and model first and import them
  pre_parser = argparse.ArgumentParser(add_help=False)
  pre_parser.add_argument('--model', default=DEFAULT_MODEL)
  pre_parsed = pre_parser.parse_known_args()[0]
  if (pre_parsed.model is not None):
    MODEL_MODULE = importlib.import_module('models.' + pre_parsed.model)


def main(unused_argv):
  # initialize
  os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.keras.backend.set_learning_phase(0)

  # model
  model = MODEL_MODULE.create_model()
  model.prepare(is_training=False, input_size=FLAGS.input_patch_size)

  # model > restore
  model.restore(restore_path=FLAGS.restore_path)
  tf.logging.info('restored the model')

  # freeze
  model.freeze(freeze_path=FLAGS.output_path)
  tf.logging.info('freezed and saved the model')
    
  # finalize
  tf.logging.info('finished')


if __name__ == '__main__':
  tf.app.run()