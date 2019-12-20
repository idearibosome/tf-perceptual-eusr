import argparse
import importlib
import os
import time

import numpy as np
import tensorflow as tf

import dataloaders
import models

FLAGS = tf.flags.FLAGS

DEFAULT_MODEL = 'base_model'

if __name__ == '__main__':
  tf.flags.DEFINE_string('model', DEFAULT_MODEL, 'Name of the model.')
  tf.flags.DEFINE_string('scales', '2,4,8', 'Scales of the input images. Use the \',\' character to specify multiple scales (e.g., 2,4,8).')

  tf.flags.DEFINE_string('restore_path', None, 'Checkpoint path to be restored.')
  tf.flags.DEFINE_string('restore_target', None, 'Target of the restoration.')
  tf.flags.DEFINE_integer('restore_global_step', 0, 'Global step of the restored model. Some models may require to specify this.')

  # parse data loader and model first and import them
  pre_parser = argparse.ArgumentParser(add_help=False)
  pre_parser.add_argument('--model', default=DEFAULT_MODEL)
  pre_parsed = pre_parser.parse_known_args()[0]
  if (pre_parsed.model is not None):
    MODEL_MODULE = importlib.import_module('models.' + pre_parsed.model)


def main(unused_argv):
  # initialize
  os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # CPU-mode
  tf.logging.set_verbosity(tf.logging.INFO)

  # model
  model = MODEL_MODULE.create_model()
  model.prepare(is_training=False, global_step=FLAGS.restore_global_step)

  # model > restore
  model.restore(ckpt_path=FLAGS.restore_path, target=FLAGS.restore_target)
  tf.logging.info('restored the model')

  # freeze
  with model.tf_session.graph.as_default():
    output_node = tf.identity(model.tf_output, name='sr_output')
  graph_def = model.tf_session.graph.as_graph_def()
  for node in graph_def.node:
    node.device = ''
  constant_graph = tf.graph_util.convert_variables_to_constants(model.tf_session, graph_def, output_node_names=['sr_output'])
  tf.io.write_graph(constant_graph, '.', 'model.pb', as_text=False)

  # finalize
  tf.logging.info('finished')


if __name__ == '__main__':
  tf.app.run()