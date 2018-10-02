import importlib
import os
import time

import tensorflow as tf

import models
from dataloader import DataLoader

FLAGS = tf.flags.FLAGS

if __name__ == '__main__':
  tf.flags.DEFINE_integer('batch_size', 16, 'Size of the batches for each training step.')
  tf.flags.DEFINE_integer('input_patch_size', 48, 'Size of each input image patch.')

  tf.flags.DEFINE_string('model', 'base_model', 'Name of the model.')
  tf.flags.DEFINE_string('scales', '2,4,8', 'Scales of the input images. Use the \',\' character to specify multiple scales (e.g., 2,4,8).')
  tf.flags.DEFINE_string('cuda_device', '0', 'CUDA device index to be used in training. This parameter may be set to the environment variable \'CUDA_VISIBLE_DEVICES\'. Specify it as -1 to disable GPUs.')

  tf.flags.DEFINE_string('train_path', './train/', 'Base path of the trained model to be saved.')
  tf.flags.DEFINE_integer('max_steps', 1000000, 'The number of maximum training steps.')
  tf.flags.DEFINE_integer('log_freq', 10, 'The frequency of logging via tf.logging.')
  tf.flags.DEFINE_integer('summary_freq', 200, 'The frequency of logging on TensorBoard.')
  tf.flags.DEFINE_integer('save_freq', 50000, 'The frequency of saving the trained model.')
  tf.flags.DEFINE_integer('save_max_keep', 100, 'The maximum number of recent trained models to keep (i.e., max_to_keep of tf.train.Saver).')
  tf.flags.DEFINE_float('sleep_ratio', 0.05, 'The ratio of sleeping time for each training step, which prevents overheating of GPUs. Specify 0 to disable sleeping.')

  tf.flags.DEFINE_string('restore_path', None, 'Checkpoint path to be restored. Specify this to resume the training or use pre-trained parameters.')
  tf.flags.DEFINE_string('restore_target', None, 'Target of the restoration.')
  tf.flags.DEFINE_integer('global_step', 0, 'Initial global step. Specify this to resume the training.')


def main(unused_argv):
  # initialize
  os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.cuda_device
  scale_list = list(map(lambda x: int(x), FLAGS.scales.split(',')))
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.gfile.MakeDirs(FLAGS.train_path)

  # data loader
  dataloader = DataLoader(scale_list=scale_list)
  dataloader.prepare()

  # model
  model_module = importlib.import_module('models.' + FLAGS.model)
  model = model_module.create_model()
  model.prepare(is_training=True, global_step=FLAGS.global_step)

  # model > restore
  if (FLAGS.restore_path is not None):
    model.restore(ckpt_path=FLAGS.restore_path, target=FLAGS.restore_target)

  # model > summary
  summary_writers = {}
  for scale in scale_list:
    summary_path = os.path.join(FLAGS.train_path, 'x%d' % (scale))
    summary_writer = tf.summary.FileWriter(summary_path, graph=model.get_session().graph)
    summary_writers[scale] = summary_writer

  # train
  local_train_step = 0
  while (model.global_step < FLAGS.max_steps):
    global_train_step = model.global_step + 1
    local_train_step += 1

    with_summary = True if (local_train_step % FLAGS.summary_freq == 0) else False

    start_time = time.time()

    scale = model.get_next_train_scale()
    input_list, truth_list = dataloader.get_batch(batch_size=FLAGS.batch_size, scale=scale, input_patch_size=FLAGS.input_patch_size)
    loss, summary = model.train_step(input_list=input_list, scale=scale, truth_list=truth_list, with_summary=with_summary)

    duration = time.time() - start_time
    if (FLAGS.sleep_ratio > 0):
      time.sleep(min(1.0, duration*FLAGS.sleep_ratio))

    if (local_train_step % FLAGS.log_freq == 0):
      tf.logging.info('step %d, scale x%d, loss %.6f (%.3f sec/batch)' % (global_train_step, scale, loss, duration))
    
    if (summary is not None):
      summary_writers[scale].add_summary(summary, global_step=global_train_step)
    
    if (local_train_step % FLAGS.save_freq == 0):
      model.save(base_path=FLAGS.train_path)
      tf.logging.info('saved a model checkpoint at step %d' % (global_train_step))

    
  # finalize
  tf.logging.info('finished')


if __name__ == '__main__':
  tf.app.run()