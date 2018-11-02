import math
import os

import numpy as np
import tensorflow as tf

from models.base_model import BaseModel

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_float('mobilenetv2_alpha', 1.0, 'Width parameter of the MobileNetV2 model.')
tf.flags.DEFINE_boolean('mobilenetv2_train_last_only', False, 'Set this argument to train the last layer only.')

def create_model():
  return MobileNetV2()

class MobileNetV2(BaseModel):
  def __init__(self):
    super().__init__()
  

  def prepare(self, is_training, input_size):
    base_model = tf.keras.applications.MobileNetV2((input_size, input_size, 3), alpha=FLAGS.mobilenetv2_alpha, include_top=False, pooling='avg')
    if (FLAGS.mobilenetv2_train_last_only):
      for layer in base_model.layers:
        layer.trainable = False
    
    x = base_model.output
    if (is_training):
      x = tf.keras.layers.Dropout(0.75)(x)
    x = tf.keras.layers.Dense(10, activation='softmax')(x)

    self.model = tf.keras.models.Model(base_model.input, x)
    self.model.summary()

    if (is_training):
      self.optimizer = tf.keras.optimizers.Adam(lr=FLAGS.learning_rate)
      self.model.compile(self.optimizer, loss=self._earth_mover_loss)
  
  def restore(self, restore_path):
    self.model.load_weights(restore_path)
  
  def train(self, train_generator, train_steps, train_epochs, validate_generator, validate_steps, save_path):
    self.checkpoint = tf.keras.callbacks.ModelCheckpoint(save_path, monitor='val_loss', mode='min', save_weights_only=True, save_best_only=False, verbose=1)

    self.model.fit_generator(
      generator=train_generator,
      steps_per_epoch=train_steps,
      epochs=train_epochs,
      callbacks=[self.checkpoint],
      validation_data=validate_generator,
      validation_steps=validate_steps,
      verbose=1
    )
  
  
  def _earth_mover_loss(self, truths, predictions):
    cdf_truths = tf.cumsum(truths, axis=-1)
    cdf_predictions = tf.cumsum(predictions, axis=-1)

    emd = tf.keras.backend.abs(cdf_truths - cdf_predictions)
    emd = tf.keras.backend.square(emd)
    emd = tf.keras.backend.mean(emd, axis=-1)
    emd = tf.keras.backend.sqrt(emd)
    emd = tf.keras.backend.mean(emd)

    return emd

