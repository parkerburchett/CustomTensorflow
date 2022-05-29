import tensorflow as tf

# this decorator will break it if you call it more than once
@tf.keras.utils.register_keras_serializable()
class FeatureReversalNoise(tf.keras.layers.Layer):
  """
    Randomly switches the sign of the input columns with probability `prob`
    It does not randomly switch exactly `prob` percent of features.
    The % of signs that are switched is normally distributed around `prob`

    Based on this Numerai forum post by https://forum.numer.ai/u/mdo:
    https://forum.numer.ai/t/feature-reversing-input-noise/1416

    example usage :
    model.add(tf.keras.Input(shape=(24)))
    model.add(FeatureReversalNoise(input_vector_length=24,prob=.2))

    Might want to rewrite this as an tf.keras.layers.Input layer. 
  """

  def __init__(self, input_vector_length:int, prob:float, name=None,**kwargs):
    """
    Syntax Source:
    https://stackoverflow.com/questions/62280161/saving-keras-models-with-custom-layers
    """
    if prob < 0 or prob > 1:
        raise ValueError(f'prob must be between (0,1) is currently {prob}')
    self.input_vector_length = input_vector_length
    self.prob = prob
    super(FeatureReversalNoise, self).__init__(**kwargs)


  def call(self, inputs, training=None):
        if training:
                probabilities = tf.random.uniform(shape=(1,self.input_vector_length))
                random_sign_switcher = (tf.cast(probabilities>self.prob,tf.float32)*2)-1
                # tensor of [1, -1, ..., 1, -1] where the odds of -1 == self.prob
                return tf.math.multiply(inputs, random_sign_switcher)
        else:
                return inputs


  def get_config(self):
        """
        Overriding this method to save and load models that include this layer
        """
        config = super().get_config().copy()
        config.update({
            'input_vector_length': self.input_vector_length,
            'prob': self.prob,
        })
        return config
