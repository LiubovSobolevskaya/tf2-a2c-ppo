"""Model for A2C and PPO"""
import tensorflow as tf
import tensorflow_probability as tfp


class Model(tf.keras.Model):
    """
        Conv Neural Network for A2C and PPO.
        Architecure taken from Openai Baselines.

        Args:
            action_dim (int): number of possible actions/Actor's outputs
    """
    def __init__(self, action_dim):

        relu_gain = tf.math.sqrt(2.0)
        relu_init = tf.initializers.Orthogonal(gain=relu_gain)

        super(Model, self).__init__()
        self.main = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32,
                                   input_shape=(84, 84, 4),
                                   kernel_size=(8, 8),
                                   strides=4,
                                   activation='relu',
                                   kernel_initializer=relu_init),
            tf.keras.layers.Conv2D(64,
                                   kernel_size=(4, 4),
                                   activation='relu',
                                   strides=2,
                                   kernel_initializer=relu_init),
            tf.keras.layers.Conv2D(32,
                                   kernel_size=(3, 3),
                                   activation='relu',
                                   strides=1,
                                   kernel_initializer=relu_init),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(512,
                                  activation='relu',
                                  kernel_initializer=relu_init)
        ])

        self.critic = tf.keras.Sequential([
            tf.keras.layers.Dense(
                1,
                activation=None,
                kernel_initializer=tf.keras.initializers.Orthogonal(gain=1.0))
        ])

        self.actor = tf.keras.Sequential([
            tf.keras.layers.Dense(
                action_dim,
                activation=None,
                kernel_initializer=tf.initializers.Orthogonal(gain=0.01))
        ])

    @tf.function
    def call(self, states, actions=None):

        main = self.main(states, training=True)

        value = tf.squeeze(self.critic(main), axis=1)
        actor_features = self.actor(main)
        dist = tfp.distributions.Categorical(logits=actor_features)
        if actions is None:
            actions = dist.sample()

        action_log_probs = dist.log_prob(actions)
        return actions, action_log_probs, dist.entropy(), value
