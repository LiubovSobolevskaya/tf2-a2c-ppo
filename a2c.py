"""Actor Critic class for A2C"""
import tensorflow as tf
import tensorflow_addons as tfa

from model import Model


class A2C():
    """
    Class that implements A2C training

        Args:
            obs_shape (tuple): Shape of the observations.
            action_n (int): Number of possible actions inr the enviroment.
            learning_rate (float): The initial learning rate.
            entropy_coef (float): Entropy coefficient.
            value_loss_coef (float):  value loss coefficient.
            gamma (float): Discount rate.
            num_steps (int): Number of steps in trajectory.
        Returns:
            bool: The return value. True for success, False otherwise.
    """
    def __init__(self, obs_shape, action_n, entropy_coef, value_loss_coef,
                 gamma, num_steps, learning_rate):

        self.num_steps = num_steps

        self.gamma = gamma
        self.obs_shape = obs_shape
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef

        self.model = Model(action_n)

        self.learning_rate = tf.Variable(learning_rate)
        self.optimizer = tfa.optimizers.RectifiedAdam(
            learning_rate=self.learning_rate,
            epsilon=1e-5,
            beta_1=0.0,
            beta_2=0.99)

    def set_learning_rate(self, learning_rate):
        """Update learning rate."""
        self.learning_rate.assign(learning_rate)

    @tf.function
    def update(self, env_step, get_obs):
        """
            Implements A2C update.

            Args:
                env_step (function): Numpy enviroment function that performs enviroment step.
                get_obs (function): Numpy enviroment function that returns current observations.

            Returns:
                 Value loss, Action_loss and Entropy_loss.
        """

        masks = []
        rewards = []
        values = []
        log_probs = []
        entropies = []
        obs = tf.numpy_function(func=get_obs, inp=[], Tout=tf.float32)

        # Shape inference is lost due to numpy_function.
        obs = tf.reshape(obs, self.obs_shape)

        with tf.GradientTape() as tape:
            for j in range(self.num_steps):

                action, log_prob, entropy, value = self.model(obs)
                obs, reward, done = tf.numpy_function(func=env_step,
                                                      inp=[action],
                                                      Tout=(tf.float32,
                                                            tf.float32,
                                                            tf.float32))

                obs = tf.reshape(obs, self.obs_shape)

                mask = 1 - done

                log_probs.append(log_prob)
                values.append(value)
                rewards.append(reward)
                masks.append(mask)
                entropies.append(entropy)

            next_value = self.model(obs)[-1]

            returns = [next_value]
            for j in reversed(range(self.num_steps)):
                returns.insert(0,
                               rewards[j] + masks[j] * self.gamma * returns[0])

            value_loss = 0.0
            action_loss = 0.0
            entropy_loss = 0.0

            for j in range(self.num_steps):
                advantages = tf.stop_gradient(returns[j]) - values[j]

                value_loss += tf.reduce_mean(tf.square(advantages))

                action_loss += -tf.reduce_mean(
                    tf.stop_gradient(advantages) * log_probs[j])

                entropy_loss += tf.reduce_mean(entropies[j])

            value_loss /= self.num_steps
            action_loss /= self.num_steps
            entropy_loss /= self.num_steps
            loss = (self.value_loss_coef * value_loss + action_loss -
                    entropy_loss * self.entropy_coef)

        gradients = tape.gradient(loss, self.model.trainable_variables)
        gradients, _ = tf.clip_by_global_norm(gradients, 0.5)

        self.optimizer.apply_gradients(
            zip(gradients, self.model.trainable_variables))
        return value_loss, action_loss, entropy_loss
