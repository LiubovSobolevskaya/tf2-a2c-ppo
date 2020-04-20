"""Actor Critic class for PPO"""
import tensorflow as tf

from model import Model


class PPO():
    """
    Class that implements PPO training
        Args:
            obs_shape (tuple): Shape of the observations.
            action_n (int): Number of possible actions inr the enviroment.
            entropy_coef (float): Entropy coefficient.
            value_loss_coef (float):  value loss coefficient.
            gamma (float): Discount rate.
            num_processes (int):
            num_steps (int): Number of parallel enviroments.
            ppo_epoch (int): Number of ppo updates
            num_mini_batch (int): Number of batches used in each ppo update
            learning_rate (float): The initial learning rate.
            clip_param (float): Clipped Surrogate Objective parameter
            gae_lambda (float): Generalized Advantage Estimation lambda
        Returns:
            tuple:  value, action, and entropy losses
    """
    def __init__(self, obs_shape, action_n, entropy_coef, value_loss_coef,
                 gamma, num_processes, num_steps, ppo_epoch, num_mini_batch,
                 learning_rate, clip_param, gae_lambda):
        self.num_steps = num_steps
        self.num_processes = num_processes
        self.gamma = gamma
        self.obs_shape = obs_shape
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.num_mini_batch = num_mini_batch
        self.model = Model(action_n)
        self.clip_param = clip_param
        self.gae_lambda = gae_lambda
        self.ppo_epoch = ppo_epoch
        self.learning_rate = tf.Variable(learning_rate)
        self.optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.learning_rate, epsilon=1e-5)

    def set_learning_rate(self, learning_rate):
        """Update learning rate."""
        self.learning_rate.assign(learning_rate)

    @tf.function
    def update(self, env_step, get_obs):
        """
            Implements PPO update.
            Args:
                env_step (function): Numpy enviroment function that performs enviroment step.
                get_obs (function): Numpy enviroment function that returns current observations.

            Returns:
                 Value loss, Action_loss and Entropy_loss.
        """
        observations = []
        masks = []
        rewards = []
        values = []
        old_log_probs = []
        actions = []
        obs = tf.numpy_function(func=get_obs, inp=[], Tout=tf.float32)

        # Shape inference is lost due to numpy_function.
        obs = tf.reshape(obs, self.obs_shape)
        observations.append(obs)

        for _ in range(self.num_steps):
            action, old_log_prob, _, value = self.model(obs)
            obs, reward, done = tf.numpy_function(func=env_step,
                                                  inp=[action],
                                                  Tout=(tf.float32, tf.float32,
                                                        tf.float32))

            obs = tf.reshape(obs, self.obs_shape)
            mask = 1.0 - done

            observations.append(obs)
            old_log_probs.append(old_log_prob)
            rewards.append(reward)
            masks.append(mask)
            values.append(value)
            actions.append(action)

        next_value = self.model(obs)[-1]

        returns = []
        values.append(next_value)
        gae = 0
        for step in reversed(range(self.num_steps)):
            delta = rewards[step] + self.gamma * values[
                step + 1] * masks[step] - values[step]
            gae = delta + self.gamma * self.gae_lambda * masks[step] * gae
            returns.insert(0, gae + values[step])

        tf_returns = tf.concat(returns, axis=0)
        tf_observations = tf.concat(observations[:-1], axis=0)

        tf_actions = tf.concat(actions, axis=0)
        tf_old_log_probs = tf.concat(old_log_probs, axis=0)
        tf_values = tf.concat(values[:-1], axis=0)

        tf_adv_target = tf_returns - tf_values
        tf_adv_target = (tf_adv_target - tf.reduce_mean(tf_adv_target)) / (
            tf.math.reduce_std(tf_adv_target) + 1e-5)

        batch_size = self.num_processes * self.num_steps
        mini_batch_size = batch_size // self.num_mini_batch

        value_loss_epoch = 0.0
        action_loss_epoch = 0.0
        entropy_loss_epoch = 0.0

        for _ in range(self.ppo_epoch):
            indx = tf.random.shuffle(tf.range(batch_size))
            indx = tf.reshape(indx, (-1, mini_batch_size))
            for sample in indx:
                obs_batch = tf.gather(tf_observations, sample)
                returns_batch = tf.gather(tf_returns, sample)
                adv_target_batch = tf.gather(tf_adv_target, sample)
                action_batch = tf.gather(tf_actions, sample)
                old_log_probs_batch = tf.gather(tf_old_log_probs, sample)
                values_batch = tf.gather(tf_values, sample)

                with tf.GradientTape() as tape:
                    _, action_log_probs, dist_entropy, value = self.model(
                        obs_batch, action_batch)

                    ratio = tf.exp(action_log_probs - old_log_probs_batch)
                    surr1 = -ratio * adv_target_batch
                    surr2 = -tf.clip_by_value(
                        ratio, 1.0 - self.clip_param,
                        1.0 + self.clip_param) * adv_target_batch
                    action_loss = tf.reduce_mean(tf.maximum(surr1, surr2))

                    value_pred_clipped = values_batch + tf.clip_by_value(
                        value - values_batch, -self.clip_param,
                        self.clip_param)
                    value_losses = tf.square(value - returns_batch)
                    value_losses_clipped = tf.square(value_pred_clipped -
                                                     returns_batch)
                    value_loss = 0.5 * tf.reduce_mean(
                        tf.maximum(value_losses, value_losses_clipped))

                    entropy_loss = tf.reduce_mean(dist_entropy)

                    loss = (self.value_loss_coef * value_loss + action_loss -
                            entropy_loss * self.entropy_coef)

                gradients = tape.gradient(loss, self.model.trainable_variables)
                gradients, _ = tf.clip_by_global_norm(gradients, 0.5)

                self.optimizer.apply_gradients(
                    zip(gradients, self.model.trainable_variables))

                value_loss_epoch += value_loss
                action_loss_epoch += action_loss
                entropy_loss_epoch += entropy_loss

        num_updates = self.ppo_epoch * self.num_mini_batch

        value_loss_epoch /= num_updates
        action_loss_epoch /= num_updates
        entropy_loss_epoch /= num_updates

        return value_loss_epoch, action_loss_epoch, entropy_loss_epoch
