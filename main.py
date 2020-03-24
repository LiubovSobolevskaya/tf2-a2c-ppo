"""main file of A2C implementation for atari"""
import os
from datetime import datetime

import numpy as np
import tensorflow as tf
from absl import app, flags
from tqdm import tqdm

from a2c import A2C
from envs import make_vec_envs

FLAGS = flags.FLAGS

flags.DEFINE_integer('seed', 42, 'Fixed random seed for training.')
flags.DEFINE_float('gamma', 0.99, 'Discount used for returns.')
flags.DEFINE_float('lr', 7e-4, 'Learning rate.')
flags.DEFINE_float('entropy_coef', 0.01, 'Entropy coeficient.')
flags.DEFINE_float('value_loss_coef', 0.5, 'Value loss coefficient.')
flags.DEFINE_string('env_name', 'PongNoFrameskip-v4', 'Name of the used env')
flags.DEFINE_integer('max_timesteps', int(1e7), 'Max timesteps to train.')
flags.DEFINE_integer('num_processes', 16, 'Num of parallel processes.')
flags.DEFINE_integer('num_steps', 5, 'Num of steps of enviroment rollout.')
flags.DEFINE_integer('log_interval', 100, 'Log every N updates.')
flags.DEFINE_string('logs_dir', 'logs/pong', 'Directory to save logs to.')
flags.DEFINE_string('save_dir', 'runs',
                    'Directory to save tensorboard logs to.')
flags.DEFINE_boolean('debug', False, 'Whether to execute all code eagerly.')


def main(_):

    if FLAGS.debug:
        tf.config.experimental_run_functions_eagerly(True)

    os.makedirs(FLAGS.logs_dir, exist_ok=True)

    tf.random.set_seed(FLAGS.seed)

    envs = make_vec_envs(FLAGS.env_name, FLAGS.seed, FLAGS.num_processes,
                         FLAGS.logs_dir)

    for gpu in tf.config.experimental.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)

    now = datetime.now()
    date_time = now.strftime("%H:%M:%S")

    hparam_str_dict = dict(date_time=date_time,
                           seed=FLAGS.seed,
                           env_name=FLAGS.env_name)

    hparam_str = ','.join([
        '%s=%s' % (k, str(hparam_str_dict[k]))
        for k in sorted(hparam_str_dict.keys())
    ])

    writer = tf.summary.create_file_writer(
        os.path.join(FLAGS.save_dir, 'tb', hparam_str))

    writer.set_as_default()

    actor_critic = A2C((-1, *envs.observation_space.shape),
                       envs.action_space.n, FLAGS.lr, FLAGS.entropy_coef,
                       FLAGS.value_loss_coef, FLAGS.gamma, FLAGS.num_steps)

    def get_obs():
        return envs.stackedobs

    def env_step(action):
        next_obs, reward, done, _ = envs.step(action)
        return next_obs, reward.astype(np.float32), done.astype(np.float32)

    batch_size = FLAGS.num_steps * FLAGS.num_processes
    num_updates = FLAGS.max_timesteps // batch_size

    envs.reset()
    val_loss, act_loss, ent_loss = 0, 0, 0

    for i in tqdm(range(num_updates), unit_scale=batch_size, smoothing=0.1):

        actor_critic.set_learning_rate(FLAGS.lr * (1.0 - i / num_updates))

        value_loss, action_loss, entropy_loss = actor_critic.update(
            env_step, get_obs)

        val_loss += value_loss
        act_loss += action_loss
        ent_loss += entropy_loss

        if i % FLAGS.log_interval == 0 and i > 0:
            tf.summary.scalar("losses/value_loss",
                              val_loss / FLAGS.log_interval,
                              step=batch_size * i)
            tf.summary.scalar("losses/action_loss",
                              act_loss / FLAGS.log_interval,
                              step=batch_size * i)
            tf.summary.scalar("losses/entropy_loss",
                              ent_loss / FLAGS.log_interval,
                              step=batch_size * i)
            tf.summary.flush()
            val_loss = 0
            act_loss = 0
            ent_loss = 0


if __name__ == '__main__':
    app.run(main)
