"""Atari Enviroment"""
import os

from baselines import bench
from baselines.common.atari_wrappers import make_atari, wrap_deepmind
from baselines.common.vec_env.shmem_vec_env import ShmemVecEnv
from baselines.common.vec_env.vec_frame_stack import VecFrameStack


def make_env(env_name, seed, rank, log_dir):
    """
        Creates atari enviroment.

        Args:
            env_name (string): Name of the enviroment.
            seed (int): Random seed for the enviroment.
            rank (int): Parallel process id.
            log_dir (string): Directory to save logs for visualize.ipynb.

        Returns:
            Function that creates an atari enviroment.
    """
    def _thunk():

        env = make_atari(env_name)

        env.seed(seed + rank)
        env = bench.Monitor(env, os.path.join(log_dir, str(rank)))

        env = wrap_deepmind(env, scale=True)
        return env

    return _thunk


def make_vec_envs(env_name, seed, num_processes, log_dir):
    """
        Creates atari enviroments.

        Args:
            env_name (string): Name of the enviroment.
            seed (int): Random seed for the enviroment.
            num_processes (int): Number of parallel enviroments.
            log_dir (string): Directory to save logs for visualize.ipynb.

        Returns:
            Parallel atari enviroments.

    """

    envs = [make_env(env_name, seed, i, log_dir) for i in range(num_processes)]
    envs = ShmemVecEnv(envs, context='fork')
    envs = VecFrameStack(envs, 4)

    return envs
