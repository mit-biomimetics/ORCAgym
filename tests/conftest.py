import isaacgym
import torch
import sys
import pytest
from unittest.mock import patch

from gym.envs import *
from gym.utils import get_args
from gym.utils import task_registry as _task_registry

from torch.multiprocessing import set_start_method

environment_dict = {}
environment_list = []


def pytest_sessionstart(session):
    set_start_method('spawn')
    with patch.object(sys, 'argv', sys.argv[:1]):
        args = get_args()
        args.headless = True
        _task_registry.make_gym()
        env_names = _task_registry.task_classes.keys()
        env_names =[]
        for env_name in env_names:
            print(env_name)
            args.task = env_name
            env_cfg, _ = _task_registry.create_cfgs(args)
            _task_registry.update_sim_cfg(args)
            _task_registry.make_sim()
            env = _task_registry.make_env(name=env_name,
                                          env_cfg=env_cfg)
            environment_dict[env_name] = env
            environment_list.append(environment_dict[env_name])
            _task_registry._gym.destroy_sim(_task_registry._sim)


@pytest.fixture
def task_registry():
    return _task_registry


@pytest.fixture
def args(monkeypatch):
    monkeypatch.setattr(sys, 'argv', sys.argv[:1])
    return get_args()


@pytest.fixture
def env_cfg():
    return env_cfg


@pytest.fixture
def train_cfg():
    return train_cfg


@pytest.fixture
def env_dict():
    return environment_dict


@pytest.fixture
def env_list():
    return environment_list


@pytest.fixture
def mini_cheetah():
    return environment_dict['mini_cheetah']