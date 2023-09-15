import importlib

from gym.utils.task_registry import task_registry

# * To add a new env:
# * 1. add the base env and env class name and location to the class dict
# * 2. add the config name and location to the config dict
# * 3. add the runner confg name and location to the runner config dict
# * 3. register the task experiment name to the env/config/ppo classes

# * from y import x where {y:x}
class_dict = {
    'LeggedRobot': '.base.legged_robot',
    'FixedRobot': '.base.fixed_robot',
    'MiniCheetah': '.mini_cheetah.mini_cheetah',
    'MiniCheetahRef': '.mini_cheetah.mini_cheetah_ref',
    'MiniCheetahOsc': '.mini_cheetah.mini_cheetah_osc',
    'MiniCheetahOscIK': '.mini_cheetah.mini_cheetah_osc_IK',
    'MiniCheetahFloquet': '.mini_cheetah.mini_cheetah_floquet',
}

config_dict = {
    'MiniCheetahCfg': '.mini_cheetah.mini_cheetah_config',
    'MiniCheetahRefCfg': '.mini_cheetah.mini_cheetah_ref_config',
    'MiniCheetahOscCfg': '.mini_cheetah.mini_cheetah_osc_config',
    'MiniCheetahOscIKCfg': '.mini_cheetah.mini_cheetah_osc_IK_config',
    'MiniCheetahFloquetCfg': '.mini_cheetah.mini_cheetah_floquet_config',
}

runner_config_dict = {
    'MiniCheetahRunnerCfg': '.mini_cheetah.mini_cheetah_config',
    'MiniCheetahRefRunnerCfg': '.mini_cheetah.mini_cheetah_ref_config',
    'MiniCheetahOscRunnerCfg': '.mini_cheetah.mini_cheetah_osc_config',
    'MiniCheetahOscIKRunnerCfg': '.mini_cheetah.mini_cheetah_osc_IK_config',
    'MiniCheetahFloquetRunnerCfg': '.mini_cheetah.mini_cheetah_floquet_config',
}

task_dict = {
    'mini_cheetah': ['MiniCheetah', 'MiniCheetahCfg', 'MiniCheetahRunnerCfg'],
    'mini_cheetah_ref':
        ['MiniCheetahRef', 'MiniCheetahRefCfg', 'MiniCheetahRefRunnerCfg'],
    'mini_cheetah_osc':
        ['MiniCheetahOsc', 'MiniCheetahOscCfg', 'MiniCheetahOscRunnerCfg'],
    'mini_cheetah_osc_IK':['MiniCheetahOscIK', 'MiniCheetahOscIKCfg',
                           'MiniCheetahOscIKRunnerCfg'],
    'mini_cheetah_floquet': ['MiniCheetahFloquet', 'MiniCheetahFloquetCfg',
                             'MiniCheetahFloquetRunnerCfg'],
}

for class_name, class_location in class_dict.items():
    locals()[class_name] = getattr(
        importlib.import_module(class_location, __name__), class_name)
for config_name, config_location in config_dict.items():
    locals()[config_name] = getattr(
        importlib.import_module(config_location, __name__), config_name)
for runner_config_name, runner_config_location in runner_config_dict.items():
    locals()[runner_config_name] = getattr(
        importlib.import_module(
            runner_config_location, __name__),
        runner_config_name)

for task_name, class_list in task_dict.items():
    task_registry.register(
        task_name,
        locals()[class_list[0]],
        locals()[class_list[1]](),
        locals()[class_list[2]]())
