class TestEnvironment:
    def test_all_rewards_have_right_shape(self, env_list):

        def generate_message(env, reward_name):
            name = env.__class__.__name__
            message = f"Wrong shape for {reward_name} in {name}"
            return message

        for env in env_list:
            for item in dir(env):
                if "_reward_" in item:
                    reward_name = item.replace("_reward_", "")
                    assert len(env._eval_reward(reward_name).shape) == 1, \
                        generate_message(env, reward_name)

    def test_extras(self, env_list):

        def generate_message(env):
            message = f"Environment {env.__class__.__name__} has extras: "
            message += f"{', '.join(list(env.extras.keys()))}"
            return message

        for env in env_list:
            assert len(env.extras) == 0, generate_message(env)
