from isaacgym import gymapi
import torch


class KeyboardInterface():
    def __init__(self, env):
        env.gym.subscribe_viewer_keyboard_event(env.viewer,
                                                gymapi.KEY_W, 'forward')
        env.gym.subscribe_viewer_keyboard_event(env.viewer,
                                                gymapi.KEY_A, 'left')
        env.gym.subscribe_viewer_keyboard_event(env.viewer,
                                                gymapi.KEY_D, 'right')
        env.gym.subscribe_viewer_keyboard_event(env.viewer,
                                                gymapi.KEY_S, 'back')
        env.gym.subscribe_viewer_keyboard_event(env.viewer,
                                                gymapi.KEY_Q, 'yaw_left')
        env.gym.subscribe_viewer_keyboard_event(env.viewer,
                                                gymapi.KEY_E, 'yaw_right')
        env.gym.subscribe_viewer_keyboard_event(env.viewer,
                                                gymapi.KEY_R, "RESET")
        env.gym.subscribe_viewer_keyboard_event(env.viewer,
                                                gymapi.KEY_ESCAPE, "QUIT")
        print("______________________________________________________________")
        print("Using keyboard interface, overriding default comand settings")
        print("commands are in 1/5 increments of max.")
        print("WASD: forward, strafe left, "
              "backward, strafe right")
        print("QE: yaw left/right")
        print("R: reset environments")
        print("ESC: quit")
        print("______________________________________________________________")

        env.commands[:] = 0.
        env.cfg.commands.resampling_time = env.max_episode_length_s + 1
        self.max_vel_backward = -1.
        self.max_vel_forward = 4.
        self.increment_x = (self.max_vel_forward-self.max_vel_backward)*0.5

        self.max_vel_sideways = 1.0
        self.increment_y = self.max_vel_sideways*0.2

        self.max_vel_yaw = 2.0
        self.increment_yaw = self.max_vel_yaw*0.2

    def update(self, env):
        for evt in env.gym.query_viewer_action_events(env.viewer):
            if evt.value == 0:
                continue
            if evt.action == 'forward':
                if env.commands[0, 0] < -1.:
                    env.commands[:, 0] = -1.
                elif env.commands [0, 0] < 0.:
                    env.commands[:, 0] = 0.
                elif env.commands [0, 0] < 1.:
                    env.commands[:, 0] = 1.
                else:
                    env.commands[:, 0] = 4.
                # env.commands[:, 0] = torch.clamp(env.commands[:, 0]
                #                                  + self.increment_x,
                #                                  max=self.max_vel_forward)
            elif evt.action == 'back':
                if env.commands[0, 0] > 1.:
                    env.commands[:, 0] = 1.
                elif env.commands [0, 0] > 0.:
                    env.commands[:, 0] = 0.
                elif env.commands [0, 0] > -1.:
                    env.commands[:, 0] = -1.
                else:
                    env.commands[:, 0] = -4.
                # env.commands[:, 0] = torch.clamp(env.commands[:, 0]
                #                                  - self.increment_x,
                #                                  min=self.max_vel_backward)
            elif evt.action == 'left':
                # env.commands[:, 1] = torch.clamp(env.commands[:, 1]
                #                                  + self.increment_y,
                #                                  min=-self.max_vel_sideways)
                # similar to above, increment at -1, 0, 1
                if env.commands[0, 1] <= 0.:
                    env.commands[:, 1] = 1.
                else:
                    env.commands[:, 1] = 0.
            elif evt.action == 'right':
                # env.commands[:, 1] = torch.clamp(env.commands[:, 1]
                #                                  - self.increment_y,
                #                                  max=self.max_vel_sideways)
                if env.commands[0, 1] >= 0.:
                    env.commands[:, 1] = 0.
                else:
                    env.commands[:, 1] = -1.
            elif evt.action == 'yaw_left':
                # env.commands[:, 2] = torch.clamp(env.commands[:, 2]
                #                                  - self.increment_yaw,
                #                                  min=-self.max_vel_yaw)
                if env.commands[0, 2] < -3.:
                    env.commands[:, 2] = -3.
                elif env.commands[0, 2] < 0.:
                    env.commands[:, 2] = 0.
                elif env.commands[0, 2] < 3.:
                    env.commands[:, 2] = 3.
                else:
                    env.commands[:, 2] = 4.

            elif evt.action == 'yaw_right':
                # env.commands[:, 2] = torch.clamp(env.commands[:, 2]
                #                                  + self.increment_yaw,
                #                                  max=self.max_vel_yaw)
                if env.commands[0, 2] > 3.:
                    env.commands[:, 2] = 3.
                elif env.commands[0, 2] > 0.:
                    env.commands[:, 2] = 0.
                elif env.commands[0, 2] > -3.:
                    env.commands[:, 2] = -3.
                else:
                    env.commands[:, 2] = -4.

            elif evt.action == "QUIT":
                exit()
            elif evt.action == "RESET":
                env.timed_out[:] = True
                env.reset()
                env.commands[:] = 0.
