import os

import gym
import numpy as np
from gym import spaces
from keras.models import load_model
from numpy import unravel_index

import CONSTANT

# X = 1
# EMPTY = 0
# O = -1

MARK_COST = 1
WIN_REWARD = 100


class Game_environment(gym.Env):
    def __init__(self):
        super(Game_environment, self).__init__()
        self.state = np.zeros(shape=(CONSTANT.FIELD_SIZE, CONSTANT.FIELD_SIZE, 1), dtype=np.int)
        self.opponent = self.get_oponent()
        self.action_space = spaces.Box(low=0, high=CONSTANT.FIELD_SIZE - 1, shape=(2,), dtype=np.int)

    def get_oponent(self):
        model_is_exist = os.path.exists(CONSTANT.AGENT_MODEL_LOSS_PATH)
        if not model_is_exist:
            model = None
        else:
            model = load_model(CONSTANT.AGENT_MODEL_LOSS_PATH)
        return model

    def step(self, action):
        my_turn = self.do_turn(action, 1)
        reward = my_turn[1]
        if not my_turn[2]:
            state = my_turn[0]
            if self.opponent:
                opponent_action = self.opponent(np.stack([state * -1], axis=0))[0, :, :, 0]
                opponent_action = opponent_action.numpy()
                a = unravel_index(opponent_action.argmax(), opponent_action.shape)
            else:
                a = self.action_space.sample()
            opponent_turn = self.do_turn(a, -1)
            if opponent_turn[2]:
                reward = -100
        else:
            return my_turn
        return opponent_turn[0], reward, opponent_turn[2], {}

    def do_turn(self, action, MARK_CHAR):
        done = False
        line = action[0]
        column = action[1]
        if np.count_nonzero(self.state[:, :, 0] == 0) == 0:
            done = True
        if self.state[line][column][0] != 0:
            return self.get_state(), -50, done, {}

        self.state[line][column][0] = MARK_CHAR
        reward = 0
        y_up = max(line - 3, 0)
        y_down = min(line + 3 + 1, CONSTANT.FIELD_SIZE)
        X_left = max(column - 3, 0)
        x_right = min(column + 3 + 1, CONSTANT.FIELD_SIZE)

        region = self.state[y_up:y_down, X_left:x_right, 0]

        local_y = line - y_up
        local_x = column - X_left

        left_before = np.count_nonzero(region[local_y, :local_x + 1] == 1)
        reward = max(reward, MARK_COST * left_before)  # line left
        done = done or left_before >= 4
        left_after = np.count_nonzero(region[local_y, local_x:] == MARK_CHAR)
        if not np.all(region[local_y, local_x:] != MARK_CHAR * (-1)):
            reward = 0
        else:
            reward = max(reward, MARK_COST * left_after)  # line right
        done = done or left_after >= 4
        column_up = np.count_nonzero(region[:local_y + 1, local_x] == MARK_CHAR)
        if not np.all(region[:local_y + 1, local_x] != MARK_CHAR * (-1)):
            reward = 0
        else:
            reward = max(reward, MARK_COST * column_up)  # column up
        done = done or column_up >= 4
        column_down = np.count_nonzero(region[local_y:, local_x] == MARK_CHAR)
        if not np.all(region[local_y:, local_x] != MARK_CHAR * (-1)):
            reward = 0
        else:
            reward = max(reward, MARK_COST * column_down)  # column down
        done = done or column_down >= 4

        # diagonals
        up_left = region[:local_y + 1, :local_x + 1]
        mark_up_left = np.count_nonzero(up_left.diagonal() == MARK_CHAR)
        if not np.all(up_left.diagonal() != MARK_CHAR * (-1)):
            reward = 0
        else:
            reward = max(reward, MARK_COST * mark_up_left)
        done = done or mark_up_left >= 4

        up_right = region[:local_y + 1, local_x:]
        mark_up_right = np.count_nonzero(np.diag(np.fliplr(up_right)) == MARK_CHAR)
        if not np.all(up_right.diagonal() != MARK_CHAR * (-1)):
            reward = 0
        else:
            reward = max(reward, MARK_COST * mark_up_right)
        done = done or mark_up_right >= 4

        down_left = region[local_y:, :local_x + 1]
        mark_down_left = np.count_nonzero(np.diag(np.fliplr(down_left)) == MARK_CHAR)
        if not np.all(down_left.diagonal() != MARK_CHAR * (-1)):
            reward = 0
        else:
            reward = max(reward, MARK_COST * mark_down_left)
        done = done or mark_down_left >= 4

        down_right = region[local_y:, local_x:]
        mark_down_right = np.count_nonzero(down_right.diagonal() == MARK_CHAR)
        if not np.all(down_right.diagonal() != MARK_CHAR * (-1)):
            reward = 0
        else:
            reward = max(reward, MARK_COST * mark_down_right)
        done = done or mark_down_right >= 4
        if done:
            reward = 100

        obs = self.get_state()
        return obs, reward, done, {}

    def reset(self):
        # Reset the state of the environment to an initial state
        self.state = np.zeros(shape=(CONSTANT.FIELD_SIZE, CONSTANT.FIELD_SIZE, 1), dtype=np.int)
        return self.get_state()

    def get_state(self):
        return self.state
