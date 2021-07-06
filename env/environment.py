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
NOT_EMPTY_PENALTY = 100
WIN_REWARD = 250
LOSE_PENALTY = 250


class Game_environment(gym.Env):
    def __init__(self):
        super(Game_environment, self).__init__()
        self.state = np.zeros(shape=(CONSTANT.FIELD_SIZE, CONSTANT.FIELD_SIZE, 1), dtype=np.int)
        self.opponent = self.get_oponent()
        self.action_space = spaces.Box(low=0, high=CONSTANT.FIELD_SIZE - 1, shape=(2,), dtype=np.int)

    def get_oponent(self):
        model_is_exist = os.path.exists(CONSTANT.AGENT_TEMP_MODEL_PATH)
        if not model_is_exist:
            model = None
        else:
            model = load_model(CONSTANT.AGENT_TEMP_MODEL_PATH)
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
                # print(a)
            else:
                a = self.action_space.sample()
            opponent_turn = self.do_turn(a, -1)
            if opponent_turn[2]:
                reward = -LOSE_PENALTY
        else:
            return my_turn
        return opponent_turn[0], reward, opponent_turn[2], {}

    def do_turn(self, action, MARK_CHAR):
        done = False
        line = action[0]
        column = action[1]
        if np.count_nonzero(self.state[:, :, 0] == 0) == 0:
            return self.get_state(), 0, done, {}
        if self.state[line][column][0] != 0:
            return self.get_state(), -NOT_EMPTY_PENALTY, done, {}

        self.state[line][column][0] = MARK_CHAR
        reward = 0
        y_up = max(line - 3, 0)
        y_down = min(line + 3 + 1, CONSTANT.FIELD_SIZE)
        X_left = max(column - 3, 0)
        x_right = min(column + 3 + 1, CONSTANT.FIELD_SIZE)

        region = self.state[y_up:y_down, X_left:x_right, 0]

        local_y = line - y_up
        local_x = column - X_left

        line = region[:, local_x]
        line_m = self.count_mark(line, MARK_CHAR)
        done = done or line_m[0]
        reward += MARK_COST * line_m[1]

        column = region[local_y, :]
        column_m = self.count_mark(column, MARK_CHAR)
        done = done or column_m[0]
        reward += MARK_COST * column_m[1]

        diagonal = np.diag(region, k=(local_x - local_y))
        diagonal_m = self.count_mark(diagonal, MARK_CHAR)
        done = done or diagonal_m[0]
        reward += MARK_COST * diagonal_m[1]

        flip_diagonal = np.diag(np.fliplr(region), k=(local_x - local_y))
        flip_diagonal_m = self.count_mark(flip_diagonal, MARK_CHAR)
        done = done or flip_diagonal_m[0]
        reward += MARK_COST * flip_diagonal_m[1]

        if done:
            reward = WIN_REWARD

        obs = self.get_state()
        return obs, reward, done, {}

    def reset(self):
        # Reset the state of the environment to an initial state
        self.state = np.zeros(shape=(CONSTANT.FIELD_SIZE, CONSTANT.FIELD_SIZE, 1), dtype=np.int)
        return self.get_state()

    def get_state(self):
        return self.state

    def count_mark(self, arr, MARK_CHAR):
        count = 0
        for i in range(0, len(arr) - 4):
            curr = arr[i:i + 5]
            if np.all(curr != MARK_CHAR * (-1)):
                count = max(count, np.count_nonzero(curr == MARK_CHAR))
                if count == 4:
                    return True, 4
        return False, count
