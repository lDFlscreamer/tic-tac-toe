import io
import os
from datetime import datetime

import gym
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from numpy import unravel_index
from tensorflow.keras.layers import *
from tensorflow.keras.models import *

import CONSTANT
from env.environment import WIN_REWARD, LOSE_PENALTY

AGENT_METRIC = 'mae'
AGENT_ACTIVATION = "linear"


class Game_agent:

    def __init__(self, log_dir=None):
        self.gamma = 0.4
        self.epsilon = 1
        self.epsilon_min = 0.002
        self.epsilon_decay = 0.995
        self.save_frequency = 50
        self.image_verbose_frequency = 50
        self.initial_NN_verbose_frequency = 50
        self.NN_verbose_frequency = self.initial_NN_verbose_frequency
        self.step = 0

        self.memory = {1: list(), -1: list()}
        self.model = self.get_bot()
        if not log_dir:
            log_dir = CONSTANT.TENSORBOARD_REINFORCEMENT_LEARNING + datetime.now().strftime("%Y%m%d-%H-%M-%S")
        self.summary_writer = tf.summary.create_file_writer(logdir=log_dir)

    def create_model(self):
        # todo: model architecture
        input: Input = Input(shape=(CONSTANT.FIELD_SIZE, CONSTANT.FIELD_SIZE, 1), name="input")

        first_part = tf.math.greater_equal(input, tf.constant(1, tf.float32))
        first_part = tf.cast(first_part, tf.float32)

        second_part = tf.math.greater_equal(input * (-1), tf.constant(1, tf.float32))
        second_part = tf.cast(second_part, tf.float32)

        two_channel_state = Concatenate(name="two_channel_state")([first_part, second_part])

        net = Flatten()(two_channel_state)

        net = Dense(input_shape=(None,98),units=98,activation=None, name="7x7_input_to_Dense")(net)
        # net = BatchNormalization()(net)
        net = Activation("linear")(net)

        net = Dense(units=256, activation=None, name="Dense_first")(net)
        # net = BatchNormalization()(net)
        net = Activation("linear")(net)
        net = Dropout(0.4)(net)

        net = Dense(units=128, activation=None, name="Dense_second")(net)
        # net = BatchNormalization()(net)
        net = Activation("linear")(net)
        net = Dropout(0.4)(net)

        net = Dense(units=CONSTANT.FIELD_SIZE * CONSTANT.FIELD_SIZE, activation=None, name="output")(net)
        net = Activation("linear")(net)

        net = Reshape((CONSTANT.FIELD_SIZE, CONSTANT.FIELD_SIZE))(net)

        model = Model(inputs=input, outputs=net)
        model.compile(loss="mse",
                      optimizer='adam',
                      metrics=[
                          AGENT_METRIC
                      ])
        model.summary()
        return model

    def get_bot(self):
        if hasattr(self, 'model'):
            return self.model

        model_is_exist = os.path.exists(CONSTANT.AGENT_TEMP_MODEL_PATH)
        if not model_is_exist:
            model = self.create_model()
        else:
            model = load_model(CONSTANT.AGENT_TEMP_MODEL_PATH)
        model.compile(loss="mse",
                      optimizer='adam',
                      metrics=[
                          AGENT_METRIC,
                          "mse"
                      ])
        return model

    def reinforcement_learning(self, env: gym.Env, num_episodes=250):
        for i in range(num_episodes):
            epoch = 0
            s = env.reset()
            self.epsilon *= self.epsilon_decay
            self.epsilon = max(self.epsilon_min, self.epsilon)
            self.NN_verbose_frequency = int(self.initial_NN_verbose_frequency * self.epsilon)
            if i % max(self.save_frequency, 1) == 0:
                print("Episode {} of {}".format(i + 1, num_episodes))
                self.model.save(CONSTANT.AGENT_TEMP_MODEL_PATH)
                print('save %d version' % i)
            done = False
            while not done:
                model_state = np.stack([s], axis=0)
                if np.random.random() < self.epsilon:
                    a = env.action_space.sample()
                else:
                    q_s = self.model(model_state)[0, :, :]
                    q_s = q_s.numpy()
                    a = unravel_index(q_s.argmax(), q_s.shape)
                new_s, r, done, opp = env.step(a)

                self.memory[1].append((model_state, a, r, done))
                if done:
                    last_O = self.memory[-1][-1]
                    self.memory[-1][-1] = (last_O[0], last_O[1], -LOSE_PENALTY, last_O[3])
                if opp:
                    self.memory[-1].append(opp)
                with self.summary_writer.as_default():
                    with tf.name_scope('debug_data'):
                        tf.summary.scalar("epsilon", data=self.epsilon, step=1)
                        tf.summary.scalar("NN_verbose_frequency", self.NN_verbose_frequency, step=1)
                epoch += 1
                s = new_s
            self.model.compiled_metrics.reset_state()
            is_verbose = (i % self.image_verbose_frequency == 0)
            is_NN_verbose = (i % self.NN_verbose_frequency == 0)
            self.train_network(self.model, self.memory[1], image_verbose=is_verbose, target_snapshot=is_verbose,
                               is_NN_verbose=is_NN_verbose)
            self.train_network(self.model, self.memory[-1], image_verbose=is_verbose, target_snapshot=is_verbose,
                               is_NN_verbose=is_NN_verbose, char='-1')

    def train_network(self, model: Model, memory, image_verbose, target_snapshot=False, is_NN_verbose=False, char=''):
        for i in range(len(memory) - 1, -1, -1):
            model_state = memory[i][0]
            a = memory[i][1]
            reward = memory[i][2]
            done = memory[i][3]
            if not done and i + 1 <= len(memory) - 1:
                new_m_state = memory[i + 1][0]
                predict_new_state = model(new_m_state)[0, :, :]
                possible_reward = self.gamma * np.max(predict_new_state)
                target = reward + self.gamma * possible_reward
            else:
                target = reward
            target_vec = model(model_state)
            target_vec = target_vec.numpy()
            abs_max = abs(max(target_vec.max(), target_vec.min(), key=abs))
            if abs_max > WIN_REWARD:
                target_vec = (target_vec / abs_max) * WIN_REWARD
            target_vec[0][a[0]][a[1]] = target
            self.train_step(model, model_state, target_vec)
            # self.model.fit(model_state, target_vec, verbose=0)
            with self.summary_writer.as_default():
                with tf.name_scope('game_data'):
                    if i == 0 and char == '':
                        tf.summary.scalar(f"game_turn_amount{char}", data=len(memory), step=1)
                if is_NN_verbose and i == 0:
                    for layer in self.model.layers:
                        if len(layer.weights) > 0:
                            with tf.name_scope(layer.name):
                                for weight in layer.weights:
                                    tf.summary.histogram(name=weight.name, data=weight, step=1)
                with tf.name_scope(f'{char}_data'):
                    if image_verbose:
                        if i == len(memory) - 1:
                            tf.summary.image(f"result_end", self.matrix_to_img(model_state[0, :, :], reward, a),
                                             step=self.step)
                        elif i == 0:
                            tf.summary.image(f"result_start", self.matrix_to_img(model_state[0, :, :], reward, a),
                                             step=self.step)
                    with tf.name_scope('target_snapshot'):
                        if target_snapshot:
                            if i == len(memory) - 1:
                                tf.summary.image(f"target_end", self.matrix_to_img(target_vec[0, :, :], reward, a),
                                                 step=self.step)
                            elif i == 0:
                                tf.summary.image(f"target_start", self.matrix_to_img(target_vec[0, :, :], reward, a),
                                                 step=self.step)
                    tf.summary.scalar(f"reward_train{char}", data=reward, step=1)
                for m in model.metrics:
                    if m.name != "loss":
                        with tf.name_scope('game_data'):
                            tf.summary.scalar(m.name, data=float(m.result()), step=1)
            self.step += 1
        memory.clear()

    def matrix_to_img(self, data, reward, action):
        fig, ax = plt.subplots()
        im = ax.imshow(data)
        # Loop over data dimensions and create text annotations.
        for i in range(len(data)):
            for j in range(len(data[i])):
                value = data[i, j]
                text = ax.text(j, i, round(value, 1),
                               ha="center", va="center", color="w")
        ax.set_title(f"reward:{reward}, action=[{action[0]}:{action[1]}]")
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close(fig)
        buf.seek(0)

        # Use tf.image.decode_png to convert the PNG buffer
        # to a TF image. Make sure you use 4 channels.
        image = tf.image.decode_png(buf.getvalue(), channels=4)

        # Use tf.expand_dims to add the batch dimension
        image = tf.expand_dims(image, 0)

        return image

    @tf.function
    def train_step(self, model: Model, x, y):
        with tf.GradientTape() as tape:
            y_pred = model(x, training=True)  # Forward pass
            y = tf.convert_to_tensor(y)
            # Compute the loss value
            # (the loss function is configured in `compile()`)
            loss = model.compiled_loss(y, y_pred, regularization_losses=model.losses)

        # Compute gradients
        trainable_vars = model.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        model.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # model.optimizer.minimize(loss, trainable_vars, tape=tape)
        # Update metrics (includes the metric that tracks the loss)
        model.compiled_metrics.update_state(y, y_pred)
        # Return a dict mapping metric names to current value
