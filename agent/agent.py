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
from env.environment import WIN_REWARD

AGENT_METRIC = 'mae'
AGENT_ACTIVATION = "linear"


class Game_agent:

    def __init__(self, log_dir=None):
        self.gamma = 0.4
        self.epsilon = 1
        self.epsilon_min = 0.06
        self.epsilon_decay = 0.999
        self.step = 0

        self.memory = {1: list(), -1: list()}
        self.model = self.get_bot()
        if not log_dir:
            log_dir = CONSTANT.TENSORBOARD_REINFORCEMENT_LEARNING + datetime.now().strftime("%Y%m%d-%H-%M-%S")
        self.summary_writer = tf.summary.create_file_writer(logdir=log_dir)

    def create_model(self):
        # todo: model architecture
        input: Input = Input(shape=(CONSTANT.FIELD_SIZE, CONSTANT.FIELD_SIZE, 1), name="input")
        net = input

        first = Conv2D(filters=8, kernel_size=(3, 3), padding="same",
                       activation=None, name="first")(net)
        first = BatchNormalization()(first)
        first = Activation(AGENT_ACTIVATION)(first)

        second = Conv2D(filters=8, kernel_size=(5, 5), padding="same",
                        activation=None, name="second")(net)
        second = BatchNormalization()(second)
        second = Activation(AGENT_ACTIVATION)(second)

        third = Conv2D(filters=8, kernel_size=(7, 7), padding="same",
                       activation=None, name="third")(net)
        third = BatchNormalization()(third)
        third = Activation(AGENT_ACTIVATION)(third)

        net = Concatenate(name="concat1")([first, second, third])

        net = Conv2D(filters=3, kernel_size=(7, 7), padding="same",
                       activation=None)(net)
        net = BatchNormalization()(net)
        net = Activation(AGENT_ACTIVATION)(net)

        net = Concatenate(name="concat2")([net, input])

        net = Flatten()(net)

        net = Dense(256, activation=None, )(net)
        net = Activation("linear")(net)

        net = Dense(128, activation=None, )(net)
        net = Activation("linear")(net)

        net = Dense(CONSTANT.FIELD_SIZE * CONSTANT.FIELD_SIZE, activation=None)(net)
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
                          AGENT_METRIC
                      ])
        return model

    def reinforcement_learning(self, env: gym.Env, num_episodes=250):
        for i in range(num_episodes):
            epoch = 0
            s = env.reset()
            self.epsilon *= self.epsilon_decay
            self.epsilon = max(self.epsilon_min, self.epsilon)
            if i % 50 == 0:
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
                if opp:
                    self.memory[-1].append(opp)
                # if not done:
                #     model_new_state = np.stack([new_s], axis=0)
                #     predict_new_state = self.model(model_new_state)[0, :, :, 0]
                #     possible_reward = self.gamma * np.max(predict_new_state)
                #     target = r + self.gamma * possible_reward
                # else:
                #     target = r
                # target_vec = self.model(model_state)
                # target_vec = target_vec.numpy()
                # target_vec[0][a[0]][a[1]][0] = target
                # self.model.fit(model_state, target_vec, initial_epoch=epoch, epochs=1, verbose=1,
                #                callbacks=self.get_callbacks(epoch))
                # self.train_step(self.model, model_state, target_vec)

                with self.summary_writer.as_default():
                    tf.summary.scalar("reward", data=r, step=1)
                    tf.summary.scalar("epsilon", data=self.epsilon, step=1)
                epoch += 1
                s = new_s
                self.model.compiled_metrics.reset_state()
            self.train_network(self.model, self.memory[1], i % 50 == 0)
            self.train_network(self.model, self.memory[-1], i % 50 == 0, '-1')

    def train_network(self, model: Model, memory, verbose, char=''):
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
            with self.summary_writer.as_default():
                if i == 0 and char == '':
                    tf.summary.scalar(f"game_turn_amount{char}", data=len(memory), step=1)
                if verbose and i == len(memory) - 1:
                    tf.summary.image(f"result{char}", self.matrix_to_img(model_state[0, :, :], reward, a),
                                     step=self.step)
                    tf.summary.image(f"target_vec{char}", self.matrix_to_img(target_vec[0, :, :], reward, a),
                                     step=self.step)
                tf.summary.scalar(f"reward_train{char}", data=reward, step=1)
                tf.summary.scalar(f"target_train{char}", data=target, step=1)
                for m in model.metrics:
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
