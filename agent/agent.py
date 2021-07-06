import os
from datetime import datetime

import gym
import numpy as np
import tensorflow as tf
from keras.callbacks import ModelCheckpoint
from numpy import unravel_index
from tensorflow.keras.layers import *
from tensorflow.keras.models import *

import CONSTANT

AGENT_METRIC = 'mae'
AGENT_ACTIVATION = "linear"


class Game_agent:

    def __init__(self):
        self.gamma = 0.25
        self.epsilon = 0.8
        self.epsilon_min = 0.06
        self.epsilon_decay = 0.9995

        self.memory = list()
        self.model = self.get_bot()

        log_dir = CONSTANT.TENSORBOARD_LOG_DIR + datetime.now().strftime("%Y%m%d-%H-%M-%S")
        self.summary_writer = tf.summary.create_file_writer(logdir=log_dir)

    def get_callbacks(self, epoch):
        call_back = []
        temp_checkpoint = ModelCheckpoint(CONSTANT.AGENT_TEMP_MODEL_PATH,
                                          verbose=1,
                                          save_freq=1)
        call_back.append(temp_checkpoint)
        loss_checkpoint = ModelCheckpoint(CONSTANT.AGENT_MODEL_LOSS_PATH,
                                          monitor='loss',
                                          mode='min',
                                          save_best_only=True,
                                          save_freq=1)
        call_back.append(loss_checkpoint)
        mae_checkpoint = ModelCheckpoint(CONSTANT.AGENT_MODEL_MAE_PATH,
                                         monitor=AGENT_METRIC,
                                         mode="min",
                                         save_best_only=True,
                                         save_freq=1)
        call_back.append(mae_checkpoint)
        return call_back

    def create_model(self):
        # todo: model architecture
        input: Input = Input(shape=(CONSTANT.FIELD_SIZE, CONSTANT.FIELD_SIZE, 1), name="input")
        net = input

        first = Conv2D(filters=8, kernel_size=(2, 2), padding="same",
                       activation=None, name="first")(net)
        first = BatchNormalization()(first)
        first = Activation(AGENT_ACTIVATION)(first)

        net_before = first
        first = Conv2D(filters=8, kernel_size=(2, 2), padding="same", activation=None, name="first_skip_1")(net_before)
        first = BatchNormalization()(first)
        first = Activation(AGENT_ACTIVATION)(first)

        net_after = Add()([net_before, first])
        first = net_after

        net_before = first
        first = Conv2D(filters=8, kernel_size=(2, 2), padding="same", activation=None, name="first_skip_2")(net_before)
        first = BatchNormalization()(first)
        first = Activation(AGENT_ACTIVATION)(first)

        net_after = Add()([net_before, first])
        first = net_after

        second = Conv2D(filters=8, kernel_size=(3, 3), padding="same",
                        activation=None, name="second")(net)
        second = BatchNormalization()(second)
        second = Activation(AGENT_ACTIVATION)(second)

        net_before = second
        second = Conv2D(filters=8, kernel_size=(3, 3), padding="same", activation=None, name="second_skip_1")(
            net_before)
        second = BatchNormalization()(second)
        second = Activation(AGENT_ACTIVATION)(second)

        net_after = Add()([net_before, second])
        second = net_after

        net_before = second
        second = Conv2D(filters=8, kernel_size=(3, 3), padding="same", activation=None, name="second_skip_2")(
            net_before)
        second = BatchNormalization()(second)
        second = Activation(AGENT_ACTIVATION)(second)

        net_after = Add()([net_before, second])
        second = net_after

        third = Conv2D(filters=8, kernel_size=(4, 4), padding="same",
                       activation=None, name="third")(net)
        third = BatchNormalization()(third)
        third = Activation(AGENT_ACTIVATION)(third)

        net_before = third
        third = Conv2D(filters=8, kernel_size=(4, 4), padding="same", activation=None, name="third_skip_1")(
            net_before)
        third = BatchNormalization()(third)
        third = Activation(AGENT_ACTIVATION)(third)

        net_after = Add()([net_before, third])
        third = net_after

        net_before = third
        third = Conv2D(filters=8, kernel_size=(4, 4), padding="same", activation=None, name="third_skip_2")(
            net_before)
        third = BatchNormalization()(third)
        third = Activation(AGENT_ACTIVATION)(third)

        net_after = Add()([net_before, third])
        third = net_after

        net = Concatenate(name="concat")([first, second, third])

        net = Conv2D(filters=16, kernel_size=(2, 2), padding="same",
                     activation=None, name="first_decoder")(net)
        net = BatchNormalization()(net)
        net = Activation("linear")(net)

        net = Conv2D(filters=8, kernel_size=(2, 2), padding="same",
                     activation=None, name="second_decoder")(net)
        net = BatchNormalization()(net)
        net = Activation("linear")(net)

        net = Conv2D(filters=1, kernel_size=(2, 2), padding="same",
                     activation=None, name="third_decoder")(net)
        net = BatchNormalization()(net)
        net = Activation("linear")(net)

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
                    q_s = self.model(model_state)[0, :, :, 0]
                    q_s = q_s.numpy()
                    a = unravel_index(q_s.argmax(), q_s.shape)
                new_s, r, done, _ = env.step(a)

                self.memory.append((model_state, a, r, done))
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
                if epoch % 10 == 0:
                    print({m.name: float(m.result()) for m in self.model.metrics})
                epoch += 1
                s = new_s
                self.model.compiled_metrics.reset_state()
            self.train_network()

    def train_network(self):
        for i in range(len(self.memory) - 1, -1, -1):
            model_state = self.memory[i][0]
            a = self.memory[i][1]
            reward = self.memory[i][2]
            done = self.memory[i][3]
            if not done:
                new_m_state = self.memory[i + 1][0]
                predict_new_state = self.model(new_m_state)[0, :, :, 0]
                possible_reward = self.gamma * np.max(predict_new_state)
                target = reward + self.gamma * possible_reward
            else:
                target = reward
            target_vec = self.model(model_state)
            target_vec = target_vec.numpy()
            target_vec[0][a[0]][a[1]][0] = target
            self.train_step(self.model, model_state, target_vec)
            with self.summary_writer.as_default():
                if i == 0:
                    tf.summary.scalar("game_turn_amount", data=len(self.memory), step=1)
                tf.summary.scalar("reward_train", data=reward, step=1)
                tf.summary.scalar("target_train", data=target, step=1)
                for m in self.model.metrics:
                    tf.summary.scalar(m.name, data=float(m.result()), step=1)
        self.memory.clear()

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
