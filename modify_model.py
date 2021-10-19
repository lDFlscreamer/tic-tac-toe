import os

import tensorflow as tf
from keras.models import load_model

import CONSTANT
from agent.agent import Game_agent


def modify_weight(w):
    w = tf.random.normal(shape=w.shape, mean=0.0, stddev=10, dtype=tf.float32)
    return w


def modify_bias(b):
    b = b - b
    return b


def modify_layer(model, start, end):
    layers = model.layers
    for layer in layers[start:end]:
        if len(layer.weights) < 2:
            continue
        w, b = layer.get_weights()
        # w = modify_weight(w)
        b = modify_bias(b)
        layer.set_weights([w, b])


just_get_summary = False
save_modified_model = True
by_name = False
if not just_get_summary:
    a = Game_agent(CONSTANT.TENSORBOARD_LOG_DIR + "modify")
    model2 = a.create_model()
    model_is_exist = os.path.exists(CONSTANT.AGENT_TEMP_MODEL_PATH)
    if model_is_exist:
        model2.load_weights(CONSTANT.AGENT_TEMP_MODEL_PATH, by_name=by_name, skip_mismatch=by_name)
    # modify_layer(model=model2,start=16,end=17)
    modify_layer(model=model2, start=None, end=None)
    if save_modified_model:
        model2.save(CONSTANT.AGENT_TEMP_MODEL_PATH)
        load_model(CONSTANT.AGENT_TEMP_MODEL_PATH).summary()
    else:
        model2.summary()
else:
    load_model(CONSTANT.AGENT_TEMP_MODEL_PATH).summary()
