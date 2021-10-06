from keras.models import load_model

import CONSTANT
from agent.agent import Game_agent

just_get_summary = False
save_modified_model = True
by_name = False
if not just_get_summary:
    a = Game_agent(CONSTANT.TENSORBOARD_LOG_DIR + "modify")
    model2 = a.create_model()
    model2.load_weights(CONSTANT.AGENT_TEMP_MODEL_PATH, by_name=by_name, skip_mismatch=by_name)
    w,b=model2.layers[16].get_weights()
    b=b-20
    model2.layers[16].set_weights([w,b])
    if save_modified_model:
        model2.save(CONSTANT.AGENT_TEMP_MODEL_PATH)
        load_model(CONSTANT.AGENT_TEMP_MODEL_PATH).summary()
    else:
        model2.summary()
else:
    load_model(CONSTANT.AGENT_TEMP_MODEL_PATH).summary()
