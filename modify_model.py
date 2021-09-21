from keras.models import load_model

import CONSTANT
from agent.agent import Game_agent

just_get_summary = False
save_modified_model = True

if not just_get_summary:
    a = Game_agent(CONSTANT.TENSORBOARD_LOG_DIR + "modify")
    model2 = a.create_model()
    model2.load_weights(CONSTANT.AGENT_TEMP_MODEL_PATH, by_name=True, skip_mismatch=True)
    if save_modified_model:
        model2.save(CONSTANT.AGENT_TEMP_MODEL_PATH)
        load_model(CONSTANT.AGENT_TEMP_MODEL_PATH).summary()
    else:
        model2.summary()
else:
    load_model(CONSTANT.AGENT_TEMP_MODEL_PATH).summary()
