import CONSTANT
from agent.agent import Game_agent

a = Game_agent()
model2 = a.create_model()
model2.load_weights(CONSTANT.AGENT_TEMP_MODEL_PATH, by_name=True)
model2.save(CONSTANT.AGENT_TEMP_MODEL_PATH)
a.get_bot().summary()
