import numpy as np

from agent.agent import Game_agent
from env.environment import Game_environment

# g = Game_environment()
# a = Game_agent()
# if not g.opponent:
#     g.opponent=a.get_bot()
#     a.reinforcement_learning(env=g, num_episodes=250)
# # # g.state = np.random.randint(low=-1, high=1, size=g.state.shape)
# # b = g.get_state()
# # bot = a.get_bot()
# # bot_ = bot(np.stack([g.get_state()], axis=0)).numpy()[0,:,:,0]
# # g.state[3][4][0]=1
# # g.state[:,:,0]
# # a.reinforcement_learning(env=g, num_episodes=500)
# print("s")

from visual.gui import TicTacToeGUI

gui = TicTacToeGUI()


