# from agent.agent import Game_agent
# from env.environment import Game_environment
#
# g = Game_environment()
# a = Game_agent()
# g.opponent = a.get_bot()
# epsilon = 1
# for i in range(0, 1000):
#     g.opponent = g.get_oponent()
#     if not g.opponent:
#         g.opponent = a.get_bot()
#         a.reinforcement_learning(env=g, num_episodes=250)
#     a.reinforcement_learning(env=g, num_episodes=200)
#     epsilon = epsilon * 0.85
#     a.epsilon = epsilon
# print("s")

from visual.gui import TicTacToeGUI

gui = TicTacToeGUI()
