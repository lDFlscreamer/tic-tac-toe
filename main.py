from agent.agent import Game_agent
from env.environment import Game_environment

g = Game_environment()
a = Game_agent()
g.opponent = a.get_bot()
if not g.opponent:
    g.opponent = a.get_bot()
    a.reinforcement_learning(env=g, num_episodes=1000)
a.reinforcement_learning(env=g, num_episodes=5000)
print("s")

# from visual.gui import TicTacToeGUI
#
# gui = TicTacToeGUI()
#
