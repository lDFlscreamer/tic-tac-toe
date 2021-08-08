from agent.agent import Game_agent
from env.environment import Game_environment

g = Game_environment()
a = Game_agent()
g.opponent = a.get_bot()
epsilon = 1
decay = 0.999
for i in range(0, 150):
    epsilon = epsilon * pow(decay, i)
    a.epsilon_min = a.epsilon_min * pow(decay, i)
    for j in range(0, 180):
        print(f'try i={i}  j={j}')
        a.epsilon = epsilon * pow(decay, j)
        g.opponent = g.get_oponent()
        if not g.opponent:
            g.opponent = a.get_bot()
            a.reinforcement_learning(env=g, num_episodes=250)
        a.reinforcement_learning(env=g, num_episodes=150)
print("END")

# from visual.gui import TicTacToeGUI
#
# gui = TicTacToeGUI()
