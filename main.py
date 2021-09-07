from agent.agent import Game_agent
from env.environment import Game_environment

g = Game_environment()
a = Game_agent()
g.opponent = a.get_bot()

# a.epsilon_decay = 1  # <1 for dynamical change inside train process
epsilon = 1
decay = 0.998
initial_i = 0  # last  i value
initial_j = 0  # last j value 42
a.save_frequency = 50
a.image_verbose_frequency = 50
for i in range(initial_i, 250):
    epsilon = epsilon * pow(decay, i)
    a.epsilon_min = a.epsilon_min * pow(decay, i)
    for j in range(initial_j, 250):
        print(f'try i={i}  j={j}')
        a.epsilon = epsilon * pow(decay, j)
        g.opponent = g.get_oponent()
        if not g.opponent:
            g.opponent = a.get_bot()
            a.reinforcement_learning(env=g, num_episodes=250)
        a.reinforcement_learning(env=g, num_episodes=300)
print("END")
