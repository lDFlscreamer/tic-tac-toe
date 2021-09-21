import tensorflow as tf

from agent.agent import Game_agent
from env.environment import Game_environment

g = Game_environment()
a = Game_agent()
g.opponent = a.get_bot()

a.epsilon_decay = 1  # <1 for dynamical change inside train process
epsilon = 1
decay = 0.99
# decay = 1
a.epsilon_min = 0.04
initial_i = 0  # last  i value
initial_j = 63  # last j value 124

a.save_frequency = 50
a.image_verbose_frequency = 50
a.NN_verbose_frequency = 50

for i in range(initial_i, 250):
    epsilon = epsilon * pow(decay, i)
    a.epsilon_min = a.epsilon_min * pow(decay, i)
    for j in range(initial_j, 250):
        a.epsilon = epsilon * pow(decay, j)
        print(f'try i={i}  j={j}')
        with a.summary_writer.as_default():
            tf.summary.text("current iteration", f'try i={i}  j={j} epsilon={a.epsilon}', step=i)
        g.opponent = g.get_oponent()
        if not g.opponent:
            g.opponent = a.get_bot()
            a.reinforcement_learning(env=g, num_episodes=250)
        a.reinforcement_learning(env=g, num_episodes=300)
print("END")

# tensorboard --logdir logs --samples_per_plugin images=100
