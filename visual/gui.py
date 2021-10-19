from datetime import datetime
from tkinter import Tk, Label, Button, StringVar, Frame

import numpy as np

import CONSTANT
from agent.agent import Game_agent
from env.environment import Game_environment


class TicTacToeGUI:
    def __init__(self, epsilon=0.06):
        self.t = Tk()
        self.t.title("TIC TAC TOE")
        self.t.configure(bg="white")
        self.app = Frame(self.t)
        self.app.grid(row=1, column=1)

        self.end = False

        self.buttons = []
        self.env = Game_environment()
        self.env.epsilon_min = 0.0
        self.env.epsilon = epsilon
        self.memory = {1: list(), -1: list()}
        self.agent = Game_agent(CONSTANT.TENSORBOARD_GUI + datetime.now().strftime("%Y%m%d-%H-%M-%S"))
        if not self.env.opponent:
            raise ValueError
        # Making the background of the window as white#Displaying the player
        self.score = 0
        self.player_label = StringVar(self.t, value="PLAYER 1 :%i" % self.score)
        l1 = Label(self.t, textvariable=self.player_label, height=3, font=("COMIC SANS MS", 10, "bold"), bg="white")
        l1.grid(row=0, column=0)  # Quit button
        exitButton = Button(self.t, text="reset", command=self.reset, font=("COMIC SANS MS", 10, "bold"))
        exitButton.grid(row=0, column=2)  # Grid buttons

        for i in range(0, CONSTANT.FIELD_SIZE):
            row = []
            for j in range(0, CONSTANT.FIELD_SIZE):
                lbl = StringVar(self.app, "")
                row.append(lbl)
                Button(self.app, textvariable=lbl, height=1, width=1, bg="white",
                       fg="black",
                       font="Times 15 bold", command=lambda i=i, j=j: self.changeVal(i, j)).grid(row=i + 1, column=j)
            self.buttons.append(row)

        self.t.mainloop()

    def changeVal(self, row, col):
        if self.end:
            return
        action = (row, col)
        step = self.env.step(action)
        state = np.stack([step[0]], axis=0)
        self.memory[1].append((state, action, step[1], step[2]))
        step_ = step[3]
        if step_:
            self.memory[-1].append(step_)
        if step[2]:
            self.end = step[2]
            self.agent.train_network(model=self.env.opponent, memory=self.memory[1], image_verbose=True, img_eps=0.0)
            self.agent.train_network(model=self.env.opponent, memory=self.memory[-1], image_verbose=True,
                                     img_eps=self.env.epsilon, char='-1')
            print('end')
        self.score += step[1]
        self.update_score()
        self.change_board(step[0])

    def change_board(self, state):
        for i in range(0, len(state)):
            for j in range(0, len(state[i])):
                btn = self.buttons[i][j]
                if state[i][j] == 0:
                    if not btn.get() == "":
                        btn.set('')
                if state[i][j] == 1:
                    if not btn.get() == "X":
                        btn.set('X')
                if state[i][j] == -1:
                    if not btn.get() == "O":
                        btn.set("O")

    def update_score(self):
        self.player_label.set(value="PLAYER 1 :%i" % self.score)

    def reset(self):
        self.end = False
        self.env.reset()
        self.change_board(self.env.get_state())
        self.score = 0
        self.update_score()
