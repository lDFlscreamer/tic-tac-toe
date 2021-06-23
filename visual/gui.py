from tkinter import Tk, Label, Button, StringVar, Frame

from env.environment import Game_environment


class TicTacToeGUI:
    def __init__(self):
        self.t = Tk()
        self.t.title("TIC TAC TOE")
        self.t.configure(bg="white")
        self.app = Frame(self.t)
        self.app.grid(row=1, column=1)

        self.end = False

        self.buttons = []
        self.env = Game_environment()
        if not self.env.opponent:
            raise ValueError
        # Making the background of the window as white#Displaying the player
        self.score = 0
        self.player_label = StringVar(self.t, value="PLAYER 1 :%i" % self.score)
        l1 = Label(self.t, textvariable=self.player_label, height=3, font=("COMIC SANS MS", 10, "bold"), bg="white")
        l1.grid(row=0, column=0)  # Quit button
        exitButton = Button(self.t, text="Quit", command=self.t.destroy, font=("COMIC SANS MS", 10, "bold"))
        exitButton.grid(row=0, column=2)  # Grid buttons

        for i in range(0, 32):
            row = []
            for j in range(0, 32):
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
        step = self.env.step((row, col))
        if step[2]:
            self.end = step[2]
            print('end')
        self.score += step[1]
        self.player_label.set(value="PLAYER 1 :%i" % self.score)
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
