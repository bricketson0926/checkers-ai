import numpy as np
import torch
import torch.nn as nn
from checkers.game import Game
import tkinter as tk
from tkinter import messagebox
import pickle
import math
from PIL import Image, ImageTk

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CELL_SIZE = 80  # Size of each square
GAME = Game()
GUI_BOARD = [[0 for _ in range(8)] for _ in range(8)]

def num_to_x(pos):
    pos -= 1
    row = pos // 4
    return(row)

def num_to_y(pos):
    pos -= 1
    row = pos // 4
    col = (pos % 4) * 2 + (1- (row % 2))
    return(col)

def x_y_to_num(x, y):
    return y*4 + math.ceil((x+1)/2)


def board_to_tensor(game, device):
    """Convert board pieces to a 32‑dim tensor."""
    arr = np.zeros(32, dtype=np.float32)
    for piece in game.board.pieces:
        if piece.position is None:
            continue
        idx = piece.position - 1
        val = 3.0 if piece.king else 1.0
        arr[idx] = val if piece.player == 1 else -val
    return torch.tensor(arr, device=device)


# Initialize the GUI and game logic
def run_checkers_gui(model):

    def draw_board(selected=-1):
        for i in range(8):
            for j in range(8):
                color = "sienna4" if (i + j) % 2 == 0 else "sandy brown"
                canvas.create_rectangle(
                    j * CELL_SIZE,
                    i * CELL_SIZE,
                    (j + 1) * CELL_SIZE,
                    (i + 1) * CELL_SIZE,
                    fill=color
                )
        for piece in GAME.board.pieces:
            if piece.captured == False:
                x = num_to_x(piece.position)
                y = num_to_y(piece.position)
                color = "red" if piece.player == 1 else "black"
                canvas.create_oval(
                    y * CELL_SIZE + 10,
                    x * CELL_SIZE + 10,
                    (y + 1) * CELL_SIZE - 10,
                    (x + 1) * CELL_SIZE - 10,
                    fill=color
                )
                #Display a ring around selected piece
                if piece.position == selected:
                    print("Selected:", selected)
                    canvas.create_oval(
                        y * CELL_SIZE + 10,
                        x * CELL_SIZE + 10,
                        (y + 1) * CELL_SIZE - 10,
                        (x + 1) * CELL_SIZE - 10,
                        outline="white",
                        width=2
                    )
                    #Display circle for possible positional moves
                    for move in piece.get_possible_positional_moves() + piece.get_possible_capture_moves():
                        if move in GAME.get_possible_moves():
                            move_x = num_to_x(move[1])
                            move_y = num_to_y(move[1])
                            canvas.create_oval(
                                move_y * CELL_SIZE + 10,
                                move_x * CELL_SIZE + 10,
                                (move_y + 1) * CELL_SIZE - 10,
                                (move_x + 1) * CELL_SIZE - 10,
                                fill="gray"
                            )

                #Place King symbols
                if piece.king:
                    canvas.create_image(
                        y * CELL_SIZE + 40,
                        x * CELL_SIZE + 40,
                        image=KING_TK,  
                    )

    def user_move(start_x, start_y, end_x, end_y):
        # Update the board based on user input

        move = [x_y_to_num(start_x, start_y), x_y_to_num(end_x, end_y)]
        if move in GAME.get_possible_moves():
            GAME.move(move)
            update_gui()

            #Check for game over then pass turn
            if(check_game_end()):
                quit()
            
            #Only if it's AI's turn
            if GAME.whose_turn() == 2:
                canvas.after(800, ai_turn)

        else:
            messagebox.showerror("Invalid Move", "No piece to move!")
                

    def ai_turn():
        #Pull legal moves and then make an AI move   
        legal_moves = GAME.get_possible_moves()
        if len(legal_moves) > 0:
            state = board_to_tensor(GAME, DEVICE)
            logits = model(state, output_size=len(legal_moves))
            action = torch.argmax(logits)
            GAME.move(legal_moves[action.item()])

        
        update_gui()

        # Check if AI move ends the game
        if check_game_end():
            quit()

        #If still AI turn, run again
        if GAME.whose_turn() == 2:
            ai_turn()

    def check_game_end():
        # Decide if anyone has won, and return true if there is a winner
        if GAME.is_over():
            if GAME.get_winner() == 2:
                messagebox.showinfo("Game Over", "AI wins!")
            elif GAME.get_winner() == 1:
                messagebox.showinfo("Game Over", "You win!")
            else:
                messagebox.showinfo("Game Over", "It's a Draw!")
            return True
        return False

    def update_gui(selected=False):
        canvas.delete("all")
        draw_board(selected)

    def piece_in_space(x, y):
        move_index = x_y_to_num(x, y)
        for piece in GAME.board.pieces:
            if piece.player == 1 and piece.position == move_index:
                return True

    #Define click handler function before passing it to buttons
    def handle_click(event):
        # Get the coordinates of the clicked square
        x, y = event.x // CELL_SIZE, event.y // CELL_SIZE

        #Ignore invalid squares
        if (x + y) %  2 == 0:
            return
        #Ignore if piece ins't in space
        

        # If it's the user's turn, allow move selection
        if GAME.whose_turn() == 1:
            if not hasattr(handle_click, "start"):
                if not piece_in_space(x, y):
                    print("No Player piece in this location")
                    return
                update_gui(x_y_to_num(x,y))
                handle_click.start = (x, y)
            else:
                start_x, start_y = handle_click.start
                
                #Click on the same piece to unselect
                if start_x == x and start_y == y:
                    print("Deselected Piece")
                    del handle_click.start
                    update_gui()
                #Tried to move a piece onto another
                elif piece_in_space(x, y):
                    print("New Piece Selected")
                    del handle_click.start
                    update_gui(x_y_to_num(x,y))
                    handle_click.start = (x, y)
                else:
                    user_move(start_x, start_y, x, y)
                    del handle_click.start  # Reset for next move

    # Initialize the Tkinter GUI window
    window = tk.Tk()
    window.title("Checkers AI")
    canvas = tk.Canvas(window, width=8 * CELL_SIZE, height=8 * CELL_SIZE)
    canvas.pack()

    image = Image.open("C:\\Users\\ricke\\OneDrive\\Documents\\2.Programming\\Python\\king.png")
    KING_TK = ImageTk.PhotoImage(image.resize((40, 40)))


    # Bind click event to canvas
    canvas.bind("<Button-1>", handle_click)

    # Draw the initial board
    draw_board()
    window.mainloop()


class PolicyNet(nn.Module):
    def __init__(self, input_size=32, hidden=128, max_out=48):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.out = nn.Linear(hidden, max_out)

    def forward(self, x, output_size):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.out(x)[:output_size]

def main():
    print("Starting")


    # path = "C:\\Users\\ricke\\OneDrive\\Documents\\2.Programming\\Python\\demo_model.pt" # Ten Games
    path = "C:\\Users\\ricke\\OneDrive\\Documents\\2.Programming\\Python\\checkersModelBatch32Iterations500.pt" # 16000 Games

    model = torch.load(path, pickle_module=pickle, map_location=torch.device("cpu"))
    model.eval()

    run_checkers_gui(model)
    


if __name__ == "__main__":
    main()