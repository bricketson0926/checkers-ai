import numpy as np
import torch as torch
import torch.nn as nn
import torch.optim as optim
import random 
from checkers.game import Game
import pickle

def num_to_x(pos):
    pos -= 1
    row = pos // 4
    return(row)

def num_to_y(pos):
    pos -= 1
    row = pos // 4
    col = (pos % 4) * 2 + (1- (row % 2))
    return(col)

def board_to_tensor(game, device):  #Expects an input if game.board.pieces
    pieces = np.zeros(32, dtype=np.float32)
    for piece in game.board.pieces:
        if piece.position is None:              #Check for None pieces which are present for some reason
            continue
        # x = num_to_x(piece.position)
        # y = num_to_y(piece.position)
        index = piece.position - 1
        if piece.king:
            if piece.player == 1:
                pieces[index] = 3.0
            elif piece.player == 2:
                pieces[index] = -3.0
        elif not piece.king:
            if piece.player == 1:
                pieces[index] = 1.0
            elif piece.player == 2:
                pieces[index] = -1.0
    return torch.tensor(pieces, dtype=torch.float, device=device)


def play_da_darn_game(model_1, model_2, optimizer_1, optimizer_2, device, max_moves):
    game = Game()
    log = []
    final_reward_1 = 0
    final_reward_2 = 0
    player = game.whose_turn    


    while not game.is_over():

        #Take the board state and put into a tensor for the model
        state = board_to_tensor(game, device)
        # state = state.unsqueeze(0).to(device)
        legal_moves = game.get_possible_moves()
        if len(legal_moves) > max_moves:
            max_moves = len(legal_moves)

        if game.whose_turn() == 2:
            game.move(random.choice(legal_moves))
            player = game.whose_turn()
            continue

        #End of the game
        if len(legal_moves) == 0:
            break

        # print("Input Tensor Shape:", state.shape)

        # Define logits based on which player is playing. We will later reward based on this as well
        logits = model_1(state, output_size=len(legal_moves)) if player==1 else model_2(state, output_size=len(legal_moves))
        # print("Logits of Model", player, ":", logits)

        #Fit output layer to a distribution with total probabilty of 1
        mini = torch.distributions.Categorical(logits=logits) 

        #Sample according to probability distribution
        action = mini.sample()

        # Pull out the desired move and make it in the game
        move = legal_moves[action.item()]
        game.move(move)
        
        #Get reward based on the previous model Calculate loss for previous player
        reward = get_reward(game) if len(game.moves) > 1 else 3 if game.whose_turn() == player else 0
        log_prob = mini.log_prob(action)
        loss = -log_prob * reward


        #Reward model 1
        if player == 1:
            optimizer_1.zero_grad()
            loss.backward()
            optimizer_1.step()
            final_reward_1 += reward

        #Reward model 2
        else:
            optimizer_2.zero_grad()
            loss.backward()
            optimizer_2.step()
            final_reward_2 += reward


        #Append to log and add to total reward for possible debugging
        log.append(log_prob)

        #Update player for next iteration
        player = game.whose_turn()

    # print("Winner: ", game.get_winner())
        

    # print("Log after iterations:", log)
    return final_reward_1, final_reward_2, max_moves

#Reward function
def get_reward(game):
    recent_move = game.moves[len(game.moves) - 1]

    reward = 0
    #Win for most recent player
    if game.get_winner() != None:
        return 10
    
    #Rewarding Player got a capture
    if abs(recent_move[0] - recent_move[1]) > 5:
        reward += 1
    
    #Central/Side Play for rewarded player
    if(recent_move[1] in [15, 18]):
        reward += 0.3
    elif(recent_move[1] in [10, 11, 14, 19, 22, 23]):
        reward += 0.2

    #Advancing deep into opponents side
    elif (recent_move[0] > 24 and game.whose_turn() == 2) or (recent_move[0] < 9 and game.whose_turn() == 1):
        reward += 0.3

    #Moved off back row
    elif (recent_move[0] < 5 and game.whose_turn() == 2) or (recent_move[0] > 28 and game.whose_turn() == 1):
        reward -= 2

    return reward

def play_against_random_moves(model, device):

    model.eval()    #Set model into evaluation mode

    wins = 0
    losses = 0
    draws = 0
    
    for i in range(100):
        game = Game()
        while not game.is_over():
            if game.whose_turn == 1:
                state = board_to_tensor(game)  
                state = state.unsqueeze(0).to(device)   #Resized tensor to fit
                logits = model(state)
        
                legal_moves = game.get_possible_moves()
                if legal_moves == []:
                    break       #No legal moves, means no more game
                
                move_logits = logits[0, :len(legal_moves)]  #cuts out all illegal move scores from original logits
        
        
                mini = torch.distributions.Categorical(logits=move_logits)      #Creates a probability distribution over all legal moves using Catergorical Distribution????
                action = mini.sample()
        
                real_move = legal_moves[action.item()]      #real_move now stores array of [original position, new position]
        
                if isinstance(real_move, list) or isinstance(real_move, tuple):
                    from_pos, to_pos = real_move[0], real_move[1]
                    game.move([from_pos,to_pos])                #Actually make the move selected above
                else:
                    game.move(real_move)
            else:
                legal_moves = game.get_possible_moves()
                game.move(random.choice(legal_moves))
        if game.get_winner() == 1:
            wins += 1
        elif game.get_winner() == 2:
            losses += 1
        else:
            draws += 1
    print("Wins:", wins,"\nLosses: ", losses,"\nDraws: ", draws)


class PolicyNet(nn.Module):
    def __init__(self, input_size, max_output_size):
        super().__init__()
        self.input_size = input_size
        self.max_output_size = max_output_size

        self.hidden_layer_1 = nn.Linear(32, 128)
        self.hidder_layer_2 = nn.Linear(128, 128)
        
        self.final_layer = nn.Linear(128, 48)

        #Define activation function
        self.relu = nn.ReLU() 

    def forward(self, x, output_size):
        x = torch.nn.functional.relu(self.hidden_layer_1(x))

        weights = self.hidder_layer_2.weight[:128, :]
        bias = self.hidder_layer_2.bias[:128]
        x = nn.functional.linear(x, weights, bias)

        x = torch.nn.functional.relu(self.hidder_layer_2(x))
        weights = self.final_layer.weight[:output_size, :]
        bias = self.final_layer.bias[:output_size]
        x = nn.functional.linear(x, weights, bias)

        return x


#MAIN
def main():
    torch.autograd.set_detect_anomaly(True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_1 = PolicyNet(input_size=32, max_output_size=20).to(device)
    model_2 = PolicyNet(input_size=32, max_output_size=20).to(device)
    optimizer_1 = optim.Adam(model_1.parameters(), lr=1e-3)
    optimizer_2 = optim.Adam(model_2.parameters(), lr=1e-3)

    iterations = 100
    print_every = 100
    total_iteration_reward_1 = 0
    total_iteration_reward_2 = 0
    total_reward_1 = 0
    total_reward_2 = 0

    max_moves = 0

    #Training Iterations
    for iti in range(1, iterations+1):
        reward_1, reward_2, max_moves = play_da_darn_game(model_1, model_2, optimizer_1, optimizer_2, device, max_moves)
        total_reward_1 += reward_1
        total_reward_2 += reward_2
        total_iteration_reward_1 += reward_1
        total_iteration_reward_2 += reward_2
        if iti % print_every == 0:
            avg_reward_1 = total_iteration_reward_1/print_every
            avg_reward_2 = total_iteration_reward_2/print_every
            print("Total reward_1 for ", iti-10,"-",iti,": ",avg_reward_1)
            print("Total reward_2 for ", iti-10,"-",iti,": ",avg_reward_2)
            total_iteration_reward_1 = 0
            total_iteration_reward_2 = 0
    print("Total change_1 over entire training", total_reward_1)
    print("Total change_2 over entire training", total_reward_2)
    print("Maximum Legal Moves Length:", max_moves)

    play_against_random_moves(model_1 if total_reward_1 > total_reward_2 else model_2, device)
    torch.save(model_1, "checkersModel.pt", pickle_module=pickle, pickle_protocol=2)

if __name__ == "__main__":
    main()