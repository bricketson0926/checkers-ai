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


def play_da_darn_game(model_1, model_2, optimizer_1, optimizer_2, device, batch_size):
    games = [Game() for _ in range(batch_size)]
    final_reward_1 = 0
    final_reward_2 = 0

    active_games = [game for game in games if not game.is_over()]
    
    while len(active_games) > 0:
        batch_legal_moves = []
        batch_games = []
        for game in active_games:
            if game.whose_turn() == 1:
                batch_legal_moves.append(game.get_possible_moves())
                batch_games.append(game)
            else:
                moves = game.get_possible_moves()
                if len(moves) > 0:
                    game.move(random.choice(moves))
        
        #Only Get moves when there are games for the model to play
        if len(batch_games) > 0:

            #Get a stack of tensors for each game where it's the model's turn to play
            batch_states = torch.stack([board_to_tensor(game, device) for game in batch_games])
            output_size = max(len(legal_moves) for legal_moves in batch_legal_moves)
            logits = model_1(batch_states, output_size=output_size)

            # print(logits)

            #Get a mask
            mask = torch.zeros(len(batch_states), output_size, device=device)
            for i, legal_moves in enumerate(batch_legal_moves):
                mask[i, :len(legal_moves)] = 1


            #Mask for how many legal moves are in each logit
            masked_logits = torch.where(mask == 1, logits, torch.tensor(-1e9).to(device))
            mini = torch.distributions.Categorical(logits=masked_logits)

            #Get an action and log_prob tensor
            action = mini.sample()
            log_prob = mini.log_prob(action)

            #Make each move in the batch
            for i in range(action.shape[0]):
                #Get and make move on each individual game
                move = batch_legal_moves[i][action[i].item()]
                batch_games[i].move(move)


            #Get reward for each game in the batch and calculate loss
            reward_tensor = torch.tensor([get_reward(game) for game in batch_games], dtype=torch.float32, device=device)
            loss = -log_prob * reward_tensor

            #Optimize and Backpropogate. Add to final_reward
            loss = loss.sum()
            loss.backward()
            optimizer_1.step()
            optimizer_1.zero_grad()
            final_reward_1 += reward_tensor.mean()

        #Update active games now that moves have been made
        active_games = [game for game in games if not game.is_over()]

    return final_reward_1, final_reward_2

#Reward function
def get_reward(game):
    if len(game.moves) == 1:
        return 0

    recent_move = game.moves[len(game.moves) - 1]

    reward = 0
    #Win for most recent player
    if game.get_winner() != None:
        return 4
    
    #Rewarding Player got a capture
    if abs(recent_move[0] - recent_move[1]) > 5:
        reward += 2
    
    #Central/Side Play for rewarded player
    if(recent_move[1] in [15, 18]):
        reward += 1
    elif(recent_move[1] in [10, 11, 14, 19, 22, 23]):
        reward += 0.6

    #Advancing deep into opponents side
    elif (recent_move[0] > 24 and game.whose_turn() == 2) or (recent_move[0] < 9 and game.whose_turn() == 1):
        reward += 0.6

    #Moved off back row
    elif (recent_move[0] < 5 and game.whose_turn() == 2) or (recent_move[0] > 28 and game.whose_turn() == 1):
        reward -= 1

    return reward

def play_against_random_moves(model, device):
    #Set model into evaluation mode
    model.eval()    

    wins = 0
    losses = 0
    draws = 0
    
    for i in range(100):
        game = Game()
        while not game.is_over():
            legal_moves = game.get_possible_moves()
            if game.whose_turn == 1:
            
                if len(legal_moves) == 0:
                    break

                state = board_to_tensor(game, device)
                logits = model(state, output_size=len(legal_moves))
        
                #This time, get the actual highest value from logits
                mini = torch.distributions.Categorical(logits=logits)
                action = mini.argmax()
        
                move = legal_moves[action.item()]
                game.move(move)
            else:
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
    optimizer_1 = optim.Adam(model_1.parameters(), lr=0.001)
    optimizer_2 = optim.Adam(model_2.parameters(), lr=0.001)

    iterations = 100
    print_every = 100
    total_iteration_reward_1 = 0
    # total_iteration_reward_2 = 0
    total_reward_1 = 0
    # total_reward_2 = 0

    #Training Iterations
    for iti in range(1, iterations+1):
        reward_1, reward_2 = play_da_darn_game(model_1, model_2, optimizer_1, optimizer_2, device, 2)
        total_reward_1 += reward_1
        # total_reward_2 += reward_2
        total_iteration_reward_1 += reward_1
        # total_iteration_reward_2 += reward_2
        if iti % print_every == 0:
            avg_reward_1 = total_iteration_reward_1/print_every
            # avg_reward_2 = total_iteration_reward_2/print_every
            print("Total reward_1 for ", iti-10,"-",iti,": ",avg_reward_1)
            # print("Total reward_2 for ", iti-10,"-",iti,": ",avg_reward_2)
            total_iteration_reward_1 = 0
            # total_iteration_reward_2 = 0
    print("Total change_1 over entire training", total_reward_1)
    # print("Total change_2 over entire training", total_reward_2)

    play_against_random_moves(model_1, device) #if total_reward_1 > total_reward_2 else model_2, device)
    torch.save(model_1, "checkersModel.pt", pickle_module=pickle, pickle_protocol=2)

if __name__ == "__main__":
    main()