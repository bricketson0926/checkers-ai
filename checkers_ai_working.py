import numpy as np
import torch
torch.autograd.set_detect_anomaly(True)
import torch.nn as nn
import torch.optim as optim
import random
from checkers.game import Game
import pickle


GAMMA = 0.99            # discount factor per iti
BATCH_SIZE = 1          # episodes per update
LR = 1e-3               # learning rate
ITERATIONS = 10          # total updates
PRINT_EVERY = 1        
MAX_STEPS = np.inf      
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


def get_reward(game):
    """Immediate reward shaping."""
    if len(game.moves) <= 1:
        return 0.0
    move = game.moves[-1]
    if game.get_winner() is not None:
        return 4.0
    r = 0.0
    if abs(move[0] - move[1]) > 5:
        r += 2.0  # capture
    if move[1] in (15, 18):
        r += 1.0  # center
    elif move[1] in (10,11,14,19,22,23):
        r += 0.6  # side
    turn = game.whose_turn()
    if (move[0] > 24 and turn==2) or (move[0] < 9 and turn==1):
        r += 0.6  # deep advance
    if (move[0] < 5 and turn==2) or (move[0] > 28 and turn==1):
        r -= 1.0  # off back row
    return r


def compute_returns(rewards, gamma):
    #Compute returns here with discount in plce
    R = 0.0
    returns = []
    for r in reversed(rewards):
        R = r + gamma * R
        returns.insert(0, R)
    return returns


def train_policy(model, optimizer, device, batch_size, gamma, max_steps):

    all_log_probs, all_returns = [], []

    for ep in range(batch_size):
        game = Game()
        log_probs, rewards = [], []
        steps = 0

        # play one game
        while not game.is_over():
            steps += 1
            if game.whose_turn() == 1:
                legal = game.get_possible_moves()
                if not legal:
                    break
                state = board_to_tensor(game, device)
                logits = model(state, output_size=len(legal))
                mask = torch.zeros_like(logits)
                mask[:len(legal)] = 1
                masked_logits = logits.masked_fill(mask==0, float('-1e9'))
                dist = torch.distributions.Categorical(logits=masked_logits)
                a = dist.sample()
                log_probs.append(dist.log_prob(a))
                game.move(legal[a.item()])
                rewards.append(get_reward(game))
            else:
                moves = game.get_possible_moves()
                if not moves:
                    break
                game.move(random.choice(moves))
                rewards.append(0.0)

        
        if not game.is_over():
            pass

        # compute and store returns
        returns = compute_returns(rewards, gamma)
        all_log_probs.extend(log_probs)
        all_returns.extend(returns)

    returns_tensor = torch.tensor(all_returns, device=device)
    baseline = returns_tensor.mean()
    loss = -torch.stack([lp * (R - baseline) for lp, R in zip(all_log_probs, returns_tensor)]).sum()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return baseline.item()

#play against random
def evaluate_against_random(model, device, games=200):
    model.eval()
    w = l = d = 0
    for _ in range(games):
        game = Game()
        while not game.is_over():
            legal = game.get_possible_moves()
            if game.whose_turn() == 1 and legal:
                state = board_to_tensor(game, device)
                logits = model(state, output_size=len(legal))
                a = torch.argmax(logits)
                game.move(legal[a.item()])
            else:
                if legal:
                    game.move(random.choice(legal))
                else:
                    break
        res = game.get_winner()
        if res==1: w+=1
        elif res==2: l+=1
        else: d+=1
    print(f"Eval vs random — Wins: {w}, Losses: {l}, Draws: {d}")
    model.train()


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
    #create model and optimizer
    model = PolicyNet().to(DEVICE)
    opt = optim.Adam(model.parameters(), lr=LR)

    #begin training
    for it in range(1, ITERATIONS+1):
        avg_ret = train_policy(model, opt, DEVICE, BATCH_SIZE, GAMMA, MAX_STEPS)
        if it % PRINT_EVERY == 0:
            print(f"Update {it}: avg return {avg_ret:.3f}")


    #evaluate
    evaluate_against_random(model, DEVICE)
    torch.save(model, "checkersModelBatch" + str(BATCH_SIZE) +"Iterations" + str(ITERATIONS) + ".pt", pickle_module=pickle, pickle_protocol=2)

if __name__ == "__main__":
    main()
