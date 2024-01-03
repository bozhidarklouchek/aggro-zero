import socket

from random import choice

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
torch.manual_seed(0)

import random
from random import choice

import math

from scipy.ndimage import label

# Set up adjacency tiles for hex
_adj = np.ones([3,3], int)
_adj[0,0] = 0
_adj[2,2] = 0


class Hex:
    def __init__(self, size):
        self.row_count = size
        self.column_count = size
        self.action_size = self.row_count * self.column_count
        
    def get_initial_state(self):
        return np.zeros((self.row_count, self.column_count))
    
    def get_next_state(self, state, action, player):
        row = action // self.column_count
        column = action % self.column_count
        state[row, column] = player
        return state
    
    def get_valid_moves(self, state):
        return (state.reshape(-1) == 0).astype(np.uint8)
    
    def check_win(self, state, action):
        if action == None:
            return False
        
        row = action // self.column_count
        column = action % self.column_count
        player = state[row, column]

        if player == 1:
            clumps = label(state > 0, _adj)[0]
        else:
            clumps = label(state.T < 0, _adj)[0]
        spanning_clumps = np.intersect1d(clumps[0], clumps[-1])
        return np.count_nonzero(spanning_clumps)
    
    def get_value_and_terminated(self, state, action):
        if self.check_win(state, action):
            return 1, True
        if np.sum(self.get_valid_moves(state)) == 0:
            return 0, True
        return 0, False
    
    def get_opponent(self, player):
        return -player
    
    def get_opponent_value(self, value):
        return -value
    
    def change_perspective(self, state, player):
        if(player == -1):
            return state.T * -1
        else:
            return state
        
    def get_encoded_state(self, state):

        encoded_state = np.stack(
            (state == -1, state == 0, state == 1)
        ).astype(np.float32)

        if len(state.shape) == 3:
            encoded_state = np.swapaxes(encoded_state, 0, 1)

        return encoded_state

    
class ResNet(nn.Module):
    def __init__(self, game, num_resBlocks, num_hidden, device):
        super().__init__()

        self.device = device
        self.startBlock = nn.Sequential(
            nn.Conv2d(3, num_hidden, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_hidden),
            nn.ReLU()
        )
        
        self.backBone = nn.ModuleList(
            [ResBlock(num_hidden) for i in range(num_resBlocks)]
        )
        
        self.policyHead = nn.Sequential(
            nn.Conv2d(num_hidden, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * game.row_count * game.column_count, game.action_size)
        )
        
        self.valueHead = nn.Sequential(
            nn.Conv2d(num_hidden, 3, kernel_size=3, padding=1),
            nn.BatchNorm2d(3),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3 * game.row_count * game.column_count, 1),
            nn.Tanh()
        )

        self.to(device)
        
    def forward(self, x):
        x = self.startBlock(x)
        for resBlock in self.backBone:
            x = resBlock(x)
        policy = self.policyHead(x)
        value = self.valueHead(x)
        return policy, value
        
        
class ResBlock(nn.Module):
    def __init__(self, num_hidden):
        super().__init__()
        self.conv1 = nn.Conv2d(num_hidden, num_hidden, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_hidden)
        self.conv2 = nn.Conv2d(num_hidden, num_hidden, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_hidden)
        
    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual
        x = F.relu(x)
        return x
    

class Node:
    def __init__(self, game, args, state, parent=None, action_taken=None, prior=0):
        self.game = game
        self.args = args
        self.state = state
        self.parent = parent
        self.action_taken = action_taken
        self.prior = prior
        
        self.children = []
        
        self.visit_count = 0
        self.value_sum = 0
        
    def is_fully_expanded(self):
        return len(self.children) > 0
    
    def select(self):
        best_child = None
        best_ucb = -np.inf
        
        for child in self.children:
            ucb = self.get_ucb(child)
            if ucb > best_ucb:
                best_child = child
                best_ucb = ucb
                
        return best_child
    
    def get_ucb(self, child):
        if child.visit_count == 0:
            q_value = 0
        else:
            q_value = 1 - ((child.value_sum / child.visit_count) + 1) / 2
        return q_value + self.args['C'] * (math.sqrt(self.visit_count) / (child.visit_count + 1)) * child.prior
    
    def expand(self, policy):
        for action, prob in enumerate(policy):
            if prob > 0:
                child_state = self.state.copy()
                child_state = self.game.get_next_state(child_state, action, 1)
                child_state = self.game.change_perspective(child_state, player=-1)

                child = Node(self.game, self.args, child_state, self, action, prob)
                self.children.append(child)
                
        return child
            
    def backpropagate(self, value):
        self.value_sum += value
        self.visit_count += 1
        
        value = self.game.get_opponent_value(value)
        if self.parent is not None:
            self.parent.backpropagate(value)  


class MCTS:
    def __init__(self, game, args, model):
        self.game = game
        self.args = args
        self.model = model
        
    @torch.no_grad()
    def search(self, state):
        root = Node(self.game, self.args, state)
        
        for search in range(self.args['num_searches']):
            node = root
            
            while node.is_fully_expanded():
                node = node.select()
                
            value, is_terminal = self.game.get_value_and_terminated(node.state, node.action_taken)
            value = self.game.get_opponent_value(value)
            
            if not is_terminal:
                policy, value = self.model(
                    torch.tensor(self.game.get_encoded_state(node.state), device=self.model.device).unsqueeze(0)
                )

                policy = torch.softmax(policy, axis=1).squeeze(0).cpu().numpy()
                valid_moves = self.game.get_valid_moves(node.state)
                policy *= valid_moves
                policy /= np.sum(policy)
                
                value = value.item()
                
                node.expand(policy)
                
            node.backpropagate(value)    
            
            
        action_probs = np.zeros(self.game.action_size)
        for child in root.children:
            action_probs[child.action_taken] = child.visit_count
        action_probs /= np.sum(action_probs)
        return action_probs
        

class AlphaZero:
    def __init__(self, model, optimizer, game, args):
        self.model = model
        self.optimizer = optimizer
        self.game = game
        self.args = args
        self.mcts = MCTS(game, args, model)
        
    def selfPlay(self):
        memory = []
        player = 1
        state = self.game.get_initial_state()
        
        while True:
            neutral_state = self.game.change_perspective(state, player)
            action_probs = self.mcts.search(neutral_state)

            memory.append((neutral_state, action_probs, player))

            if(player == -1):
                action_probs = action_probs.reshape(self.game.column_count, self.game.column_count).T.reshape(self.game.column_count**2,)
            
            action = np.random.choice(self.game.action_size, p=action_probs)
            
            state = self.game.get_next_state(state, action, player)
            
            value, is_terminal = self.game.get_value_and_terminated(state, action)
            
            if is_terminal:
                returnMemory = []
                for hist_neutral_state, hist_action_probs, hist_player in memory:
                    hist_outcome = value if hist_player == player else self.game.get_opponent_value(value)
                    returnMemory.append((
                        self.game.get_encoded_state(hist_neutral_state),
                        hist_action_probs,
                        hist_outcome
                    ))
                return returnMemory
            
            player = self.game.get_opponent(player)
                
    def train(self, memory):
        random.shuffle(memory)
        for batchIdx in range(0, len(memory), self.args['batch_size']):
            sample = memory[batchIdx:min(len(memory) - 1, batchIdx + self.args['batch_size'])]
            state, policy_targets, value_targets = zip(*sample)
            
            state, policy_targets, value_targets = np.array(state), np.array(policy_targets), np.array(value_targets).reshape(-1, 1)
            
            state = torch.tensor(state, dtype=torch.float32, device=self.model.device)
            policy_targets = torch.tensor(policy_targets, dtype=torch.float32, device=self.model.device)
            value_targets = torch.tensor(value_targets, dtype=torch.float32, device=self.model.device)
            
            out_policy, out_value = self.model(state)
            
            policy_loss = F.cross_entropy(out_policy, policy_targets)
            value_loss = F.mse_loss(out_value, value_targets)
            loss = policy_loss + value_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    def learn(self):
        for iteration in range(self.args['num_iterations']):
            memory = []
            
            self.model.eval()
            for selfPlay_iteration in range(self.args['num_selfPlay_iterations']):
                memory += self.selfPlay()
                
            self.model.train()
            for epoch in range(self.args['num_epochs']):
                self.train(memory)
            
            torch.save(self.model.state_dict(), f"model_{self.game.column_count}x{self.game.column_count}_{iteration}.pt")
            torch.save(self.optimizer.state_dict(), f"optimizer_{self.game.column_count}x{self.game.column_count}_{iteration}.pt")

def is_there_winning_move(global_hex, game_size, state, player):
    player_moves = np.sum(state == player)
    valid_moves = []
    if(player_moves < game_size - 1):
        return (False, None)
    else:
        all_actions = np.array(range(game_size**2))
        valid_moves = global_hex.get_valid_moves(state)
        valid_actions = all_actions[valid_moves == 1]
        

        for action in valid_actions:
            potential_state = global_hex.get_next_state(state.copy(), action, player)
            if(global_hex.check_win(
                    potential_state,
                    action)):
                return (True, action)
    return (False, None)

def print_game(state):
    for i in range(state.shape[0]):
            print(i*'  ', end='')
            for j in state[i]:
                if(j == 0 or j == 1 or j==-1):
                    print(' ',int(j), end='')
                else:
                    print(int(j),end='')
            print()


def distance_from_center(matrix, row, col):
    rows, cols = matrix.shape
    center_row, center_col = rows // 2, cols // 2
    return abs(row - center_row) + abs(col - center_col)
    
def paint_center_importance(state, follow_opponent, local_action):
    state[state == -1] = 1
    dim = state.shape[0]

    if(follow_opponent):
        state = np.zeros((dim, dim))
        state[local_action[0], local_action[1]] = 1

    for i in range(dim):
        for j in range(dim):
            distance = distance_from_center(state.copy(), i, j)
            state[i, j] *= dim - distance
    return state

def get_subregion_score(subregion, follow_opponent, local_action, player):

    # If moves less than required to win, win is impossible
    player_moves_count = np.sum(subregion == player)
    if(player_moves_count >= subregion.shape[0] - 1):

        # if subregion won, ignore
        win = False
        if player == 1:
            clumps = label(subregion > 0, _adj)[0]
        else:
            clumps = label(subregion.T < 0, _adj)[0]
        spanning_clumps = np.intersect1d(clumps[0], clumps[-1])
        win = np.count_nonzero(spanning_clumps)
        if(win):
            return -1

    # The negatives no longer matter
    subregion[subregion == -1] = 1
    subregion = paint_center_importance(subregion.copy(), follow_opponent, local_action)
    return np.sum(subregion)


def get_best_subregion(state, last_action, local_board_dimension, player, follow_opponent=True):
    global_board_dimension = state.shape[0]
    subregions_dimension = global_board_dimension - local_board_dimension + 1

    best_i = -1
    best_j = -1
    best_score = 0

    # If all subregions would be with 0, return centre
    state_copy = state.copy()
    state_copy[state == -1] = 1
    if(np.sum(state_copy) == 0):
        return ((global_board_dimension-local_board_dimension)//2, (global_board_dimension-local_board_dimension)//2)

    # Get all subregions
    for i in range(subregions_dimension):
        if(follow_opponent):
            if(not(i <= last_action[0] and last_action[0] < i + local_board_dimension)):
                continue
        for j in range(subregions_dimension):
            if(follow_opponent):
                if(not(j <= last_action[1] and last_action[1] < j + local_board_dimension)):
                    continue
            curr_subregion = state[i:i+local_board_dimension, j:j+local_board_dimension]
            local_action = (last_action[0] - i, last_action[1] - j)
            curr_score = get_subregion_score(curr_subregion.copy(), follow_opponent, local_action, player)
            if(curr_score >= best_score):
                best_score = curr_score
                best_i = i
                best_j = j
    return (best_i, best_j)


class AggroHexAgent():
    """This class describes the AggroHexAgent Hex agent."""

    HOST = "127.0.0.1"
    PORT = 1234

    def __init__(self, board_size=11):
        self.s = socket.socket(
            socket.AF_INET, socket.SOCK_STREAM
        )

        self.s.connect((self.HOST, self.PORT))

        self.board_size = board_size
        self.local_board_size = 7
        self.args = {
                    'C': 1.41,
                    'num_searches': 600
                    }
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.colour = ""
        self.can_swap = False

    def run(self):
        """Reads data until it receives an END message or the socket closes."""

        while True:
            data = self.s.recv(1024)
            if not data:
                break
            if (self.interpret_data(data)):
                break


    def interpret_data(self, data):
        """Checks the type of message and responds accordingly. Returns True
        if the game ended, False otherwise.
        """

        messages = data.decode("utf-8").strip().split("\n")
        messages = [x.split(";") for x in messages]
        for s in messages:
            if s[0] == "START":
                self.board_size = int(s[1])
                self.colour = s[2]
                if(s[2] == "B"):
                    self.can_swap = True

                # Set up both hex instances (local and global)
                self.global_hex = Hex(self.board_size)
                self.local_hex = Hex(self.local_board_size)

                # Load in model weights & MCTS
                self.model = ResNet(self.local_hex, 8, 128, self.device)
                self.model.load_state_dict(torch.load("model_7x7_7.pt", map_location=self.device))
                self.model.eval()
                self.mcts = MCTS(self.local_hex, self.args, self.model)

                self.state = self.global_hex.get_initial_state()

                if self.colour == "R":
                    self.make_move(opponent_last_move = None)

            elif s[0] == "END":
                return True

            elif s[0] == "CHANGE":
                if s[3] == "END":
                    return True

                elif s[1] == "SWAP":
                    self.colour = self.opp_colour()
                    
                    # If swapped to B because of opponent
                    if(self.colour == "B"):
                        moves = s[2].split(',')
                        for row_i in range(len(moves)):
                            for col_j in range(len(moves[row_i])):
                                if(moves[row_i][col_j] == 'R'):
                                    oppontent_last_move = (int(row_i), int(col_j))
                    
                    if s[3] == self.colour:
                        self.make_move(opponent_last_move = oppontent_last_move)

                elif s[3] == self.colour:
                    moves = s[1].split(',')
                    oppontent_last_move = (int(moves[0]), int(moves[1]))
                    action = [int(x) for x in s[1].split(",")]
                    action = action[0] * self.board_size + action[1]
                    if(self.colour == "R"):
                        self.state = self.global_hex.get_next_state(self.state, action, -1)
                    elif(self.colour == "B"):
                        self.state = self.global_hex.get_next_state(self.state, action, 1)
                    self.make_move(opponent_last_move = oppontent_last_move)

        return False

    def make_move(self, opponent_last_move = None):

        action = None
        global_row = 0
        global_col = 0
        player = 1
        if(self.colour == "R"):
            player = 1
        elif(self.colour == "B"):
            player = -1

        # If there is a winning move, play it
        winning_move = is_there_winning_move(self.global_hex, self.board_size, self.state.copy(), player)
        if winning_move[0]:
            action = winning_move[1]
            global_row = action // self.board_size
            global_col = action % self.board_size
        else:
            # If the enemy has a winning move, block them by playing it
            enemy_winning_move = is_there_winning_move(self.global_hex, self.board_size, self.state.copy(), -player)
            if(enemy_winning_move[0]):
                action = enemy_winning_move[1]
                global_row = action // self.board_size
                global_col = action % self.board_size  
            else:
                # If playing as R choose a random opener
                if(self.colour == "R" and opponent_last_move == None):
                    action = choice([2, 12, 13, 22, 23, 33, 87, 97, 98, 107, 108, 117, 118, ])
                    global_row = action // self.board_size
                    global_col = action % self.board_size
                
                # If playing as B and can swap, choose whether to swap based on the
                # move that R chose
                elif(self.can_swap):
                    if(self.colour == "B"):
                        opponent_last_action = int(opponent_last_move[0]) * self.board_size + int(opponent_last_move[1])
                        non_swappable_positions = [0, 1, 11, 109, 119, 120]
                        
                        # If the opener isn't in the bad positions, swap
                        if(opponent_last_action not in non_swappable_positions):
                            action = 'SWAP'
                        self.can_swap = False
                
                # If an action hasn't been chosen yet, utilise AggroZero strategy
                if(action == None):
                    i, j = get_best_subregion(self.state.copy(), opponent_last_move, self.local_board_size, player, True)
                    if i == -1 or j == -1:
                        i, j = get_best_subregion(self.state.copy(), opponent_last_move, self.local_board_size, player, False)
                    subregion = self.state.copy()[i : i + self.local_board_size, j : j + self.local_board_size]
                    neutral_state = self.local_hex.change_perspective(subregion, player)
                    
                    # Need to transpose if playing as B
                    mcts_probs = None
                    if(self.colour == "R"):
                        mcts_probs = self.mcts.search(neutral_state)
                    elif(self.colour == "B"):
                        mcts_probs = self.mcts.search(neutral_state).reshape(self.local_board_size, self.local_board_size).T.reshape(self.local_board_size * self.local_board_size,)
                    local_action = np.argmax(mcts_probs)

                    local_row = local_action // self.local_board_size
                    local_col = local_action % self.local_board_size

                    global_row = local_row + i
                    global_col = local_col + j

                    action = global_row * self.board_size + global_col
        
        # Choose what message to send the engine
        if(action == "SWAP"):
            self.s.sendall(bytes(f"SWAP\n", "utf-8"))
        else:
            self.state = self.global_hex.get_next_state(self.state, action, player)
            self.s.sendall(bytes(f"{global_row},{global_col}\n", "utf-8"))

    def opp_colour(self):
        """Returns the char representation of the colour opposite to the
        current one.
        """
        if self.colour == "R":
            return "B"
        elif self.colour == "B":
            return "R"
        else:
            return "None"


if (__name__ == "__main__"):
    agent = AggroHexAgent()
    agent.run()



