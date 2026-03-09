import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
from tqdm import trange
import random

class TicTacToe:

    def __init__(self):
        self.row = 3
        self.col = 3
        self.action_size = 9

    def init_game(self):
        return np.zeros((3,3))

    def get_next_state(self,state,action,player):

        # FIX: never mutate state in-place
        next_state = state.copy()

        row = action // self.col
        col = action % self.col

        next_state[row,col] = player
        return next_state


    def get_valid_moves(self,state):
        return (state.reshape(-1)==0).astype(np.uint8)


    def check_win(self,state,action):

        if action is None:
            return False

        row = action // self.col
        col = action % self.col

        player = state[row,col]

        return (

            np.sum(state[row,:]) == player*self.col or
            np.sum(state[:,col]) == player*self.row or
            np.sum(np.diag(state)) == player*self.row or
            np.sum(np.diag(np.flip(state,axis=0))) == player*self.row

        )


    def get_value_and_is_terminated(self,state,action):

        if self.check_win(state,action):
            return 1,True

        if np.sum(self.get_valid_moves(state))==0:
            return 0,True

        return 0,False


    def get_opponent(self,player):
        return -player


    def get_opponent_value(self,value):
        return -value


    def change_perspective(self,state,player):
        return state * player


    def get_encoded_state(self,state):

        encoded = np.stack(

            (state==-1,state==0,state==1)

        ).astype(np.float32)

        return encoded



# =========================================================
# MCTS NODE
# =========================================================

class Node:

    def __init__(self,game,args,state,parent=None,action_taken=None,prior=0):

        self.game = game
        self.args = args

        self.state = state
        self.parent = parent
        self.action_taken = action_taken

        self.children = []

        self.visit_count = 0
        self.value_sum = 0
        self.prior = prior

        self.C = args["C"]


    def is_fully_expanded(self):

        return len(self.children) > 0


    def select(self):

        best_child = None
        best_ucb = -np.inf

        for child in self.children:

            ucb = self.get_ucb(child)

            if ucb > best_ucb:
                best_ucb = ucb
                best_child = child

        return best_child


    def get_ucb(self,child):

        # FIX: correct AlphaZero PUCT formula

        if child.visit_count == 0:
            q = 0
        else:
            q = child.value_sum / child.visit_count

        u = self.C * child.prior * np.sqrt(self.visit_count) / (1 + child.visit_count)

        return q + u


    def expand(self,policy):

        for action,prob in enumerate(policy):

            if prob > 0:

                child_state = self.game.get_next_state(self.state,action,1)

                child_state = self.game.change_perspective(child_state,-1)

                child = Node(self.game,self.args,child_state,self,action,prob)

                self.children.append(child)


    def backpropagate(self,value):

        self.value_sum += value
        self.visit_count += 1

        value = self.game.get_opponent_value(value)

        if self.parent is not None:

            self.parent.backpropagate(value)





class MonteCarloTreeSearch:

    def __init__(self,game,args,model):

        self.game = game
        self.args = args
        self.model = model


    def search(self,state):

        root = Node(self.game,self.args,state)

        for _ in range(self.args["num_searches"]):

            node = root

            # SELECT
            while node.is_fully_expanded():

                node = node.select()


            value,is_terminal = self.game.get_value_and_is_terminated(
                node.state,node.action_taken
            )

            value = self.game.get_opponent_value(value)


            if not is_terminal:

                encoded = self.game.get_encoded_state(node.state)

                tensor = torch.tensor(encoded).unsqueeze(0)

                # FIX: no_grad prevents autograd overhead
                with torch.no_grad():

                    policy,value = self.model(tensor)

                policy = torch.softmax(policy,dim=1).squeeze(0).cpu().numpy()

                valid_moves = self.game.get_valid_moves(node.state)

                policy *= valid_moves

                # FIX: avoid divide by zero
                if np.sum(policy) == 0:
                    policy = valid_moves / np.sum(valid_moves)
                else:
                    policy /= np.sum(policy)

                value = value.item()

                node.expand(policy)

            node.backpropagate(value)


        action_probs = np.zeros(self.game.action_size)

        for child in root.children:

            action_probs[child.action_taken] = child.visit_count


        action_probs /= np.sum(action_probs)

        return action_probs



# =========================================================
# NETWORK
# =========================================================

class ResBlock(nn.Module):

    def __init__(self,num_hidden):

        super().__init__()

        self.conv1 = nn.Conv2d(num_hidden,num_hidden,3,padding=1)
        self.bn1 = nn.BatchNorm2d(num_hidden)

        self.conv2 = nn.Conv2d(num_hidden,num_hidden,3,padding=1)
        self.bn2 = nn.BatchNorm2d(num_hidden)


    def forward(self,x):

        residual = x

        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))

        x += residual

        return F.relu(x)



class ResNet(nn.Module):

    def __init__(self,game,num_res,num_hidden):

        super().__init__()

        self.start = nn.Sequential(

            nn.Conv2d(3,num_hidden,3,padding=1),
            nn.BatchNorm2d(num_hidden),
            nn.ReLU()

        )

        self.backbone = nn.ModuleList(

            [ResBlock(num_hidden) for _ in range(num_res)]

        )

        self.policy_head = nn.Sequential(

            nn.Conv2d(num_hidden,32,3,padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32*9,9)

        )

        self.value_head = nn.Sequential(

            nn.Conv2d(num_hidden,3,3,padding=1),
            nn.BatchNorm2d(3),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(27,1),
            nn.Tanh()

        )


    def forward(self,x):

        x = self.start(x)

        for block in self.backbone:
            x = block(x)

        policy = self.policy_head(x)
        value = self.value_head(x)

        return policy,value



# =========================================================
# ALPHAZERO
# =========================================================

class AlphaZero:

    def __init__(self,model,optimizer,game,args):

        self.model = model
        self.optimizer = optimizer
        self.game = game
        self.args = args

        self.mcts = MonteCarloTreeSearch(game,args,model)

        self.memory = []


    def self_play(self):

        history = []

        player = 1
        state = self.game.init_game()


        while True:

            neutral = self.game.change_perspective(state,player)

            probs = self.mcts.search(neutral)

            history.append((self.game.get_encoded_state(neutral),probs,player))


            # FIX: temperature sampling early
            if len(history) < 6:
                action = np.random.choice(9,p=probs)
            else:
                action = np.argmax(probs)


            state = self.game.get_next_state(state,action,player)

            value,terminal = self.game.get_value_and_is_terminated(state,action)


            if terminal:

                for hist_state,hist_probs,hist_player in history:

                    outcome = value if hist_player == player else -value

                    self.memory.append((hist_state,hist_probs,outcome))

                return


            player = -player


    def train(self):

        batch = random.sample(self.memory,len(self.memory))

        for i in range(0,len(batch),self.args["batch_size"]):

            sample = batch[i:i+self.args["batch_size"]]

            state,policy,value = zip(*sample)

            state = torch.tensor(np.array(state))
            policy = torch.tensor(np.array(policy))
            value = torch.tensor(np.array(value)).unsqueeze(1).float()

            pred_policy,pred_value = self.model(state)

            # FIX: correct policy loss for distributions
            log_probs = F.log_softmax(pred_policy,dim=1)
            policy_loss = -(policy * log_probs).sum(dim=1).mean()
            value_loss = F.mse_loss(pred_value,value)
            loss = policy_loss + value_loss
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()


    def learn(self):

        for i in range(self.args["num_iterations"]):

            print("Iteration",i)

            self.memory = []

            self.model.eval()

            for _ in trange(self.args["num_episode"]):

                self.self_play()

            self.model.train()

            for _ in trange(self.args["num_epochs"]):

                self.train()

            torch.save(self.model.state_dict(),f"model_{i}.pth")


def main():

    # =========================
    # GAME + MODEL SETUP
    # =========================
    game = TicTacToe()

    model = ResNet(game, num_res=4, num_hidden=64)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    args = {
        "C": 2,
        "num_searches": 200,
        "num_iterations": 5,
        "num_episode": 500,
        "num_epochs": 4,
        "batch_size": 64
    }

    # =========================
    # TRAIN ALPHAZERO
    # =========================
    alphazero = AlphaZero(model, optimizer, game, args)

    print("Starting Training...")
    #alphazero.learn()

    print("Training Finished")


    # =========================
    # LOAD TRAINED MODEL
    # =========================
    model.load_state_dict(torch.load("model_4.pth"))

    model.eval()

    mcts = MonteCarloTreeSearch(game, args, model)


    # =========================
    # HUMAN VS AI GAMEPLAY
    # =========================
    state = game.init_game()

    player = 1

    while True:

        print("\nBoard State:")
        print(state)

        if player == 1:

            valid_moves = game.get_valid_moves(state)

            print("Valid moves:", [i for i in range(9) if valid_moves[i] == 1])

            action = int(input("Your move (0-8): "))

            if valid_moves[action] == 0:
                print("Invalid move")
                continue

        else:

            neutral_state = game.change_perspective(state, player)

            probs = mcts.search(neutral_state)

            action = np.argmax(probs)

            print("AI move:", action)


        state = game.get_next_state(state, action, player)

        value, terminal = game.get_value_and_is_terminated(state, action)

        if terminal:

            print("\nFinal Board:")
            print(state)

            if value == 1:
                print("Player", player, "wins")
            else:
                print("Draw")

            break


        player = game.get_opponent(player)



if __name__ == "__main__":
    main()