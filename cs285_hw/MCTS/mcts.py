import numpy as np

class TicTacToe:
    def __init__(self):
        self.row=3
        self.col=3
        self.action_size=self.row*self.col
    
    def init_game(self):
        return np.zeros((self.row,self.col))
    
    def get_next_state(self,state,action,player):
        row=action//self.col
        col=action % self.col
        state[row,col]=player
        return state
    
    def get_valid_moves(self,state):
        return (state.reshape(-1)==0).astype(np.uint8)
    
    def check_win(self,state,action):
        if(action==None):
            return False
        row = action//self.col
        col = action%self.col
        player = state[row,col]

        return (
            np.sum(state[row,:])==player*self.col or
            np.sum(state[:,col]) == player*self.row or
            np.sum(np.diag(state)) == player*self.row or
            np.sum(np.diag(np.flip(state,axis=0))) == player*self.row
        )
    
    def get_value_and_is_terminated(self,state,action):
        #action was taken by parent, so the player in check win will be parent.
        if self.check_win(state,action):
            return 1, True
        if np.sum(self.get_valid_moves(state))==0:
            return 0, True
        return 0,False
    
    def get_opponent(self,player):
        return -player

    def get_opponent_value(self,value):
        return -value
    
    def change_perspective(self,state,player):
        return state*player

#MCTS
class Node:
    def __init__(self,game,args,state,parent=None,action_taken=None):
        self.game = game
        self.args = args
        self.state = state
        self.parent:Node = parent
        self.action_taken = action_taken

        self.children = []
        self.expandable_states =  game.get_valid_moves(state)

        self.visit_count = 0
        self.value_sum = 0
        # CHANGED: exploration constant is configurable through args.
        self.C = self.args.get("C", np.sqrt(2))

    def is_fully_expanded(self):
        #game terminated and no children for the node
        return np.sum(self.expandable_states)==0 and len(self.children)!= 0
    
    def select(self):
        best_child = None
        best_ucb = -np.inf

        for child in self.children:
            ucb = self.get_ucb(child)
            if ucb>best_ucb:
                best_child = child
                best_ucb = ucb
        return best_child
    
    def get_ucb(self,child):
        # CHANGED: always visit unvisited children first and avoid divide-by-zero.
        if child.visit_count == 0:
            return np.inf
        # CHANGED: child value is from child perspective, so negate for parent selection.
        q_val = -child.value_sum / child.visit_count
        return q_val + self.C * np.sqrt(np.log(max(1, self.visit_count)) / child.visit_count)

    def expand(self):
        action = np.random.choice(np.where(self.expandable_states==1)[0])
        #that action has been selected, make it unselectable for future mc trials
        self.expandable_states[action]=0
        child_state = self.state.copy()
        child_state = self.game.get_next_state(child_state,action,1)
        child_state = self.game.change_perspective(child_state,-1)

        child = Node(self.game,self.args,child_state,self,action)
        self.children.append(child)
        return child

    def simulate(self):
        #this will work on the expanded node!!!
        #perorm monte carlo simulations , random rollouts to check the results
        value, is_terminal = self.game.get_value_and_is_terminated(self.state,self.action_taken)
        value = self.game.get_opponent_value(value)

        if is_terminal:
            return value
        
        rollout_state = self.state.copy()
        # CHANGED: track value perspective while rollout alternates players.
        perspective = 1
        while True:
            valid_moves = self.game.get_valid_moves(rollout_state)
            action = np.random.choice(np.where(valid_moves==1)[0])
            rollout_state = self.game.get_next_state(rollout_state,action,1)
            value, is_terminal = self.game.get_value_and_is_terminated(rollout_state,action)
            #print(is_terminal)
            if is_terminal:
                # CHANGED: convert terminal value to the node's perspective.
                return value * perspective
            rollout_state = self.game.change_perspective(rollout_state,-1)
            # CHANGED: perspective flips every ply after changing viewpoint.
            perspective *= -1

    def backpropagate(self,value):
        self.value_sum+=value
        self.visit_count+=1
        value = self.game.get_opponent_value(value)
        if self.parent is not None:
            self.parent.backpropagate(value)

class MonteCarloTreeSearch:
    def __init__(self,game,args):
        self.game = game
        self.args = args
    
    def search(self,state):
        #define root node with beginning state
        root = Node(self.game,self.args, state)
        for i in range(self.args["num_iter"]):
            node = root
            #select
            while node.is_fully_expanded():
                #select best node based on ucb
                node = node.select()
                
            #see if the node is a terminal state of the game, action taken was from the parent (aka the opponent)
            value, is_terminal = self.game.get_value_and_is_terminated(node.state,node.action_taken)
            #the previous line gives the value of the opponent from the childs perspecitve its the opposite
            value = self.game.get_opponent_value(value)

            if not is_terminal:
                #expansion
                node= node.expand()
                #simulation (rollouts)
                value = node.simulate()
            #backprop
            node.backpropagate(value)
        
        action_probs = np.zeros(self.game.action_size)
        #from the root we need to make only one move, to select the child (select the action most visited)
        for child in root.children:
            action_probs[child.action_taken]=child.visit_count
        # CHANGED: normalize safely in case no visits were recorded.
        total_visits = np.sum(action_probs)
        if total_visits > 0:
            action_probs /= total_visits
        return action_probs
        #return visit_counts


    

if __name__ == "__main__":
    tct = TicTacToe()
    player = 1
    args = {'num_iter': 5000, 'C': np.sqrt(2)}
    mcts = MonteCarloTreeSearch(tct,args)
    state = tct.init_game()

    while True:
        print(state)
        if player==1:
            valid_moves = tct.get_valid_moves(state)
            print("valid moves: ", [i for i in range(tct.action_size) if valid_moves[i]==1])
            action = int(input(f"{player}:"))
            if valid_moves[action]==0:
                print("action not valid")
                continue
        else:
            #according to any player, they are 1, and their opponent is -1
            neutral_state = tct.change_perspective(state,player)
            mcts_probs = mcts.search(neutral_state)
            action = np.argmax(mcts_probs)

        state = tct.get_next_state(state,action,player)
        value, is_terminal = tct.get_value_and_is_terminated(state,action)
        if is_terminal:
            print(state)
            if value ==1:
                print(player," won")
            else:
                print("draw")
            break
        player = tct.get_opponent(player)
