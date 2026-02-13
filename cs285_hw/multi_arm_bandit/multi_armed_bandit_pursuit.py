import numpy as np
import matplotlib.pyplot as plt


class Environment:
    #slot machine 10 states
    def __init__(self,probs:np.ndarray):
        self.probs = probs
    def step(self,slot):
        return 1 if (np.random.random()<self.probs[slot]) else 0

class MAB_Agent:
    def __init__(self,n_actions,beta:int):
        self.beta = beta
        self.n_actions = n_actions
        self.n = np.zeros(n_actions, dtype=int)
        self.Q = np.zeros(n_actions, dtype=float)
        self.policy = np.ones(self.n_actions)*1/self.n_actions

    def update_Q(self,action,reward):
        self.n[action]+=1
        self.Q[action]+=(1.0/self.n[action])*(reward-self.Q[action])
        
        #update policy
        greedy_action = np.argmax(self.Q)
        #nominally decrease others
        self.policy-=self.beta*self.policy
        #make argmax better
        self.policy[greedy_action]+=self.beta

    def get_action(self,):
        return np.random.choice(self.n_actions,p=self.policy)

class MultiArmedBandit:
    def __init__(self,n_arms,probs,beta):
        self.n_arms = n_arms
        self.probs = probs
        self.beta= beta

    def step(self,N_steps):
        env = Environment(self.probs)
        agent = MAB_Agent(self.n_arms,self.beta)
        actions,rewards = [],[]
        for i in range(N_steps):
            action = agent.get_action()
            reward = env.step(action)
            agent.update_Q(action,reward)
            actions.append(action)
            rewards.append(reward)
        return np.array(actions),np.array(rewards)
    
    def multi_step(self,N_steps,N_experiments):
        # 10000 EXPERIMENTS, WITH 500 STEPS EACH
        R = np.zeros((N_steps,))  
        A = np.zeros((N_steps, len(self.probs)))  
        for i in range(N_experiments):
            actions, rewards = self.step(N_steps) 
            if (i + 1) % (N_experiments / 100) == 0:
                print("[Experiment {}/{}] ".format(i + 1, N_experiments) +
                    "beta = {}, ".format(self.beta) +
                    "n_steps = {}, ".format(N_steps) +
                    "reward_avg = {}".format(np.sum(rewards) / len(rewards)))
            R += rewards
            for j, a in enumerate(actions):
                A[j][a] += 1
        return R,A
    
import os

probs=[0.10, 0.50, 0.60, 0.80, 0.10, 0.25, 0.60, 0.45, 0.75, 0.65]
betas = [0.01,0.05,0.1,0.3,0.5] #reward across varying beta
N_experiments=10000 
N_steps=500 
Rewards:list[np.ndarray] =[]
Actions:list[np.ndarray] =[]
for beta in betas:
    solve=MultiArmedBandit(len(probs), probs,beta)
    R,A = solve.multi_step(N_steps,N_experiments)
    Rewards.append(R)
    Actions.append(A)


for i,R in enumerate(Rewards):
    R_avg = R/np.float32(N_experiments)
    print(R_avg.shape)
    plt.plot(R_avg,".",label="T = {}".format(betas[i]))

plt.xlabel("Step")
plt.ylabel("Average Reward")
plt.grid()
plt.xlim([1,N_steps])


save_fig = True 
output_dir = os.getcwd()
plt.legend()

if save_fig:
    if not os.path.exists(output_dir): os.mkdir(output_dir)
    plt.savefig(os.path.join(output_dir, "rewards_pursuit.png"), bbox_inches="tight")
else:
    plt.show()
plt.close()

optimal_arm = np.argmax(probs)
#plotting only the optimal action
for idx,A in enumerate(Actions):
    slot_i_actions = 100 * A[:,optimal_arm]/N_experiments
    steps = list(np.array(range(len(slot_i_actions)))+1)
    plt.plot(steps, slot_i_actions, "-",
            linewidth=3,
            label="T = {}".format(betas[idx]))

plt.xlabel("Step")
plt.ylabel("Count Percentage (%)")
leg = plt.legend(loc='upper left', shadow=True)
plt.xlim([1, N_steps])
plt.ylim([0, 100])

for legobj in leg.legendHandles:
    legobj.set_linewidth(4.0)
if save_fig:
    if not os.path.exists(output_dir): os.mkdir(output_dir)
    plt.savefig(os.path.join(output_dir, "actions_pursuit.png"), bbox_inches="tight")
else:
    plt.show()
plt.close()