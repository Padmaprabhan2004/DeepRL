import numpy as np
import matplotlib.pyplot as plt


class Environment:
    #slot machine 10 states
    def __init__(self,probs:np.ndarray):
        self.probs = probs
    def step(self,slot):
        return 1 if (np.random.random()<self.probs[slot]) else 0

class MAB_Agent:
    def __init__(self,eps,n_actions):
        self.eps = eps
        self.n_actions = n_actions
        self.n = np.zeros(n_actions, dtype=int)
        self.Q = np.zeros(n_actions, dtype=float)

    def update_Q(self,action,reward):
        self.n[action]+=1
        self.Q[action]+= (1.0/self.n[action]) * (reward - self.Q[action])

    def get_action(self,):
        if np.random.random() < self.eps:
            #exploration
            return np.random.randint(self.n_actions)
        else:
            #exploitation
            #uniform over max returns, dont always go for argmax slots
            best_val = np.max(self.Q)
            best_slots = np.where(self.Q == best_val)[0]
            return np.random.choice(best_slots)

class MultiArmedBandit:
    def __init__(self,n_arms,probs,eps=0.1):
        self.n_arms = n_arms
        self.probs = probs
        self.eps = eps

    def step(self,N_steps):
        env = Environment(self.probs)
        agent = MAB_Agent(self.eps,self.n_arms)
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
                    "n_steps = {}, ".format(N_steps) +
                    "reward_avg = {}".format(np.sum(rewards) / len(rewards)))
            R += rewards
            for j, a in enumerate(actions):
                A[j][a] += 1
        return R,A
    
import os

probs=[0.10, 0.50, 0.60, 0.80, 0.10, 0.25, 0.60, 0.45, 0.75, 0.65]
eps=0.3
solve=MultiArmedBandit(10, probs, eps=eps)
N_experiments=10000 
N_steps=500 


R,A = solve.multi_step(N_steps,N_experiments)

save_fig = True 
output_dir = os.getcwd()


#plots
R_avg = R/np.float32(N_experiments)
plt.plot(R_avg,".")
plt.xlabel("Step")
plt.ylabel("Average Reward")
plt.grid()
plt.xlim([1,N_steps])
if save_fig:
    if not os.path.exists(output_dir): os.mkdir(output_dir)
    plt.savefig(os.path.join(output_dir, "rewards_{}.png".format(eps)), bbox_inches="tight")
else:
    plt.show()
plt.close()

for i in range(len(probs)):
    slot_i_actions = 100 * A[:,i]/N_experiments
    steps = list(np.array(range(len(slot_i_actions)))+1)
    plt.plot(steps, slot_i_actions, "-",
             linewidth=4,
             label="Slot {} ({:.0f}%)".format(i+1, 100*probs[i]))

plt.xlabel("Step")
plt.ylabel("Count Percentage (%)")
leg = plt.legend(loc='upper left', shadow=True)
plt.xlim([1, N_steps])
plt.ylim([0, 100])

for legobj in leg.legendHandles:
    legobj.set_linewidth(4.0)
if save_fig:
    if not os.path.exists(output_dir): os.mkdir(output_dir)
    plt.savefig(os.path.join(output_dir, "actions_{}.png".format(eps)), bbox_inches="tight")
else:
    plt.show()
plt.close()