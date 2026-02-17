import numpy as np
import matplotlib.pyplot as plt

#Thomposon sampling and upper confidence bound for multi arm bandits

class Environment:
    #slot machine 10 states
    def __init__(self,probs:np.ndarray):
        self.probs = probs
    def step(self,slot):
        return 1 if (np.random.random()<self.probs[slot]) else 0


class ThompsonAgent:
    def __init__(self,n_actions):
        self.n_actions = n_actions
        self.alpha = np.ones(n_actions)
        self.beta = np.ones(n_actions)
    
    def update(self,action,reward):
        if reward == 1:
            self.alpha[action] +=1
        else:
            self.beta[action] +=1

    def get_action(self,):
        samples = np.random.beta(self.alpha,self.beta)
        return np.argmax(samples)


class MultiArmedBandit:
    def __init__(self,n_arms,probs):
        self.n_arms = n_arms
        self.probs = probs

    def step(self,N_steps):
        env = Environment(self.probs)
        agent = ThompsonAgent(self.n_arms)
        actions,rewards = [],[]
        for i in range(N_steps):
            action = agent.get_action()
            reward = env.step(action)
            agent.update(action,reward)
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
                    "reward_avg = {}".format(np.sum(rewards) / len(rewards)))
            R+=rewards
            for j, a in enumerate(actions):
                A[j][a] += 1

        return R,A
    
import os

probs=[0.10, 0.50, 0.60, 0.80, 0.10, 0.25, 0.60, 0.45, 0.75, 0.65]
eps=0.3
solve=MultiArmedBandit(len(probs), probs)
N_experiments=10000 
N_steps=500 


R,A = solve.multi_step(N_steps,N_experiments)

save_fig = True 
output_dir = os.getcwd()


#plots
R_avg = R/np.float32(N_experiments)
print(R_avg.shape)
plt.plot(R_avg,".")
plt.xlabel("Step")
plt.ylabel("Average Reward")
plt.grid()
plt.xlim([1,N_steps])
if save_fig:
    if not os.path.exists(output_dir): os.mkdir(output_dir)
    plt.savefig(os.path.join(output_dir, "rewards_thompson.png"), bbox_inches="tight")
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
    plt.savefig(os.path.join(output_dir, "actions_thompson.png"), bbox_inches="tight")
else:
    plt.show()
plt.close()