import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gammaln, digamma




# max info gain, minimal regret posterior sampling for multi armed bandits

class Environment:
    #slot machine 10 states
    def __init__(self,probs:np.ndarray):
        self.probs = probs
    def step(self,slot):
        return 1 if (np.random.random()<self.probs[slot]) else 0


class InformationGainAgent:
    def __init__(self,n_actions):
        self.n_actions =  n_actions
        self.alpha = np.ones(n_actions)
        self.beta = np.ones(n_actions)
        self.mc_samples = 50
    def update(self,action,reward):
        if reward == 1:
            self.alpha[action] +=1
        else:
            self.beta[action] +=1

    def beta_entropy(self,alpha,beta):
        log_B=gammaln(alpha) + gammaln(beta) - gammaln(alpha + beta)
        term1=log_B
        term2=-(alpha-1)*digamma(alpha)
        term3=-(beta-1)*digamma(beta)
        term4=(alpha+beta-2)*digamma(alpha + beta)
        
        return term1 + term2 + term3 + term4

    def get_action(self,):
        #monte carlo estimate of regret for each arm
        deltas = np.zeros(self.n_actions)
        #vectorize this part(try)
        for s in range (self.mc_samples):
            arm_probs=np.random.beta(self.alpha,self.beta)
            best_arm =np.argmax(arm_probs)
            for a in range(self.n_actions):
                deltas[a]+=arm_probs[best_arm]-arm_probs[a]
        #norm
        deltas/=self.mc_samples

        #ig estimate for each arm
        p1 = self.alpha/(self.alpha+self.beta)
        p2 = self.beta/(self.alpha+self.beta)
        H_prior = self.beta_entropy(self.alpha,self.beta)
        H_post_win = self.beta_entropy(self.alpha+1,self.beta)
        H_post_loss = self.beta_entropy(self.alpha,self.beta+1)
        info_gain= H_prior - (p1*H_post_win+p2*H_post_loss)
        info_gain = np.maximum(info_gain,1e-8) #for stability during division

        score = (deltas ** 2) / info_gain
        return np.argmin(score)

class MultiArmedBandit:
    def __init__(self,n_arms,probs):
        self.n_arms = n_arms
        self.probs = probs

    def step(self,N_steps):
        env = Environment(self.probs)
        agent = InformationGainAgent(self.n_arms)
        actions,rewards = [],[]
        for i in range(N_steps):
            action = agent.get_action()
            reward = env.step(action)
            agent.update(action,reward)
            actions.append(action)
            rewards.append(reward)
        return np.array(actions),np.array(rewards)
    
    def multi_step(self,N_steps,N_experiments):
        print("Running {} experiments with {} steps each...".format(N_experiments, N_steps))
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
    plt.savefig(os.path.join(output_dir, "rewards_ig.png"), bbox_inches="tight")
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
    plt.savefig(os.path.join(output_dir, "actions_ig.png"), bbox_inches="tight")
else:
    plt.show()
plt.close()