from typing import Callable, Optional, Tuple
import numpy as np
import torch.nn as nn
import torch
import gym
from cs285.infrastructure import pytorch_util as ptu

#question : where are discounted rewards????
class ModelBasedAgent(nn.Module):
    def __init__(
        self,
        env: gym.Env,
        make_dynamics_model: Callable[[Tuple[int, ...], int], nn.Module],
        make_optimizer: Callable[[nn.ParameterList], torch.optim.Optimizer],
        ensemble_size: int,
        mpc_horizon: int,
        mpc_strategy: str,
        mpc_num_action_sequences: int,
        cem_num_iters: Optional[int] = None,
        cem_num_elites: Optional[int] = None,
        cem_alpha: Optional[float] = None,
    ):
        super().__init__()
        self.env = env
        self.mpc_horizon = mpc_horizon
        self.mpc_strategy = mpc_strategy
        self.mpc_num_action_sequences = mpc_num_action_sequences
        self.cem_num_iters = cem_num_iters
        self.cem_num_elites = cem_num_elites
        self.cem_alpha = cem_alpha

        assert mpc_strategy in (
            "random",
            "cem",
        ), f"'{mpc_strategy}' is not a valid MPC strategy"

        # ensure the environment is state-based
        assert len(env.observation_space.shape) == 1
        assert len(env.action_space.shape) == 1

        self.eps = 1e-9
        self.ob_dim = env.observation_space.shape[0]
        self.ac_dim = env.action_space.shape[0]

        self.ensemble_size = ensemble_size
        self.dynamics_models = nn.ModuleList(
            [
                make_dynamics_model(
                    self.ob_dim,
                    self.ac_dim,
                )
                for _ in range(ensemble_size)
            ]
        )
        self.optimizer = make_optimizer(self.dynamics_models.parameters())
        self.loss_fn = nn.MSELoss()

        # keep track of statistics for both the model input (obs & act) and
        # output (obs delta)
        self.register_buffer(
            "obs_acs_mean", torch.zeros(self.ob_dim + self.ac_dim, device=ptu.device)
        )
        self.register_buffer(
            "obs_acs_std", torch.ones(self.ob_dim + self.ac_dim, device=ptu.device)
        )
        self.register_buffer(
            "obs_delta_mean", torch.zeros(self.ob_dim, device=ptu.device)
        )
        self.register_buffer(
            "obs_delta_std", torch.ones(self.ob_dim, device=ptu.device)
        )

    def update(self, i: int, obs: np.ndarray, acs: np.ndarray, next_obs: np.ndarray):
        """
        Update self.dynamics_models[i] using the given batch of data.

        Args:
            i: index of the dynamics model to update
            obs: (batch_size, ob_dim)
            acs: (batch_size, ac_dim)
            next_obs: (batch_size, ob_dim)
        """
        obs = ptu.from_numpy(obs)
        acs = ptu.from_numpy(acs)
        next_obs = ptu.from_numpy(next_obs)
        # TODO(student): update self.dynamics_models[i] using the given batch of data
        # HINT: make sure to normalize the NN input (observations and actions)
        # *and* train it with normalized outputs (observation deltas) 
        # HINT 2: make sure to train it with observation *deltas*, not next_obs
        # directly
        # HINT 3: make sure to avoid any risk of dividing by zero when
        # normalizing vectors by adding a small number to the denominator!
        #only update self.dynamics_models[i]
        delta_obs = next_obs-obs# train with delta obs, ref mb-mf paper
        delta_obs = (delta_obs - self.obs_delta_mean)/(self.obs_delta_std)


        input_states = torch.concat((obs,acs),dim=1) #(batch_size,obs+acd dim concatenated)
        input_states = (input_states-self.obs_acs_mean)/(self.obs_acs_std)

        delta_pred = self.dynamics_models[i](input_states)
        loss = self.loss_fn(delta_pred,delta_obs)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return ptu.to_numpy(loss)

    @torch.no_grad()
    def update_statistics(self, obs: np.ndarray, acs: np.ndarray, next_obs: np.ndarray):
        """
        Update the statistics used to normalize the inputs and outputs of the dynamics models.

        Args:
            obs: (n, ob_dim)
            acs: (n, ac_dim)
            next_obs: (n, ob_dim)
        """
        obs = ptu.from_numpy(obs)
        acs = ptu.from_numpy(acs)
        next_obs = ptu.from_numpy(next_obs)

        obs_acs = torch.concat((obs,acs),dim=1)
        obs_delta = next_obs - obs
        self.obs_acs_mean = torch.mean(obs_acs,dim=0)
        self.obs_acs_std = torch.std(obs_acs,dim=0)+self.eps
        self.obs_delta_mean = torch.mean(obs_delta,dim=0)
        self.obs_delta_std = torch.std(obs_delta,dim=0)+self.eps

    @torch.no_grad()
    def get_dynamics_predictions(
        self, i: int, obs: np.ndarray, acs: np.ndarray
    ) -> np.ndarray:
        """
        Takes a batch of each current observation and action and outputs the
        predicted next observations from self.dynamics_models[i].

        Args:
            obs: (batch_size, ob_dim)
            acs: (batch_size, ac_dim)
        Returns: (batch_size, ob_dim)
        """
        obs = ptu.from_numpy(obs)
        acs = ptu.from_numpy(acs)
        # TODO(student): get the model's predicted `next_obs`
        # HINT: make sure to *unnormalize* the NN outputs (observation deltas)
        # Same hints as `update` above, avoid nasty divide-by-zero errors when
        # normalizing inputs!
        input_states = torch.concat((obs,acs),dim=1) #(obs+acs,1)
        input_states = (input_states-self.obs_acs_mean)/(self.obs_acs_std)
        delta_pred = self.dynamics_models[i](input_states)
        pred_next_obs = obs+ delta_pred*self.obs_delta_std+self.obs_delta_mean #unnormalize the nn outputs and add

        return ptu.to_numpy(pred_next_obs)

    def evaluate_action_sequences(self, obs: np.ndarray, action_sequences: np.ndarray):
        """
        Evaluate a batch of action sequences using the ensemble of dynamics models.

        Args:
            obs: starting observation, shape (ob_dim,)
            action_sequences: shape (mpc_num_action_sequences, horizon, ac_dim)
        Returns:
            sum_of_rewards: shape (mpc_num_action_sequences,)
        """
        # We are going to predict (ensemble_size * mpc_num_action_sequences)
        # distinct rollouts, and then average over the ensemble dimension to get
        # the reward for each action sequence.

        # We start by initializing an array to keep track of the reward for each
        # of these rollouts.
        sum_of_rewards = np.zeros(
            (self.ensemble_size, self.mpc_num_action_sequences), dtype=np.float32
        )
        # We need to repeat our starting obs for each of the rollouts.
        obs = np.tile(obs, (self.ensemble_size, self.mpc_num_action_sequences, 1))

        # TODO(student): for each batch of actions in in the horizon...
        for i in range(action_sequences.shape[1]):
            acs = action_sequences[:,i,:]
            assert acs.shape == (self.mpc_num_action_sequences, self.ac_dim)
            assert obs.shape == (
                self.ensemble_size,
                self.mpc_num_action_sequences,
                self.ob_dim,
            )

            # TODO(student): predict the next_obs for each rollout
            # HINT: use self.get_dynamics_predictions
            next_obs = np.array([
                self.get_dynamics_predictions(j, obs[j], acs)
                for j in range(self.ensemble_size)
            ])
            assert next_obs.shape == (
                self.ensemble_size,
                self.mpc_num_action_sequences,
                self.ob_dim,
            )

            # TODO(student): get the reward for the current step in each rollout
            # HINT: use `self.env.get_reward`. `get_reward` takes 2 arguments:
            # `next_obs` and `acs` with shape (n, ob_dim) and (n, ac_dim),
            # respectively, and returns a tuple of `(rewards, dones)`. You can 
            # ignore `dones`. You might want to do some reshaping to make
            # `next_obs` and `acs` 2-dimensional.
            rewards= self.env.get_reward(next_obs.reshape(self.ensemble_size*self.mpc_num_action_sequences,-1),acs.reshape(self.ensemble_size*self.mpc_num_action_sequences,-1))
            rewards =  rewards.reshape(self.ensemble_size,self.mpc_num_action_sequences)
            assert rewards.shape == (self.ensemble_size, self.mpc_num_action_sequences)

            sum_of_rewards += rewards

            obs = next_obs

        # now we average over the ensemble dimension
        return sum_of_rewards.mean(axis=0)

    def get_action(self, obs: np.ndarray):
        """
        Choose the best action using model-predictive control.

        Args:
            obs: (ob_dim,)
        """
        # always start with uniformly random actions
        action_sequences = np.random.uniform(
            self.env.action_space.low,
            self.env.action_space.high,
            size=(self.mpc_num_action_sequences, self.mpc_horizon, self.ac_dim),
        )

        if self.mpc_strategy == "random":
            # evaluate each action sequence and return the best one
            rewards = self.evaluate_action_sequences(obs, action_sequences)
            assert rewards.shape == (self.mpc_num_action_sequences,)
            best_index = np.argmax(rewards)
            return action_sequences[best_index][0]
        elif self.mpc_strategy == "cem":
            #self.cem_num_iters = cem_num_iters
            #self.cem_num_elites = cem_num_elites
            #self.cem_alpha = cem_alpha
            rewards =  self.evaluate_action_sequences(obs,action_sequences)
            top_j_indices = np.argsort(rewards)[-self.cem_num_elites:]
            top_j_seq = action_sequences[top_j_indices,:,:]
            elite_mean = np.mean(top_j_seq, axis=0)
            elite_std = np.std(top_j_seq, axis=0)
            for i in range(1,self.cem_num_iters):
                # TODO(student): implement the CEM algorithm
                # HINT: you need a special case for i == 0 to initialize
                # the elite mean and std
                act_sequences = np.random.normal(
                    elite_mean,
                    elite_std,
                    size=(self.mpc_num_action_sequences, self.mpc_horizon, self.ac_dim)
                )
                rewards =  self.evaluate_action_sequences(obs,act_sequences)
                top_j_indices = np.argsort(rewards)[-self.cem_num_elites:]
                top_j_seq = act_sequences[top_j_indices,:,:]
                elite_mean, elite_std = np.mean(top_j_seq,dim=0),np.std(top_j_seq,dim=0)    
            
            fin_act_sequences = np.random.normal(
                elite_mean,elite_std,size=(self.mpc_num_action_sequences,self.mpc_horizon,self.ac_dim)
            )
            rewards = self.evaluate_action_sequences(obs,fin_act_sequences)
            best_index= np.argmax(rewards)
            return fin_act_sequences[best_index][0]


        else:
            raise ValueError(f"Invalid MPC strategy '{self.mpc_strategy}'")
