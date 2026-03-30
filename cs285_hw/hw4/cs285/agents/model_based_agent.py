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
        # Preserve single-state calls as 1D outputs so downstream rollout code
        # does not accidentally accumulate an extra batch dimension.
        single_input = obs.ndim == 1
        obs = ptu.from_numpy(obs)
        acs = ptu.from_numpy(acs)
        if single_input:
            obs = obs[None]
            acs = acs[None]
        # TODO(student): get the model's predicted `next_obs`
        # HINT: make sure to *unnormalize* the NN outputs (observation deltas)
        # Same hints as `update` above, avoid nasty divide-by-zero errors when
        # normalizing inputs!
        input_states = torch.concat((obs, acs), dim=1)
        input_states = (input_states - self.obs_acs_mean) / (self.obs_acs_std)
        delta_pred = self.dynamics_models[i](input_states)
        pred_next_obs = obs + delta_pred * self.obs_delta_std + self.obs_delta_mean

        pred_next_obs = ptu.to_numpy(pred_next_obs)
        if single_input:
            pred_next_obs = pred_next_obs[0]
        return pred_next_obs

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

            flat_next_obs = next_obs.reshape(
                self.ensemble_size * self.mpc_num_action_sequences, self.ob_dim
            )
            # Each candidate action sequence is shared across all ensemble models,
            # so we repeat the same actions along the ensemble dimension before
            # flattening into the batch expected by env.get_reward.
            flat_acs = np.broadcast_to(acs, next_obs.shape[:2] + (self.ac_dim,)).reshape(
                self.ensemble_size * self.mpc_num_action_sequences, self.ac_dim
            )
            rewards, _ = self.env.get_reward(flat_next_obs, flat_acs)
            rewards = rewards.reshape(self.ensemble_size, self.mpc_num_action_sequences)
            assert rewards.shape == (self.ensemble_size, self.mpc_num_action_sequences)

            sum_of_rewards += rewards

            obs = next_obs

        # now we average over the ensemble dimension
        return sum_of_rewards.mean(axis=0)#(mpc_num_action_seq,)

    def get_action(self, obs: np.ndarray):
        """
        Choose the best action using model-predictive control.

        Args:
            obs: (ob_dim,)
        """
        # always start with uniformly random actions, be it random or cem strategy.
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
            # CEM keeps a factorized Gaussian over the full action sequence.
            # That means the mean/std must preserve `(horizon, ac_dim)` shape;
            # collapsing them to scalars destroys the per-step plan structure.
            elite_mean = None
            elite_std = None

            for i in range(self.cem_num_iters):
                if i == 0:
                    candidate_action_sequences = action_sequences
                else:
                    candidate_action_sequences = np.random.normal(
                        loc=elite_mean,
                        scale=elite_std,
                        size=(
                            self.mpc_num_action_sequences,
                            self.mpc_horizon,
                            self.ac_dim,
                        ),
                    )
                    candidate_action_sequences = np.clip(
                        candidate_action_sequences,
                        self.env.action_space.low,
                        self.env.action_space.high,
                    )

                rewards = self.evaluate_action_sequences(
                    obs, candidate_action_sequences
                )
                elite_indices = np.argsort(rewards)[-self.cem_num_elites :]
                elite_action_sequences = candidate_action_sequences[elite_indices]

                new_elite_mean = np.mean(elite_action_sequences, axis=0)
                new_elite_std = np.std(elite_action_sequences, axis=0)

                if elite_mean is None:
                    elite_mean = new_elite_mean
                    elite_std = new_elite_std
                else:
                    # Alpha smooths the distribution update so one noisy CEM
                    # iteration does not completely overwrite the previous plan.
                    elite_mean = (
                        self.cem_alpha * new_elite_mean
                        + (1 - self.cem_alpha) * elite_mean
                    )
                    elite_std = (
                        self.cem_alpha * new_elite_std
                        + (1 - self.cem_alpha) * elite_std
                    )

                # Keep a small floor on std so the search does not collapse too
                # early and get stuck re-sampling nearly identical sequences.
                elite_std = np.maximum(elite_std, 1e-6)

            return elite_mean[0]


        else:
            raise ValueError(f"Invalid MPC strategy '{self.mpc_strategy}'")
