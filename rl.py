import torch
import numpy as np
from abc import abstractmethod
from scipy import special
from util import argmax_random_tiebreak

# RL ENVIRONMENT OBJECT ============================================================
class BinaryRewardEnv:
    def __init__(self, success_p, num_arms=100, rng_gen=None):
        self.num_arms = num_arms
        self.success_p = success_p

        if rng_gen is None:
            self.seed = 49248204
            self.rng_gen = np.random.default_rng(self.seed)
        else:
            self.rng_gen = rng_gen
        
    def generate_reward(self, arms, t):
        tmp_success_p = self.success_p[arms]
        reward = self.rng_gen.binomial(1, tmp_success_p)
        return reward

    def get_expected_reward(self, arm, t):
        return self.success_p[arm]


class BinaryRewardEnv_horizonDependent:
    def __init__(self, success_p, T, num_arms=100, rng_gen=None):
        self.num_arms = num_arms
        self.success_p = success_p

        if rng_gen is None:
            self.seed = 49248204
            self.rng_gen = np.random.default_rng(self.seed)
        else:
            self.rng_gen = rng_gen

        # generate table of potential outcomes
        potential_outcomes = []
        for arm in range(num_arms):
            PO_row = self.rng_gen.binomial(1, self.success_p[arm], size=(T))
            potential_outcomes.append(PO_row)

        self.potential_outcomes = np.vstack(potential_outcomes)        
        
    def generate_reward(self, arm, t):
        return self.potential_outcomes[arm][t]

    def get_expected_reward(self, arm, t):
        return self.generate_reward(arm, t)
        
# GET BANDIT ENVS ============================================================

def get_bandit_envs(num_arms, T, N_monte_carlo, success_p_all, seed=879437260, horizonDependent=False):
    # deterministic, and in the sense that if you re-run with different
    # values of N_monte_carlo = N1 < N2, then the first N1 bandit envs
    # will be the same for both runs
    all_bandit_envs = []
    rng_env = np.random.default_rng(seed)

    for i in range(N_monte_carlo):
        chosen_arms = rng_env.choice(np.arange(len(success_p_all)), num_arms)
        if horizonDependent:
            bandit_env_tmp = BinaryRewardEnv_horizonDependent(success_p_all[chosen_arms], T, num_arms, rng_env)
        else:
            bandit_env_tmp = BinaryRewardEnv(success_p_all[chosen_arms], num_arms, rng_env)
        all_bandit_envs.append( (bandit_env_tmp, chosen_arms) )
    return all_bandit_envs


# ABSTRACT BANDIT ALGORITHM CLASS ============================================================

class BanditAlgorithm:
    def __init__(self, num_arms=100, seed=None):
        self.num_arms = num_arms
        
        if seed is None:
            self.seed = 457323948
        else:
            self.seed = seed
        self.rng_gen = np.random.default_rng(self.seed)

    
    @abstractmethod
    def update_algorithm(self, arms, rewards):
        pass

    @abstractmethod        
    def sample_action(self):
        pass


# POSTERIOR HALLUCINATION BANDIT ALGORITHMS ============================================================


class GreedyPosteriorMeanAlg(BanditAlgorithm):
    def __init__(self, seq_model, Z_representation, num_arms):
        self.seq_model = seq_model
        self.Z_representation = Z_representation
        self.curr_state = seq_model.init_model_states(batch_size=num_arms)
        if Z_representation is not None:
            assert len(Z_representation) == num_arms

    def update_algorithm(self, arm, reward):
        arm_curr_state = self.curr_state[[arm]]
        arm_new_state = self.seq_model.update_state(arm_curr_state, 
                                                    torch.tensor([reward]))
        self.curr_state[arm] = arm_new_state[0]
    
    def sample_action(self, return_extra=False):
        if self.Z_representation is not None:
            state = torch.cat([self.Z_representation, self.curr_state], 1)
        else:
            state = self.curr_state
        p_pred = self.seq_model.top_layer(state)
        best_arm = torch.argmax(p_pred).item()
        if return_extra:
            return best_arm, p_pred
        return best_arm


class SampledGreedyPosteriorMeanAlg(BanditAlgorithm):
    def __init__(self, seq_model, Z_representation, num_arms, num_samples):
        self.seq_model = seq_model
        self.num_samples = num_samples
        self.Z_representation = Z_representation
        self.curr_state = seq_model.init_model_states(batch_size=num_arms)
        if Z_representation is not None:
            assert len(Z_representation) == num_arms

    def update_algorithm(self, arm, reward):
        arm_curr_state = self.curr_state[[arm]]
        arm_new_state = self.seq_model.update_state(arm_curr_state, 
                                                    torch.tensor([reward]))
        self.curr_state[arm] = arm_new_state[0]
    
    def sample_action(self, return_extra=False):
        if self.Z_representation is not None:
            state = torch.cat([self.Z_representation, self.curr_state], 1)
        else:
            state = self.curr_state
        p_pred = self.seq_model.top_layer(state)
        sampled_pred = torch.bernoulli(p_pred.flatten().unsqueeze(-1).repeat(1, self.num_samples)).mean(1)
        best_arm = torch.argmax(sampled_pred).item()
        if return_extra:
            return best_arm, sampled_pred
        return best_arm

    
class PosteriorHallucinationAlg(BanditAlgorithm):
    def __init__(self, seq_model, Z_representation, num_arms, 
                 num_imagined=100, seed=None, randomly_break_ties=False):
        super(PosteriorHallucinationAlg, self).__init__(num_arms, seed)
        self.seq_model = seq_model
        self.Z_representation = Z_representation
        self.curr_state = seq_model.init_model_states(batch_size=num_arms)
        self.num_imagined = num_imagined
        self.randomly_break_ties = randomly_break_ties
        if Z_representation is not None:
            assert len(Z_representation) == num_arms

    
    def update_algorithm(self, arm, reward):
        arm_curr_state = self.curr_state[[arm]]
        arm_new_state = self.seq_model.update_state(arm_curr_state, 
                                                    torch.tensor([reward]))
        self.curr_state[arm] = arm_new_state[0]

    
    def sample_action(self, return_extra=False):
        post_draws = self.seq_model.get_posterior_draws(Z_input=self.Z_representation, 
                                                     curr_state=self.curr_state, 
                                                     num_imagined=self.num_imagined,
                                                     num_repetitions=1)
        if self.randomly_break_ties: 
            best_arms = argmax_random_tiebreak(post_draws).item()
        else:
            best_arms = torch.argmax(post_draws).item()
        if return_extra:
            return best_arms, post_draws
        return best_arms


class PosteriorHallucinationAlg_horizonDependent(BanditAlgorithm):
    def __init__(self, seq_model, Z_representation, num_arms, T, seed=None):
        super(PosteriorHallucinationAlg_horizonDependent, self).__init__(num_arms, seed)
        self.seq_model = seq_model
        self.Z_representation = Z_representation
        self.curr_state = seq_model.init_model_states(batch_size=num_arms)
        self.T = T
        self.past_rewards = { k : [] for k in range(num_arms)}
        if Z_representation is not None:
            assert len(Z_representation) == num_arms

    def update_algorithm(self, arm, reward):
        arm_curr_state = self.curr_state[[arm]]
        arm_new_state = self.seq_model.update_state(arm_curr_state, 
                                                    torch.tensor([reward]))
        self.curr_state[arm] = arm_new_state[0]
        self.past_rewards[arm].append(reward)

    def sample_action(self, return_extra=False):
        all_rewards = [ torch.Tensor(self.past_rewards[k]) for k in range(self.num_arms) ]
        
        post_draws = self.seq_model.get_posterior_draws_horizonDependent(Z_input=self.Z_representation, 
                                                     curr_state=self.curr_state, T=self.T,
                                                     past_obs=all_rewards,
                                                     num_repetitions=1)
        if return_extra:
            return torch.argmax(post_draws).item(), post_draws
        return torch.argmax(post_draws).item()


# SQUARE-CB BANDIT ALGORITHMS ============================================================

class SquareCB(BanditAlgorithm):
    # follows https://arxiv.org/pdf/2002.04926.pdf
    # learning rate schedule follows https://arxiv.org/pdf/2010.03104
        # doesn't follow https://proceedings.mlr.press/v80/foster18a/foster18a.pdf
    
    def __init__(self, seq_model, Z_representation, num_arms, T, seed=None, hyparam_dict=None):
        super(SquareCB, self).__init__(num_arms, seed)
        self.mu = num_arms                          # they prove regret bound for this case
        self.seq_model = seq_model
        self.Z_representation = Z_representation
        self.curr_state = seq_model.init_model_states(batch_size=num_arms)
        self.T = T
        self.t = 0.0

        # learning rate gamma is set this way in https://arxiv.org/pdf/2010.03104
        self.gamma0 = hyparam_dict['gamma0']
        self.rho = hyparam_dict['rho']
        if Z_representation is not None:
            assert len(Z_representation) == num_arms
    
    def update_algorithm(self, arm, reward):
        arm_curr_state = self.curr_state[[arm]]
        arm_new_state = self.seq_model.update_state(arm_curr_state, 
                                                    torch.tensor([reward]))
        self.curr_state[arm] = arm_new_state[0]
        self.t += 1

    def sample_action(self):
        self.lr = self.gamma0 * self.t ** self.rho
        with torch.no_grad():
            if self.Z_representation is not None:
                state = torch.cat([self.Z_representation, self.curr_state], 1) 
            else:
                state = self.curr_state
            p_hats = self.seq_model.top_layer(state).squeeze().numpy()
            
            b = np.argmax(p_hats)
            phat_max = p_hats[b]
    
            new_phats = 1/ (self.mu + self.lr*(phat_max - p_hats) )
            not_max_ind = np.not_equal( np.arange(self.num_arms), b )
            new_phats[b] = 1 - np.dot(new_phats.squeeze(), not_max_ind)
    
            assert np.isclose( np.sum( new_phats ), 1 )

        return np.argmax( self.rng_gen.multinomial(1, new_phats) )


# NEURAL LINEAR BANDIT ALGORITHMS ============================================================

class NeuralLinearAlg(BanditAlgorithm):
    def __init__(self, marginal_preds, num_arms, hyparam_dict, seed=None):
        super(NeuralLinearAlg, self).__init__(num_arms, seed)
        self.prior_means = marginal_preds

        if 'sigma_squared' in hyparam_dict.keys():
            hyparam_dict['prior_var'] = hyparam_dict['sigma_squared']
            hyparam_dict['noise_var'] = hyparam_dict['s_squared']
        self.prior_vars = np.ones(num_arms) * hyparam_dict['prior_var']
        self.noise_var = hyparam_dict['noise_var']

        assert len(marginal_preds) == num_arms

        self.post_means = np.copy(self.prior_means)   # set to prior mean
        self.post_vars = np.copy(self.prior_vars)     # set to prior variance
        
        self.num_obs = np.zeros(num_arms)
        self.reward_sum = np.zeros(num_arms)
        
    def update_algorithm(self, arm, reward):
        self.num_obs[arm] += 1
        self.reward_sum[arm] += reward
        
        self.post_vars[arm] = 1 / ( 1/self.prior_vars[arm] + self.num_obs[arm] / self.noise_var )
        self.post_means[arm] = self.post_vars[arm] * (self.prior_means[arm] / self.prior_vars[arm] + self.reward_sum[arm] / self.noise_var)
    
    def sample_action(self):        
        samples = self.rng_gen.normal(self.post_means, self.post_vars ** 0.5)
        return samples.argmax().item()


# BETA BERNOULLI BANDIT ALGORITHMS ============================================================
from torch.distributions import Beta

class BetaBernoulliAlg(BanditAlgorithm):
    def __init__(self, num_arms, hyparam_dict, seed=None):
        super(BetaBernoulliAlg, self).__init__(num_arms, seed)
        self.ones = torch.zeros(num_arms)
        self.zeros = torch.zeros(num_arms)
        self.alpha = hyparam_dict['alpha']
        self.beta = hyparam_dict['beta']
    
    def update_algorithm(self, arm, reward):
        if not isinstance(reward, int):
            reward = reward.item()
        if reward == 1:
            self.ones[arm] += 1
        elif reward == 0:
            self.zeros[arm] += 1
        else:
            raise ValueError('Reward must be binary for beta bernoulli bandit')
    
    def sample_action(self):
        m = Beta(self.alpha + self.ones, self.beta + self.zeros)
        samples = m.sample()
        return samples.argmax()

class BetaBernoulliMixtureAlg(BanditAlgorithm):
    def __init__(self, num_arms, hyparam_dict, seed=None):
        super(BetaBernoulliMixtureAlg, self).__init__(num_arms, seed)
        self.ones = torch.zeros(num_arms)
        self.zeros = torch.zeros(num_arms)
        self.alpha1 = hyparam_dict['alpha1']
        self.alpha2 = hyparam_dict['alpha2']
        self.beta1 = hyparam_dict['beta1']
        self.beta2 = hyparam_dict['beta2']
        self.mixweight = hyparam_dict['mixweight']

    def update_algorithm(self, arm, reward):
        if not isinstance(reward, int):
            reward = reward.item()
        if reward == 1:
            self.ones[arm] += 1
        elif reward == 0:
            self.zeros[arm] += 1
        else:
            raise ValueError('Reward must be binary for beta bernoulli bandit')
        
    def sample_action(self):
        post_alpha1 = self.alpha1 + self.ones
        post_alpha2 = self.alpha2 + self.ones
        post_beta1 = self.beta1 + self.zeros
        post_beta2 = self.beta2 + self.zeros
        # Formula for marginal likelihood is here on page 24 of
        # https://www2.stat.duke.edu/~rcs46/modern_bayes17/lecturesModernBayes17/lecture-1/01-intro-to-Bayes.pdf
        clog1 = special.betaln(post_alpha1, post_beta1) - special.betaln(self.alpha1, self.beta1)
        clog2 = special.betaln(post_alpha2, post_beta2) - special.betaln(self.alpha2, self.beta2)
        bigconst = -clog1
        c1 = self.mixweight * torch.exp( clog1 + bigconst )
        c2 = (1-self.mixweight) * torch.exp( clog2 + bigconst )
        post_mixweight = c1 / (c1+c2)

        beta1 = torch.distributions.beta.Beta(post_alpha1, post_beta1)
        beta2 = torch.distributions.beta.Beta(post_alpha2, post_beta2)
        samples1 = beta1.sample()
        samples2 = beta2.sample()
        ind = torch.bernoulli(post_mixweight * torch.ones_like(post_mixweight))
        post_samples = samples1 * ind + samples2 * (1-ind)
        return post_samples.argmax()


class UCBAlg(BanditAlgorithm):
    # following https://papers.nips.cc/paper_files/paper/2011/file/e1d5be1c7f2f456670de3d53c7b54f4a-Paper.pdf

    def __init__(self, num_arms, delta=0.1, seed=None):
        super(UCBAlg, self).__init__(num_arms, seed)
        self.arm_means = np.zeros(num_arms)
        self.arm_counts = np.zeros(num_arms)
        self.delta = delta
        self.subgaussian_sigma = 0.5

    def update_algorithm(self, arm, reward):
        new_arm_sum = self.arm_means[arm] * self.arm_counts[arm] + reward
        self.arm_means[arm] = new_arm_sum / ( self.arm_counts[arm] + 1 )
        self.arm_counts[arm] += 1

    def compute_ucb(self):
        part1 = (1+self.arm_counts) / np.square(self.arm_counts)
        part2 = 1 + 2 * np.log( self.num_arms * np.sqrt(1+self.arm_counts) / self.delta )
        return self.subgaussian_sigma * np.sqrt( part1 * part2 )

    def sample_action(self):
        not_selected_ind = np.equal( self.arm_counts, 0 )*1
        if np.any(not_selected_ind):
            return np.argmax( not_selected_ind )

        # compute UCB
        arm_scores = self.arm_means + self.compute_ucb()
        return np.argmax( arm_scores )


def run_bandit(env, alg, T, num_round_robin=0, return_extra=False):
    all_rewards = []
    all_exp_rewards = []
    action_arms = []
    all_extras = []
    for t in range(T):
        extras = None
        if t < num_round_robin * env.num_arms:
            arm = t % env.num_arms
        else:
            if return_extra:
                arm, extras = alg.sample_action(return_extra=True)
            else:
                arm = alg.sample_action()
        #print("arm", arm)
        reward = env.generate_reward(arm, t)
        alg.update_algorithm(arm, reward)
        all_rewards.append(reward)

        # Get a less noisy estimate of the reward
        exp_reward = env.get_expected_reward(arm, t)
        all_exp_rewards.append(exp_reward)
        action_arms.append(arm)
        if extras is not None:
            all_extras.append(extras)
        
    res = { 'rewards':np.array(all_rewards), 
             'expected_rewards': np.array(all_exp_rewards), 
             'action_arms': np.array(action_arms) }
    if return_extra:
        res['extras'] = all_extras
    return res


############## DPT ##############

class DPTSequenceAlg(BanditAlgorithm):
    def __init__(self, seq_model, Z_representation, num_arms):
        super(DPTSequenceAlg, self).__init__()
        self.seq_model = seq_model
        self.Z_representation = Z_representation
        self.curr_state = seq_model.init_model_states(batch_size=num_arms)
        if Z_representation is not None:
            assert len(Z_representation) == num_arms

    
    def update_algorithm(self, arm, reward):
        arm_curr_state = self.curr_state[[arm]]
        arm_new_state = self.seq_model.update_state(arm_curr_state, 
                                                    torch.tensor([reward]))
        self.curr_state[arm] = arm_new_state[0]

    
    def sample_action(self, return_extra=False):
        post_draws = self.seq_model.dpt_sampling(self.curr_state,self.Z_representation) 
        arm = torch.multinomial(torch.softmax(post_draws.flatten(),0),1).item()
        if return_extra:
            return arm, post_draws
        return arm
