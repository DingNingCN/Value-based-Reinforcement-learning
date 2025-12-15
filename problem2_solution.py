# ============================================================================
# EL2805 - Reinforcement Learning
# Computer Lab 1 - Problem 2
# Author: Ning Ding(Personal number 20030604-0254) and Longwei Xiao(Personal number 20020505-4836)
# ============================================================================

import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
import itertools
import pickle


class FourierBasis:
    def __init__(self, state_dim, order):
        """
        初始化傅里叶基
        :param state_dim: 状态维度 (MountainCar 为 2)
        :param order: 傅里叶级数的阶数 p (建议 2 或 3)
        """
        self.order = order
        self.state_dim = state_dim
        self.eta = np.array(list(itertools.product(range(order + 1), repeat=state_dim)))
        self.n_features = self.eta.shape[0]

        norms = np.linalg.norm(self.eta, axis=1)
        norms[norms == 0] = 1.0
        self.alpha_scale = 1.0 / norms

    def get_features(self, state):
        """
        计算状态 s 的特征向量 phi(s)
        :param state: 归一化后的状态 [0, 1]^d
        :return: 特征向量 (n_features, )
        """
        # phi_i(s) = cos(pi * eta_i . s)
        return np.cos(np.pi * np.dot(self.eta, state))

def scale_state_variables(s, low, high):
    ''' 将状态 s 归一化到 [0,1]^2 '''
    return (s - low) / (high - low)

def get_q_value(w, features, action):
    ''' 计算 Q(s,a) = w_a^T . phi(s) '''
    return np.dot(w[action], features)

def choose_action(w, features, n_actions, epsilon):
    ''' Epsilon-Greedy 动作选择 '''
    if np.random.rand() < epsilon:
        return np.random.randint(n_actions)
    else:
        # 计算所有动作的 Q 值
        q_values = np.dot(w, features)
        # 随机打破平局
        max_q = np.max(q_values)
        actions = np.where(q_values == max_q)[0]
        return np.random.choice(actions)

if __name__ == "__main__":
    env = gym.make('MountainCar-v0')
    k = env.action_space.n
    low, high = env.observation_space.low, env.observation_space.high

    N_EPISODES = 1000       
    ORDER = 2              
    ALPHA = 0.001          
    LAMBDA = 0.9           
    GAMMA = 0.99           
    EPSILON = 0.0          

    basis = FourierBasis(state_dim=2, order=ORDER)
    
    W = np.zeros((k, basis.n_features))

    print(f"Features dimension: {basis.n_features}")
    print(f"Starting training for {N_EPISODES} episodes...")

    episode_reward_list = []

    for i in range(N_EPISODES):
        state_raw = env.reset()[0]
        state = scale_state_variables(state_raw, low, high)
        features = basis.get_features(state)
        
        action = choose_action(W, features, k, EPSILON)
        Z = np.zeros((k, basis.n_features))
        
        total_episode_reward = 0.
        done = False
        truncated = False
        
        while not (done or truncated):
            next_state_raw, reward, done, truncated, _ = env.step(action)
            next_state = scale_state_variables(next_state_raw, low, high)
            next_features = basis.get_features(next_state)
            
            next_action = choose_action(W, next_features, k, EPSILON)
            

            q_current = np.dot(W[action], features)
            q_next = np.dot(W[next_action], next_features)
            
            delta = reward + GAMMA * q_next - q_current
            if done: 
                delta = reward - q_current

            Z *= (GAMMA * LAMBDA)
            Z[action] += features
            
            Z = np.clip(Z, -5, 5)

            effective_alpha = ALPHA * basis.alpha_scale
            W += effective_alpha * delta * Z
            
            state = next_state
            features = next_features
            action = next_action
            
            total_episode_reward += reward

        episode_reward_list.append(total_episode_reward)
        
        if (i+1) % 10 == 0:
            print(f"Episode {i+1}: Reward = {total_episode_reward}")

    env.close()

    # ----------------------------
    save_data = {'W': W, 'N': basis.eta}
    with open('weights.pkl', 'wb') as f:
        pickle.dump(save_data, f)
    print("Weights saved to weights.pkl")

    plt.figure(figsize=(10, 5))
    plt.plot(episode_reward_list, label='Episode Reward')
    window = 10
    if len(episode_reward_list) >= window:
        avg_rewards = np.convolve(episode_reward_list, np.ones(window)/window, mode='valid')
        plt.plot(np.arange(window-1, len(episode_reward_list)), avg_rewards, label='Moving Average (10)')
    
    plt.xlabel('Episodes')
    plt.ylabel('Total Reward')
    plt.title('MountainCar Sarsa(lambda) with Fourier Basis')
    plt.legend()
    plt.grid(True)
    plt.show()