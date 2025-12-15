# ============================================================================
# EL2805 - Reinforcement Learning
# Computer Lab 1 - Problem 1
# Author: Ning Ding(Personal number 20030604-0254) and Longwei Xiao(Personal number 20020505-4836)
# ============================================================================


import numpy as np
import matplotlib.pyplot as plt
import copy
import random

from maze import Maze, dynamic_programming, value_iteration
from maze_key import MazeWithKey


def create_env_with_stay(original_env):

    env_stay = copy.deepcopy(original_env)
    move_func = getattr(original_env, '_Maze__move')
    
    S, A = env_stay.n_states, env_stay.n_actions
    P_new = np.zeros((S, S, A))
    
    for s in range(S):
        for a in range(A):
            next_states_desc = move_func(s, a)
            outcomes = list(next_states_desc) if isinstance(next_states_desc, list) else [next_states_desc]
            
            curr_state_desc = env_stay.states[s]
            if curr_state_desc != 'Eaten' and curr_state_desc != 'Win':
                ((pr, pc), (mr, mc)) = curr_state_desc
                
                first_outcome = outcomes[0]
                if first_outcome != 'Eaten' and first_outcome != 'Win':
                    ((p_new_r, p_new_c), _) = first_outcome
                    
                    if (p_new_r, p_new_c) == (mr, mc):
                        stay_outcome = 'Eaten'
                    else:
                        stay_outcome = ((p_new_r, p_new_c), (mr, mc))
                    outcomes.append(stay_outcome)
            
            prob = 1.0 / len(outcomes)
            for out in outcomes:
                s_prime = env_stay.map[out]
                P_new[s, s_prime, a] += prob
                
    env_stay.transition_probabilities = P_new
    return env_stay

# ============================================================================

def q_learning(env, num_episodes, epsilon, gamma=0.98):
    Q = np.zeros((env.n_states, env.n_actions))
    N = np.zeros((env.n_states, env.n_actions))
    V_history = []
    start_state_idx = env.map[((0,0), (6,5), 0)] 

    for i in range(num_episodes):
        state = env.reset()
        done = False
        while not done:
            # Epsilon-Greedy
            if random.random() < epsilon:
                action = random.randint(0, env.n_actions - 1)
            else:
                max_q = np.max(Q[state, :])
                actions = np.where(Q[state, :] == max_q)[0]
                action = random.choice(actions)
            
            next_state, reward, done = env.step(state, action)
            
            # Update
            N[state, action] += 1
            alpha = 1.0 / (N[state, action] ** (2/3))
            best_next = np.max(Q[next_state, :])
            Q[state, action] += alpha * (reward + gamma * best_next - Q[state, action])
            
            state = next_state
        
        V_history.append(np.max(Q[start_state_idx, :]))
    return Q, V_history

def sarsa(env, num_episodes, epsilon, gamma=0.98):
    Q = np.zeros((env.n_states, env.n_actions))
    N = np.zeros((env.n_states, env.n_actions))
    V_history = []
    start_state_idx = env.map[((0,0), (6,5), 0)]
    
    def choose_action(s):
        if random.random() < epsilon:
            return random.randint(0, env.n_actions - 1)
        max_q = np.max(Q[s, :])
        return random.choice(np.where(Q[s, :] == max_q)[0])

    for i in range(num_episodes):
        state = env.reset()
        action = choose_action(state)
        done = False
        while not done:
            next_state, reward, done = env.step(state, action)
            
            if done:
                next_action = None
                q_next = 0.0
            else:
                next_action = choose_action(next_state)
                q_next = Q[next_state, next_action]
            
            N[state, action] += 1
            alpha = 1.0 / (N[state, action] ** (2/3))
            
            # SARSA Update
            Q[state, action] += alpha * (reward + gamma * q_next - Q[state, action])
            
            state = next_state
            action = next_action
        
        V_history.append(np.max(Q[start_state_idx, :]))
    return Q, V_history

def evaluate_policy_win_rate(env, Q_table, num_games=5000):
    wins = 0
    for _ in range(num_games):
        state = env.reset()
        done = False
        steps = 0
        while not done:
            max_q = np.max(Q_table[state, :])
            actions = np.where(Q_table[state, :] == max_q)[0]
            action = random.choice(actions)
            state, _, done = env.step(state, action)
            steps += 1
            if steps > 5000: break
        
        if env.states[state] == 'Win':
            wins += 1
    return wins / num_games

# ============================================================================

def run_mandatory_part():
    print("\n=== MANDATORY PART (c - f) ===")
    
    maze_map = np.array([
        [0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0, 1, 1, 1],
        [0, 0, 1, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 1, 1, 0],
        [0, 0, 0, 0, 1, 2, 0, 0]
    ])
    
    env = Maze(maze_map)
    start_idx = env.map[((0,0), (6,5))]

    # Question (c) & (d)
    print("\n[Question c & d] Finite Horizon Analysis")
    env.rewards.fill(-0.0001)
    env.rewards[env.map['Win'], :] = 1.0
    env.rewards[env.map['Eaten'], :] = 0.0
    
    env_stay = create_env_with_stay(env)
    env_stay.rewards = env.rewards.copy()
    
    T_range = range(1, 31)
    probs_normal, probs_stay = [], []
    
    for T in T_range:
        V, _ = dynamic_programming(env, T)
        probs_normal.append(V[start_idx, 0])
        V_s, _ = dynamic_programming(env_stay, T)
        probs_stay.append(V_s[start_idx, 0])
        if T == 20: print(f"  T=20, Exit Prob: {V[start_idx, 0]:.4f}")

    plt.figure(figsize=(8, 5))
    plt.plot(T_range, probs_normal, label='Minotaur Moves')
    plt.plot(T_range, probs_stay, label='Minotaur Stays')
    plt.title('Question (d): Survival Probability')
    plt.legend(); plt.grid(True); plt.show()

    # Question (e) & (f)
    print("\n[Question e & f] Poisoned Scenario")
    gamma = 1.0 - (1.0 / 30)
    env.rewards.fill(0.0)
    env.rewards[env.map['Win'], :] = 1.0
    
    V_vi, policy_vi = value_iteration(env, gamma, 1e-6)
    print(f"  Theoretical Prob: {V_vi[start_idx] * (1-gamma):.4f}")

def run_bonus_part():
    print("\n=== BONUS PART (h - k) ===")
    
    maze_map = np.array([
        [0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0, 1, 1, 1],
        [0, 0, 1, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 1, 1, 0],
        [0, 0, 0, 0, 1, 2, 0, 0]
    ])
    
    env = MazeWithKey(maze_map, minotaur_stay=False)
    gamma = 0.98
    episodes = 50000

    # Question (i)
    print(f"\n[Question i] Q-Learning ({episodes} eps)")
    Q_q_02, V_q_02 = q_learning(env, episodes, 0.2, gamma)
    Q_q_01, V_q_01 = q_learning(env, episodes, 0.01, gamma)
    
    plt.figure(figsize=(8, 4))
    plt.plot(V_q_02, label='e=0.2')
    plt.plot(V_q_01, label='e=0.01')
    plt.title('Q-Learning Convergence')
    plt.legend(); plt.grid(True); plt.show()

    # Question (j)
    # Re-init environment to be safe
    env = MazeWithKey(maze_map, minotaur_stay=False)
    print(f"\n[Question j] SARSA ({episodes} eps)")
    Q_s_02, V_s_02 = sarsa(env, episodes, 0.2, gamma)
    Q_s_01, V_s_01 = sarsa(env, episodes, 0.1, gamma)
    
    plt.figure(figsize=(8, 4))
    plt.plot(V_s_02, label='e=0.2')
    plt.plot(V_s_01, label='e=0.1')
    plt.title('SARSA Convergence')
    plt.legend(); plt.grid(True); plt.show()

    # Question (k)
    print("\n[Question k] Evaluation")
    wr = evaluate_policy_win_rate(env, Q_q_02)
    print(f"  Q-Learning Win Rate: {wr:.2%}")
    wr = evaluate_policy_win_rate(env, Q_s_02)
    print(f"  SARSA Win Rate: {wr:.2%}")
    print(f"  Q-Learning V(s0): {np.max(Q_q_02[env.map[((0,0),(6,5),0)], :]):.2f}")
    print(f"  SARSA V(s0): {np.max(Q_s_02[env.map[((0,0),(6,5),0)], :]):.2f}")

if __name__ == "__main__":
    run_mandatory_part()
    run_bonus_part()