# ============================================================================
# EL2805 - Reinforcement Learning
# Computer Lab 1 - Problem 1
# Author: Ning Ding(Personal number 20030604-0254) and Longwei Xiao(Personal number 20020505-4836)
# ============================================================================


import numpy as np
import random
import matplotlib.pyplot as plt
from IPython import display
import time

class MazeWithKey:
    # 动作定义
    STAY       = 0
    MOVE_LEFT  = 1
    MOVE_RIGHT = 2
    MOVE_UP    = 3
    MOVE_DOWN  = 4

    # 奖励定义 (根据问题 h 的设定，可以调整)
    # 为了 Q-learning 收敛，通常到达目标给大正奖励，被抓给大负奖励，每一步给小负奖励
    STEP_REWARD = -1
    GOAL_REWARD = 100        # 到达 B 且有钥匙
    EATEN_REWARD = -100      # 被抓
    IMPOSSIBLE_REWARD = -10  # 撞墙

    def __init__(self, maze, minotaur_stay=False):
        self.maze = maze
        self.minotaur_stay = minotaur_stay # 问题 h 中怪物不能停留，设为 False
        
        # 定义关键位置
        # A: (0,0), B: (6,5) from maze.py main block
        # C: (0,7) 假设 C 在右上角 (Figure 2)
        self.start_pos_p = (0, 0)
        self.start_pos_m = (6, 5)
        self.key_pos     = (0, 7) 
        self.exit_pos    = (6, 5)

        self.actions = self.__actions()
        self.n_actions = len(self.actions)
        
        # 预计算状态映射表
        # 状态元组: ((pr, pc), (mr, mc), has_key) -> state_index
        self.map = {}
        self.states = {}
        self.n_states = 0
        self.__build_state_map()

    def __actions(self):
        actions = dict()
        actions[self.STAY]       = (0, 0)
        actions[self.MOVE_LEFT]  = (0, -1)
        actions[self.MOVE_RIGHT] = (0, 1)
        actions[self.MOVE_UP]    = (-1, 0)
        actions[self.MOVE_DOWN]  = (1, 0)
        return actions

    def __build_state_map(self):
        # 遍历所有可能的 P, M 位置以及 Key 状态
        idx = 0
        rows, cols = self.maze.shape
        
        for pr in range(rows):
            for pc in range(cols):
                if self.maze[pr, pc] == 1: continue # 玩家不能在墙里
                
                for mr in range(rows):
                    for mc in range(cols):
                        # 米诺陶可以穿墙，所以遍历所有格子
                        # Key 状态: 0 (无), 1 (有)
                        for k in [0, 1]:
                            state_tuple = ((pr, pc), (mr, mc), k)
                            self.map[state_tuple] = idx
                            self.states[idx] = state_tuple
                            idx += 1
        
        # 添加终止状态
        self.map['Eaten'] = idx
        self.states[idx] = 'Eaten'
        idx += 1
        
        self.map['Win'] = idx
        self.states[idx] = 'Win'
        idx += 1
        
        self.n_states = idx

    def reset(self):
        """重置环境，返回初始状态索引"""
        # 初始状态: P在A, M在B, Key=0
        state_tuple = (self.start_pos_p, self.start_pos_m, 0)
        return self.map[state_tuple]

    def step(self, state_idx, action):
        """
        执行一步动作
        返回: next_state_idx, reward, done
        """
        current_state = self.states[state_idx]
        
        # 1. 检查是否已经是终止状态
        if current_state == 'Eaten' or current_state == 'Win':
            return state_idx, 0, True

        (pr, pc), (mr, mc), has_key = current_state

        # 2. 玩家移动 (确定性)
        dy, dx = self.actions[action]
        npr, npc = pr + dy, pc + dx

        # 检查玩家是否撞墙或出界
        if npr < 0 or npr >= self.maze.shape[0] or \
           npc < 0 or npc >= self.maze.shape[1] or \
           self.maze[npr, npc] == 1:
            # 撞墙，停在原地
            npr, npc = pr, pc
            player_hit_wall = True
        else:
            player_hit_wall = False

        # 3. 米诺陶移动 (概率性)
        # 获取米诺陶所有可能的下一步位置
        possible_m_moves = []
        m_deltas = [(-1,0), (1,0), (0,-1), (0,1)] # 上下左右
        if self.minotaur_stay:
            m_deltas.append((0,0))

        rows, cols = self.maze.shape
        for dr, dc in m_deltas:
            nmr, nmc = mr + dr, mc + dc
            # 仅检查边界，不检查墙
            if 0 <= nmr < rows and 0 <= nmc < cols:
                possible_m_moves.append((nmr, nmc))

        # 决策逻辑: 35% 朝向玩家, 65% 随机
        # "朝向玩家" 定义为: 能够减小曼哈顿距离的移动
        # 注意：这里计算距离用的是玩家的新位置 (npr, npc) 还是旧位置？
        # 题目说 "You and the Minotaur move simultaneously"[cite: 38]. 
        # 但在 step 函数模拟中，我们通常计算完双方意图后判定结果。
        # 为了符合 MDP 定义，怪物是根据当前状态 (pr, pc) 还是预测玩家？
        # 通常简化为根据当前时刻 t 的距离判断。
        
        # 计算当前距离
        current_dist = abs(pr - mr) + abs(pc - mc)
        
        # 找出能减小距离的移动
        better_moves = []
        for (tr, tc) in possible_m_moves:
            dist = abs(pr - tr) + abs(pc - tc)
            if dist < current_dist:
                better_moves.append((tr, tc))
        
        # 掷骰子决定怪物行为
        is_smart_move = (random.random() < 0.35)
        
        final_m_pos = None
        
        if is_smart_move and len(better_moves) > 0:
            # 朝向玩家移动 (在最优移动中随机选一个)
            final_m_pos = random.choice(better_moves)
        else:
            # 随机移动 (在所有可行移动中随机选一个)
            final_m_pos = random.choice(possible_m_moves)

        nmr, nmc = final_m_pos

        # 4. 判定事件
        next_state_desc = None
        reward = 0
        done = False

        # 判定被吃: 玩家和怪物位置重合
        if (npr, npc) == (nmr, nmc):
            next_state_desc = 'Eaten'
            reward = self.EATEN_REWARD
            done = True
        
        # 判定拿钥匙
        elif (npr, npc) == self.key_pos and has_key == 0:
            # 拿到钥匙了！状态改变
            next_state_desc = ((npr, npc), (nmr, nmc), 1)
            reward = self.STEP_REWARD # 或者给个小奖励鼓励拿到钥匙？通常 STEP_REWARD 即可
            done = False
            
        # 判定胜利: 有钥匙 且 到达出口
        elif (npr, npc) == self.exit_pos and has_key == 1:
            next_state_desc = 'Win'
            reward = self.GOAL_REWARD
            done = True
            
        else:
            # 普通移动
            next_state_desc = ((npr, npc), (nmr, nmc), has_key)
            if player_hit_wall:
                reward = self.IMPOSSIBLE_REWARD
            else:
                reward = self.STEP_REWARD
            done = False

        return self.map[next_state_desc], reward, done