import gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

# 神经网络定义
class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class ValueNetwork(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# PPO算法实现
class PPO:
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99, epsilon=0.2, epochs=10, batch_size=64):
        self.policy = PolicyNetwork(state_dim, action_dim)
        self.value = ValueNetwork(state_dim)
        self.optimizer = optim.Adam([
            {'params': self.policy.parameters()},
            {'params': self.value.parameters()}
        ], lr=lr)
        self.gamma = gamma
        self.epsilon = epsilon
        self.epochs = epochs
        self.batch_size = batch_size

    def act(self, state):
        state = torch.FloatTensor(state)
        logits = self.policy(state)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob.item()

    def update(self, states, actions, old_log_probs, rewards, dones):
        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(np.array(actions))
        old_log_probs = torch.FloatTensor(np.array(old_log_probs))
        rewards = torch.FloatTensor(np.array(rewards))
        dones = torch.FloatTensor(np.array(dones))

        # 计算蒙特卡洛回报
        returns = []
        R = 0
        for r, done in zip(reversed(rewards), reversed(dones)):
            R = r + self.gamma * R * (1 - done)
            returns.insert(0, R)
        returns = torch.FloatTensor(returns)

        # 计算优势
        with torch.no_grad():
            values = self.value(states).squeeze()
            advantages = returns - values
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # 创建数据集
        dataset = torch.utils.data.TensorDataset(states, actions, old_log_probs, advantages, returns)
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # 更新参数
        for _ in range(self.epochs):
            for batch in loader:
                s, a, old_lp, adv, ret = batch

                # 计算新策略
                logits = self.policy(s)
                dist = torch.distributions.Categorical(logits=logits)
                new_log_probs = dist.log_prob(a)
                ratio = torch.exp(new_log_probs - old_lp)

                # 策略损失
                surr1 = ratio * adv
                surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * adv
                policy_loss = -torch.min(surr1, surr2).mean()

                # 价值损失
                value_loss = F.mse_loss(self.value(s).squeeze(), ret)

                # 熵奖励
                entropy = dist.entropy().mean()

                # 总损失
                loss = policy_loss + 0.5 * value_loss - 0.01 * entropy

                # 反向传播
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        return loss.item()

# 训练过程
env = gym.make('CartPole-v1', render_mode='human')  
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

ppo = PPO(state_dim, action_dim)
episode_rewards = []
loss_history = []

num_episodes = 300
batch_size = 24  # 每次更新收集的步数
states = []
actions = []
log_probs = []
rewards = []
dones = []
total_reward  = 0
for ep in range(num_episodes):
    # 兼容性reset
    reset_result = env.reset()
    state = reset_result[0] if isinstance(reset_result, tuple) else reset_result
    
    done = False
    while not done:
        action, log_prob = ppo.act(state)
        
        # 兼容性step
        step_result = env.step(action)
        if len(step_result) == 4:
            next_state, reward, done, info = step_result
            truncated = False
        else:
            next_state, reward, terminated, truncated, info = step_result
            done = terminated or truncated
        
        # 存储经验
        states.append(state)
        actions.append(action)
        log_probs.append(log_prob)
        rewards.append(reward)
        dones.append(done)

        state = next_state
        total_reward += reward

        if len(states) >= batch_size:
            loss = ppo.update(states, actions, log_probs, rewards, dones)
            loss_history.append(loss)
            # 清空缓冲区
            states = []
            actions = []
            states = []
            log_probs = []
            rewards = []
            dones = []

    episode_rewards.append(total_reward)
    print(f"Episode {ep+1}, Reward: {total_reward}")

    # 可视化训练过程
    if (ep+1) % 10 == 0:
        plt.figure(figsize=(12, 5))
        plt.subplot(121)
        plt.plot(episode_rewards)
        plt.title("Episode Rewards")
        plt.xlabel("Episode")
        
        plt.subplot(122)
        plt.plot(loss_history)
        plt.title("Training Loss")
        plt.xlabel("Update Step")
        plt.tight_layout()
        plt.pause(1)
        plt.close()

# 训练后演示
state = env.reset()
done = False
while not done:
    action, _ = ppo.act(state)
    state, _, done, _ = env.step(action)
    env.render()
env.close()