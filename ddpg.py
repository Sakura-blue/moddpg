import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

###############################  DDPG  ####################################

class Actor(nn.Module):
    def __init__(self, s_dim):
        super().__init__()
        self.fc1 = nn.Linear(s_dim, 400)
        self.fc2 = nn.Linear(400, 300)
        self.fc3_v = nn.Linear(300, 300)
        self.fc3_angle = nn.Linear(300, 300)
        self.v_out = nn.Linear(300, 1)
        self.angle_out = nn.Linear(300, 1)
        self.relu6 = nn.ReLU6()

        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_normal_(self.fc1.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.fc2.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.fc3_v.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.fc3_angle.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.v_out.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.angle_out.weight, nonlinearity='relu')
        nn.init.constant_(self.fc1.bias, 0.001)
        nn.init.constant_(self.fc2.bias, 0.001)
        nn.init.constant_(self.fc3_v.bias, 0.001)
        nn.init.constant_(self.fc3_angle.bias, 0.001)
        nn.init.constant_(self.v_out.bias, 0.0)
        nn.init.constant_(self.angle_out.bias, 0.0)

    def forward(self, s):
        x = self.relu6(self.fc1(s))
        x = self.relu6(self.fc2(x))
        x_v = self.relu6(self.fc3_v(x))
        x_angle = self.relu6(self.fc3_angle(x))
        v = torch.sigmoid(self.v_out(x_v))
        angle = torch.tanh(self.angle_out(x_angle))
        return torch.cat([v, angle], dim=1)


class Critic(nn.Module):
    def __init__(self, s_dim, a_dim):
        super().__init__()
        self.fc1 = nn.Linear(s_dim + a_dim, 400)
        self.fc2 = nn.Linear(400, 300)
        self.q_out = nn.Linear(300, 1)
        self.relu6 = nn.ReLU6()

        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_normal_(self.fc1.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.fc2.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.q_out.weight, nonlinearity='relu')
        nn.init.constant_(self.fc1.bias, 0.01)
        nn.init.constant_(self.fc2.bias, 0.01)
        nn.init.constant_(self.q_out.bias, 0.01)

    def forward(self, s, a):
        x = torch.cat([s, a], dim=1)
        x = self.relu6(self.fc1(x))
        x = self.relu6(self.fc2(x))
        return self.q_out(x)


class AGENT(object):
    """
    DDPG class
    """

    def __init__(self, args, a_num, a_dim, s_dim, a_bound, is_train):
        self.memory = np.zeros((args.mem_size, s_dim * 2 + a_dim + 1), dtype=np.float32)
        self.pointer = 0
        self.mem_size = args.mem_size
        self.batch_size = args.batch_size
        self.replace_tau = args.replace_tau
        self.a_num = a_num
        self.a_dim, self.s_dim, self.a_bound = a_dim, s_dim, a_bound.high
        self.is_train = is_train
        self.gamma = args.gamma
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.actor = Actor(s_dim).to(self.device)
        self.critic = Critic(s_dim, a_dim).to(self.device)
        self.actor.train()
        self.critic.train()

        # 建立actor_target网络，并和actor参数一致，不能训练
        self.actor_target = Actor(s_dim).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_target.eval()

        # 建立critic_target网络，并和critic参数一致，不能训练
        self.critic_target = Critic(s_dim, a_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_target.eval()

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=args.lr_actor)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=args.lr_critic)

    def ema_update(self):
        """
        滑动平均更新
        """
        self.soft_replace()

    def soft_replace(self):
        with torch.no_grad():
            for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
                target_param.mul_(1 - self.replace_tau).add_(param, alpha=self.replace_tau)
            for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
                target_param.mul_(1 - self.replace_tau).add_(param, alpha=self.replace_tau)

    def choose_action(self, s):
        """
        Choose action
        :param s: state
        :return: act
        """
        s = np.array([s], dtype=np.float32)
        s_tensor = torch.as_tensor(s, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            a = self.actor(s_tensor)
        return a.cpu().numpy().squeeze()

    def learn(self):
        indices = np.random.choice(self.mem_size, size=self.batch_size)  # 随机BATCH_SIZE个随机数
        bt = self.memory[indices, :]  # 根据indices，选取数据bt，相当于随机
        bs = torch.as_tensor(bt[:, :self.s_dim], dtype=torch.float32, device=self.device)  # 从bt获得数据s
        ba = torch.as_tensor(bt[:, self.s_dim:self.s_dim + self.a_dim], dtype=torch.float32, device=self.device)  # 从bt获得数据a
        br = torch.as_tensor(bt[:, self.s_dim + self.a_dim:-self.s_dim], dtype=torch.float32, device=self.device)  # 从bt获得数据r
        bs_ = torch.as_tensor(bt[:, -self.s_dim:], dtype=torch.float32, device=self.device)  # 从bt获得数据s_
        br = br.view(-1, 1)

        # Critic：
        # Critic更新和DQN很像，不过target不是argmax了，是用critic_target计算出来的。
        # br + GAMMA * q_
        with torch.no_grad():
            a_ = self.actor_target(bs_)
            q_ = self.critic_target(bs_, a_)
            y = br + self.gamma * q_
        q = self.critic(bs, ba)
        td_error = F.mse_loss(q, y)
        self.critic_opt.zero_grad()
        td_error.backward()
        self.critic_opt.step()

        # Actor：
        # Actor的目标就是获取最多Q值的。
        a = self.actor(bs)
        q = self.critic(bs, a)
        a_loss = -torch.mean(q)  # 【敲黑板】：注意这里用负号，是梯度上升！也就是离目标会越来越远的，就是越来越大。
        self.actor_opt.zero_grad()
        a_loss.backward()
        self.actor_opt.step()

        # self.ema_update()
        self.soft_replace()
        return td_error.detach().cpu().item(), (-a_loss).detach().cpu().item()

    # 保存s，a，r，s_
    def store_transition(self, s, a, r, s_):
        """
        Store data in data buffer
        :param s: state
        :param a: act
        :param r: reward
        :param s_: next state
        :return: None
        """
        s = s.astype(np.float32)
        s_ = s_.astype(np.float32)
        # 把s, a, [r], s_横向堆叠
        transition = np.hstack((s, a, [r], s_))

        # pointer是记录了曾经有多少数据进来。
        # index是记录当前最新进来的数据位置。
        # 所以是一个循环，当MEMORY_CAPACITY满了以后，index就重新在最底开始了
        index = self.pointer % self.mem_size  # replace the old memory with new memory
        # 把transition，也就是s, a, [r], s_存进去。
        self.memory[index, :] = transition
        self.pointer += 1

    def save_ckpt(self, model_path, eps):
        """
        save trained weights
        :return: None
        """
        model_path = os.getcwd() + model_path
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        torch.save(self.actor.state_dict(), '{}actor_{}.hdf5'.format(model_path, str(eps)))
        torch.save(self.actor_target.state_dict(), '{}actor_target_{}.hdf5'.format(model_path, str(eps)))
        torch.save(self.critic.state_dict(), '{}critic_{}.hdf5'.format(model_path, str(eps)))
        torch.save(self.critic_target.state_dict(), '{}critic_target_{}.hdf5'.format(model_path, str(eps)))

    def load_ckpt(self, version, eps):
        """
        load trained weights
        :return: None
        """
        model_path = os.getcwd() + '/{}/{}/'.format(version, 'models')
        self.actor.load_state_dict(torch.load('{}actor_{}.hdf5'.format(model_path, str(eps)), map_location=self.device))
        self.actor_target.load_state_dict(torch.load('{}actor_target_{}.hdf5'.format(model_path, str(eps)), map_location=self.device))
        self.critic.load_state_dict(torch.load('{}critic_{}.hdf5'.format(model_path, str(eps)), map_location=self.device))
        self.critic_target.load_state_dict(torch.load('{}critic_target_{}.hdf5'.format(model_path, str(eps)), map_location=self.device))
