import copy
import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.optimizer import Adam
from paddle import to_tensor
from LPRB import device
import math

def build_net(layer_shape, activation, output_activation):
    '''build net with for loop'''
    layers = []
    for j in range(len(layer_shape)-1):
        act = activation if j < len(layer_shape)-2 else output_activation
        layers += [nn.Linear(layer_shape[j], layer_shape[j+1]), act()]
    return nn.Sequential(*layers)

class Q_Net(nn.Layer):
    def __init__(self, state_dim, action_dim, hid_shape):
        super(Q_Net, self).__init__()
        layers = [state_dim] + list(hid_shape) + [action_dim]
        self.Q = build_net(layers, nn.ReLU, nn.Identity)

    def forward(self, s):
        q = self.Q(s)
        return q

class DQN_Agent(object):
    def __init__(self, opt):
        self.q_net = Q_Net(opt.state_dim, opt.action_dim, (opt.net_width, opt.net_width))
        self.q_net_optimizer = paddle.optimizer.Adam(parameters=self.q_net.parameters(), learning_rate=opt.lr_init)
        self.q_target = copy.deepcopy(self.q_net)
        for p in self.q_target.parameters(): p.stop_gradient = True
        self.gamma = opt.gamma
        self.tau = 0.005
        self.batch_size = opt.batch_size
        self.exp_noise = opt.exp_noise_init
        self.action_dim = opt.action_dim
        self.DDQN = opt.DDQN

    def select_action(self, state, deterministic):
        with paddle.no_grad():
            state = to_tensor(state.reshape(1, -1), dtype='float32')
            if deterministic:
                a = paddle.argmax(self.q_net(state)).item()
                return a
            else:
                Q = self.q_net(state)
                if np.random.rand() < self.exp_noise:
                    a = np.random.randint(0, self.action_dim)
                    q_a = Q[0, a]
                else:
                    a = paddle.argmax(Q).item()
                    q_a = Q[0, a]
                return a, q_a

    def gather_2d(self, x, indices):
        b, _ = x.shape
        gathered = []
        for i in range(b):
            gathered.append(paddle.gather(x[i], indices[i]))
        return paddle.stack(gathered)

    # def custom_gather(self, x, index, dim):
    #     if dim < 0:
    #         dim += len(x.shape)
    #
    #     transposed_x = paddle.transpose(x, list(range(dim)) + list(range(dim + 1, len(x.shape))) + [dim])
    #     transposed_index = paddle.transpose(index, list(range(dim)) + list(range(dim + 1, len(index.shape))) + [dim])
    #
    #     flat_x = paddle.reshape(transposed_x, [-1, transposed_x.shape[-1]])
    #     flat_index = paddle.reshape(transposed_index, [-1])
    #
    #     gathered = paddle.gather(flat_x, flat_index, axis=0)
    #
    #     original_shape = index.shape
    #     result_shape = original_shape[:-1] + [gathered.shape[-1]]
    #     result = paddle.reshape(gathered, result_shape)
    #
    #     return result
    def custom_gather(self,tensorA, tensorB):
        """
        从给定的tensorA和tensorB生成tensorC。
        tensorA: 形状为[x, y]的张量。
        tensorB: 形状为[x, 1]的张量。
        返回: 形状为[x, 1]的张量，其中第i行数据为[A[i, B[i]]]
        """
        # 确保tensorB的形状为[x]
        tensorB = paddle.squeeze(tensorB, axis=1)

        # 获取行索引
        indices = paddle.arange(tensorA.shape[0])

        # 生成索引对
        gather_indices = paddle.stack([indices, tensorB], axis=1)

        # 从tensorA中提取tensorB所指示的列的值
        gathered_values = paddle.gather_nd(tensorA, gather_indices)

        # 将gathered_values重塑为[x, 1]
        tensorC = gathered_values.reshape([-1, 1])

        return tensorC
    def train(self, replay_buffer):
        s, a, r, s_next, dw, tr, ind, Normed_IS_weight = replay_buffer.sample(self.batch_size)

        # Compute the target Q value
        with paddle.no_grad():
            if self.DDQN:
                argmax_a = paddle.argmax(self.q_net(s_next), axis=1).unsqueeze(-1)
                #max_q_prime = self.custom_gather(self.q_target(s_next), argmax_a, 1)
                max_q_prime = self.custom_gather(self.q_target(s_next), argmax_a)
            else:
                max_q_prime = paddle.max(self.q_target(s_next), axis=1, keepdim=True)

            # Avoid impacts caused by reaching max episode steps
            Q_target = r + (~dw.astype('bool')).astype('float32') * self.gamma * max_q_prime

        # Get current Q estimates
        #current_Q = self.custom_gather(self.q_net(s), a, 1)
        current_Q = self.custom_gather(self.q_net(s), a)
        # BP
        q_loss = paddle.mean(paddle.square((~tr.astype('bool')).astype('float32') * Normed_IS_weight * (Q_target - current_Q)))
        self.q_net_optimizer.clear_grad()
        q_loss.backward()
        paddle.nn.utils.clip_grad_norm_(self.q_net.parameters(), 10.0)
        self.q_net_optimizer.step()

        # update priorities of the current batch
        with paddle.no_grad():
            batch_priorities = ((paddle.abs(Q_target - current_Q) + 0.01) ** replay_buffer.alpha).squeeze(-1)
            replay_buffer.priorities[ind] = batch_priorities

        # Update the frozen target models
        for param, target_param in zip(self.q_net.parameters(), self.q_target.parameters()):
            target_param.set_value(self.tau * param + (1 - self.tau) * target_param)
    # def train(self, replay_buffer):
    #     s, a, r, s_next, dw, tr, ind, Normed_IS_weight = replay_buffer.sample(self.batch_size)
    #     s = to_tensor(s, dtype='float32')
    #     a = to_tensor(a, dtype='int64') # 将a转换为1D
    #     r = to_tensor(r, dtype='float32')
    #     s_next = to_tensor(s_next, dtype='float32')
    #     dw = to_tensor(dw, dtype='float32')  # 将布尔类型转换为浮点数
    #     tr = to_tensor(tr, dtype='float32')  # 将布尔类型转换为浮点数
    #     Normed_IS_weight = to_tensor(Normed_IS_weight, dtype='float32')
    #
    #     with paddle.no_grad():
    #         if self.DDQN:
    #             argmax_a = paddle.argmax(self.q_net(s_next), axis=1, keepdim=True)
    #             max_q_prime = paddle.gather(self.q_target(s_next), axis=1, index=argmax_a)
    #         else:
    #             max_q_prime = paddle.max(self.q_target(s_next), axis=1, keepdim=True)
    #         Q_target = r + (1.0 - dw) * self.gamma * max_q_prime
    #
    #     current_Q = self.gather_2d(self.q_net(s), a)
    #     q_loss = paddle.square((1.0 - tr) * Normed_IS_weight * (Q_target - current_Q)).mean()  # 确保类型一致
    #     self.q_net_optimizer.clear_grad()
    #     q_loss.backward()
    #     nn.utils.clip_grad_norm_(self.q_net.parameters(), 10.0)
    #     self.q_net_optimizer.step()
    #
    #
    #
    #     with paddle.no_grad():
    #         # 确保Q_target和current_Q形状匹配
    #         assert Q_target.shape == current_Q.shape, "Q_target and current_Q must have the same shape"
    #
    #         batch_priorities = ((paddle.abs(Q_target - current_Q) + 0.01) ** replay_buffer.alpha).squeeze(-1)
    #
    #         # 确保ind不会超出replay_buffer.priorities的索引范围
    #         assert len(ind) == len(batch_priorities), "Number of indices must match number of batch priorities"
    #         assert max(ind) < len(replay_buffer.priorities), "Index out of range for replay buffer priorities"
    #
    #         # 更新replay_buffer的priorities
    #         replay_buffer.priorities[ind] = batch_priorities
    #
    #     for param, target_param in zip(self.q_net.parameters(), self.q_target.parameters()):
    #         target_param.set_value(self.tau * param + (1 - self.tau) * target_param)
    # def gather_2d(self, x, indices):
    #     b, _ = x.shape
    #     gathered = []
    #     for i in range(b):
    #         gathered.append(paddle.gather(x[i], indices[i]))
    #     return paddle.stack(gathered)
    def save(self, algo, EnvName, steps):
        paddle.save(self.q_net.state_dict(), "./model/{}_{}_{}.pdparams".format(algo, EnvName, steps))

    def load(self, algo, EnvName, steps):
        self.q_net.set_state_dict(paddle.load("./model/{}_{}_{}.pdparams".format(algo, EnvName, steps)))
        self.q_target.set_state_dict(paddle.load("./model/{}_{}_{}.pdparams".format(algo, EnvName, steps)))



