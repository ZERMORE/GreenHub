import paddle
import numpy as np

device = paddle.set_device("gpu:0" if paddle.is_compiled_with_cuda() else "cpu")


class LightPriorReplayBuffer():
    def __init__(self, opt):
        self.device = device

        self.ptr = 0
        self.size = 0

        self.state = paddle.zeros((opt.buffer_size, opt.state_dim), dtype='float32')
        self.action = paddle.zeros((opt.buffer_size, 1), dtype='int64')#, place=device
        self.reward = paddle.zeros((opt.buffer_size, 1), dtype='float32')
        self.dw = paddle.zeros((opt.buffer_size, 1), dtype='bool')
        self.tr = paddle.zeros((opt.buffer_size, 1), dtype='bool')
        self.priorities = paddle.zeros(opt.buffer_size, dtype='float32')
        self.buffer_size = opt.buffer_size

        self.alpha = opt.alpha
        self.beta = opt.beta_init
        self.replacement = opt.replacement

    def add(self, state, action, reward, dw, tr, priority):
        self.state[self.ptr] = paddle.to_tensor(state, dtype='float32')
        self.action[self.ptr] = paddle.to_tensor(action, dtype='int64')
        self.reward[self.ptr] = paddle.to_tensor(reward, dtype='float32')
        self.dw[self.ptr] = paddle.to_tensor(dw, dtype='bool')
        self.tr[self.ptr] = paddle.to_tensor(tr, dtype='bool')
        self.priorities[self.ptr] = paddle.to_tensor(priority, dtype='float32')

        self.ptr = (self.ptr + 1) % self.buffer_size
        self.size = min(self.size + 1, self.buffer_size)

    def sample(self, batch_size):
        Prob_paddle = self.priorities[0: self.size - 1].clone()
        if self.ptr < self.size:
            Prob_paddle[self.ptr - 1] = 0

        ind = paddle.multinomial(Prob_paddle, num_samples=batch_size, replacement=self.replacement)

        IS_weight = (self.size * Prob_paddle[ind]) ** (-self.beta)
        Normed_IS_weight = (IS_weight / IS_weight.max()).unsqueeze(-1)

        return (self.state[ind], self.action[ind], self.reward[ind],
                self.state[ind + 1], self.dw[ind], self.tr[ind], ind, Normed_IS_weight)
