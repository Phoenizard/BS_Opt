import torch
from modules.BS_MixG_model import C_MixG_Torch
from modules.BS_Theoretical_Model import BS_Theoretical_Value, add_noise_to_option_values
from utils import loss_torch, constraint_loss
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

class Args:
    def __init__(self, n, r, tau, d):
        self.n = n  # Example: 25
        self.r = r  # Example: 0.045
        self.tau = tau  # Example: 30 / 365
        self.d = d

class Model(torch.nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        # self.pi = torch.nn.Parameter(torch.ones(args.n + 1) / (args.n + 1), requires_grad=True)
        self.pi_raw = torch.nn.Parameter(torch.ones(args.n + 1), requires_grad=True)
        # mu 0, 1正态初始化
        self.mu = torch.nn.Parameter(torch.FloatTensor(n + 1).uniform_(7.107, 7.265), requires_grad=True)
        self.sigma =torch.nn.Parameter(torch.tensor(0.07), requires_grad=True)
        self.r = torch.tensor(args.r)
        self.tau = args.tau
        self.d = torch.tensor(args.d)
    
    @property
    def pi(self):
        # 将 pi_raw 转换为满足约束的 pi，使用 softmax 确保 pi 大于0且总和为1
        return torch.softmax(self.pi_raw, dim=0)
    
    def constraint(self, X):
        exp_mu = torch.exp(self.mu)
        weighted_exp_mu = torch.matmul(exp_mu, self.pi)
        sigma_term = torch.exp(self.sigma ** 2 / 2)
        left_term = weighted_exp_mu * sigma_term
        right_term = torch.mean(torch.exp((self.r - self.d) * self.tau) * X)
        return left_term, right_term

    def forward(self, X):
        return C_MixG_Torch(X, self.r, self.tau, self.sigma, self.mu, self.pi)

# 定义数据生成函数
def data_generate(X, ST, T, r, d, sigma_1000, sigma_1700):
    BS_Value = BS_Theoretical_Value(X, ST, T, r, d, sigma_1000, sigma_1700)
    C_obs = add_noise_to_option_values(BS_Value, X)
    # Convert to torch.Tensor
    C_obs = torch.tensor(C_obs, dtype=torch.float32)
    return C_obs

if __name__ == '__main__':
    r = 0.045
    d = 0.025
    tau = 30 / 365
    sigma = 0.07
    n = 25
    ST = 1365
    T = 30
    sigma_1000 = 0.2
    sigma_1700 = 0.4
    args = Args(n=n, r=r, tau=tau, d=d)
    X = torch.linspace(1000, 1700, n)
    C_obs = data_generate(X, ST, T, r, d, sigma_1000, sigma_1700)
    model = Model(args)
    lr = 1e-6
    losses = []
    for epoch in tqdm(range(30)):
        loss = loss_torch(model(X), C_obs)
        left_term, right_term = model.constraint(X)
        cons_loss = constraint_loss(left_term, right_term, weight=5)
        total_loss = loss + cons_loss
        total_loss.backward()
        losses.append(loss.detach().numpy())

        with torch.no_grad():
            model.mu -= lr * model.mu.grad
            model.pi_raw -= lr * model.pi_raw.grad
            model.sigma -= lr * model.sigma.grad
            model.mu.grad.zero_()
            model.pi_raw.grad.zero_()
            model.sigma.grad.zero_()
    print("Training Finished")
    # 查看约束是否满足
    left_term, right_term = model.constraint(X)
    print("Left term:", left_term)
    print("Right term:", right_term)
    # 查看pi参数
    print("Pi:", model.pi.data, model.pi.data.sum())
    print("Sigma:", model.sigma.data)
    print("Mu:", model.mu.data)
    print("Loss", losses[-1])
    # 画出 loss 曲线
    plt.plot(losses)
    plt.show()
    # 画出模型的拟合结果
    plt.plot(X, C_obs, label="True")
    plt.plot(X, model(X).detach().numpy(), label="Predicted")
    plt.legend()
    plt.show()
