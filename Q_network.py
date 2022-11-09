import numpy as np
import torch
import torch.nn as nn

seed = 2
#넘파이 랜덤 시드 고정
np.random.seed(seed)
#파이토치 랜덤 시드 고정
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


# class Score(nn.Module):
#     def __init__(self, state1_dim=5, output_dim=1):
#         super().__init__()
#
#         self.state1_dim = state1_dim
#         self.output_dim = output_dim
#
#         self.layer1 = nn.Linear(state1_dim, 128)
#         self.layer2 = nn.Linear(128, 64)
#         self.layer3 = nn.Linear(64, output_dim)
#
#         self.hidden_act = nn.ReLU()
#         self.out_act = nn.Identity()
#
#         nn.init.kaiming_normal_(self.layer1.weight, nonlinearity="relu")
#         nn.init.kaiming_normal_(self.layer2.weight, nonlinearity="relu")
#         nn.init.kaiming_normal_(self.layer3.weight, nonlinearity="relu")
#
#     def forward(self, s1):
#         x = self.layer1(s1)
#         x = self.hidden_act(x)
#         x = self.layer2(x)
#         x = self.hidden_act(x)
#         x = self.layer3(x)
#         x = self.out_act(x)
#         return x

class Score(nn.Module):
    def __init__(self, state1_dim=5, output_dim=1):
        super().__init__()

        self.layer1 = nn.Linear(state1_dim, output_dim)
        nn.init.constant_(self.layer1.weight, 0.01)

    def forward(self, s1):
        x = self.layer1(s1)
        return x


class Qnet(nn.Module):
    def __init__(self, score_net, K):
        super().__init__()
        self.score_net = score_net
        self.K = K

        # output_dim * K + state2_dim
        self.layer1 = nn.Linear(1*K + 1+K, 128)
        self.layer2 = nn.Linear(128, 64)
        self.layer3 = nn.Linear(64, 5**K)
        self.hidden_act = nn.ReLU()

        nn.init.constant_(self.layer1.weight, 0.01)
        nn.init.constant_(self.layer2.weight, 0.01)
        nn.init.constant_(self.layer3.weight, 0.01)

    def forward(self, s1_tensor, portfolio):

        for i in range(self.K):
            globals()[f"x{i+1}"] = self.score_net(s1_tensor[:,i,:])

        for k in range(self.K):
            x_list = list() if k == 0 else x_list
            x_list.append(globals()[f"x{k+1}"])

        # header
        x = torch.cat(x_list + [portfolio], dim=-1)
        x = self.layer1(x)
        x = self.hidden_act(x)
        x = self.layer2(x)
        x = self.hidden_act(x)
        q = self.layer3(x)
        return q


if __name__ == "__main__":
    s1_tensor = torch.rand(size=(10, 3, 5))
    portfolio = torch.rand(size=(10, 4))

    score_net = Score()
    qnet = Qnet(score_net, K=3)
    cnet = Qnet(score_net, K=3)
    q_value = qnet(s1_tensor, portfolio)
    c_value = cnet(s1_tensor, portfolio)
    print(q_value.shape)
    print(c_value.shape)