import torch
import torch.nn as nn
import numpy as np

from itertools import product

seed = 2

#넘파이 랜덤 시드 고정
np.random.seed(seed)
#파이토치 랜덤 시드 고정
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

class agent(nn.Module):
    # TRADING_CHARGE = 0.00015
    # TRADING_TEX = 0.0025
    # TRADING_CHARGE = 0.0
    # TRADING_TEX = 0.0

    ACTIONS = []
    NUM_ASSETS = 0
    NUM_ACTIONS = 0

    def __init__(self, environment,
                 qnet:nn.Module,
                 qnet_target:nn.Module,
                 cnet:nn.Module,
                 cnet_target:nn.Module,
                 cost:float, delta:float, lam:float, lr2:float,
                 lr:float, tau:float, K:int, alpha:float,
                 discount_factor:float,
                 min_trading_price:int,
                 max_trading_price:int):

        super().__init__()

        agent.ACTIONS = list(product(range(1, 6), repeat=K))
        agent.ACTIONS = torch.tensor(agent.ACTIONS).float().view(-1, K)
        cash_bias = torch.ones(size=(agent.ACTIONS.shape[0], 1)) * 4.5
        agent.ACTIONS = torch.cat([cash_bias, agent.ACTIONS], dim=-1)
        agent.ACTIONS = torch.softmax(agent.ACTIONS, dim=-1).numpy()
        agent.NUM_ASSETS = K
        agent.NUM_ACTIONS = agent.ACTIONS.shape[0]

        self.environment = environment
        self.min_trading_price = min_trading_price
        self.max_trading_price = max_trading_price

        self.qnet = qnet
        self.cnet = cnet
        self.qnet_target = qnet_target
        self.cnet_target = cnet_target
        self.alpha = alpha
        self.lr = lr
        self.lr2 = lr2
        self.tau = tau
        self.lam = lam
        self.delta = delta
        self.cost = cost
        self.K = K
        self.epsilon = 0.0
        self.discount_factor = discount_factor

        self.qnet_opt = torch.optim.Adam(params=self.qnet.parameters(), lr=lr)
        self.cnet_opt = torch.optim.Adam(params=self.cnet.parameters(), lr=lr)
        self.huber = nn.SmoothL1Loss()
        self.qnet.load_state_dict(self.qnet_target.state_dict())
        self.cnet.load_state_dict(self.cnet_target.state_dict())

        self.TRADING_CHARGE = cost
        self.TRADING_TEX = 0.0

        self.num_stocks = np.array([0] * self.K)
        self.portfolio = np.array([0] * (self.K + 1), dtype=float)
        self.portfolio_value = 0.0
        self.initial_balance = 0.0
        self.balance = 0.0
        self.profitloss = np.array([0] * self.K) #종목별 수익률

    def set_balance(self, balance):
        self.initial_balance = balance

    def reset(self):
        self.num_stocks = np.array([0] * self.K)
        self.portfolio = np.array([0] * (self.K + 1), dtype=float)
        self.portfolio[0] = 1
        self.balance = self.initial_balance
        self.portfolio_value = self.initial_balance
        self.profitloss = 0

    def get_action(self, state1, state2):
        with torch.no_grad():
            self.qnet.eval()
            self.q_value = self.qnet(state1, state2).cpu()
            self.qnet.train()

        prob = np.random.uniform(low=0.0, high=1.0, size=1)
        if prob < self.epsilon:
            index = np.random.choice(agent.NUM_ACTIONS)
            action = agent.ACTIONS[index].copy().reshape(-1, self.K+1)
            trading = (action[0] - self.portfolio)[1:]
            confidence = abs(trading)

        else:
            index = np.array(self.q_value.argmax(dim=-1))
            action = agent.ACTIONS[index].copy().reshape(-1, self.K+1)
            trading = (action[0] - self.portfolio)[1:]
            confidence = abs(trading)
        return index, action, trading, confidence

    def decide_trading_unit(self, confidence, price):
        trading_price = self.portfolio_value * confidence
        trading_unit = int(np.array(trading_price)/price)
        return trading_unit

    def validate_action(self, action, delta):
        m_action = action.copy()
        for i in range(action.shape[0]):
            if delta < action[i] <= 1:
                # 매수인 경우 적어도 1주를 살 수 있는지 확인
                if self.balance < self.environment.get_price()[i] * (1 + self.TRADING_CHARGE):
                    m_action[i] = 0.0 #Hold

            elif -1 <= action[i] < -delta:
                # 매도인 경우 주식 잔고가 있는지 확인
                if self.num_stocks[i] == 0:
                    m_action[i] = 0.0 #Hold
        return m_action

    def pi_operator(self, change_rate):
        pi_vector = np.zeros(len(change_rate)+1)
        pi_vector[0] = 1
        pi_vector[1:] = change_rate + 1
        return pi_vector

    def get_portfolio_value(self, close_p1, close_p2, portfolio):
        close_p2 = np.array(close_p2)
        close_p1 = np.array(close_p1)
        change_rate = (close_p2 - close_p1)/close_p1
        pi_vector = self.pi_operator(change_rate)
        portfolio_value = np.dot(self.portfolio_value * portfolio, pi_vector)
        return portfolio_value

    def get_portfolio(self, close_p1, close_p2):
        close_p2 = np.array(close_p2)
        close_p1 = np.array(close_p1)
        change_rate = (close_p2 - close_p1)/close_p1
        pi_vector = self.pi_operator(change_rate)
        portfolio = (self.portfolio * pi_vector)/(np.dot(self.portfolio, pi_vector))
        return portfolio

    def get_reward(self, pv, pv_static):
        # reward = (pv-pv_static)/pv_static
        reward = np.log(pv) - np.log(self.initial_balance)
        return reward

    def step(self, action, confidences):
        assert action.shape[0] == confidences.shape[0]
        assert 0 <= self.delta < 1

        fee = 0
        close_p1 = self.environment.get_price()
        m_action = self.validate_action(action, self.delta)
        self.portfolio_value_static_ = self.portfolio * self.portfolio_value

        #우선 전체적으로 종목별 매도 수행을 먼저한다.
        for i in range(action.shape[0]):
            p1_price = close_p1[i]
            # Sell
            if -1 <= m_action[i] < -self.delta:
                cost = self.TRADING_CHARGE + self.TRADING_TEX
                trading_unit = self.decide_trading_unit(confidences[i], p1_price)
                trading_unit = min(trading_unit, self.num_stocks[i])
                invest_amount = p1_price * trading_unit

                fee += invest_amount * cost
                self.num_stocks[i] -= trading_unit
                self.balance += invest_amount * (1-cost)
                self.portfolio[0] += invest_amount * (1-cost)/self.portfolio_value
                self.portfolio[i+1] -= invest_amount/self.portfolio_value
                m_action[i] = -invest_amount/self.portfolio_value


        #다음으로 종목별 매수 수행
        for i in range(action.shape[0]):
            p1_price = close_p1[i]

            if abs(m_action[i]) > 1.0:
                raise Exception("Action is out of bound")
            # Buy
            if self.delta < m_action[i] <= 1:
                cost = self.TRADING_CHARGE
                trading_unit = self.decide_trading_unit(confidences[i], p1_price)
                cal_balance = (self.balance - p1_price * trading_unit * (1+cost))

                #돈 부족 한 경우
                if cal_balance < 0:
                    trading_unit = min(
                        int(self.balance / (p1_price * (1+cost))),
                        int(self.max_trading_price / p1_price * (1+cost)))

                # 수수료 적용하여 총 매수 금액 산정
                invest_amount = p1_price * trading_unit
                fee += invest_amount * cost
                self.num_stocks[i] += trading_unit
                self.balance -= invest_amount * (1+cost)
                self.portfolio[0] -= invest_amount * (1+cost)/self.portfolio_value
                self.portfolio[i+1] += invest_amount/self.portfolio_value
                m_action[i] = invest_amount/self.portfolio_value

            elif -self.delta <= m_action[i] <= self.delta:
                m_action[i] = 0.0

        """
        거래로 인한 PV와 PF 변동 계산
        """
        self.portfolio_value -= fee
        self.portfolio = self.portfolio / np.sum(self.portfolio) #sum = 1

        """
        다음 Time step 으로 진행 함에 따라
        생기는 가격 변동에 의한 PV와 PF 계산
        """
        next_state1 = self.environment.observe()
        next_state2 = self.portfolio.reshape(1, -1)
        close_p2 = self.environment.get_price()

        self.change = (np.array(close_p2)-np.array(close_p1))/np.array(close_p1)
        self.portfolio = self.get_portfolio(close_p1=close_p1, close_p2=close_p2)
        self.portfolio_value = self.get_portfolio_value(close_p1=close_p1, close_p2=close_p2, portfolio=self.portfolio)
        self.portfolio_value_static = np.dot(self.portfolio_value_static_, self.pi_operator(self.change))
        self.profitloss = ((self.portfolio_value / self.initial_balance) - 1)*100

        reward = self.get_reward(self.portfolio_value, self.portfolio_value_static)
        # reward = reward*100

        if len(self.environment.chart_data)-1 <= self.environment.idx:
            done = 1
        else:
            done = 0
        return m_action, next_state1, next_state2, reward, done

    def update(self, state1, state2, indice, action, reward, cost, next_state1, next_state2, done):
        s1, s2, i, a, r, c, ns1, ns2 = state1, state2, indice, action, reward, cost, next_state1, next_state2

        with torch.no_grad():
            q_value = self.qnet_target(ns1, ns2)
            q_max, _ = q_value.max(dim=-1)
            q_max = q_max.view(-1, 1)
            target = r + self.discount_factor * q_max * (1-done)

            c_value = self.cnet_target(ns1, ns2)
            c_min, _ = c_value.min(dim=-1)
            c_min = c_min.view(-1, 1)
            target_c = c + self.discount_factor * c_min * (1-done)

        q = self.qnet(s1, s2).gather(1, i)
        self.q_loss = self.huber(q, target - self.lam * c_min)
        self.qnet_opt.zero_grad()
        self.q_loss.backward()
        self.qnet_opt.step()
        self.q_ = q

        c = self.cnet(s1, s2).gather(1, i)
        self.c_loss = self.huber(c, target_c)
        self.cnet_opt.zero_grad()
        self.c_loss.backward()
        self.cnet_opt.step()
        self.c_ = c

        grad_lam = -(torch.mean(c.detach(), dim=0) - self.alpha)
        self.grad_lam = grad_lam
        self.lam -= self.lr2 * grad_lam
        self.lam = self.lam if self.lam >= 0 else 0

    def soft_target_update(self, params, target_params):
        for param, target_param in zip(params, target_params):
            target_param.data.copy_(self.tau*param.data + (1-self.tau)*target_param.data)

    def hard_target_update(self):
        self.qnet_target.load_state_dict(self.qnet.state_dict())
        self.cnet_target.load_state_dict(self.cnet.state_dict())


