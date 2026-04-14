import numpy as np
import matplotlib.pyplot as plt
class BernoulliBandit:
    # 伯努利多臂老虎机,K为拉杆数
    def __init__(self,K):
        self.probs = np.random.uniform(size=K) #初始化每根拉杆的获奖概率
        self.best_idx = np.argmax(self.probs)  #获奖最大的拉杆
        self.best_prob = self.probs[self.best_idx] #最大获奖概率
        self.K=K
    def step(self,k):
        if np.random.rand() < self.probs[k]: #概率小于获奖概率认定为获奖
            return 1
        else:
            return 0
class Solver:
    '''多臂老虎机算法基本框架'''
    def __init__(self,bandit):
        self.bandit = bandit 
        self.counts = np.zeros(self.bandit.K) #每根杆拉的次数统计
        self.regret = 0.
        self.actions = []
        self.regrets = []
    def update_regret(self,k):
        '''用k号拉杆,更新累计的懊悔'''
        self.regret+=self.bandit.best_prob-self.bandit.probs[k] #计算regret
        self.regrets.append(self.regret)
    def run_one_step(self): #返回决定拉的杆
        return NotImplementedError
    def run(self,num_steps): #完成num_steps次的拉杆
        for i in range(num_steps):
            k = self.run_one_step()
            self.counts[k] += 1
            self.actions.append(k)
            self.update_regret(k)

class EpsilonGreedy(Solver):
    '''继承solver'''
    def __init__(self,bandit,epsilon=0.01,init_prob=1.0):
        super(EpsilonGreedy,self).__init__(bandit)
        self.epsilon = epsilon
        self.estimates = np.array([init_prob]*self.bandit.K) #为什么初始化每根拉杆的期望奖励为1(乐观初始化，让每一根杆子都有机会尝试)
    def run_one_step(self):
        if np.random.rand() < self.epsilon:
            k = np.random.randint(0,self.bandit.K)
        else:
            k = np.argmax(self.estimates)
        r = self.bandit.step(k)
        self.estimates[k] += 1.0/(self.counts[k]+1)*(r-self.estimates[k])
        return k
    
class DecayingEpsilonGreedy(Solver):
    '''继承solver'''
    def __init__(self,bandit,init_prob=1.0):
        super(DecayingEpsilonGreedy,self).__init__(bandit)
        self.estimates = np.array([init_prob]*self.bandit.K) #为什么初始化每根拉杆的期望为1
        self.total_count = 0
    def run_one_step(self):
        self.total_count +=1
        if np.random.rand() < 1/self.total_count:
            k = np.random.randint(0,self.bandit.K)
        else:
            k = np.argmax(self.estimates)
        r = self.bandit.step(k)
        self.estimates[k] += 1.0/(self.counts[k]+1)*(r-self.estimates[k])
        return k
    
class UCB(Solver):
    def __init__(self,bandit,coef,init_prob=1.0):
        super(UCB,self).__init__(bandit)
        self.total_count=0
        self.estimates = np.array([init_prob]*self.bandit.K)
        self.coef = coef
    def run_one_step(self):
        self.total_count+=1
        ucb=self.estimates + self.coef*np.sqrt(np.log(self.total_count)/(2*(self.counts+1)))
        k=np.argmax(ucb)
        r=self.bandit.step(k)
        self.estimates[k]+= 1./(self.counts[k]+1)*(r-self.estimates[k])
        return k

class ThompsonSampling(Solver):
    def __init__(self,bandit):
        super(ThompsonSampling,self).__init__(bandit)
        self._a =np.ones(self.bandit.K)
        self._b =np.ones(self.bandit.K)
    def run_one_step(self):
        samples = np.random.beta(self._a,self._b)
        k=np.argmax(samples)
        r=self.bandit.step(k)
        self._a[k]+=r
        self._b[k]+=(1-r)
        return k

def plot_result(solvers,solver_names):
    for idx,solver in enumerate(solvers):
        time_list =range(len(solver.regrets))
        plt.plot(time_list,solver.regrets,label=solver_names[idx])
    plt.xlabel('Time_steps')
    plt.ylabel('Cumulative_regrets')
    plt.title('%d -armed_bandit' % solvers[0].bandit.K)
    plt.legend()
    plt.show()
def main():
    np.random.seed(1)
    K=10
    bandit_10_arm = BernoulliBandit(K)
    print("随机生成一%d臂伯努利老虎机" %K)
    print("获奖概率最大的拉杆是%d,获奖概率是%.4f" %(bandit_10_arm.best_idx,bandit_10_arm.best_prob))

    # EpslionGreedy
    np.random.seed(1)
    epsilon_greedy_solver = EpsilonGreedy(bandit_10_arm,epsilon=0.01)
    epsilon_greedy_solver.run(5000)
    print('epsilon-greedy 累计懊悔为:',epsilon_greedy_solver.regret)
    plot_result([epsilon_greedy_solver],["EpsilonGreedy"])

    # DecayingEpsilonGreedy
    np.random.seed(1)
    decaying_epsilon_greedy_solver =DecayingEpsilonGreedy(bandit_10_arm)
    decaying_epsilon_greedy_solver.run(5000)
    print('epsilon衰减的累计懊悔值: ',decaying_epsilon_greedy_solver.regret)
    plot_result([decaying_epsilon_greedy_solver],["DecayingEpsilonGreedy"])

    #UCB
    np.random.seed(1)
    coef=1
    UCB_solver = UCB(bandit_10_arm,coef)
    UCB_solver.run(5000)
    print('UCB的累积懊悔为: ',UCB_solver.regret)
    plot_result([UCB_solver],["UCB"])

    #ThompsonSampling
    np.random.seed(1)
    thompson_sampling_solver =ThompsonSampling(bandit_10_arm)
    thompson_sampling_solver.run(5000)
    print('thompson_sampling的累计懊悔:',thompson_sampling_solver.regret)
    plot_result([thompson_sampling_solver],["ThompsonSampling"])
    
if __name__ == "__main__":
    main()
