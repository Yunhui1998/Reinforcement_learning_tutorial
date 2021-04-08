### 第三章 Policy Iteration 和Value Iteration 

#### Policy evalution

##### 定义

- Objective: Evaluate a given policy $\pi$ for a MDP ：为了评价MDP中一个策略的好坏
- Output: the value function under policy $v^{\pi}$  ：输出是价值函数
- Solution: iteration on Bellman expectation backup ：不断做贝尔曼更新
- Algorithm: Synchronous backup
  At each iteration $t+1$ update $v_{t+1}(s)$ from $v_{t}\left(s^{\prime}\right)$ for all states $s \in \mathcal{S}$ where $s^{\prime}$ is a successor state of $s$

$$
\begin{array}{c}
v_{t+1}(s)=\sum_{a \in \mathcal{A}} \pi(a \mid s)\left(R(s, a)+\gamma \sum_{s^{\prime} \in \mathcal{S}} P\left(s^{\prime} \mid s, a\right) v_{t}\left(s^{\prime}\right)\right) \\
\end{array}
$$

$$
\begin{array}{c}
\text { Convergence: } v_{1} \rightarrow v_{2} \rightarrow \ldots \rightarrow v^{\pi}
\end{array}
$$

Or if in the form of $\mathrm{MRP}<\mathcal{S}, \mathcal{P}^{\pi}, \mathcal{R}, \gamma>$
$$
v_{t+1}(s)=R^{\pi}(s)+\gamma P^{\pi}\left(s^{\prime} \mid s\right) v_{t}\left(s^{\prime}\right)
$$

##### 最优策略和最优价值函数

解决强化学习任务大概意味着要从长远的角度找到一个取得很大回报策略。 对于有限MDP，我们可以通过以下方式精确地定义一个最优策略。价值函数对策略进行部分排序。 **如果策略 π 所有状态的预期返回值大于或等于策略 π′ 的值， 则该策略 π 被定义为优于或等于策略 π′**。 换句话说，对所有 s∈S，π≥π′， 当且仅当 vπ(s)≥vπ′(s) 时，成立。 总是至少有一个策略优于或等于所有其他策略。这个策略称为 **最优策略**。 虽然可能有不止一个，我们用 π∗ 表示所有最优策略。 它们共享同样的状态值函数，称为 **最优状态价值函数**，表示为 v∗，并定义为

​                                                                                                         $v^{*}(s)=\max _{\pi} v^{\pi}(s)$

**可以说，当我们知道$v^{*}(s)$的时候，这个MDP问题就被解决了，**（因为我们就可以把所有动作遍历一遍，然后求出最大的q（s,a)。

最优的策略：

​                                                                                                        $\pi^{*}(s)=\underset{\pi}{\arg \max } v^{\pi}(s)$

最优动作价值函数：

​                                                                                                        $q_{*}(s, a) \doteq \max _{\pi} q_{\pi}(s, a)$

​                                                                                          $q_{*}(s, a)=\mathbb{E}\left[R_{t+1}+\gamma v_{*}\left(S_{t+1}\right) \mid S_{t}=s, A_{t}=a\right]$

当我们知道$q^{*}(s)$的时候，我们马上就得到了最优策略。

#### Policy Iteration和Value Iteration

##### Value Iteration

其实这篇[博客](https://zhuanlan.zhihu.com/p/33229439)里面给的例子我觉得很好帮助理解，下面的例子都是来自于这篇博客。假设我们有一个3 x 3的棋盘：

- 有一个单元格是超级玛丽，每回合可以往上、下、左、右四个方向移动

- 有一个单元格是宝藏，超级玛丽找到宝藏则游戏结束，目标是让超级玛丽以最快的速度找到宝藏

- 假设游戏开始时，宝藏的位置一定是(1, 2)

  ![image-20200828222444907](https://raw.githubusercontent.com/Yunhui1998/markdown_image/main/RL/image-20200828222444907.png)

  这个一个标准的马尔科夫决策过程(MDP)：

- **状态空间State**：超级玛丽当前的坐标

- **决策空间Action**: 上、下、左、右四个动作

- **Action对State的影响和回报 P(State', Reward | State, Action)**：本文认为该关系是已知的

- - 超级玛丽每移动一步，reward = -1
  - 超级玛丽得到宝箱，reward = 0并且游戏结束

- 利用value iteration解决这个问题：![image-20200828222610049](https://raw.githubusercontent.com/Yunhui1998/markdown_image/main/RL/image-20200828222610049.png)

- 结合上图可以非常简单的理解价值迭代：

- - **初始化**：所有state的价值V(s) = 0

  - 第一轮迭代：**对于每个state，逐一尝试上、下、左、右四个Action**

  - - 记录Action带来的Reward、以及新状态 V(s')
    - 选择最优的Action，更新V(s) = Reward + V(s') = -1 + 0 （**我觉得这里其实有点问题？应该宝箱旁边的三个空格的reward应该是0？**）
    - 第一轮结束后，所有状态都有V(s) = -1，即从当前位置出发走一步获得Reward=-1

  - 第二轮迭代：**对于每个state，逐一尝试上、下、左、右四个Action**

  - - 记录Action带来的Reward、以及新状态 V(s')

    - 选择最优的Action，更新V(s) = Reward + V(s')

    - - 对于宝箱周围的State，最优的Action是一步到达宝箱，V(s) = Reward + V(s') = -1 + 0
      - 对于其他State，所有的Action都是一样的，V(s) = Reward + V(s') = -1 + -1

    - 第二轮结束后，宝箱周围的State的价值保持不变 V(s) = -1，其他State的价值 V(s) = -2

  - 第三轮迭代：**对于每个state，逐一尝试上、下、左、右四个Action**

  - - 记录Action带来的Reward、以及新状态 V(s')

    - 选择最优的Action，更新V(s) = Reward + V(s')

    - - 对于宝箱周围的State，最优的Action是一步到达宝箱，V(s) = Reward + V(s') = -1 + 0
      - 对于宝箱两步距离的State，最优的Action是先一步到达宝箱周边的State，V(s) = Reward + V(s') = -1 + -1
      - 对于宝箱三步距离的State，所有Action都是一样的，V(s) = Reward + V(s') = -1 + -2

  - 第四轮迭代：**对于每个state，逐一尝试上、下、左、右四个Action**

  - - 记录Action带来的Reward、以及新状态 V(s')

  - 选择最优的Action，更新V(s) = Reward + V(s')

    - - 对于宝箱周围的State，最优的Action是一步到达宝箱，V(s) = Reward + V(s') = -1 + 0
    - 对于宝箱两步距离的State，最优的Action是先一步到达宝箱周边的State，V(s) = Reward + V(s') = -1 + -1
      - 对于宝箱三步距离的State，最优的Action是所有Action都是一样的，V(s) = Reward + V(s') = -1 + -2
    - **在第四轮迭代中，所有V(s)更新前后都没有任何变化，价值迭代已经找到了最优策略**

    **需要注意的是，对于Value Iteration来说，在每一轮迭代中，我们都对每一个状态尝试了所有他可能可以尝试的动作，并对这个状态的价值函数进行更新，直到价值函数不再发生变化。**

    上面的迭代过程实际上运用了贝尔曼方程 (Bellman Equation)，对每个位置的价值进行更新：

  ​                                                                                       $V_{*}(s)=\max _{a} \sum_{s^{\prime}, r} p\left(s^{\prime}, r \mid s, a\right)\left[r+\gamma V_{*}\left(s^{\prime}\right)\right]$                

  贝尔曼方程 (Bellman Equation)是非常好理解的 --- 状态s的价值V(s)由两部分组成：

  - 采取action=a后带来的reward

  - 采取action=a后到达的新的状态的价值V(s')

  Value Iteration的正式过程如下：

  - Iteration on the Bellman optimality backup：第一步就是使用贝尔曼方程去更新价值函数

- $$
  v_{i+1}(s) \leftarrow \max _{a \in \mathcal{A}} R(s, a)+\gamma \sum_{s^{\prime} \in \mathcal{S}} P\left(s^{\prime} \mid s, a\right) v_{i}\left(s^{\prime}\right)
  $$

- - To retrieve the optimal policy after the value iteration:第二步就是根据得到的收敛的价值函数去计算最优策略

- $$
  \pi^{*}(s) \leftarrow \underset{a}{\arg \max } R(s, a)+\gamma \sum_{s^{\prime} \in S} P\left(s^{\prime} \mid s, a\right) v_{e n d}\left(s^{\prime}\right)
  $$

  具体的伪码如下：

- ![image-20200828225649488](https://raw.githubusercontent.com/Yunhui1998/markdown_image/main/RL/image-20200828225649488.png)

##### Value Iteration的适用场景

- Action对State的影响和回报 P(State', Reward | State, Action)是已知的，然而绝大多数实际问题中P(State', Reward | State, Action)是未知的
- State和Action都是离散取值，无法应对Action或者State是连续取值的
- **State和Action都是低维度离散取值，因为计算复杂度是随着维度的升高而迅速变大**—— O(|State| x |Action| x |State|)

##### Policy Iteration

Policy iteration的大致过程为：

- **初始化：**随机选择一个策略作为初始值，比如说不管什么状态，一律朝下走，即P(Action = 朝下走 | State) = 1，P(Action = 其他Action | State) = 0

- **第一步 策略评估 (Policy Evaluation)**：根据当前的策略计算V(s)

- **第二步 策略提升 (Policy Improvement)：**计算当前状态的最好Action，更新策略，

  $\pi(s)=\operatorname{argmax}_{a} \sum_{s^{\prime}, r}\left(r+\gamma V\left(s^{\prime}\right)\right)$

- 不停的重复**策略评估**和**策略提升**，直到策略不再变化为止

  还是以上面的藏宝图的举例说明Policy Iteration的过程：

![image-20200828223950910](https://raw.githubusercontent.com/Yunhui1998/markdown_image/main/RL/image-20200828223950910.png)

- 初始化：无论超级玛丽在哪个位置，策略默认为向下走

- - **策略评估**：计算V(s)

  - - 如果宝藏恰好在正下方，则期望价值等于到达宝藏的距离(-2或者-1）
    - 如果宝藏不在正下方，则永远也不可能找到宝藏，期望价值为负无穷

  - **策略提升**：根据V(s)找到更好的策略

  - - 如果宝藏恰好在正下方，则策略已经最优，保持不变
    - 如果宝藏不在正下方，根据 $\operatorname{argmax}_{a} \sum_{s^{\prime}, r}\left(r+\gamma V\left(s^{\prime}\right)\right)$ 可以得出最优策略为横向移动一步

- 第一轮迭代：通过上一轮的策略提升，这一轮的策略变成了横向移动或者向下移动（如图所示)

- - **策略评估**：计算V(s)

  - - 如果宝藏恰好在正下方，则期望价值等于到达宝藏的距离(-2或者-1）
    - 如果宝藏不在正下方，当前策略会选择横向移动，期望价值为-3, -2, -1

  - **策略提升**：根据V(s)找到更好的策略

  - - 如果宝藏恰好在正下方，则策略已经最优，保持不变

    - 如果宝藏不在正下方，根据$\operatorname{argmax}_{a} \sum_{s^{\prime}, r}\left(r+\gamma V\left(s^{\prime}\right)\right)$ 可以得出当前策略已经最优，保持不变

      整个过程的[源码实现](https://github.com/whitepaper/RL-Zoo/blob/master/policy_iteration.ipynb)

Policy Iteration正式一点的表示如下，主要分为两步：

- Iterate through the two steps:

  - Policy evaluation: iteration on the Bellman expectation backup：第一步是策略评估
    $$
    v_{i}(s)=\sum_{a \in \mathcal{A}} \pi(a \mid s)\left(R(s, a)+\gamma \sum_{s^{\prime} \in \mathcal{S}} P\left(s^{\prime} \mid s, a\right) v_{i-1}\left(s^{\prime}\right)\right)
    $$

  - Policy improvement: greedy on action-value function $q$：第二步是策略提升
    $$
    \begin{array}{l}
    q_{\pi_{i}}(s, a)=R(s, a)+\gamma \sum_{s^{\prime} \in S} P\left(s^{\prime} \mid s, a\right) v_{\pi_{i}}\left(s^{\prime}\right) \\
    \pi_{i+1}(s)=\arg \max _{a} q_{\pi_{i}}(s, a)
    \end{array}
    $$



![image-20200828224825929](https://raw.githubusercontent.com/Yunhui1998/markdown_image/main/RL/image-20200828224825929.png)

##### Policy Iteration的适用场景

使用策略迭代求解MDP问题时，需要满足一下条件（跟价值迭代的应用条件非常类似）：

- Action对State的影响和回报 P(State', Reward | State, Action)是已知的，然而绝大多数实际问题中P(State', Reward | State, Action)是未知的
- State和Action都是离散取值，无法应对Action或者State是连续取值的
- State和Action都是低维度离散取值，因为计算复杂度是随着维度的升高而迅速变大的—— O(|State| x |Action| x |State|)

#####  Value Iteration和Policy Iteration的对比

- 在Value Iteration中

  - 第一步 Policy Eval：**一直迭代至收敛，获得准确的V(s)**
  - 第二步 Policy Improvement：**根据准确的V(s)，求解最好的Action**

  对比之下，在Policy Iteration中

  - 第一步 "Policy Eval"：**迭代只做一步，获得不太准确的V(s)**

  - 第二步 "Policy Improvement"：**根据不太准确的V(s)，求解最好的Action**

- Policy iteration includes: policy evaluation + policy improvement and the two are repeated iteratively until policy converges. Policy Iteration包含策略评估和策略提升两个步骤， 两个不走反复迭代直到策略收敛。

  Value iteration includes: finding optimal value function + one policy extraction. There is no repeat of the two because once the
  value function is optimal, then the policy out of it should also be optimal.    Value Iteration 包含找到最优的价值函数和一次策略提升两个步骤，两个步骤之间是没有重复的，因为一旦价值函数是最优的，根据其得出的策略也是最优的。

- Finding optimal value function can also be seen as a combination of policy improvement (due to max) and truncated policy evaluation
  (the reassignment of v(s) after just one sweep of all states regardless of convergence)

  找最优价值函数可以被看成是策略提升和缩略的策略评估的结合（价值函数的再赋值并没有保证收敛）

- **Policy iteration和Value iteration都是针对known MDP的，也就是你知道状态转移概率矩阵的情况，而且你还需要知道环境的奖励。**，但是现实生活中我们大量情况是不知道状态转移矩阵的

- In a lot of real-world problems, MDP model is either unknown or known by too big or too complex to use：
  A Atari Game, Game of Go, Helicopter, Portfolio management, etc   ：现实世界中的问题中的MDP都是不可知或者太大以致于很难实现的。

- 值迭代的收敛原因在于贝尔曼最优方程（Bellman optimality equation）具有压缩映射的性质。因为巴拿赫不动点定理，值迭代收敛。

  而策略迭代收敛的原因在于每次更新策略以后，目标函数都变好了，也就是累积的reward变大了。这样就构造了一个单调序列。根据单调有界序列收敛定理，策略迭代收敛。这个其实没怎么理解，下次把值迭代和策略迭代的收敛性证明复习一下放到笔记里面。

- 还有一种策略评估和策略提升过程交互进行的想法就是广义策略迭代（generalized policy iteration ，GPI）

- 有一个stanford的[demo](https://cs.stanford.edu/people/karpathy/reinforcejs/gridworld_dp.html)可以帮助很好地理解 Value Iteration和Policy Iteration两个过程。

##### Policy Iteration 和Value Iteration的python实现

policy iteration 去解决 Frozenlake：

```python
# 作者：Yunhui
# 创建时间：2020/9/10 9:51
# IDE：PyCharm
import gym
import numpy as np
from matplotlib import pyplot as plt


def run_episode(env, policy, gamma=1.0, render=False):
    """
    这一个函数的作用就是从头到尾跑一遍策略，然后你就可以用来评估策略比如说策略多快就done了，获得了多少reward
    :param env: 环境
    :param policy: 策略
    :param gamma:  奖励折扣
    :param render: 是否仿真环境
    :return:  total_reward , step_idx :总的奖励以及最终走完的步数
    """
    obs = env.reset()
    total_reward = 0
    step_idx = 0
    while True:
        if render:
            env.render()
        obs, reward, done, _ = env.step(int(policy[obs]))
        total_reward += (gamma ** step_idx * reward)
        step_idx += 1
        if done:
            print("总共花了%d步走完" % step_idx)
            break
    return total_reward, step_idx


def evaluate_policy(env, policy, gamma=1.0, n=100):
    scores = np.array([run_episode(env, policy, gamma=1.0, render=False) for _ in range(n)])[
        ..., 0]  # 跑一百次算平均reward作为分数
    step_idexs = np.array([run_episode(env, policy, gamma=1.0, render=False) for _ in range(n)])[..., 1]  # 看看大概需要多少步数

    return scores, step_idexs


def computer_policy_v(env, policy, gamma):
    """
    在给定策略的状态下，计算每个状态的v值，这个主要是通过贝尔曼方程，当前状态的v值=sum（去状态s_的概率*（去状态s_得到的reward+前一次计算得出的状态s_的v值）
    当这一次与前一次计算的每个状态的v值的差的和小于一个数字时，我们就认为在策略policy下估计v值的这个行为已经完成
    问题1：贝尔曼方程是怎么实现这一点的
    问题2：如果去算v值的时候不从0开始进行迭代会不会快一点？？？
    :param env:
    :param policy:
    :param gamma:
    :return: v  返回的是一个shape为(env.env.nS）的ndarry
    """
    v = np.zeros(env.env.nS)
    eps = 1e-10
    while True:
        prev_v = np.copy(v)
        for s in range(env.env.nS):  # 对于这里的一次for循环来说，就给每一个状态算了一次v值
            policy_a = policy[s]
            v[s] = sum([p * (r + gamma * prev_v[s_]) for p, s_, r, _ in env.env.P[s][policy_a]])
            # P[s][a] == [(probability, nextstate, reward, done), ...]
            # 这里在给每个状态算v值的时候，用到了前一次对每个状态的v值prev_v,比如说在算第一块砖的v值时，
            # 第一块有一定概率去第二块，有一定概率去第五块，分别计算一些这两种情况下的r + gamma * prev_v[s_]
        if np.sum(np.abs(prev_v - v)) < eps:  # 当这一次与前一次计算的每个状态的v值的差的和小于一个数字时，我们就认为在策略policy下估计v值的这个行为已经完成
            break
    return v


def extract_policy(v, gamma):
    """
    这个函数是用来在知道v的前提下去提取最优策略，方法就是去把所有的q_sa都算一遍，然后对于每一个状态S，选择让q_sa值最大的那个a
    :param v:
    :param gamma:
    :return:
    """
    policy = np.zeros(env.env.nS)
    for s in range(env.env.nS):
        q_sa = np.zeros(env.env.nA)
        for a in range(env.env.nA):
            q_sa[a] = sum([p * (r + gamma * v[s_]) for p, s_, r, _ in env.env.P[s][a]])
        policy[s] = np.argmax(q_sa)  # 返回沿轴的最大值的索引
    return policy


def policy_iteration(env, gamma):
    """
    通过迭代的方法进行，主要分为两步，首先是根据当前策略进行v值的计算，第二部是根据计算得到的v值去更新策略，反复迭代，当两次策略没有变化的时候我们就认为策略收敛到最优策略
    问题1：为什么这方式一定可以保证收连到最优策略？
    问题2：两次策略相同就能说明策略已经收敛了？需不需要多比较几次？
    :param env:
    :param gamma: 折扣因子
    :return: 最优策略policy，这个策略其实就是一个shape为env.env.nS的nDarray，也就是在这里是每个状态只给出一个最优动作
    """
    policy = np.random.choice(env.env.nA, size=(env.env.nS))
    max_iteration = 1000
    for i in range(max_iteration):
        old_policy_v = computer_policy_v(env, policy, gamma)
        new_policy = extract_policy(old_policy_v, gamma)
        if np.all(policy == new_policy):
            print("Policy Iteration converged at step %i" % (i + 1))
            break
        policy = new_policy
    return policy


if __name__ == '__main__':
    # ，if __name__ == '__main__'的意思是：
    # 当.py文件被直接运行时，if __name__ == '__main__'之下的代码块将被运行；
    # 当.py文件以模块形式被导入时，if __name__ == '__main__'之下的代码块不被运行。
    env_name = 'FrozenLake-v0'
    env = gym.make(env_name)
    gamma = 1.0
    optimal_policy = policy_iteration(env, gamma)
    print("optimal policy:", optimal_policy)
    scores, steps = evaluate_policy(env, optimal_policy, gamma=1.0)
    print('The average reward of optimal policy is', np.mean(scores))
    print('The average step of optimal policy is', np.mean(steps))
    x = np.array(range(1, 101))
    plt.plot(x, steps)
    plt.xlabel("num")
    plt.ylabel("step")
    plt.title("how steps change")
    plt.show()

```

Value Iteration 去解决Frozenlake：

```python
# 作者：Yunhui
# 创建时间：2020/9/10 15:51
# IDE：PyCharm

import gym
import numpy as np
import matplotlib.pyplot as plt

gamma = 1.0


def value_iteration(env, gamma=1.0):
    """
    和policy iteration中的算当前policy下的v值不同，这里的value iteration是直接求出了最优的v值，方法就是先把状态空间中
    的所有的q_sa都算一遍，然后直接取一个状态下最大的那个q_sa作为v值
    :param env:
    :param gamma:
    :return:
    """
    v = np.zeros(env.env.nS)
    max_iterations = 10000
    eps = 1e-20
    for i in range(max_iterations):
        prev_v = np.copy(v)
        for s in range(env.env.nS):
            q_sa = [sum([p * (r + gamma * prev_v[s_]) for p, s_, r, _ in env.env.P[s][a]]) for a in range(env.env.nA)]
            v[s] = max(q_sa)
        if np.sum(np.fabs(prev_v - v)) < eps:
            print("Value function converged at iteration %d" % (i + 1))
            break
    return v


def extract_policy(optimal_v, gamma=1.0):
    """
    这个提取策略的过程其实和policy_iteration 是一样的，都是在你知道v[s]之后，去算q_sa,然后对q_sa取最大值的索引，这里试了一下不让
    最开始的策略都往左走而是一个随机的策略看看会不会带来差别
    :param optimal_v:
    :param gamma:
    :return:
    """
    policy = np.random.choice(env.env.nA, size=(env.env.nS))  # policy=np.zeros(env.env.ns)
    for s in range(env.env.nS):
        q_sa = np.zeros(env.env.nA)
        for a in range(env.env.nA):
            q_sa[a] = sum(p * (r + gamma * optimal_v[s_]) for p, s_, r, _ in env.env.P[s][a])
        policy[s] = np.argmax(q_sa)
    return policy


def run_episode(env, policy, gamma=1.0, render=False):
    """
    这个和policy iteration里面的是完全一样的
    :param env:
    :param policy:
    :param gamma:
    :param render:
    :return:
    """
    obs = env.reset()
    total_reward = 0
    step_idx = 0
    while True:
        if render:
            env.render()
        obs, reward, done, _ = env.step(int(policy[obs]))
        total_reward += (gamma ** step_idx * reward)
        step_idx += 1
        if done:
            print("总共花了%d步走完" % step_idx)
            break
    return total_reward, step_idx


def evaluate_policy(env, policy, gamma=1.0, n=100):
    """
    这里和policy iteration里面的也是完全一样的
    :param env:
    :param policy:
    :param gamma:
    :param n:
    :return:
    """
    scores = np.array([run_episode(env, policy, gamma=1.0, render=False) for _ in range(n)])[
        ..., 0]  # 跑一百次算平均reward作为分数
    step_idexs = np.array([run_episode(env, policy, gamma=1.0, render=False) for _ in range(n)])[..., 1]  # 看看大概需要多少步数

    return scores, step_idexs


if __name__ == '__main__':
    env_name = 'FrozenLake-v0'
    env = gym.make(env_name)
    optimal_v = value_iteration(env, gamma)
    optimal_policy = extract_policy(optimal_v, gamma)
    print(optimal_policy)
    scores, steps = evaluate_policy(env, optimal_policy, gamma=1.0)
    print('The average reward of optimal policy is', np.mean(scores))
    print('The average step of optimal policy is', np.mean(steps))
    x = np.array(range(1, 101))
    plt.plot(x, steps)
    plt.xlabel("num")
    plt.ylabel("step")
    plt.title("how steps change")
    plt.show()
```

完整代码见[github](https://github.com/Yunhui1998/How-do-I-learn-RL)。