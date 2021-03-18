### 第四章- DP、MC、TD

#### 动态规划（DP）、蒙特卡罗（MC）、时间差分（TD）

##### Dynamic Programming（利用贝尔曼方程迭代）

![image-20200827193937569](https://raw.githubusercontent.com/Yunhui1998/markdown_image/main/RL/image-20200827193937569.png)

​       其实可以把MC、TD都理解成DP的一种近似，只不过降低了计算复杂度以及削弱了对环境模型完备性的假设。

- [动态规划的使用条件](https://sites.google.com/a/chaoskey.com/algorithm/03/03)：

  - 最优化原理：如果问题的最优解所包含的子问题的解也是最优的，就称该问题具有最优子结构，即满足最优化原理。
  - 无后效性：即某阶段状态一旦确定，就不受这个状态以后决策的影响。也就是说，某状态以后的过程不会影响以前的状态，只与当前状态有关。

- 有重叠子问题：即子问题之间是不独立的，一个子问题在下一阶段决策中可能被多次使用到。（**该性质并不是动态规划适用的必要条件，但是如果没有这条性质，动态规划算法同其他算法相比就不具备优势**）

- [动态规划的步骤](https://www.cnblogs.com/steven_oyj/archive/2010/05/22/1741374.html)

  - **划分阶段**：按照问题的时间或空间特征，把问题分为若干个阶段。在划分阶段时，注意划分后的阶段一定要是有序的或者是可排序的，否则问题就无法求解。

  - **确定状态和状态变量**：将问题发展到各个阶段时所处于的各种客观情况用不同的状态表示出来。当然，状态的选择要满足无后效性。

  - **确定决策并写出状态转移方程**：因为决策和状态转移有着天然的联系，状态转移就是根据上一阶段的状态和决策来导出本阶段的状态。所以如果确定了决策，状态转移方程也就可写出。但事实上常常是反过来做，根据相邻两个阶段的状态之间的关系来确定决策方法和状态转移方程。

  - **寻找边界条件**：给出的状态转移方程是一个递推式，需要一个递推的终止条件或边界条件。

    一般，只要解决问题的阶段、状态和状态转移决策确定了，就可以写出状态转移方程（包括边界条件）。

- 动态规划三要素

  - 问题的阶段 
  - 每个阶段的状态
  - 从前一个阶段转化到后一个阶段之间的递推关系

##### [异步的动态规划](https://zhuanlan.zhihu.com/p/30518290)：Asynchronous Dynamic Programming

在我们之前的算法中，我们每一次的迭代都会完全更新所有的，这样对于程序资源需求特别大。这样的做法叫做同步备份(synchronous backups)。异步备份的思想就是通过某种方式，使得每一次迭代不需要更新所有的，因为事实上，很多的也不需要被更新。异步备份有以下几种方案

1. **In-place 动态规划所做的改进，是直接去掉了原来的副本 $v_{k}$, 只保留最新的副本**(也就是说，在 一次更新过程中，存在着有些用的是 $v_{k},$ 有些用的是 $v_{k+1}$ )。具体而言，我们可以这样表示，对于所有的状态s：
   $$
   v(s) \leftarrow \max _{a \in A}\left(R_{s}^{a}+\gamma \sum_{s^{\prime} \in S} P_{s s^{\prime}}^{a} v\left(s^{\prime}\right)\right)
   $$

2. **加权扫描动态规划：Prioritized Sweeping Dynamic Programming**

Prioritized Sweeping 的思想是，根据某种方式，来确定每一个状态现在是否重要，**对于重要的状态进行更多的更新，对于不重要的状态更新的次数就比较少。**

- 更新顺序：可以使用priority queue 来有效的确定更新次序(按照优先权排队，每次把优先权最高的拿出来更新
- 权重设计规则：可以使用Bellman error 来确定优先权，这个公式是通过两次的value的差异来作为state的评估标准，**如果某个状态上次的value和这次的value相差不大，我们有理由认为他即将达到稳定，则更新他的价值就比较小**，反之则比较大。具体公式如下：

​                                                                                                            $$\max _{a \in A}\left(R_{s}^{a}+\gamma \sum_{s^{\prime} \in S} P_{s s^{\prime}}^{a} v\left(s^{\prime}\right)-v(s) \mid\right.$$

​        所以说，这个方法需要我们进行反向规划，因为我们需要知道当前状态之前的value是多少

3. **实时动态规划：Real-Time Dynamic Programming**

​       实时动态规划的思想是只有Agent现在关注的状态会被更新。与当前无关的状态暂时不被更新。

​        就比如我们在时间步t进入状态 $S_{t}$ ，进行动作 $A_{t}$，结果是得到反馈  $R_{t+1}$，那么我们要做的就是，仅仅更新 $S_{t}$ 的value function，公式如下：

​                                                                                                            $$v\left(S_{t}\right) \leftarrow \max _{a \in A}\left(R_{S_{t}}^{a}+\gamma \sum_{s^{\prime} \in S} P_{S_{t} s^{\prime}}^{a} v\left(s^{\prime}\right)\right) $$

4. **Full-Width Backups and Sample Backups**

​          Full-Width 和 Sample Backups的区别在于更新状态时考虑的后继状态的个数的区别，他和同步DP，异步DP思考的是两个维度的优化方向。

​          Full-Width Backups 介绍的是：当我们在考虑更新某一个state 的value function的时候，我们需要遍历这个state的所有可能的action，和每一个action所可能达到的后继state，这个遍历的开销非常大，对于每一次迭代，如果有m个action和n个state，则时间复杂度为 $O\left(m n^{2}\right)$，也就是说，遍历次数会随着n而指数增长，这在大型的DP问题中，代价是无法接受的，所以提出了sample backups。

​       sample backups 的思路是将state-to-state的过程进行采样，也就是说，我们以整个MDP$<S, A, R, S^{\prime}>$为单位，得到很多的样本，也就是说，对**于一个样本，一个state对应一个action，通过采样多个MDP过程，来估计当前的策略的优劣，而不是每个节点直接遍历所有的可能性**，我们可以用下图表示：

![img](https://raw.githubusercontent.com/Yunhui1998/markdown_image/main/RL/v2-90a0552f35411aa45017cbcd3361187d_720w.jpg)

​       这样做有如下的优点：

- Model-free：一个非常重要的好处就是，由于我们不再需要计算所有的action可到达的状态，就意味着我们不需要使用状态转换概率矩阵，也就是说，我们不再需要提前完全搞明白环境的变化模型，这便是一个model-free的算法！
- 假设我们采样的个数为a，那么我们一次迭代的时间复杂度就是 $O(a m n)$，随着state的增加，我们的时间复杂度仅仅是常数增长。一定程度避免了维度灾难。

##### Monte-Carlo evalution（采样的一种办法）

​        蒙特卡罗方法是一种基于采样的方法，也就是我们采样得到很多轨迹，然后根据采样得到的结果平均去算V（s）

- MC simulation: we can simply sample a lot of trajectories, compute the actual returns for all the trajectories, then average them
- To evaluate state $v(s)$

1. Every time-step $t$ that state s is visited in an episode,
          2. Increment counter $N(s) \leftarrow N(s)+1$
             3. Increment total return $S(s) \leftarrow S(s)+G_{t}$
             4. Value is estimated by mean return $v(s)=S(s) / N(s)$,**这里计算这个均值的时候，我们其实可以用Incremental Mean的方式，也就是新估计←旧估计+步长[目标−旧估计]**  $v\left(S_{t}\right) \leftarrow v\left(S_{t}\right)+\frac{1}{N\left(S_{t}\right)}\left(G_{t}-v\left(S_{t}\right)\right)$
                **By law of large numbers, $v(s) \rightarrow v^{\pi}(s)$ as $N(s) \rightarrow \infty$**

![image-20200827192441184](https://raw.githubusercontent.com/Yunhui1998/markdown_image/main/RL/image-20200827192441184.png)

##### Tempor-Difference learning

- TD methods learn directly from episodes of experience  TD方法从序列的经验里面进行学习

- TD is model-free: no knowledge of MDP transitions/rewards   没有状态概率转移矩阵

- TD learns from incomplete episodes, by bootstrapping   通过bootstrapping从不完全的轨迹学习

- Objective: learn $v_{\pi}$ online from experience under policy $\pi$ Simplest TD algorithm: $\operatorname{TD}(0)$   也就是往前走一步进行估计
  U Undate $v\left(S_{t}\right)$ toward estimated return $R_{t+1}+\gamma v\left(S_{t+1}\right)$
  $$
  v\left(S_{t}\right) \leftarrow v\left(S_{t}\right)+\alpha\left(R_{t+1}+\gamma v\left(S_{t+1}\right)-v\left(S_{t}\right)\right)
  $$
  $R_{t+1}+\gamma v\left(S_{t+1}\right)$ is called **TD target**
  $\delta_{t}=R_{t+1}+\gamma v\left(S_{t+1}\right)-v\left(S_{t}\right)$ is called the **TD error**
  Comparison: Incremental Monte-Carlo
  $$
  v\left(S_{t}\right) \leftarrow v\left(S_{t}\right)+\alpha\left(G_{i, t}-v\left(S_{t}\right)\right)
  $$

##### n-step TD

n-step TD像是控制n的大小在TD(0)和MC中找一个平衡

$\begin{array}{ll}n=1(T D) & G_{t}^{(1)}=R_{t+1}+\gamma v\left(S_{t+1}\right) \\ n=2 & G_{t}^{(2)}=R_{t+1}+\gamma R_{t+2}+\gamma^{2} v\left(S_{t+2}\right) \\ & \vdots \\ n=\infty(M C) & G_{t}^{\infty}=R_{t+1}+\gamma R_{t+2}+\ldots+\gamma^{T-t-1} R_{T}\end{array}$

<img src="https://raw.githubusercontent.com/Yunhui1998/markdown_image/main/RL/image-20200829165754594.png" alt="image-20200829165754594" style="zoom: 33%;" />

Thus the n-step return is defined as
$$
G_{t}^{n}=R_{t+1}+\gamma R_{t+2}+\ldots+\gamma^{n-1} R_{t+n}+\gamma^{n} v\left(S_{t+n}\right)
$$
n-step $\mathrm{TD}: v\left(S_{t}\right) \leftarrow v\left(S_{t}\right)+\alpha\left(G_{t}^{n}-v\left(S_{t}\right)\right)$

##### MC、DP以及TD算法的对比

Dynamic Programming $(\mathrm{DP})$ computes $v_{i}$ by **bootstrapping** the rest of the expected return by the value estimate $v_{i-1}$  Iteration on Bellman expectation backup:
$$
v_{i}(s) \leftarrow \sum_{a \in \mathcal{A}} \pi(a \mid s)\left(R(s, a)+\gamma \sum_{s^{\prime} \in \mathcal{S}} P\left(s^{\prime} \mid s, a\right) v_{i-1}\left(s^{\prime}\right)\right)
$$
<img src="https://raw.githubusercontent.com/Yunhui1998/markdown_image/main/RL/image-20200829103402838.png" alt="image-20200829103402838" style="zoom:30%;" />

$\mathrm{MC}$ updates the empirical mean return with one sampled episode
$$
v\left(S_{t}\right) \leftarrow v\left(S_{t}\right)+\alpha\left(G_{i, t}-v\left(S_{t}\right)\right)
$$
<img src="https://raw.githubusercontent.com/Yunhui1998/markdown_image/main/RL/image-20200829103510012.png" alt="image-20200829103510012" style="zoom:33%;" />

<img src="https://raw.githubusercontent.com/Yunhui1998/markdown_image/main/RL/image-20200829170841169.png" alt="image-20200829164545152" style="zoom:40%;" />

###### MC相比于DP的优点

- **$\mathrm{M} \mathrm{C}$ works when the environment is unknown **  当环境未知时，MC更管用
- Working with sample episodes has a huge advantage, even when one has complete knowledge of the environment's dynamics, for example, **transition probability is complex to compute** 可以从采样的轨迹中进行学习总是好的，即使是知道环境的动态性，比如说，转移矩阵很难计算
- Cost of estimating a single state's value is independent of the total number of states. So you can sample episodes starting from the states of interest then average returns     不管整体的状态数量有多少，计算一个状态的价值是相对容易的。

###### TD相比于MC的优点

1、TD不需要等到epsilon结束后才学习（Incomplete sequences）

- TD can learn online after every step
  $\mathrm{MC}$ must wait until end of episode before return is known

- TD can learn from incomplete sequences
  $\mathrm{M} \mathrm{C}$ can only learn from complete sequences
- TD works in continuing (non-terminating) environments
  $\mathrm{MC}$ only works for episodic (terminating) environment

2、TD在马尔科夫的环境中更有效（因为用了bootstraping）

- TD exploits Markov property, more efficient in Markov environments
  $\mathrm{MC}$ does not exploit Markov property, more effective in non-Markov environments

3、Lower variance

4、Online

- 总结：由于MC的高方差，无偏差的特性，有以下几个特点：

1. 他有更好的收敛性质。他总能够很好的拟合函数（他能够更容易接近真实的价值函数）；
2. 对初始化数据不敏感（因为他的标注是真实的，所以最后总会调整到正确的轨道上）；
3. 收敛速度比较慢

- 由于TD算法的有偏差，低方差的特性，他有以下几个特点：

1. 他通常速度比较快（因为数据的方差比较小，而我们一般认为收敛的准则是：当数据的波动比较小，则认为他收敛了）；
2. 但是对初始化数据比较敏感（如果有一个不好的初始化值，那么他虽然可以很快收敛，但不是收敛到正确的解）；

###### 是否有Bootstrapping和Sampling

- Bootstrapping：update involves an estimate
  - MC does not bootstrap
  - DP bootstraps
  - TD bootstraps
- Sampling:update samples an expectation
  - MC samples
  - DP does not sample
  - TD samples

###### 画图理解

DP：$v\left(S_{t}\right) \leftarrow \mathbb{E}_{\pi}\left[R_{t+1}+\gamma v\left(S_{t+1}\right)\right]$

<img src="https://raw.githubusercontent.com/Yunhui1998/markdown_image/main/RL/image-20200829170841169.png" alt="image-20200829170841169" style="zoom:33%;" />

MC：$v\left(S_{t}\right) \leftarrow v\left(S_{t}\right)+\alpha\left(G_{t}-v\left(S_{t}\right)\right)$

<img src="https://raw.githubusercontent.com/Yunhui1998/markdown_image/main/RL/image-20200829170920684.png" alt="image-20200829170920684" style="zoom:33%;" />

TD(0):$T D(0): v\left(S_{t}\right) \leftarrow v\left(S_{t}\right)+\alpha\left(R_{t+1}+\gamma v\left(s_{t+1}\right)-v\left(S_{t}\right)\right)$

<img src="https://raw.githubusercontent.com/Yunhui1998/markdown_image/main/RL/image-20200829171010565.png" alt="image-20200829171010565" style="zoom:33%;" />

<img src="https://raw.githubusercontent.com/Yunhui1998/markdown_image/main/RL/image-20200829171235088.png" alt="image-20200829171235088" style="zoom:33%;" />

