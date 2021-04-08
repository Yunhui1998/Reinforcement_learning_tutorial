## 第七章-Policy Optimazation

#### Policy Optimazation 简介

先说一个我觉得很好的观点，可以把策略梯度法看成一个加权了的最大似然估计法，**加的这个权重是轨迹得到的回报，也就是说，我们不是希望学出来的策略和采样得到的轨迹的概率分布接近，而是我们希望最大化那些回报大的轨迹出现的概率，也就是希望策略去实施得到的轨迹尽可能能获得大的回报。**

##### 定义

与基于价值的策略不同，基于策略的优化不再去求价值函数，而是直接去算策略参数的梯度去进行优化。也就是说输入是比如前面说的游戏的图片，输出的直接就是每个动作的概率。

![image-20200922111603182](https://raw.githubusercontent.com/Yunhui1998/markdown_image/main/RL/image-20200922111603182.png)

##### Valued-based和Policy-based RL对比

- Value-based RL
  - to learn value function    去学价值函数
  - implicit policy based on the value function  通过价值函数隐含地学出策略
- Policy-based RL
  - no value function   没有价值函数
  - to learn policy directly   直接去学策略
- Actor-critic
  - to learn both policy and value function   即学价值函数，也学策略

<img src="https://raw.githubusercontent.com/Yunhui1998/markdown_image/main/RL/image-20200922111809605.png" alt="image-20200922111809605" style="zoom: 50%;" />



##### Advantages of Policy-based RL

-  Advantages:
   - better convergence properties: we are guaranteed to converge on a local optimum (worst case) or global optimum (best case)   更好的收敛性，保证起码收敛到一个局部最优点
   - Policy gradient is more effective in high-dimensional action space   在高维空间中更有效
   - Policy gradient can learn stochastic policies, while value function can't    基于策略的方法可以学出随机策略，而基于值的方法不行
-  Disadvantages:
   - typically converges to a local optimum    总是收敛到局部最优点   
   - evaluating a policy has high variance    评估策略的时候总是方差很大

##### Policy Optimazation的方法

- Policy-based RL is an optimization problem that find $\theta$ that maximizes $J(\theta)$
- If $J(\theta)$ is differentiable, we can use gradient-based methods:    如果目标函数是可导的，那我们就可以用基于梯度的方式去求解基于策略的强化学习方法
  -  gradient ascend
  -  conjugate gradient
  -  quasi-newton
- If $J(\theta)$ is non-differentiable or hard to compute the derivative, some derivative-free black-box optimization methods:
  - Cross-entropy method (CEM)
  - Hill climbing
  - Evolution algorithm

##### Cross-Entropy Method

![image-20200922152959654](https://raw.githubusercontent.com/Yunhui1998/markdown_image/main/RL/image-20200922152959654.png)

类似于采样枚举？？？？

##### Approximate Gradients by Finite Difference

![image-20200922155129929](https://raw.githubusercontent.com/Yunhui1998/markdown_image/main/RL/image-20200922155129929.png)

相当于间接地去求了微分。

#### REINFOECE

先回顾一下强化学习的目标，最大化累计奖励：

$$\theta^{\star}=\arg \max _{\theta} E_{\tau \sim p_{\theta}(\tau)}\left[\sum_{t} r\left(\mathbf{s}_{t}, \mathbf{a}_{t}\right)\right]$$

在有限长度轨迹的情况下：

$$\theta^{\star}=\arg \max _{\theta} E_{(\mathbf{s}, \mathbf{a}) \sim p_{\theta}(\mathbf{s}, \mathbf{a})}[r(\mathbf{s}, \mathbf{a})]$$

在无限长度轨迹的情况下：

$$\theta^{\star}=\arg \max _{\theta} \sum_{t=1}^{T} E_{\left(\mathbf{s}_{t}, \mathbf{a}_{t}\right) \sim p_{\theta}\left(\mathbf{s}_{t}, \mathbf{a}_{t}\right)}\left[r\left(\mathbf{s}_{t}, \mathbf{a}_{t}\right)\right]$$

我们令目标函数：

$$J(\theta)=E_{\tau \sim p_{\theta}(\tau)}\left[\sum_{t} r\left(\mathbf{s}_{t}, \mathbf{a}_{t}\right)\right] \approx \frac{1}{N} \sum_{i} \sum_{t} r\left(\mathbf{s}_{i, t}, \mathbf{a}_{i, t}\right)$$，令第一项中括号中的为$$r(\tau)$$，

$$\begin{array}{c}
J(\theta)=E_{\tau \sim p_{\theta}(\tau)}[\underbrace{r(\tau)]}{\hookrightarrow}=\int p_{\theta}(\tau) r(\tau) d \tau \\
\sum_{t=1}^{T} r\left(\mathbf{s}_{t}, \mathbf{a}_{t}\right)
\end{array}$$

对它直接进行梯度计算:

$$\nabla_{\theta} J(\theta)=\int \nabla_{\theta} p_{\theta}(\tau) r(\tau) d \tau=\int p_{\theta}(\tau) \nabla_{\theta} \log p_{\theta}(\tau) r(\tau) d \tau=E_{\tau \sim p_{\theta}(\tau)}\left[\nabla_{\theta} \log p_{\theta}(\tau) r(\tau)\right]$$

在这里的第二个等号中，使用了一个恒等式:

$$p_{\theta}(\tau) \nabla_{\theta} \log p_{\theta}(\tau)=p_{\theta}(\tau) \frac{\nabla_{\theta} p_{\theta}(\tau)}{p_{\theta}(\tau)}=\nabla_{\theta} p_{\theta}(\tau)$$

而     $$\underbrace{p_{\theta}\left(\mathbf{s}_{1}, \mathbf{a}_{1}, \ldots, \mathbf{s}_{T}, \mathbf{a}_{T}\right)}_{p_{\theta}(\tau)}=p\left(\mathbf{s}_{1}\right) \prod_{t=1} \pi_{\theta}\left(\mathbf{a}_{t} \mid \mathbf{s}_{t}\right) p\left(\mathbf{s}_{t+1} \mid \mathbf{s}_{t}, \mathbf{a}_{t}\right)$$

两边取log：

$$\log p_{\theta}(\tau)=\log p\left(\mathbf{s}_{1}\right)+\sum_{t=1}^{T} \log \pi_{\theta}\left(\mathbf{a}_{t} \mid \mathbf{s}_{t}\right)+\log p\left(\mathbf{s}_{t+1} \mid \mathbf{s}_{t}, \mathbf{a}_{t}\right)$$

所以：

$$\nabla_{\theta} \log p_{\theta}(\tau) r(\tau)$$= $$\nabla_{\theta}\left[\log \not p\left(\mathbf{s}_{1}\right)+\sum_{t=1}^{T} \log \pi_{\theta}\left(\mathbf{a}_{t} \mid \mathbf{s}_{t}\right)+\log p\left(\mathbf{s}_{t+1} \widehat{\mathbf{s}_{t}}, \mathbf{a}_{t}\right)\right]$$

第一项和第三项都和$\theta$没有关系，所以

$$\nabla_{\theta} J(\theta)=E_{\tau \sim p_{\theta}(\tau)}\left[\nabla_{\theta} \log p_{\theta}(\tau) r(\tau)\right]$$  =   $$E_{\tau \sim p_{\theta}(\tau)}\left[\left(\sum_{t=1}^{T} \nabla_{\theta} \log \pi_{\theta}\left(\mathbf{a}_{t} \mid \mathbf{s}_{t}\right)\right)\left(\sum_{t=1}^{T} r\left(\mathbf{s}_{t}, \mathbf{a}_{t}\right)\right)\right]$$

​                                                                   $$\approx \frac{1}{N} \sum_{i=1}^{N}\left(\sum_{t=1}^{T} \nabla_{\theta} \log \pi_{\theta}\left(\mathbf{a}_{i, t} \mid \mathbf{s}_{i, t}\right)\right)\left(\sum_{t=1}^{T} r\left(\mathbf{s}_{i, t}, \mathbf{a}_{i, t}\right)\right)$$

终于！这两项都可以通过采样然后通过样本算出来了！ 然后根据强化学习算法的经典的流程图：

![image-20210223160135811](https://raw.githubusercontent.com/Yunhui1998/markdown_image/main/RL/image-20210223160135811.png)

我们就可以得到REINFORCE 算法，也就是根据当前策略进行采样，采样之后算出梯度，算出梯度之后对策略进行更新，然后继续采样，如此往复：

![image-20210223160210899](https://raw.githubusercontent.com/Yunhui1998/markdown_image/main/RL/image-20210223160210899.png)

值得注意的是，在上述推导过程中，我们并没有用到Markov property，所以REINFORCE对partially observed MDPs而言也是可以使用的，因为它针对的是trajectory。

在离散动作空间下，可以跟分类任务一样，使用多个输出结点作为不同action的选择。

而在连续动作空间下，往往会使用Gaussian polices进行action选择：通过神经网络对state 进行处理，得到高斯分布的均值（方差可以得到也可以指定），再通过这个分布进行抽样。
$$
\pi_{\theta}\left(\mathbf{a}_{t} \mid \mathbf{s}_{t}\right)=\mathcal{N}\left(f_{\text {neural network }}\left(\mathbf{s}_{t}\right) ; \Sigma\right)
$$
而由于高斯分布梯度易求，所以参数优化也比较方便
$$
\begin{array}{l}
\log \pi_{\theta}\left(\mathbf{a}_{t} \mid \mathbf{s}_{t}\right)=-\frac{1}{2}\left\|f\left(\mathbf{s}_{t}\right)-\mathbf{a}_{t}\right\|_{\Sigma}^{2}+\text { const } \\
\nabla_{\theta} \log \pi_{\theta}\left(\mathbf{a}_{t} \mid \mathbf{s}_{t}\right)=-\frac{1}{2} \Sigma^{-1}\left(f\left(\mathbf{s}_{t}\right)-\mathbf{a}_{t}\right) \frac{d f}{d \theta}
\end{array}
$$

#### Differences between RL and Imitation learning

在这里我们将最大似然估计(maximum likelihood)中求解参数的梯度和REINFORCE算法求解的梯度进行对比：

​                                                                                   $$\begin{array}{c}
\nabla_{\theta} J_{\mathrm{ML}}(\theta) \approx \frac{1}{N} \sum_{i=1}^{N}\left(\sum_{t=1}^{T} \nabla_{\theta} \log \pi_{\theta}\left(\mathbf{a}_{i, t} \mid \mathbf{s}_{i, t}\right)\right) \\
\nabla_{\theta} J(\theta) \approx \frac{1}{N} \sum_{i=1}^{N}\left(\sum_{t=1}^{T} \nabla_{\theta} \log \pi_{\theta}\left(\mathbf{a}_{i, t} \mid \mathbf{s}_{i, t}\right)\right)\left(\sum_{t=1}^{T} r\left(\mathbf{s}_{i, t}, \mathbf{a}_{i, t}\right)\right)
\end{array}$$

这里我们可以看出，policy gradient相当于是加了权重的maximum likelihood，权重就是奖励值，也就是说ML是希望以后的策略都接近我们采样的那些数据，但是policy gradient 是希望策略去增大那些累积奖励高的轨迹出现的概率而抑制那些累积奖励低的轨迹出现的概率，这也是强化学习和模仿学习的差别，因为模仿学习里面我们拿到的是专家样本，我们默认它就是最好的或者说是累积奖励很高的，因此我们要做的只是去让我们的策略更像专家策略，也就是让我们学出来的策略的轨迹和专家样本的轨迹更像。

可以通过下面的这个图来理解：**在强化学习中我们希望奖励高的轨迹出现的概率更大**。

![image-20210223161318267](https://raw.githubusercontent.com/Yunhui1998/markdown_image/main/RL/image-20210223161318267.png)



#### Drawback of REINFORCE  and solutions

##### No causality

首先，我们再回顾一下求解策略价值梯度的公式:
$$
\nabla_{\theta} J(\theta)=\frac{1}{N} \sum_{i=1}^{N} \sum_{t=0}^{T} \nabla_{\theta} \log \pi_{\theta}\left(\boldsymbol{a}_{i, t} \mid \boldsymbol{s}_{i, t}\right)\left(\sum_{t=0}^{T} r\left(s_{i, t}, \boldsymbol{a}_{i, t}\right)\right)
$$
从这个公式中我们看出了一个问题，不论是哪个时间段，我们都要用策略的梯度除以所有时刻的回报值总和，这样的设计显然是不合理的。因为理论上，在 $t$ 时刻我们完成了决策后，**它最多只能影响 $t$ 时刻之后的所有回报**（这也被称之为**causality**，表示在t时刻做出的策略只会影响t时刻之后获得的reward），并不会影响 $t$ 时刻之前的回报，因为我们无法通过现在的决策影响已经发生过的事情，所以这一部分的回报值不应该计算在梯度中。我们的公式应该改写成：

​                                                                               $$\nabla_{\theta} J(\theta)=\frac{1}{N} \sum_{i=1}^{N} \sum_{t=0}^{T}\left[\nabla_{\theta} \log \pi_{\theta}\left(\boldsymbol{a}_{i, t} \mid \boldsymbol{s}_{i, t}\right)\left(\sum_{t^{\prime}=t}^{T} r\left(s_{i, t^{\prime}}, a_{i, t^{\prime}}\right)\right)\right]$$

##### High Variance

因为在REINFORCE中，$$\nabla_{\theta} J(\theta) \approx \frac{1}{N} \sum_{i=1}^{N} \nabla_{\theta} \log \pi_{\theta}(\tau) r(\tau)$$，也就是后面有一个对轨迹奖励的一个加权，看下面这个例子：

![image-20210407020013556](https://raw.githubusercontent.com/Yunhui1998/markdown_image/main/RL/image-20210407020013556.png)

我们先看黄色的bar作为第一种情况，如果我们有三条轨迹，他们的奖励值分别长成黄色的bar这种情况，那么我们最终得到的策略应该是图中的第二条线。

然后我们再来看第二种情况，我们把第一种情况下轨迹的奖励的值减去一个常数，得到的奖励分布变成了绿色的bar，那我们最终得到的策略应该是图中的第三条线。

这就很有问题了，我们只是让奖励函数减去一个常量，得到的策略变化就差这么大，说明policy gradient方法的方差很大，当然如果你有无限个样本的时候，这个问题就不是问题，但是如果你像上面一样只有有限个样本，那就会影响很大了，而现实中我们的样本数量肯定是有限的。

李宏毅老师课上也举过一个很好理解的例子，就是如果我们的奖励函数都是正数，那么对于我们所有采集的轨迹，它的概率都会上升，对于未采样到的trajectory，即使它是非常好的trajectory，它的概率是下降的，这显然很不合理。

![image-20210407015624088](https://raw.githubusercontent.com/Yunhui1998/markdown_image/main/RL/88.png)

那么针对这个问题我们能有什么方法去缓解呢？

我们知道，上面的问题在于“当reward为正时，我们会增强所有的策略，只是对实际效果不好的策略，我们为其提升的幅度会有所下降”，但我们希望的是：让能够最大化长期回报策略的“权重”为正且尽可能大，让不能最大化长期回报策略的“权重" 为负且尽可能小。于是公式就变成：

​                                                                                       $$\nabla_{\theta} J(\theta)=\frac{1}{N} \sum_{i=1}^{N} \sum_{t=0}^{T}\left[\nabla_{\theta} \log \pi_{\theta}\left(\boldsymbol{a}_{i, t} \mid \boldsymbol{s}_{i, t}\right)\left(\sum_{t^{\prime}=t}^{T} r\left(\boldsymbol{s}_{i, t^{\prime}}, \boldsymbol{a}_{i, t^{\prime}}\right)-\boldsymbol{b}_{i, t^{\prime}}\right)\right]$$

这个变量可以设计为同一起始点的不同序列在同一时刻的长期回报均值，它的公式形式如下所示：:
$$
\boldsymbol{b}_{i, t^{\prime}}=\frac{1}{N} \sum_{i=1}^{N} \sum_{t^{\prime}=t}^{T} r\left(\boldsymbol{s}_{i, t^{\prime}}, \boldsymbol{a}_{i, t^{\prime}}\right)
$$
这个时候$$E\left[\nabla_{\theta} \log p_{\theta}(\tau) b\right]=\int p_{\theta}(\tau) \nabla_{\theta} \log p_{\theta}(\tau) b d \tau=\int \nabla_{\theta} p_{\theta}(\tau) b d \tau=b \nabla_{\theta} \int p_{\theta}(\tau) d \tau=b \nabla_{\theta} 1=0$$，也就是说增加这一项不会使得原有的计算值变得有偏。

当然，我们可以有更好的做法让方差更小：可以对方差进行求导判断导数为0的点：

$$\nabla_{\theta} J(\theta)=E_{\tau \sim p_{\theta}(\tau)}\left[\nabla_{\theta} \log p_{\theta}(\tau)(r(\tau)-b)\right]$$

$$\operatorname{Var}=E_{\tau \sim p_{\theta}(\tau)}\left[\left(\nabla_{\theta} \log p_{\theta}(\tau)(r(\tau)-b)\right)^{2}\right]-E_{\tau \sim p_{\theta}(\tau)}\left[\nabla_{\theta} \log p_{\theta}(\tau)(r(\tau)-b)\right]^{2}$$

有因为b是无偏的，所有第二项可以省略，所以

$$\left.\frac{d \mathrm{Var}}{d b}=\frac{d}{d b} E\left[g(\tau)^{2}(r(\tau)-b)^{2}\right]=\frac{d}{d b}\left(E\left[q(\tau)^{2} r(\tau)\right)^{2}\right]-2 E\left[g(\tau)^{2} r(\tau) b\right]+b^{2} E\left[g(\tau)^{2}\right]\right)$$

$$=-2 E\left[g(\tau)^{2} r(\tau)\right]+2 b E\left[g(\tau)^{2}\right]=0$$

所以：$$b=\frac{E\left[g(\tau)^{2} r(\tau)\right]}{E\left[g(\tau)^{2}\right]}$$ = $$\frac{E\left[\nabla_{\theta} \log \pi(\tau)^{2} r(\tau)\right]}{E\left[\nabla_{\theta} \log \pi(\tau)^{2}\right]}$$

这一项可以理解成： expected reward weighted by gradient magnitudes，不知道该怎么翻译成中文。在实践中，我们一般都直接使用前面的相对简单的算术平均的形式作为baseline。

#### Off-Policy policy gradient

由于上面这种On-policy的policy gradient的方法在策略更新之后就需要把过去采集到的样本丢弃，因此效率非常低，所以我们就想着能不能将其改变成为off-policy的方法。

#####  Importance Sampling

Importance Sampling（下面简称IS）是一个使用易采样的数据的样本估计难采样的样本的期望的方法。基于这个工作，则可以利用来自其他policy的数据来提高当前的policy，从而达到off-policy的作用。

假设可以从q(x)中进行采样，需要求的是f(x)关于p(x)的期望，那么可以通过如下的方式得到：

​                                                                                          $$\begin{aligned}
E_{x \sim p(x)}[f(x)] &=\int p(x) f(x) d x \\
&=\int \frac{q(x)}{q(x)} p(x) f(x) d x \\
&=\int q(x) \frac{p(x)}{q(x)} f(x) d x \\
&=E_{x \sim q(x)}\left[\frac{p(x)}{q(x)} f(x)\right]
\end{aligned}$$

##### Off-policy policy gradient

将importance sampling引入policy gradient，假设我们有来自 $\bar{\pi}(\tau)$ 的数据, 那么计算 $\pi(\tau)$ 关于这些数据的目标函数则可以转化为：
$$
J(\theta)=E_{\tau \sim \bar{\pi}(\tau)}\left[\frac{\pi_{\theta}(\tau)}{\bar{\pi}(\tau)} r(\tau)\right]
$$
由于 $\pi_{\theta}(\tau)$ 可以表示为：
$$
\pi_{\theta}(\tau)=p\left(\mathbf{s}_{1}\right) \prod_{t=1}^{T} \pi_{\theta}\left(\mathbf{a}_{t} \mid \mathbf{s}_{t}\right) p\left(\mathbf{s}_{t+1} \mid \mathbf{s}_{t}, \mathbf{a}_{t}\right)
$$
约掉来自environment的initial state和transition probability, 两个policy的比值可以转化为:
$$
\frac{\pi_{\theta}(\tau)}{\bar{\pi}(\tau)}=\frac{\prod_{t=1}^{T} \pi_{\theta}\left(\mathbf{a}_{t} \mid \mathbf{s}_{t}\right)}{\prod_{t=1}^{T} \bar{\pi}\left(\mathbf{a}_{t} \mid \mathbf{s}_{t}\right)}
$$
令$$\frac{p_{\theta^{\prime}}(\tau)}{p_{\theta}(\tau)}=\frac{\prod_{t=1}^{T} \pi_{\theta^{\prime}}\left(\mathbf{a}_{t} \mid \mathbf{s}_{t}\right)}{\prod_{t=1}^{T} \pi_{\theta}\left(\mathbf{a}_{t} \mid \mathbf{s}_{t}\right)}$$， 

​                                                                                          $$\nabla_{\theta^{\prime}} J\left(\theta^{\prime}\right)=E_{\tau \sim p_{\theta}(\tau)}\left[\frac{p_{\theta^{\prime}}(\tau)}{p_{\theta}(\tau)} \nabla_{\theta^{\prime}} \log \pi_{\theta^{\prime}}(\tau) r(\tau)\right] \quad \text { when } \theta \neq \theta^{\prime}            $$

​                                                                                         $$=E_{\tau \sim p_{\theta}(\tau)}\left[\left(\prod_{t=1}^{T} \frac{\pi_{\theta^{\prime}}\left(\mathbf{a}_{t} \mid \mathbf{s}_{t}\right)}{\pi_{\theta}\left(\mathbf{a}_{t} \mid \mathbf{s}_{t}\right)}\right)\left(\sum_{t=1}^{T} \nabla_{\theta^{\prime}} \log \pi_{\theta^{\prime}}\left(\mathbf{a}_{t} \mid \mathbf{s}_{t}\right)\right)\left(\sum_{t=1}^{T} r\left(\mathbf{s}_{t}, \mathbf{a}_{t}\right)\right)\right]$$

<img src="https://raw.githubusercontent.com/Yunhui1998/markdown_image/main/RL/image-20210223185445781.png" alt="image-20210223185445781" style="zoom:50%;" />

​                                                                            $$\nabla_{\theta^{\prime}} J\left(\theta^{\prime}\right)=E_{\tau \sim p_{\theta}(\tau)}\left[\sum_{t=1}^{T} \nabla_{\theta^{\prime}} \log \pi_{\theta^{\prime}}\left(\mathbf{a}_{t} \mid \mathbf{s}_{t}\right)\left(\prod_{t^{\prime}=1}^{t} \frac{\pi_{\theta^{\prime}}\left(\mathbf{a}_{t^{\prime}} \mid \mathbf{s}_{t^{\prime}}\right)}{\pi_{\theta}\left(\mathbf{a}_{t^{\prime}} \mid \mathbf{s}_{t^{\prime}}\right)}\right)\left(\sum_{t^{\prime}=t}^{T} r\left(\mathbf{s}_{t^{\prime}}, \mathbf{a}_{t^{\prime}}\right)\right)\right]$$

![image-20210223185754568](https://raw.githubusercontent.com/Yunhui1998/markdown_image/main/RL/image-20210223185754568.png)

其实这里上面的图片上中上面的标注错了，图片中那一项可以去掉的原因不是因为两个策略不能差太多，而是因为如果看他们初始化的概率的话和那个策略是无关的。

##### Tips in Policy Gradient Descent

- 首先policy gradient的variance是非常大的，它的gradient噪声比较大（来自trajectory的累积reward）。为了克服它，可以尝试将训练的batch size加到非常大。
- 另外，由于gradient的noise很大，学习率会比较难调节，很多时候使用ADAM勉强可行。而一般而言，我们会使用类似PPO/TRPO那样的专用于policy gradient的自动确定学习步长的方法