## 第十章 Off-policy Policy gradient

#### Retrace

从本节开始，我们要开始介绍off-policy的策略梯度法，我们首先来介绍一下Retrace，Retrace来自DeepMind在NIPS2016发表的论文Safe and efficient off-policy reinforcement learning。它主要有以下四个特点：

- 低方差。
- 不管有什么样的动作策略进行采样，总是能“安全”地利用这些动作策略采样得到的样本，这里的“安全”我理解是当behavior policy和target policy差很多的时候，依然能保障策略最终的收敛性？
- 对样本的高效使用。
- 第一个不需要GLIE（Greedy in the limit with infinite exploration)假设就能保证收敛的returned-based off-policy control algorithm，其中return-based是之折扣奖励的累积和，它的重点在于一条轨迹或者一段时间，而不是一般的一个点。

然后我们来具体地介绍以下Retrace方法。一般基于 Off-Policy 的价值估计方法主要使用重要性采样的方法实现，我们可以用一个
$\mathrm{R}$ 表示这一类计算方法的基本形式:
$$
\mathrm{R} Q_{\pi}(\boldsymbol{x}, \boldsymbol{a})=Q_{\pi}(\boldsymbol{x}, \boldsymbol{a})+E_{\mu}\left[\sum_{t \geqslant 0} \gamma^{t}\left(\prod_{s=1}^{t} c_{s}\right)\left(\boldsymbol{r}_{t}+\gamma Q_{\pi}\left(\boldsymbol{x}_{t+1}, \cdot\right)-Q_{\mu}\left(\boldsymbol{x}_{t}, \boldsymbol{a}_{t}\right)\right)\right]
$$
其中：

- R被称为operator
-  $Q(\boldsymbol{x}, \boldsymbol{a})$ 表示值函数估计值
-  $\mu$ 表示参与交互的策略
-  $\pi$ 表示待学习的策略
-  $\gamma$ 表示回报的打折率
- $c_{s}$ 是非负的系数，被称为trace of the operator

我们接下来就来讨论当$c_{s}$不同时得到的不同的算法。

1. 当target policy   $\pi$ 和 behaviour policy   $\mu$完全相同时：

   此时$\prod_{s=1}^{t} c_{s}=1$ ，当t=0时，上面的公式就变成了 Actor Critic 中 TD-Error 的计算公式:
   $$
   \mathrm{R} Q_{\pi}(\boldsymbol{x}, \boldsymbol{a})=\boldsymbol{r}_{t}+\gamma Q_{\pi}\left(\boldsymbol{x}_{t+1}, \cdot\right)
   $$
   如果时间长度进一步拉长，我们可以得到
   $$
   \begin{aligned}
   \mathrm{R}_{t=1} Q_{\pi}(\boldsymbol{x}, \boldsymbol{a})=& Q_{\pi}(\boldsymbol{x}, \boldsymbol{a})+E_{\pi}\left[\boldsymbol{r}_{t}+\gamma Q_{\pi}\left(\boldsymbol{x}_{t+1} \mid \cdot\right)-Q_{\mu}\left(\boldsymbol{x}_{t}, \boldsymbol{a}_{t}\right)+\gamma\left(\boldsymbol{r}_{t+1}+\gamma Q_{\pi}\left(\boldsymbol{x}_{t+2} \mid \cdot\right)\right.\right.\\
   &\left.\left.-Q_{\mu}\left(\boldsymbol{x}_{t+1}, \boldsymbol{a}_{t+1}\right)\right)\right] \\
   =& Q_{\pi}(\boldsymbol{x}, \boldsymbol{a})+E_{\pi}\left[\sum_{d=0}^{1} \gamma^{d}\left(\boldsymbol{r}_{t+d}+\gamma Q_{\pi}\left(\boldsymbol{x}_{t+d+1} \mid \cdot\right)-Q_{\mu}\left(\boldsymbol{x}_{t+d}, \boldsymbol{a}_{t+d}\right)\right)\right.
   \end{aligned}
   $$
   此时的公式形式和  GAE 的计算公式比较接近。

2. 当$c_{s}=\pi\left(a_{s} \mid x_{s}\right) / \mu\left(a_{s} \mid x_{s}\right)$，上面的式子其实就是重要性采样(Importance Sampling)，重要性采样是当target policy  $\pi$  和 behaviour policy $\mu$  不同时去修正偏差的最简单的一种方式，主要时通过$\pi$和$\mu$之间的似然率的内积（也可以理解为修正的时sample path的概率）来修正$\pi$和$\mu$不同时带来的问题。但是重要性采样即使在path有限的情况下也会存在很大的方差，这主要是因为$\frac{\pi\left(a_{1} \mid x_{1}\right)}{\mu\left(a_{1} \mid x_{1}\right)} \cdots \frac{\pi\left(a_{t} \mid x_{t}\right)}{\mu\left(a_{t} \mid x_{t}\right)}$的方差造成的。

3. 当$c_{s}=\lambda$时，这是另外一种off-policy correction的方法。由 于 $\lambda$ 是一个稳定的数值, 所以不会出现IS中的那种连积后有可能会很大的情况。但是一方面这个 数值比较难定, 要有一定的实际经验; 另一方面这种方法并不能保证对任意的 $\pi$ 和 $\mu$ **安全**, 这个方法比较适用于 $\pi$ 和 $\mu$ 区别不大的时候。

   ![image-20210329223239050](https://raw.githubusercontent.com/Yunhui1998/markdown_image/main/RL/image-20210329223239050.png)

4. 当$c_{s}=\lambda \pi\left(a_{s} \mid x_{s}\right)$，这里用上了target policy $\pi_{\circ}$ 保障了算法的安全性, 但是对于两种策略相近时 (称为near on-policy) 的样本利用效率下降了。因为它同样会将一些比较远的轨迹切掉。而在near on-policy的情况下，通常是不希望这样去做的。

5. $\operatorname{Retrace}(\lambda): c_{s}=\lambda \min \left(1, \pi\left(a_{s} \mid x_{s}\right) / \mu\left(a_{s} \mid x_{s}\right)\right)$ 。Retrace算是前面提到的算法的优点整合的一个算法，它不仅在 $\pi$ 和 $\mu$ 相差比较大时保障了算法的安全性，而且当 $\pi$ 和 $\mu$ 比较接近时也不会出现切掉较远轨迹，造成效率低的问题。而且由于$c_s$由于最大值是1，所以也不会出现 $\Pi_{s=1}^{t} c_{s}$ 数值很大的情况。

在Retrace的原始论文中，作者对上面提到的几种情况做了总结：

![image-20210329230702993](https://raw.githubusercontent.com/Yunhui1998/markdown_image/main/RL/image-20210329230702993.png)

下面是论文中提到的Retrace在60种Artri游戏上的表现，可见Retrace的方法相对于原始的Q-Learning领先非常明显。

![image-20210329231021171](https://raw.githubusercontent.com/Yunhui1998/markdown_image/main/RL/image-20210329231021171.png)

#### ACER

ACER来自2017ICLR的论文Sample Efficient Actor-Critic with Experience Replay，是一种为了提高sample-efficiency的Off-policy Actor-Critic的方法。ACER结合了以下三种方法：

- truncated importance sampling with bias correction   带修正的截断重要性采样
- stochastic dueling network architectures  随机dueling网络架构
- and a new trust region policy optimization method 一个新的trust region 策略优化方法

在论文中，作者还对离散动作空间和连续动作空间分别进行了分析，而且都做了实验。我们在这里介绍以下带修正的截断重要性采样和新的trust region 策略优化方法、retrace Q函数（我觉得这个也挺重要的）和dueling网络架构。

1. 首先介绍带修正的截断重要性采样：

我们首先将Actor-Critic的动作分解成on-policy和off-policy两个部分如下：
$$
\begin{aligned}
g^{\operatorname{marg}} &=E_{s_{t} \sim \beta, a_{t} \sim \mu}\left[\rho_{t} \nabla_{\theta} \log \pi_{\theta}\left(a_{t} \mid s_{t}\right) Q^{\pi}\left(s_{t}, a_{t}\right)\right] \\
&=E_{s_{t} \sim \beta, a_{t} \sim \mu}\left[\left(\left(1-\frac{c}{\rho_{t}}+\frac{c}{\rho_{t}}\right) \rho_{t} \nabla_{\theta} \log \pi_{\theta}\left(a_{t} \mid s_{t}\right) Q^{\pi}\left(s_{t}, a_{t}\right)\right]\right.\\
&=E_{s_{t} \sim \beta}\left[E_{a_{t} \sim \mu}\left[\left(\left(1-\frac{c}{\rho_{t}}+\frac{c}{\rho_{t}}\right) \rho_{t} \nabla_{\theta} \log \pi_{\theta}\left(a_{t} \mid s_{t}\right) Q^{\pi}\left(s_{t}, a_{t}\right)\right]\right]\right.\\
&=E_{s_{t} \sim \beta}\left[E_{a_{t} \sim \mu}\left[\frac{c}{\rho_{t}} \rho_{t} \nabla_{\theta} \log \pi_{\theta}\left(a_{t} \mid s_{t}\right) Q^{\pi}\left(s_{t}, a_{t}\right)\right]\right.
\end{aligned}
$$

$$
\begin{aligned}
&\left.+E_{\boldsymbol{a}_{t} \sim \mu}\left[\left(1-\frac{c}{\rho_{t}}\right) \rho_{t} \nabla_{\theta} \log \pi_{\theta}\left(\boldsymbol{a}_{t} \mid s_{t}\right) Q^{\pi}\left(s_{t}, \boldsymbol{a}_{t}\right)\right]\right] \\
=& E_{\boldsymbol{s}_{t} \sim \beta}\left[E_{\boldsymbol{a}_{t} \sim \mu}\left[\frac{c}{\rho_{t}} \rho_{t} \nabla_{\theta} \log \pi_{\theta}\left(\boldsymbol{a}_{t} \mid \boldsymbol{s}_{t}\right) Q^{\pi}\left(\boldsymbol{s}_{t}, \boldsymbol{a}_{t}\right)\right]\right.\\
&\left.+E_{\boldsymbol{a}_{t} \sim \pi}\left[\left(\frac{\rho_{t}-c}{\rho_{t}}\right) \nabla_{\theta} \log \pi_{\theta}\left(\boldsymbol{a}_{t} \mid \boldsymbol{s}_{t}\right) Q^{\pi}\left(\boldsymbol{s}_{t}, \boldsymbol{a}_{t}\right)\right]\right]
\end{aligned}
$$

如果我们用一个经过限定的比率 $\tilde{\rho}_{t}$ 替换其中一部分的 $\rho_{t},$ 同时限定 $\tilde{\rho}_{t}=\max \left(\rho_{t}, c\right),$
就有
$$
\begin{aligned}
\frac{c}{\tilde{\rho_{t}}} &=\min \left(\frac{c}{\rho_{t}}, 1\right), \quad \frac{c}{\tilde{\rho_{t}}} \rho_{t}=\min \left(c, \rho_{t}\right) \\
\tilde{\rho_{t}}-c &=\max \left(\rho_{t}-c, 0\right), \frac{\tilde{\rho_{t}}-c}{\rho_{t}}=\max \left(\frac{\rho_{t}-c}{\rho_{t}}, 0\right)
\end{aligned}
$$
我们就可以将上面的公式替换为 ACER算法的公式:
$$
\begin{aligned}
=& E_{s_{t} \sim \beta}\left[E_{a_{t} \sim \mu}\left[\min \left(c, \rho_{t}\right) \nabla_{\theta} \log \pi_{\theta}\left(a_{t} \mid s_{t}\right) Q^{\pi}\left(s_{t}, a_{t}\right)\right]\right.\\
&+E_{a_{t} \sim \pi}\left[\left(\max \left(\frac{\rho_{t}-c}{\rho_{t}}, 0\right) \nabla_{\theta} \log \pi_{\theta}\left(a_{t} \mid s_{t}\right) Q^{\pi}\left(s_{t}, a_{t}\right)\right]\right]
\end{aligned}
$$
公式中的两项对算法起到不同的作用：

第一项被称为截断的重要性采样 ( Truncated Importance Sampling )，和 Retrace 算法的思想类似，它限制了重要性采样的比率上限，我们可以确保模型不会造成太大的波动。

第二项被称为偏差纠正项（Bias Correction for the Truncation )，作为前面一项的纠正项，它可以保证算法是无偏的，这样算法就在偏差和方差之间得到了一定的平衡。实际上，只有 $\rho_{t}>c$ 时，第二项才会 发挥作用，而此时第一项的比率被限定为 $c,$ 这时第二项会对第一项的限制做一定的补充，从而保证算法没有较大的偏差。

   2.retrace Q函数

上面的ACER的公式第一项是off-policy的期望计算，我们可以使用 $\operatorname{Retrace}(\lambda)$ 算法计算价值估计值 $Q^{\mathrm{ret}}\left(s_{t}, a_{t}\right),$ 而第二项由于是 On-Policy 的价值计算 , 因此直接使用价值模型的估计值即可，模型的参数为 $\theta_{v}$ 。这样公式就变成了下面的形式:
$$
\begin{aligned}
=& E_{s_{t} \sim \beta}\left[E_{\boldsymbol{a}_{t} \sim \mu}\left[\min \left(c, \rho_{t}\right) \nabla_{\theta} \log \pi_{\theta}\left(\boldsymbol{a}_{t} \mid \boldsymbol{s}_{t}\right) Q^{\mathrm{ret}}\left(s_{t}, \boldsymbol{a}_{t}\right)\right]\right.\\
&+E_{\boldsymbol{a}_{t} \sim \pi}\left[\left(\max \left(\frac{\rho_{t}-c}{\rho_{t}}, 0\right) \nabla_{\theta} \log \pi_{\theta}\left(\boldsymbol{a}_{t} \mid \boldsymbol{s}_{t}\right) Q_{\theta_{v}}\left(s_{t}, \boldsymbol{a}_{t}\right)\right]\right]
\end{aligned}
$$
最后一步是 Actor-Critic 算法中常见的步骤，为算法添加一个 Baselines，以降低目标函数的方差，我们可以用价值模型估计当前的状态值函数 $V(s),$ 最终的模型变为：

​                                                                            $\begin{aligned}=& E_{\boldsymbol{s}_{t} \sim \beta}\left[E_{a_{t} \sim \mu}\left[\min \left(c, \rho_{t}\right) \nabla_{\theta} \log \pi_{\theta}\left(\boldsymbol{a}_{t} \mid \boldsymbol{s}_{t}\right)\left(Q^{\mathrm{ret}}\left(s_{t}, \boldsymbol{a}_{t}\right)-V\left(s_{t}\right)\right)\right]\right.\\ &+E_{\boldsymbol{a}_{t} \sim \pi}\left[\left(\max \left(\frac{\rho_{t}-c}{\rho_{t}}, 0\right) \nabla_{\theta} \log \pi_{\theta}\left(\boldsymbol{a}_{t} \mid s_{t}\right)\left(Q_{\theta_{v}}\left(s_{t}, \boldsymbol{a}_{t}\right)-V_{\theta_{v}}\left(s_{t}\right)\right)\right]\right] \end{aligned}$



  3.新的trust region 策略优化方法

提到trust region策略优化方法，大家肯定会想到TRPO，它能够解决Actor-Critic中模型效果波动较大的问题。ACER使用了更保守的参数更新方式和相对简单的目标函数两种方式来实现类似TRPO的效果。

第一个改变是参数更新方式。ACER 维护了一个滑动平均的策略网络，每一次模型更新时，新的参数仅会以一个很小比例进行更新。这样我们就可以确保每一次更新后的参数和之前的参数保持较近的距离。这样算法可以以较小的代价实现类似 TRPO 约束的效果。我们令平均策略的参数为 $\theta_{a},$ 优化后的策略参数为 $\theta,$ 那么平均策略的更新公式为
$$
\theta_{a} \leftarrow \alpha \theta+(1-\alpha) \theta_{a}
$$
第二个改变是目标函数。ACER 目标函数中计算一阶梯度的部分与 TRPO 比较相似，只要将前面得到的公式求导就可以得到
$$
\begin{aligned}
\hat{\boldsymbol{g}}_{t}^{\text {acer }}=& \min \left(c, \rho_{t}\right) \nabla_{\theta} \log \pi_{\theta}\left(\boldsymbol{a}_{t} \mid s_{t}\right)\left[Q^{\mathrm{ret}}\left(s_{t}, \boldsymbol{a}_{t}\right)-V_{\theta_{v}}\left(s_{t}\right)\right] \\
&+E_{a_{t} \sim \pi}\left[\left(\max \left(\frac{\rho_{t}-c}{\rho_{t}}, 0\right) \nabla_{\theta} \log \pi_{\theta}\left(\boldsymbol{a}_{t} \mid \boldsymbol{s}_{t}\right)\left(Q_{\theta_{v}}\left(\boldsymbol{s}_{t}, \boldsymbol{a}_{t}\right)-V_{\theta_{v}}\left(\boldsymbol{s}_{t}\right)\right)\right]\right.
\end{aligned}
$$
ACER 的约束条件与 TRPO不同，没有使用二阶导数做约束，而是使用了 KL 散度的一阶导数，这样目标函数可以变为下面的形式:
$$
\begin{array}{l}
\operatorname{minimize}_{\boldsymbol{z}} \frac{1}{2}\left\|\hat{\boldsymbol{g}}_{t}^{\text {acer }}-\boldsymbol{z}\right\|_{2}^{2} \\
\text { s.t. } \quad \nabla_{\pi_{\theta}\left(x_{t}\right)} D_{\mathrm{KL}}\left[\pi_{\theta_{a}} \| \pi_{\theta}\right]^{\mathrm{T}} \boldsymbol{z}<\delta
\end{array}
$$
可以看出，这个目标函数虽然也拥有基于 KL散度的约束条件，但是它与 TRPO 的 形式完全不同。经过这样的变换，我们不再需要计算复杂的 Fisher 信息矩阵。最终的目标是求解的参数更新量 $z$ ，我们希望它在满足约束条件的同时尽可能地靠近 $\hat{g}_{t}^{\text {acer }}$ 值。

4. 动态的Dueling网路结构

上述的Q和V的估计使用了dueling network的结构，对于连续的行动空间，文章提出了 stochastic dueling network $_{\circ}$ 主要是因为行动空间连续的时候之前dueling network的 $\sum_{a} A_{\theta_{v}}(x, a)$ 没法算了，因此这里用采样的方法来计算。
$$
\widetilde{Q}_{\theta_{v}}\left(x_{t}, a_{t}\right) \sim V_{\theta_{v}}\left(x_{t}\right)+A_{\theta_{v}}\left(x_{t}, a_{t}\right)-\frac{1}{n} \sum_{i=1}^{n} A_{\theta_{v}}\left(x_{t}, u_{i}\right), \text { and } u_{i} \sim \pi_{\theta}\left(\cdot \mid x_{t}\right)
$$
​                                                                    $V^{\text {target }}\left(x_{t}\right)=\min \left\{1, \frac{\pi\left(a_{t} \mid x_{t}\right)}{\mu\left(a_{t} \mid x_{t}\right)}\right\}\left(Q^{\mathrm{ret}}\left(x_{t}, a_{t}\right)-Q_{\theta_{v}}\left(x_{t}, a_{t}\right)\right)+V_{\theta_{v}}\left(x_{t}\right)$

网络输出的是 $Q_{\theta_{v}}$ 和 $A_{\theta_{v}},$ 然后通过这个方法拼起来得到Q 。

![image-20210330104453098](https://raw.githubusercontent.com/Yunhui1998/markdown_image/main/RL/image-20210330104453098.png)

在这里放一下最终的算法的伪码：

![image-20210330104817508](https://raw.githubusercontent.com/Yunhui1998/markdown_image/main/RL/image-20210330104817508.png)

![image-20210330104838344](https://raw.githubusercontent.com/Yunhui1998/markdown_image/main/RL/image-20210330104838344.png)

![image-20210330104853028](https://raw.githubusercontent.com/Yunhui1998/markdown_image/main/RL/image-20210330104853028.png)

作者还做了消融实验来分析Retrace / Q($\lambda$) off-policy correction, SDNs, trust region, and truncation with bias correction到底是哪个部分在发挥作用：

![image-20210330112426651](https://raw.githubusercontent.com/Yunhui1998/markdown_image/main/RL/image-20210330112426651.png)

红色的线表示的是正常的ACER，绿色的线表示的是去掉某个部分的ACER。

可见Retrace / Q($\lambda$) off-policy correction, SDNs, trust region, and truncation with bias correction都是比较重要的，去掉他们中的任意一个效果都变差了。

Truncation with bias correction没有改变Fish和 Walker2d 任务上的结果。 然而, 在动作空间很高的人型机器人任务上, truncation and bias correction 使得跪着的人型机器人站起来了（但是从结果上看效果没太大变化？）。推测是因为动作空间维度的增加增大了重要性采样的方差，使得bias correction变得更加重要。从图中看来说明Retrace和trust region是真的很重要啊！

#### DPG

DPG（Deterministic Policy Gradient）来自David Silver在ICML2014年发表的论文Deterministic Policy Gradient Algorithms。这个论文的很大一个贡献在于**在这个论文之前大家都觉得环境模型无关的确定性策略是不存在的，而David Silver等通过严密的数学推导证明了DPG的存在。根据DPG论文的证明，当概率策略的方差趋近于0的时候，就是确定性策略**。

确定性策略：在状态St时，每次采取的动作都是一个确定的action, $a=\mu(s)$;
随机策略：在状态St时，每次采取的动作很可能不一样, 随机选择动作, $\pi(a \mid s)=P(a \mid s)$ 。

DPG的学习框架采用AC的方法, DPG求解时少了重要性权重，这是因为重要性采样是用简单的概率分布去估计复杂的概率分布，DPG的action是确定值而不是概率分布。另外DPG的值函数评估用的是Q-learning的方法, 即用TD error来估计动作值函数并忽略重要性权重。确定性策略 AC方法的梯度公式和随机策略的梯度公式如下图所示。**跟随机策略梯度相比，确定性策略少了对action的积分, 多了reward对action的导数。**

**Stochastic Policy Gradient:**
$$
\nabla_{\theta} J\left(\pi_{\theta}\right)=E_{s \sim \rho^{\pi}, a \sim \pi_{\theta}}\left[\nabla_{\theta} \log \pi_{\theta}(a \mid s) Q^{\pi}(s, a)\right]
$$
 **DPG:**

​                                                                                             $\nabla_{\theta} J\left(\mu_{\theta}\right)=E_{s \sim \rho^{\mu}}\left[\left.\nabla_{\theta} \mu_{\theta}(s) \nabla_{a} Q^{\mu}(s, a)\right|_{a=\mu_{\theta}(s)}\right]$

然后在原始论文当中还有两点是比较有趣的：

1. 作者为了防止采用确定性的策略造成策略的探索性变弱，作者在论文中提出了一种off-policy的方法，通过随机的动作策略去选动作，但是去学一个确定的目标策略，提升确定性策略下的探索。
2.  对于目标函数中的价值估计部分，我们可以使用一个值函数模型进行拟合，这样价值模型不需要遵从某个策略，这个结论作者在论文的4.3部分进行了证明。

#### DDPG

DDPG（Deep DPG）来自ICLR2016的论文Contious control with deep reinforcement learning，是一种model-free、off-policy、actor-critic架构的算法，主要是利用了我们前面讲的DPG结合一些trick在连续的动作空间的环境下取得了比较好的效果，因为之前一年DQN利用深度神经网络做函数近似在离散的、低维的动作空间的任务上取得了很好的效果，因此作者在DDPG中借鉴了很多DQN的一些成功的trick。

我们先来推导一些DDPG的目标函数：

并不包含其他的限定，例如是否为 On-Policy 或者 Off-Policy。对于 On-Policy 的 Deterministic Actor-Critc算法，值函数为 $Q^{w}(s, a),$ 确定策略为 $\mu_{\theta}(s)$, 我们可以建立如下目标函数:
$$
\begin{aligned}
J(w) &=\operatorname{minimize}_{w} E_{\pi}\left[\frac{1}{2}\left(r_{t}+\gamma Q^{w}\left(s_{t+1}, a_{t+1}\right)-Q^{w}\left(s_{t}, a_{t}\right)\right)^{2}\right] \\
J(\theta) &=\operatorname{maximize}_{\theta} E_{\pi}\left[Q^{w}\left(s_{t}, \mu\left(s_{t}\right)\right)\right]
\end{aligned}
$$
对其进行求解，可以得到
$$
\begin{array}{l}
\Delta w=\alpha E_{\pi}\left[\left(\boldsymbol{r}_{t}+\gamma Q^{w}\left(s_{t+1}, \boldsymbol{a}_{t+1}\right)-Q^{w}\left(s_{t}, \boldsymbol{a}_{t}\right)\right) \nabla_{w} Q^{w}\left(\boldsymbol{s}_{t}, \boldsymbol{a}_{t}\right)\right] \\
\Delta \theta=\alpha E_{\pi}\left[\left.\nabla_{w} Q^{w}\left(\boldsymbol{s}_{t}, \boldsymbol{a}_{t}\right)\right|_{\boldsymbol{a}=\mu_{\theta}(\boldsymbol{s})} \nabla_{\theta} \mu_{\theta}\left(s_{t}\right)\right]
\end{array}
$$
对于 Off-Policy 的算法，我们同样可以建立目标函数。由于我们使用了确定的策略， 同时值函数不依赖任何策略，那么在计算时我们就不需要向随机策略那样进行重要性采样计算。假设样本来自策略 $\beta$ ，我们的目标函数为
$$
\begin{aligned}
J(w) &=\operatorname{minimize}_{w} E_{\beta}\left[\frac{1}{2}\left(\boldsymbol{r}_{t}+\gamma Q^{w}\left(s_{t+1}, \boldsymbol{a}_{t+1}\right)-Q^{w}\left(s_{t}, \boldsymbol{a}_{t}\right)\right)^{2}\right] \\
J(\theta) &=\operatorname{maximize}_{\theta} E_{\beta}\left[Q^{w}\left(s_{t}, \mu\left(s_{t}\right)\right)\right]
\end{aligned}
$$
对其进行求解，可以得到类似的结果:
$$
\begin{array}{l}
\Delta w=\alpha E_{\beta}\left[\left(r_{t}+\gamma Q^{w}\left(s_{t+1}, a_{t+1}\right)-Q^{w}\left(s_{t}, a_{t}\right)\right) \nabla_{w} Q^{w}\left(s_{t}, a_{t}\right)\right] \\
\Delta \theta=\alpha E_{\beta}\left[\left.\nabla_{w} Q^{w}\left(s_{t}, a_{t}\right)\right|_{a=\mu_{\theta}(s)} \nabla_{\theta} \mu_{\theta}\left(s_{t}\right)\right]
\end{array}
$$
然后我们总结一下DDPG中的几个关键的trick：

- DDPG可以看做是Nature DQN, Actor-Critic和DPG三种方法的组合算法

- Critic部分的输入为states和action
- Actor部分不再使用自己的Loss函数和Reward进行更新，而是使用DPG的思想, 使用critic部分Q值对action的梯度来对actor进行更新：我们的actor的目的是尽量得到一个高Q值的action，因此actor的损失可以简单的理解为得到的反馈Q值越大损失越小，得到的反馈Q值越小损失越大
- 使用了Nature DQN的思想，加入了经验池、随机抽样和目标网络, real Q值使用两个target 网络共同计算
- target网络更新改为软更新, 在每个batch缓慢更新target网络的参数。

其中大部分我们在前面介绍Nature DQN和Actor-Critic的时候就介绍过了，这里我们重点讲一下软更新和Ornstein-Uhlenbeck噪声。

**软更新：**

在论文中，作者介绍采用滑动平均的方法更新 Target Network: $\theta_{t+1} \leftarrow \tau \theta_{t}+(1-\tau) \theta_{t}^{\prime}, \tau$ 一般设置为非常接近 1 的数，这样 Target 网络的参数 $\theta$ 不会发生太大的变化，每次只会受一点训练模型 $\theta^{\prime}$ 的影响。

**Ornstein-Uhlenbeck噪声：**

由于 DPG 采用确定策略， 如果它在与环境进行交互时只采用确空的策略，那么必然会导致对环境的探索不够充分。在 DDPG 模型中，作者采用 Ornstein-Uhlenbeck 噪声增加模型的探索能力。Ornstein-Uhlenbeck 噪声是一种基 于 Urnstein-Uhlenbeck 过程的随机变量，已可以被用于模拟与时间有关联的噪声数据。它的公式为
$$
\mathrm{d} x_{t}=\theta\left(\mu-x_{t}\right)+\sigma W_{t}
$$
其中 $x$ 是要生成的数据， $\mu$ 是设定的随机变量的期望值， $W$ 是一个由 Wiener 过程生成 的随机变量，一般我们用一个简单的随机函数代替就可以， $\theta$ 和 $\sigma$ 是随机过程中的参数， $\mathrm{d} x$ 是每一时刻数据的变化量，真正的采样值等于上一时刻的采样值加上求出的变化量。从公式中中以看出，每一时刻数据的变化量与当前时刻存在关联，公式右边的第公式右边的第一项将为随机数据提供朝向均值μ 的变化，第二项才是常见的随机变化。

这里再总结一下DDPG和DPG、DQN的差别：

**1、与DPG的不同之处**
(1) 采用卷积神经网络作为策略函数 $\mu$ 和Q函数的近似, 即策略网络和Q网络; 然后使用深度学习的方法来训练上述神经网络。
(2) 网络结构和个数不同：
从DPG到DDPG的过程，可以类比于DQN到DDQN的过程。除了经验回放之外，还有双网络，即当前网络和目标网络的概念。而由于现在本就有actor和critic两个网络，那么双网络就变成了4个网络，分别是：actor当前网络、actor目标网络、critic当前网络、critic目标网络。

**2、与DQN不同**

1） DDPG中没有使用硬更新，而是选择了软更新，即每次参数只更新一点点。

(2) 增加随机性操作。

(3) 损失函数：critic当前网络的损失函数还是均方误差, 没有变化，但是actor当前网络的损失函数改变了。

在此重复一下，因为DDPG采用了DQN的很多trick，所以后面针对DQN的一些改进比如优先级采样等也是可以用到DDPG中的，感兴趣的可以关注这篇论文：

[Leveraging Demonstrations for Deep Reinforcement Learning on Robotics Problems with Sparse Rewards](https://arxiv.org/pdf/1707.08817.pdf)。

