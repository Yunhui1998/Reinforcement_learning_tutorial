## 第八章-Actor-Critic

#### Reducing variance with critic

策略梯度法的梯度如下：

$$\nabla_{\theta} J(\theta)=\frac{1}{N} \sum_{i=1}^{N} \sum_{t=0}^{T}\left[\nabla_{\theta} \log \pi_{\theta}\left(\boldsymbol{a}_{i, t} \mid \boldsymbol{s}_{i, t}\right)\left(\sum_{t^{\prime}=t}^{T} r\left(\boldsymbol{s}_{i, t^{\prime}}, \boldsymbol{a}_{i, t^{\prime}}\right)-\boldsymbol{b}_{i}\right)\right]$$

我们用轨迹的回报表示整个序列的价值，这个表示是准确无偏的，但是在真实的训练过程中，为了尽可能地控制学习时间，我们无法完成非常多次的交互，往往只会进行有限次数的交互，这些交互有时并不能很好地代表轨迹真实的期望。每一个交互得到的序列都会有一定的差异，即使他们从同一起点出发对应的回报也会有一定的差异，**因此不充足的交互会给轨迹回报带来较大的方差**。

![img](https://raw.githubusercontent.com/Yunhui1998/markdown_image/main/RL/v2-ab04f8dc9f02cec436a61e307784249a_720w.jpg)

而在这里我们选择用牺牲一定的偏差来换取方差变小：Actor-Critic 用一个独立的模型估计轨迹的长期回报，而不再直接使用轨迹的真实回报。类似于基于模型的Q-Learning 算法，在估计时使用模型估计轨迹价值，在更新时利用轨迹的回报得到目标价值，然后将模型的估计值和目标值进行比较，从而改进模型。

公式中可以被模型替代的部分有两个，其中 $\sum_{t^{\prime}=t}^{T} r\left(s_{i, t^{\prime}}, a_{i, t^{\prime}}\right)$ 代表从 $t$ 时刻出发所获得的长期回报， $b_{i}$ 代表待减去的偏移量。根据这两个部分被模型替代的情况，我们可以得到以下几种方案：

- 采用策略梯度法的方法：公式由 $\sum_{t^{\prime}=t}^{T} r\left(s_{i, t^{\prime}}, \boldsymbol{a}_{i, t^{\prime}}\right)-\boldsymbol{b}_{i}$ 表示。
- 使用状态值函数估计轨迹的回报 : $q(s, a)_{\circ}$
- 直接使用优势函数估计轨迹的回报：A $(s, a)=q(s, a)-V(s)$ 。
- 使用 TD-Error 估计轨迹的回报: 公式由 $r\left(s_{t}, \boldsymbol{a}_{t}\right)+v\left(s_{t+1}\right)-v(s)$ 表示。

可以看出四种方法都可以在一定程度上降低算法的方差，实际中 Actor Critic 算法最终选择了第 4 种方案，这种方案在计算量（**只需要算一个价值函数V，而V函数和动作无关，又可以用来计算Q函数和A函数**）和减少方差方面具有一定的优势。由于引入了状态价值模型，算法整体包含了两个模型，一个是策略模型，一个是价值模型，所以这个算法被称为 Actor-Critic，其中 Actor 表示策略模型，Critic 表示价值模型。

因此，我们可以得到我们的Actor-Critic算法为如下：

![image-20210224013111468](https://raw.githubusercontent.com/Yunhui1998/markdown_image/main/RL/image-20210224013111468.png)

#### Evaluation for value function

在上面我们讲到，我们需要去拟合V来构建Actor-Critic的结构，那么我们具体该怎么做呢？其实也就是我们前面讲到的用MC和TD方法，这里再简要介绍一下：

##### Monte Carlo

最直观的方法就是根据V的定义, 采用Monte Carlo的方法, 计算每个state到terminal state的 trajectory上会有多少reward, 经过无数条reward的平均之后得到V值:
$$
V^{\pi}\left(\mathbf{s}_{t}\right) \approx \frac{1}{N} \sum_{i=1}^{N} \sum_{t^{\prime}=t}^{T} r\left(\mathbf{s}_{t^{\prime}}, \mathbf{a}_{t^{\prime}}\right)
$$
在实际中引入无穷大条trajectory是不可能的, 所以这样也采用sampling替代expectation的方法进行替换, 这样盟然并不完美，但是在实际中也是适用的。
$$
V^{\pi}\left(\mathbf{s}_{t}\right) \approx \sum_{t^{\prime}=t}^{T} r\left(\mathbf{s}_{t^{\prime}}, \mathbf{a}_{t^{\prime}}\right)
$$
转化为监督学习，如果要利用神经网络来approximate $\mathrm{V},$ 则可以转化在如下数据集下：
$$
\text { training data: }\left\{(\mathbf{s}_{i, t}, \underbrace{\sum_{t^{\prime}=t}^{T} r\left(\mathbf{s}_{i, t^{\prime}}, \mathbf{a}_{i, t^{\prime}}\right)}_{y_{i, t}})\right\}
$$
给予如下的目标函数, 训练网络:
$$
\mathcal{L}(\phi)=\frac{1}{2} \sum_{i}\left\|\hat{V}_{\phi}^{\pi}\left(\mathbf{s}_{i}\right)-y_{i}\right\|^{2}
$$

##### Temporal difference

相对Monte Carlo方法直接使用整条轨迹来计算, Temporal difference引入了bootstrapped的方法。在前面target y是由整段轨迹的reward累积来确定的, 但是如果在V估计准确的情况下，
它 其实也等于当前state采取某个action的reward加上下一个state的V值。这个也就是temporal difference方法降低采样需求的思路, 故而给予同样的目标函数下, target的值发生了改变：
$$
\text { training data: }\left\{(\mathbf{s}_{i, t}, \underbrace{r\left(\mathbf{s}_{i, t}, \mathbf{a}_{i, t}\right)+\hat{V}_{\phi}^{\pi}\left(\mathbf{s}_{i, t+1}\right)}_{y_{i, t}})\right\}
$$
这两种方法的对比其实非常像REINFORCE方法与Actor-Critic方法的对比。 MC方法使用了整条轨迹作为target, 它可以理解为是unbiased的，但是由于估计中存在policy与dynamic的随机性, 所以variance非常高。而TD则是引入了适当的bias, 大幅度减少了variance, 也提高了训练速度。

#### Discount factor

Discount factor其实在我们前面讲return（Goal函数）的时候就已经讲到了，当trajectory的步长变成无穷时，估计中就会出现无穷大量, 这就会导致算法整体难以分析。所以我们就引入discount factor这个量, 对后续的reward进行discount, 而避免无穷的问题：

![image-20210224112446682](https://raw.githubusercontent.com/Yunhui1998/markdown_image/main/RL/image-20210224112446682.png)

在TD方法中, target可以转化为如下的形式：.
$$
y_{i, t} \approx r\left(\mathbf{s}_{i, t}, \mathbf{a}_{i, t}\right)+\gamma \hat{V}_{\phi}^{\pi}\left(\mathbf{s}_{i, t+1}\right)
$$
据此得到gradient
$$
\nabla_{\theta} J(\theta) \approx \frac{1}{N} \sum_{i=1}^{N} \sum_{t=1}^{T} \nabla_{\theta} \log \pi_{\theta}\left(\mathbf{a}_{i, t} \mid \mathbf{s}_{i, t}\right)\left(r\left(\mathbf{s}_{i, t}, \mathbf{a}_{i, t}\right)+\gamma \hat{V}_{\phi}^{\pi}\left(\mathbf{s}_{i, t+1}\right)-\hat{V}_{\phi}^{\pi}\left(\mathbf{s}_{i, t}\right)\right)
$$
在Monte Carlo中，这个问题就有两种不同的情况：

![image-20210224145909313](https://raw.githubusercontent.com/Yunhui1998/markdown_image/main/RL/image-20210224145909313.png)

在第一种形式中，对causality后的reward从舍去部分开始计算discount，也就是说前面切掉的部分对discount的累乘没有影响。而第二种形式中，先引入了discount，再计算causality，故而对于被舍去的部分它的discount效应仍然存在，在这种情况下，越往后的steps的对结果的影响会相对第一种更小。

在实际中我们通常是使用第一种方式，因为对于后面的状态我们也是希望对网络有贡献的，如果factor过小，那么就会导致后面的状态一直不太准确，影响整体结果。

我们最终得到的Actor-Critic的算法为下图所示：前者采样一个batch再进行学习，后者按照单个样本进行学习，学完之后就把这个样本丢弃。

![image-20210224150135049](https://raw.githubusercontent.com/Yunhui1998/markdown_image/main/image-20210224150135049.png)

####  Actor Critic Design Decisions

首先是我们是我们需要决定是用两个网络分别去拟合Actor网络和Critic网络还是用同一个网络去拟合：

使用两个网络的优势是容易训练且稳定，缺点是没有共享feature，导致参数量增大，计算量也增大。而使用一个网络解决了两个网络的优势，但是有可能会出现两个部分冲突的问题

![image-20210225105201833](https://raw.githubusercontent.com/Yunhui1998/markdown_image/main/image-20210225105201833.png)



其次就是选择online模式还是batch模式：

如果是连续的state形成的batch，state之间会存在很强的相关性；但是一般情况下，采用batch模式都会更有效地帮助我们降低方差。

![image-20210225105518394](https://raw.githubusercontent.com/Yunhui1998/markdown_image/main/image-20210225105518394.png)

在采样数据的时候我们一般有两种方式：第一种是同步采样，在多个worker上采集样本，等到达到一定数量后，进行更新，等待更新完毕，在获取新的模型参数，进行下一个batch的采样。第二种则是异步采样，在各个worker上采集样本后，计算梯度后，将梯度传给learner，进行更新，learner在更新后分发新的参数。目前大型的学习系统通常是采用异步的模式，主要似乎因为速度比较快，但是这种方式可能会出现梯度延迟或者对于on-policy方法带来样本off-policy的问题，后面有许多算法都在解决这个问题。

![image-20210225105731166](https://raw.githubusercontent.com/Yunhui1998/markdown_image/main/image-20210225105731166.png)

#### Critics as baselines

前面提到的Actor-Critic方法和Policy Gradient方法各有优劣：Actor-Critic方法方差小但是有偏，Policy-Gradient无偏但是方差大：

![image-20210225111621338](https://raw.githubusercontent.com/Yunhui1998/markdown_image/main/image-20210225111621338.png)

那我们就会有有一个想法是说能不能把这两种方法结合起来，形成类似下面的这种形式，在引入value function降低variance的情况下保持unbiased：

![image-20210225111742618](https://raw.githubusercontent.com/Yunhui1998/markdown_image/main/image-20210225111742618.png)

传统的Advantage函数如下：

​                                                                              $$A^{\pi}\left(\mathbf{s}_{t}, \mathbf{a}_{t}\right)=Q^{\pi}\left(\mathbf{s}_{t}, \mathbf{a}_{t}\right)-V^{\pi}\left(\mathbf{s}_{t}\right)$$

如果我们用MC的方法来估计Q函数，这个估计是unbiased的，但是由于是single-sample estimate，故而variance比较高：

​                                                                               $$\hat{A}^{\pi}\left(\mathbf{s}_{t}, \mathbf{a}_{t}\right)=\sum_{t^{\prime}=t}^{\infty} \gamma^{t^{\prime}-t} r\left(\mathbf{s}_{t^{\prime}}, \mathbf{a}_{t^{\prime}}\right)-V_{\phi}^{\pi}\left(\mathbf{s}_{t}\right)$$

我们再换成下面这种形式：如果critic是正确的时候, 它的期望应该是零,。
$$
\hat{A}^{\pi}\left(\mathbf{s}_{t}, \mathbf{a}_{t}\right)=\sum_{t^{\prime}=t}^{\infty} \gamma^{t^{\prime}-t} r\left(\mathbf{s}_{t^{\prime}}, \mathbf{a}_{t^{\prime}}\right)-Q_{\phi}^{\pi}\left(\mathbf{s}_{t}, \mathbf{a}_{t}\right)
$$
留意到上面一个奏出来的式子相对而言是少了一个bias项, 那么我们是否可以将它补充回来呢? 这 个也就是Q-Prop的思路了，他的第二项就是这么一个目的：
$$
\nabla_{\theta} J(\theta) \approx \frac{1}{N} \sum_{i=1}^{N} \sum_{t=1}^{T} \nabla_{\theta} \log \pi_{\theta}\left(\mathbf{a}_{i, t} \mid \mathbf{s}_{i, t}\right)\left(\hat{Q}_{i, t}-Q_{\phi}^{\pi}\left(\mathbf{s}_{i, t}, \mathbf{a}_{i, t}\right)\right)+\frac{1}{N} \sum_{i=1}^{N} \sum_{t=1}^{T} \nabla_{\theta} E_{\mathbf{a} \sim \pi_{\theta}\left(\mathbf{a}_{t} \mid s_{i, t}\right)}\left[Q_{\phi}^{\pi}\left(\mathbf{s}_{i, t}, \mathbf{a}_{t}\right)\right]
$$
接下来我们再引入n-step的形式：

![image-20210225112249569](https://raw.githubusercontent.com/Yunhui1998/markdown_image/main/image-20210225112249569.png)

我们发现上面的两种形式刚好是两个极端，前者使用了整条trajectory，后者仅仅使用了一个step，那么是否可以有个折中，从而达到base/variance的tradeoff？

![image-20210225112351699](https://raw.githubusercontent.com/Yunhui1998/markdown_image/main/image-20210225112351699.png)

于是我们就得到了下面这种n-step的形式：

$$\hat{A}_{n}^{\pi}\left(\mathbf{s}_{t}, \mathbf{a}_{t}\right)=\sum_{t^{\prime}=t}^{t+n} \gamma^{t^{\prime}-t} r\left(\mathbf{s}_{t^{\prime}}, \mathbf{a}_{t^{\prime}}\right)-\hat{V}_{\phi}^{\pi}\left(\mathbf{s}_{t}\right)+\gamma^{n} \hat{V}_{\phi}^{\pi}\left(\mathbf{s}_{t+n}\right)$$

