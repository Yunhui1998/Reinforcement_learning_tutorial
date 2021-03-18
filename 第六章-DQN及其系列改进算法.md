## 第六章 DQN及其系列改进算法

##### DQN

DQN整体对比Q-learning有以下改进：

- 数据预处理

  - Atari 游戏的原生尺寸为210 × 160,每个像素有128 种颜色，将其转换成84 × 84 维度的灰度图。变换后的图像依然保留了主要的信息，同时减轻了数据处理的负担。
  - 虽然Atari 游戏是一个动态的游戏，但是每一时刻Agent 只能从环境中获得1帧信息，而这种静态的图像信息很难表示游戏的动态信息。以乒乓球游戏为例，当画面静止时我们无法判断球要飞向哪一方。为此，**算法将收集从当前时刻起的前N帧画面**，并将这些信息结合起来作为模型的输入。获得了一定时间内集合的状态信息，模型可以学习到更准确的行动价值，在实验中N 被设置为4 。
  - 不同的Atari 游戏的得分系统不同，有的得分可以上万，有的得分只有几十。为了模型能够更好地拟合长期回报，我们需要
    将得分压缩到模型擅长处理的范围中，我们将每一轮得到的回报压缩到［－ 1, 1］ 。   **---后面的这两个改进是必须的吗？？？？**

- 环境交互

  Atari 游戏的可行状态数量非常多，因此如何更好地探索更多的状态变得十分关键。DQN 采用了$\epsilon-$ greedy 的策略，一开始策略以100% 的概率随机产生行动，随着训练的不断进行，这个概率将不断衰减，最终衰减至10% 。也就是说，有90% 的概率执行当前最优策略。**这样，从以探索为主的策略逐渐转变成以利用为主的策略，两者得到了很好的结合。**

- 模型结构

  前面提到价值模型是从 $|S| \times|A|$ 到 $R$ 的映射，采用这种方法构建模型自然是可以的，但是它还有一些缺点。当模型需要通过值函数求解最优策略时，我们需要计算 |A|次才能求出，在实际中这样的计算方法效率较低。为了简化计算，我们可以将模型变成 $S \rightarrow\left[R_{i}\right]_{i=1}^{|A|}$ 这样的形式，模型的输出为长度为 $|\boldsymbol{A}|$ 的向量，向量中的每一个值表示对 应行动的价值估计。这样我们只需要一次计算就可以求出所有行动的价值，无论行动有多少，我们评估价值的时间是一样的。两个模型的表示形式差异如下图：

![image-20200922083535674](https://raw.githubusercontent.com/Yunhui1998/markdown_image/main/RL/image-20200922083535674.png)

模型的主体采用了卷积神经网络的结构：

   第一层卷积层的卷积核为 $8 \times 8,$ stride $=4,$ 输出的通道数为 32，此后应用 ReLU非线性层。
  第二层卷积层的卷积核为 $4 \times 4,$ stride $=2,$ 输出的通道数为 64，此后应用 ReLU 非线性层。
  第三层卷积层的卷积核为 $3 \times 3,$ stride $=1,$ 输出的通道数为 64，此后应用 ReLU 非线性层。
  第四层是全连接层，输出的维度为 512，此后应用 ReLU 非线性层。
  最后一层全连接层得到对应行动的价值估计，输出维度与可行的行动数量相关，一 般在 4 ~18 个。

- Random Start
  有些Atari 游戏的起始画面是完全一样的，例如Space Invader ，如果每次我们都从一个固定的场景开始做决策，那么Agent 总要对这些相同的画面进行决策，这显然不利于我们探索更多的画面进行学习。为了增强探索性同时不至于使模型效果变糟，我们可以设定在游戏开始很短的一段时间内，让Agent 执行随机的行动，这样可以最大程度地获得不同的场景样本。

- Frame-Skipping

  模拟器可以达到每秒60 帧的显示速率，但是实际上人类玩家无法实现如此高频率的操作，因此我们可以设定每隔一定帧数执行一次行动，例如每隔4帧进行一次行动选择，那么中间的几帧将重复执行前面选择的行动，这样相当于模仿了人类按下某个按钮并持续一段时间的效果。

  此外，相邻帧之间的画面存在着极大的相似性，对于十分相似的画面，我们通常可以采用相同的行动，因此这样跳过一定帧数的判断也是合理。

- Replay Buffer

  Q-Leaming 方法基于当前策略进行交互和改进，更像是一种在线学习（ **Online Learning** ）的方法，每一次模型利用交互生成的数据进行学习，学习后的样本被直接丢弃。但如果使用机器学习模型代替表格式模型后再采用这样的在线学习方法，就有可能遇到两个问题。
  ( 1 ）**交互得到的序列存在一定的相关性**。交互序列中的状态行动存在着一定的相关性，而对于基于最大似然法的机器学习模型来说，我们有一个很重要的假设：训练样本是独立且来自相同分布的，一旦这个假设不成立，模型的效果就会大打折扣。而上面提到的相关性恰好打破了独立同分布的假设，那么学习得到的值函数模型可能存在很大的波动。
  ( 2 ）**交互数据的使用效率**。采用梯度下降法进行模型更新时，模型训练往往需要经过多轮迭代才能收敛。每一次迭代都需要使用一定数量的样本计算梯度， 如果每次计算的样本在计算一次梯度后就被丢弃，那么我们就需要花费更多的时间与环境交互并收集样本。

  因此DQN采用了如下图所示的Replay Buffer结构：

![image-20210225152207186](https://raw.githubusercontent.com/Yunhui1998/markdown_image/main/image-20210225152207186.png)

从上图中可以看出，Replay Buffer 保存了交互的样本信息，一般来说每一个样还会保存其他的信息。**Replay Buffer 的大小通常会设置得比较大, 例如，将上限设置为可以存储 100 万个样本，这样较长一段时间的样本都可以被保存 起来**。在训练值函数时，我们就可以从中取出一定数量的样本，根据样本记录的信息进 行训练。
总的来说，Replay Buffer 包含了收集样本和采样样本两个过程。收集的样本按照时 间先后顺序存人结构中，如果 Replay Buffer 已经存满样本，那么新的样本会将时间上最久远的样本覆盖。而对采样来说，如果每次都取出最新的样本，那么算法就和在线学习相差不多; 一般来说，Replay Buffer 会从缓存中**均匀地随机采样一批样本**进行学习。
均匀采样的好处是什么呢？前面提到我们交互得到的序列在时间维度上存在一定 的相关性。我们希望学习得到的值函数能够表示在当前状态行动下的长期收益的期望， 然而每一次交互得到的序列，只能代表当前状态-行动下的一次采样轨迹，并不能代表 所有可能的轨迹，这样估计的结果就和期望的结果存在一定的差距。随着交互时间不 断拉长，这个差距的累积会越来越大。如果完全使用序列的估计值进行训练，某一轮训练时模型会朝着一个序列的估计训练，另一轮训练又会朝着另一个序列的估计训练， 那么模型很容易产生较大波动。**采用均匀采样后，每次训练的样本通常来自多次交互序列，这样单一序列的波动就被减轻很多，训练效果也就稳定了很多。同时，一份样本也可以被多次训练，提高了样本的利用率。**

- Target Network

  模型不稳定的另外一个原因来自算法本身。从 Q-Learning 的计算公式可以看出, 算法可以分成如下两个步骤。
  （1 ）计算当前状态行动下的价值目标值: $\left.\Delta q(s, a)=r\left(s^{\prime}\right)+\max _{a^{\prime}} q^{T-1}\left(s^{\prime}, a^{\prime}\right)_{\text { }}\right)$
  （2）网络模型的更新： $q^{T}(s, a)=q^{T-1}(s, a)+\frac{1}{N}\left[\Delta q(s, \boldsymbol{a})-q^{T-1}(s, a)\right]_{\text { }}$
  可以看出模型通过当前时刻的回报和下一时刻的价值估计进行更新，这就像一场猫捉老鼠的游戏，猫在快速移动，老鼠也在快速移动，最终得到的轨迹自然是杂乱无章不稳定的。

  ![image-20200922090136933](https://raw.githubusercontent.com/Yunhui1998/markdown_image/main/RL/image-20200922090136933.png)

  ![image-20200922090152714](https://raw.githubusercontent.com/Yunhui1998/markdown_image/main/RL/image-20200922090152714.png)

  这里存在一些隐患，前面提到数据样本差异可能造成一定的波动，由于数据本身存在着不稳定性，每一轮迭代都可能产生一些波动，如果按照上面的计算公式，**这些波动会立刻反映到下一个迭代的计算中，这样我们就很难得到一个平稳的模型**。为了减轻相关问题带来的影响，我们需要尽可能地将两个部分解耦。
  为此论文作者引人了目标网络（Target Network ），它能缓解上面提到的波动性问题。这个方法引人了另一个结构完全一样的模型，这个模型被称为 Target Network，而原本的模型被称为表现模型（Behavior Network ）。两个模型的训练过程如下所示。
  （1）在训练开始时，两个模型使用完全相同的参数。
  （2）在训练过程中，Behavior Network 负责与环境交互，得到交互样本。
  （3）在学习过程中，由 Q-Learning 得到的目标价值由 Target Network 计算得到; 然 后用它和 Behavior Network 的估计值进行比较得出目标值并更新 Behavior Network。
  （4）每当训练完成一定轮数的迭代，Behavior Network 模型的参数就会同步给 Target Network，这样就可以进行下一个阶段的学习了。
  通过使用 Target Network，计算目标价值的模型在一段时间内将被固定，这样模型 可以减轻模型的波动性。
  以上就是 DQN 模型的主要内容，完整的算法流程如下所示。

![image-20210225143125779](https://raw.githubusercontent.com/Yunhui1998/markdown_image/main/RL/image-20210225143125779.png)

##### Double Q-learning

算法来自论文：Deep Reinforcement Learning with Double Q-learning

DQN 使用了两个结构相同的网络：Behavior Network 和 Target Network。通过在一段时间内固定 Target Network 的参数，Q-Learning 方法的目标价值能够得到一定的固定，这 样模型也能够获得一定的稳定性。
虽然这个方法提升了模型的稳定性，但它并没有解决另外一个问题: Q-Learning 对价值过高估计的问题。在许多实验中，我们发现DQN的估计总会偏高：

![image-20210225143224968](https://raw.githubusercontent.com/Yunhui1998/markdown_image/main/RL/image-20210225143224968.png)

我们知道 Q-Learning 在计算时利用了下一时刻的最优价值，所以它通常在计算时给出了一个状态行动的估计上限。**由于训练过程中模型并不够稳定， 因此对上限的估计可能会存在一定的偏差。如果偏差是一致的，也就是说，每个行动都拥有相近的偏差，那么偏差对模型效果的影响相对较小; 如果偏差是不一致的，那么这 个偏差会造成模型对行动优劣的判断偏差**，这样就会影响模型的效果。直观一点进行解释：如果Q网络的估计是有噪声的，那么在计算target的时候的max操作会倾向于取较大的值，就导致估计一直往较大值的方向，从而出现over-estimate的问题。

严谨一点, 假设存在两个随机变量, 它们的期望就是真实的Q-value, 这个时候我们需要的是两个 变量的期望的max (式子中的右方)，而实际上由于一次只能拿到一个数据, 所以实际估计的是两 个量最大值的期望。这样就使得得到的结果会偏大：
$$
E\left[\max \left(X_{1}, X_{2}\right)\right] \geq \max \left(E\left[X_{1}\right], E\left[X_{2}\right]\right)
$$
我们已经知道 Target Network 求解价值目标值的公式
$$
\boldsymbol{y}_{j}=\boldsymbol{r}_{j+1}+\gamma \max _{\boldsymbol{a}^{\prime}} Q\left(s_{j+1}, \boldsymbol{a}^{\prime} ; \theta^{-}\right)
$$
其中 $\theta^{-}$ 表示 Target Network 的参数。将公式进一步展开，可以得到更详细的公式内容 :
$$
\boldsymbol{y}_{j}=\boldsymbol{r}_{j+1}+\gamma Q\left(\boldsymbol{s}_{j+1}, \operatorname{argmax}_{\boldsymbol{a}^{\prime}} Q\left(\boldsymbol{s}_{j+1}, \boldsymbol{a}^{\prime} ; \theta^{-}\right) ; \theta^{-}\right)
$$
公式展开后，我们发现采用 Target Network 后，模型在选择最优行动和计算目标值时依然使用了同样的参数模型，根据Q去选择一个最佳的action，然后得到这个最佳action的Q值。前面之所以出现over-estimation的问题，正是因为这个选到的action必然会导致得到的Q值最大。为了尽可能地减少过高估计的影响，一个简单的办法就是把选择最优行动和估计最优行动两部分的工作分离，**这样就可以保证当一个网络过估计时，它其实只是相当于提案的功能，另一个网络并不一定采取它，所以另一个网络并不一定会继续过估计这个值**，我们用 Behavior Network 完成最优行动的选择工作，这样就保证了引入的网络和当前网络的差距不太大，这样就可以得到
$$
\boldsymbol{y}_{j}=\boldsymbol{r}_{j+1}+\gamma Q\left(\boldsymbol{s}_{j+1}, \operatorname{argmax}_{\boldsymbol{a}^{\prime}} Q\left(\boldsymbol{s}_{j+1}, \boldsymbol{a}^{\prime} ; \theta\right) ; \theta^{-}\right)
$$
通过这样的变化，算法在三个环节的模型安排如下。
（1 ) 采样: Behavior Network $Q(\theta)$ 。
（2）选择最优行动: Behavior NetworkQ $(\theta)$ 。
（3）计算目标价值: Target Network $Q\left(\theta^{-}\right)$
经过这样的变换，模型在过高估计的问题上得到了缓解，稳定性也就得到了提高。

##### Multi-step DQN

Multi-step的思想在前面已经多次提到了，这里就不再赘述了，也就是用n-steps return 来替代reward：

​                                                                   $$y_{j, t}=\sum_{t^{\prime}=t}^{t+N-1} \gamma^{t-t^{\prime}} r_{j, t^{\prime}}+\gamma^{N} \max _{\mathbf{a}_{j, t+N}} Q_{\phi^{\prime}}\left(\mathbf{s}_{j, t+N}, \mathbf{a}_{j, t+N}\right)$$

它的好处是减少了Q值估计不准的时候的biased，通常也能加速训练。但是问题也是来自于n-steps，既然出现了n-steps，那么自然就涉及到状态转移了，在不同policy下，当前的n-steps trajectory出现的概率就不同，这也就导致了在off-policy的buffer data存在出现distribution mismatch的问题,所以导致它只在on-policy的情况下有用：

![image-20210225145214623](https://raw.githubusercontent.com/Yunhui1998/markdown_image/main/RL/image-20210225145214623.png)

那么怎么解决这个问题呢？大致有以下三种思路：

- ignore the problem often works very well    忽略他们，算法一般也能有用，在现在大部分使用n-steps的算法，包括GORILA、Ape-X，R2D2等都是这样操作的。
- cut the trace - dynamically choose $\mathrm{N}$ to get only on-policy data   利用某些手段去cut掉这些trace，一般都是通过两个policy生成这个trace的probability比值是否超过某个阈值来判定，比较著名的就是Retrace以及基于它的ACER，IMPALA等
- works well when data mostly on-policy, and action space is small
- importance sampling   重要性采样

##### Priority Replay Buffer

Priority Replay Buffer 是一种针对 Replay Buffer 的改进结构，它来自论文 Prioritized Experience Replay 。在最传统的DQN中，从 Replay Buffer中采集样本时，每一个样本都 会以相同的概率被采样出来。也就是说，每一个样本，都会以相同的频率被学习。**但实际上，每一个样本的难度是不同的，学习每一个样本得到的收获也不同。有的样本相对简单，表现相对较好，学习得到的收获不会太高; 有的样本则有些困难，表现也相对较差，经过学习后模型会有显著的提高，就像人脑中的记忆也会存在重要性的差别。**如果平等地看待每一个样本，就会在那些简单的样本上花费比较多的时间，而学习的潜力没有被充分挖掘出来。

Priority Replay Buffer 则很好地解决了这个问题。它会根据模型对当前样本的表现情况，给样本一定的权重，在采样时样本被采样的概率就和这个权重有关。交互时表现得越差，对应的权重越高，采样的概率也就越高：反过来，交互时表现得越好，那么权 重也就越低，采样的概率也就越低。

这样，那些使模型表现不好的样本可以有更高的概率被重新学习，模型就会把更 多的精力放在这些样本上，模型的学习效率就会有一定的提升。现在，Priority Replay Buffer 已经被广泛应用在 Q-Learning 的算法中，而且这个思想也被神经科学界证实，在人脑上也存在类似的学习机制。

接下来。从算法原理上看，两者的差别在以下两个方面。
（1）为每一个存入 Replay Buffer 的样本设定一个权重。
（2）使用这个权重完成采样过程。

我们先来解决第 2 个问题。假设权重已经得到，我们就可以采用轮盘赌的方法进行样本的采样。如果像 Replay Buffer 一样直接使用数组保存每一个样本和权重，它的更改样本和采样的流程如图下图所示：

![image-20200922092356573](https://raw.githubusercontent.com/Yunhui1998/markdown_image/main/RL/image-20200922092356573.png)

流程分为 3 步: 在添加样本到 Replay Buffer 时，直接将其写入而不做任何处理，同时更新 Replay Buffer 中所有样本的权重总和 sum; 在采样时随机得到一个 [0,sum] 之间 的数，那么从 Replay Buffer 算起，累积权重和超过随机值的第一个样本就被选择为待训练的样本，它的运算复杂度如下。

- 更改复杂度: O(1)，只需要在指定的位置写人信息。

- 采样复杂度: O(N)，需要按序扫描指定的位置才能结束，最坏的情况下可能会扫描整个数组。

  从上面的分析可以看出，采样的复杂度有些高，Replay Buffer 的大小一般比较大， 有时可能会接近达到上百万，那么扫描整个数组的代价就会比较大。 为了能快速实现样本集合中一部分样本的权重修改和集合的采样，我们可以采用线段树这个数据结构实现这个功能。**线段树能够以比较快的速度更新局部的统计信息，同时可以利用这些信息快速查找满足某些特征的数据**。线段树的结构如图下图所示：

![image-20200922092538209](C:\Users\yunhu\Desktop\RL_code_from_scratch\强化学习的基本介绍.assets\image-20200922092538209.png)

线段树的主体是由一棵树构成的。它的叶子节点存放着每一个元素个体的信息，例如它的权重，而线段树的非叶子节点则存放着其子节点的权重汇总。

如果这是一颗求和的线段树，那么非叶子节点存放着其叶子节点的权重总和; 

如果这是一颗求最小值的线段树，那么非叶子节点存放着其叶子节点的最小值。使用线段树后的更新与采样的流程如下图所示：

![image-20200922092608567](https://raw.githubusercontent.com/Yunhui1998/markdown_image/main/RL/image-20200922092608567.png)

它的流程同样有 3 步。

在添加样本到 Replay Buffer 时，不光记录样本信息，还要沿着样本叶子出发，依次找到叶子节点的父节点，更新父节点的权值;

 在采样时同样得到一个随机数，然后利用树的结构二分查找就可以快速定位被选到的样本，它的运算复杂度如下:

- 更改复杂度: O(log N)，需要更新叶子节点的所有父节点的信息。
- 采样复杂度: $O(\log N),$ 只需要从根出发访问指定的叶子。
  对比发现，线段树在采样上明显快于数组，而更新的代价又没有增加太多，因此它可以替代数组使用。

我们又知道，Priority Replay Buffer 的主要思路是增加有价值样本出现的频率。那么我们该如何定义“有价值”这个概念呢？

最直观的想法是，能让模型学习到更多的内容，这个样本一定更有价值。那么，什么样的样本能让模型学习到更多呢？ 当然是能产生更大的 TD-Error 的，即估计值与目标值的差距越大，样本越有价值。所以我们可以将 TD-Error 作为样本重要性的衡量指标。**TD-Error 越高，样本出现的概率就越高。**

虽然 TD-Error 是一个很合理的指标，但是由于我们的模型处在学习的过程中，在样本放茹 Priority Replay Buffer 时样本的 TD-Error 和它被取出时的 TD-Error 有可能不一样。

所以可能会出现一种情况：一个样本在放人时 TD-Error 很高，但是随着模型的 不断学习，模型对这个样本的实际 TD-Error 已经降低，但它还是会以很高的概率出现。

 一个最合理的方法，是在我们从 Priority Replay Buffer 取出样本时，将它的 TD-Error 重新计算一遍，但是这样做的代价实在太大**，所以我们只好使用样本放人 Priority Replay Buffer 时的 TD-Error 作为近似。**

既然我们已经知道这个指标并不完全可靠，就需要其他的方法辅助。因此我们要 确保 TD-Error 较高的样本出现的概率更高，同时也要使那些 TD-Error 较低的样本以一定的概率出现。论文中定义的一个计算样本出现概率的方法为
$$
P(i)=\frac{p_{i}^{\alpha}}{\sum_{k} p_{k}^{\alpha}}
$$
其中 $p_{i}$ 表示每个样本在计算时的 TD-Error。 可以调整样本 TD-Error 的重要性。

当 $\alpha=1$ 时，相当于我们直接使用 TD-Error 数值 ; 

当 $\alpha<1$ 时，我们就可以削弱高 TD-Error 样本的影响，增强低 TD-Error 样本的影响。（感觉这里就和softmax刚好相反）。

这样我们就可以通过这个参数调整Priority Replay Buffer 的表现。
除了上面提到的两个不同，实际上 Priority Replay Buffer 还做了第三个改变，它提 供了一个参数，用于调整每个样本对模型更新的影响。

对 Replay Buffer 来说，每一个样本是被等概率取出的，它们对模型的更新也是等权重的; 而 Priority Replay Buffer 的 样本是非等概率取出的，它的样本服从另一个难以清楚地描述的分布，所以我们对模型的更新是有偏差的。

为了使我们的更新无偏，可以采用重要性采样的方法使更新变得无偏，对应的公式为
$$
\begin{aligned}
E_{i \sim \mathrm{RB}}[\nabla J] &=E_{i \sim \mathrm{PRB}}\left[\frac{P_{\mathrm{RB}}(i)}{P_{\mathrm{PRB}}(i)} \nabla J\right] \\
&=E_{i \sim \mathrm{PRB}}\left[\frac{\frac{1}{N}}{P_{\mathrm{PRB}}(i)} \nabla J\right] \\
&=E_{i \sim \mathrm{PRB}}\left[\frac{1}{N \cdot P_{\mathrm{PRB}}(i)} \nabla J\right]
\end{aligned}
$$
其中 N 表示 Replay Buffer 存放的样本数量。所以我们可以在每一个被学习的样本前增 加一个权重 $\frac{1}{N \cdot P_{\text {PRB }(i)}},$ 这样就可以使更新变得无偏。
我们使用 Priority Replay Buffer 的目的不正是让更新变得有偏吗？为什么还要削弱它的优势呢?实际上，这个纠正是为了说明我们可以通过一些设定让它变回 Replay Buffer 那样的更新方式，这样虽然没有带来任何好处，但也没有任何坏处。

**那么我们就可以根据实际问题调整这个权重，让它在两种更新效果之间做一个权衡，既能确保提升样本的利用率，又能确保结果不会带来太大的偏差**，新的权重更新公式变为
$$
w_{i}=\left(\frac{1}{N \cdot P_{\mathrm{PRB}}(i)}\right)^{\beta}
$$
这样当 $\beta=1$ 时，更新效果实际上等同于 Replay Buffer; 当 $\beta<1$ 时，Priority Replay Buffer 就能够发挥作用了。

以上是 Priority Replay Buffer 的算法原理，在此我们可以做一个总结。
（1）在样本存人 Replay Buffer 时，计算 $P(i)=\sum_{j} p(i)^{\alpha}$
（2）在样本取出时，以第 1 步计算得到的概率进行采样。
（3）在更新时，为每一个样本添加 $w_{i}=\left(\frac{1}{N \cdot P_{\text {PBB }(i)}}\right)^{\beta}$ 的权重。
（4）随着训练的进行，让 $\beta$ 从某个小于 1 的值渐进地靠近 1。

##### Dueling DQN

Dueling DQN 是一种基于 DQN 的改进算法，它的主要突破点在于利用模型结构 将值函数表示成更细致的形式，这使得模型能够拥有更好的表现。它来自论文 Dueling network architectures for deep reinforcement learning 。接下来，我们就来看看它的改进 形式。
经过前面的介绍，我们可以给出这个公式并定义一个新的变量：
$$
q\left(s_{t}, \boldsymbol{a}_{t}\right)=v\left(\boldsymbol{s}_{t}\right)+A\left(\boldsymbol{s}_{t}, \boldsymbol{a}_{t}\right)
$$
也就是说，基于状态和行动的值函数 q（以下简称为 $q$ ) 可以分解成基于状态的值 函数 $v$ (以下简称为 $v$ ) 和优势函数（Advantage Function ）A（以下简称为 $A$ )。由于存 在
$$
E_{a_{t}}\left[q\left(s_{t}, a_{t}\right)\right]=v\left(s_{t}\right)
$$
所以如果所有状态行动的值尼数不相同，一些状态行动的价值 $q(s, a)$ 必然会高于状态的价值 $v(s),$ 当然也会有一些状态行动对低于价值，**于是优势函数可以表示出当前行动和平均表现之间的区别：如果优于平均表现，那么优势函数为正; 反之则为负。**

于是我们在保持网络主体结构不变的基础上，将原本网络中的单一输出变成两路输出，一个输出用于输出 $v,$ 它是一个一维的标量，另一个输出用于输出 A，它的维度和行动数量相同。

最后将两部分的输出加起来，就是原本的 q 值。改变输出结构 后，我们只需要对模型做很少的改变即可实现功能：模型前面的部分可以保持不变，模 型后面的部分从一路输出改变为两路输出，最后再合并成一个结果。模型结构的改变如下图所示：

![image-20210307152740300](https://raw.githubusercontent.com/Yunhui1998/markdown_image/main/RL/image-20210307152740300.png)



我们还需要对这两部分的输 出做一定的限制。如果我们不对这两个输出做限制，那么当 $q$ 值一定时， $v$ 和 $a$ 有无穷种可行组合，而实际上只有很小一部分的组合是合乎情理、接近真实数值的。为了解决这个问题，我们可以对 A 函数做限定。我们知道 A 函数的期望值为 0:
$$
\begin{aligned}
E_{a}\left[A\left(s_{t}, a_{t}\right)\right] &=E_{a}\left[q\left(s_{t}, a_{t}\right)-v\left(s_{t}\right)\right] \\
&=v\left(s_{t}\right)-v\left(s_{t}\right) \\
&=0
\end{aligned}
$$
这样做有什么好处呢？就是上面这种情况网络可能会学出一种策略，让V函数等于0，然后A函数等于Q函数，这样dueling就根本没有发挥作用，而且其实我们在实际操作中实际有一些时候更希望能尽可能地去更新V，因为更新一个V会对很多个Q值造成影响，所以我们就会给A函数加一些限制，比如说让它期望为0，也就是通过加一些normalization的操作，让网络去倾向更新V的值。

![image-20210307153824058](https://raw.githubusercontent.com/Yunhui1998/markdown_image/main/RL/image-20210307153824058.png)

就可以对输出的 A 值进行约束，例如将公式变成
$$
q\left(s_{t}, a_{t}\right)=v\left(s_{t}\right)+\left(A\left(s_{t}, a_{t}\right)-\frac{1}{|A|} \sum_{a^{\prime}} A\left(s_{t}, a_{t}^{\prime}\right)\right)
$$
让每一个 A 值减去当前状态下所有 A 值的平均数，就可以保证前面提到的期望值 为 0 的约束，从而增加了 $v$ 和 $A$ 的输出稳定性。

进行这样的输出分解有什么好处呢?

 首先, 通过这样的分解，我们不但可以得到给定状态和行动的 q 值，还可以同时得到 $v$ 值和 $A$ 值。**这样如果在某些场景需要使用 V 值时，我们同样可以获得 v 值而不用再训练一个网络。**

同时，通过显式地给出 v 函数的 输出值，每一次更新时，我们都会显式地更新 v 函数。**这样 $V$ 函数的更新频率就会得到确定性的增加**。

其次，从网络训练的角度来看，我们从原本需要训练 |A| 个取值为 [0,$\infty$] 的数值 变成了训练一个取值为 $[0, \infty]$ 的数值，和 $|A|$ 个均值为 0，实际取值为 $[-C, C]$ 的数值, **对网络训练来说，后者显然是更友好且容易的**。

同时，对一些强化学习的问题来说，A值的取值范围远比 v 值小，这样将**两者分开训练更容易保持行动之间的排列顺序。**

由于我们需要从所有的行动中挑选出价值最高的行动，因此不同行动之间的价值需要保持一定的区分度。**由于 A 值的数值范围比较小，因此它对模型更新更敏感，**这样模型在更新时会更容易考量与其他行动的相对变化量，也就不会因为某一次的更新使得原本的行动排序被意外打破。如果采用 q 值进行更新，由于 q 值相对于 A 值可能会很大,因此 q 值对微小的更新不敏感，某一次的更新可能会影响行动的原本排序，从而对策略造成一定的波动。

将值函数分解后，每一部分的结果都具有实际的意义，我们也可以从中挖掘出很多有价值的信息。从论文中给出的实验效果图可以看出，将模型分为 $A$ 函数和 $v$ 函数后，两个函数会展示出不同的特点。通过**反卷积操作**得到两个函数值对原始图像输入的梯度后，**可以发现V函数对游戏中的所有关键信息都十分敏感，而A函数只对和行动相关的信息敏感。**

##### DQN from Demonstrations（DQfD解决DQN冷启动问题）

对于以值函数为核心的 Q-Learning 算法来说，前期的算法迭代很难让模型快速进入一个相对理想的环境，更何况由于前期值函数估计存在较大偏差，与环境交互得到的采样与最优策略存在一定的差别，这更增加了学习的难度。于是 论文 Deep Q-learning from Demonstrations 提出了一种在与real-world交互之前，先用准备好的优质采样轨迹进行预训练，用value function来模仿示范者，以实现agent与env在交互之初就可以用上较为完善的TD，这样就相当于直接站在巨人的肩膀上， 学习速度自然会快很多。只要我们有办法取得一定数量的优质轨迹，就可以通过监督学习完成与环境交匹前的预训练。除了使用监督学习预训练，我们还要使用强化学习完成 DQN 模型原本方法中的训 练。早终模型的目标函数变成了多个学习目标结合的形式：
$$
J(q)=J_{\mathrm{DQ}}(q)+\lambda_{1} J_{n}(q)+\lambda_{2} J_{E}(q)+\lambda_{3} J_{L 2}(q)
$$
这个目标函数包含了4个子目标。

- $J_{\mathrm{DQ}}(q):$ Deep Q-Learning 的目标函数。
- $J_{n}(q):$ 以 $n$ 步回报估计法为目标的 Q-Learning 目标函数。
- $J_{E}(q):$ 利用准备数据进行监督学习的目标函数，用于示范动作的分类。
- $J_{L 2}(q):$ 参数L2的正则

其中， $\lambda_{n}$ 用于平衡不同目标函数之间的权重。

其中,第三项 supervised large margin classification loss至关重要, **因为示范集的一大问题就是只包含 了一小部分的状态空间**, 很多状态-动作根本就没有数据。如果只是用Q-learning update的方式 更新, 网络会朝着那些ungrounded variables的方向更新，并且受到bootstrap的影响, 这将传播到其他state。同时，准备好的行动也存在一定的噪声，其中的行动并不是真正的行动，为了避免上面提到的问题，这里用了大间距分类损失作为监督学习的loss:
$$
J_{E}(Q)=\max _{a \in A}\left[Q(s, a)+l\left(a_{E}, a\right)\right]-Q\left(s, a_{E}\right)
$$
$a_{E}$ 是expert的示范动作; $l\left(a_{E}, a\right)$ 是margin function, $a=a_{E}$ 的时候为0, 其余为正值。
**与以往的imitation learning有很大的不同，这里学的是action的Q值, 而不是单纯的模仿 action**。这个loss迫使agent的动作的Q值至少比示范动作Q值低一个margin。引入了这个loss就可以使未发生的动作的value确定为合理的value, 并使greedy的policy能够受到这个模仿了示范者的value function的引导。

权衡一下，利用SL与RL中的Bellman equation, **将专家数据看成一个软的初始化约束**, 在pretraining的时候, 约束专家数据中的action要比这个state下其他的action高一个 $l$ 值。这里其 实是做了一个loss的权衡:  **这个 $l$ 值导致的action差别的loss高，还是不同action导致达到下个状态的 $s^{\prime}$ 的产生的loss高，如果专家的SL带来的Ioss高的话，那么以专家的loss为主, 如果是 RL的loss高的话, 那么这个约束就会被放宽, 选择RL中的action**。简要来说，如果最终选出的动作与专家动作不同，说明其他某个动作的价值至少不会弱于专家动作太多，这样对模型的约束相对来说是一个比较合理的约束。

第一项与第二项中1步Q-learning loss用于约束连续状态之间的value并使学习到的Q网络满足Bellman方程。n步Qlearning loss用于将 expert的trajectory传递到更早的状态, 以实现更好地pre-training，这是因为价值需要一定的时间的训练才能跳过前面波动比较大的时期，进入相对平稳的更新时期，所以我们不仅使用下一时刻的回报，而且还将此后更多时刻的回报加入目标值中，使模型更新更快。
$$
r_{t}+\gamma r_{t+1}+\ldots+\gamma^{n-1} r_{t+n-1}+\max _{a} \gamma^{n} Q\left(s_{t+n}, a\right)
$$
第四项目标函数用来防止模型过拟合。

除此之外，算法还有一些其他的设定：

- 由于为**预训练准备的样本质量比较高**，我们可以对其进行反复利用，因此它们在训练过程中不会被换出来，全部永久存在于 Replay Buffer 中。
- 同样，由于准备数据和交互数据的来源和质量不同，因此在从 Replay Buffer 中随机抽取样本时，准备数据和交互得到的数据拥有不同的采样概率; 模型一开始只接受准备数据的学习，不进行模型模拟采样，这样确保了模型前期的快速成长。
- 算法也使用了常见的 Target Network、Priority Replay Buffer 等技巧。

完整的算法流程图如下：

![image-20210225194421748](https://raw.githubusercontent.com/Yunhui1998/markdown_image/main/RL/image-20210225194421748.png)

##### Distributional DQN：A Distributional Perspective on Reinforcement Learning

###### 输出价值分布形式

我们这次先介绍论文：A Distributional Perspective on Reinforcement Learning。这篇论文开创了一个新的方向将我们的“值函数”扩展到值分布。

让我们先回忆Bellman 公式的形式:
$$
\begin{aligned}
q(s, a) &=r(s, a)+\sum_{s^{\prime}} p\left(s^{\prime} \mid s, a\right) \sum_{a^{\prime}} \pi\left(a^{\prime} \mid s\right) \gamma q\left(s^{\prime}, a^{\prime}\right) \\
&=r(s, a)+\sum_{s^{\prime}} p\left(s^{\prime} \mid s, a\right) \gamma E_{a^{\prime} \sim \pi\left(a^{\prime} \mid s\right)}\left[q\left(s^{\prime}, a^{\prime}\right)\right] \\
&=r(s, a)+\gamma E_{s^{\prime}, a^{\prime} \sim p, \pi}\left[q\left(s^{\prime}, a^{\prime}\right)\right]
\end{aligned}
$$
由此可见，目前模型的估计结果是基于策略和状态转移两个概率分布的**价值期望**,由于综合了太多的信息，**甚至不同的distribution可以得到相同的期望，**这个数值就显得有些不够具体，如果想要得到更详细的信息，那么模型就不能仅输出一个期望值，而需要将价值的分布估计出来如下图所示：

![image-20210225224357331](https://raw.githubusercontent.com/Yunhui1998/markdown_image/main/RL/image-20210225224357331.png)

为了确保分布不受太多的限制，同时又可以减少分布的计算量，我们选择一种简单直观的方法：直方图。假设绝大多数的价值最终值落在区间 $\left[V_{\min }, V_{\max }\right]$ 之内，同时限定区间内每一段的价值范围相等，那么我们只需让模型输出给定值函数区间的概率, 也就是 $P(V),$ 就可以表示出直方图分布的形式。这样我们的价值分布估计转变如下，这样我们就可以用直方图来近似很多不太常规的分布形式，distributional DQN除了获得能获得更多的信息外，它还存在一个问题，就是它可能会under-estimate，因为我们在用distributional DQN的时候，会把reward限制在一定范围之内，那其实有可能那些极端高的reward就被去掉了，当然极端低的reward也可能被去掉了？

![image-20210225225151575](https://raw.githubusercontent.com/Yunhui1998/markdown_image/main/RL/image-20210225225151575.png)

其中每一小段的间隔为：

​                                                                                                         $$\Delta z=\frac{V_{\mathrm{MAX}}-V_{\mathrm{MIN}}}{N-1}$$

 

于是直方图中表示的数值采样集合为 $\left\{\boldsymbol{z}_{i}=V_{\mathrm{MIN}}+i \Delta \boldsymbol{z}: 0 \leqslant i<N\right\},$ 这个集合共有 $N$ 个元素。我们的模型要输出 $N$ 个值的向量，每一个值表示这其中一个价值采样点出现的概率，于是模型可以用下面的映射表示
$$
\theta: X \times A \rightarrow R^{N}
$$
不同于 DQN 输出实数，直方图模型会通过 Softmax 层输出每一个价值采样点的概率，这样可以得到价值为 $z_{i}$ 时, 对应的概率为 $p_{i}(x, a)=\frac{\mathrm{e}^{\theta_{i}(x, a)}}{\sum_{i} e^{\theta_{j}(x, a)}}$。

###### 分布更新

解决了输出的表示问题, 还要解决另一个十分重要的问题, 那就是更新。在前面的 DQN 模型中，我们可以使用 Q-Learning 公式计算目标价值，得到的结果为目标期望价值; 而在 Distributional DQN 模型中，我们的计算过程则变得有些复杂：我们如何定义损失函数，或者说我们怎么定义两个分布之间的距离，一般我们会使用使用Wasserstein Metric：
                                                                                                            $d_{p}(F, G):=\inf _{U, V}\|U-V\|_{p}$
这样可以保证我们的Bellman operator是 $\gamma-$ contraction, 也就是可以收敘到唯一的不动点。但是对于Q-learning用到的optimality operator, 并没有这样的理论保证。更槽糕的是, 在实际训练中, SGD是没办法保持这个Wasserstein Metric的。于是, 作者们提出了一个启发式的方法: 干脆用KL散度去衡量两个分布的距离。
我们之前介绍过了KL散度离散形式下的计算，对于分布 $p, q,$ 它们的KL散度定义为
                                                                                                        $\mathrm{KL}(p \| q)=\int p(x) \log \frac{p(x)}{q(x)} d x$              
对于离散的情况 , 我们还可以展开
                                                                                  $\mathrm{KL}(p \| q)=\sum_{i=1}^{N} p\left(x_{i}\right) \log \frac{p\left(x_{i}\right)}{q\left(x_{i}\right)}=\sum_{i=1}^{N} p\left(x_{i}\right)\left[\log p\left(x_{i}\right)-\log q\left(x_{i}\right)\right]$

这样我们就可以得到我们的计算过程如下：
( 1 ) 计算 $t+1$ 时刻每一个行动的价值分布。
( 2 ) 将第 1 步的价值期望和 $t+1$ 时刻的回报相加，并选出期望最大的行动。
( 3 ) 将期望最大的行动的价值分布表示出来。
我们发现得到的分布可能和原始的分布形式不同。不光是价值范围不同，价值采样点也会发生偏移，这样目标价值分布和估计分布将很难比较。为了简化计算，我们可以选择一种近似计算的方法，那就是将价值范围和采样点对齐。只要对齐了直方图，就可以使用之前的目标函数进行计算。

###### 分布对齐

![image-20210225231649634](https://raw.githubusercontent.com/Yunhui1998/markdown_image/main/RL/image-20210225231649634.png)

在对齐前，我们要先计算出最优的行动。由于价值的范围和采样点已经确定，那么 对于不同的行动，可以计算出每一个价值采样点 $z_{i}$ 的概率 $p\left(z_{i}\right),$ 这样也可以求出当前行动 $a$ 的价值期望:
$$
q(s, \boldsymbol{a})=\sum_{i} p\left(\boldsymbol{z}_{i} \mid \boldsymbol{s}, \boldsymbol{a}\right) \boldsymbol{z}_{i}
$$
我们从所有的行动中选择使期望最高的行动，可以得到
$$
a^{*}=\operatorname{argmax}_{a} q(s, a)
$$
然后我们可以开始对齐的计算了。如上图所示，计算过程分为三步：

- 计算出真实的目标价值分布。由于价值采样点是确定的，所以我们可以直接使用公式计算出目标价值的价值点，即进行下面的计算:

$$
\boldsymbol{z}_{i}^{\prime}=r_{t}+\gamma_{t} z_{i}
$$

​        由于每个采样点是固定的，所以这一步对于每一步更新来说都是相同的。每一个新的价值点的概率和旧的价值点相同，也就是说 $p\left(z_{i}^{\prime} \mid s, a^{*}\right)=p\left(z_{i} \mid s, a^{*}\right)$

- 完成价值限定的操作。其中落入价值范围之外的值将被强制投影到范围之内。大于最大值的价值采样点会被投影到最大价值处，小于最小值的价值采样点会 被投影到最小价值处。
- 第完成价值采样点的投影。最终我们需要把价值采样点 $z_{i}^{\prime}$ 投影回原始的采 样点 $z_{i \circ}$ 这由于第 2 步我们已经将采样点 $z_{i}^{\prime}$ 限定在价值范围内，**所以就可以找到离采样点 $z_{i}^{\prime}$ 最近的两个原始采样点 $z_{i},$ 那么下一步就可以以采样点到原始采样点的距离为权重将采样点 $z_{i}^{\prime}$ 的概率分配到原始采样点上，**这样就得到了 基于原始采样点的目标概率。

完整的更新算法如下图：

![image-20210225232152189](C:\Users\yunhu\AppData\Roaming\Typora\typora-user-images\image-20210225232152189.png

##### Distributional DQN：Distributional Reinforcement Learning with Quantile Regression

首先我们来介绍一个数学概念, Wasserstein距离。
Wasserstein距离度量两个概率分布之间的距离, $\quad$ (狭义的) 定义如下：
$W(P, Q)=\min _{\gamma \in \Pi} \sum_{x_{p}, x_{q}} \gamma\left(x_{p}, x_{q}\right)\left\|x_{p}-x_{q}\right\|$
直接看这个式子可能过于抽象了，因为它和我们熟悉的度量不一样, 它好像不是确定性的，而是带有一个 $\min$ 。
这里的Wasserstein距离又叫推土机距离，看下面的图，你就能很形象地理解Wasserstein距离：

![微信截图_20210226095823](https://raw.githubusercontent.com/Yunhui1998/markdown_image/main/RL/微信截图_20210226095823.png)

它的意思是, 将一个分布转变为另一个分布，所需要移动的最少的“土" 的量。
注意，因为是分布，概率的和为1，也就是说 “土" 的总量是相同的。同时，这个移动的量是指 "土" 的“距离*数量" 。
可以看到，又很多种移动的方案, **而Wasserstein距离指的是最少的那种**，当然可能有多个方案都 是最少, 但是这不重要, 重要的是移动的值。

然而上述的定义只算是一个特例，标准的Wasserstein Metric的定义更为复杂, 如果我有两个分布 $U, Y$, 那么它们的p-Wasserstein Metric为
                                                                                   $W_{p}(U, Y)=\left(\int_{0}^{1}\left|F_{Y}^{-1}(\omega)-F_{U}^{-1}(\omega)\right|^{p} d \omega\right)^{1 / p}$
其中
                                                                                   $F_{Y}^{-1}(\omega):=\inf \left\{y \in \mathbb{R}: \omega \leq F_{Y}(y)\right\}$
                                                                                   $F_{Y}(y)=\operatorname{Pr}(Y \leq y)$
当 $p=1$ 的时候, 上面的公式就退化成为我们最开始看到的推土机距离。
当 $p=1$ 的时候这个式子还是容易理解的, 这里的 $F_{Y}(y)$ 就是 $y$ 的CDF函数, 而 $F_{Y}^{-1}(\omega)$ 可以理解为计算 $P_{Y}$ 的 $w$ 分位数。
而 $W_{p}(U, Y)$ 的表达式, 则是将这个代表分位数的 $w$ 从 0 到 1 积分。

下图形象的描述了 ![[公式]](https://www.zhihu.com/equation?tex=p%3D1) 情况下的Wasserstein Metric，这不过这个定义是**连续**的，刚才的定义是**离散**的。

![2](https://raw.githubusercontent.com/Yunhui1998/markdown_image/main/RL/2.jpg)

上图中红色和蓝色的线分别是 $P_{X}$ 和 $P_{Y}$ 的CDF函数，对于某一个分位数 $\tau$, 我们可以计算得到两个值, 分别是 $F_{X}^{-1}(\tau)$ 和 $F_{Y}^{-1}(\tau)$ 。
它们的差值的绝对值就是上图中黑线的长度，**这个长度积分就是青色部分的面积, 这就代表了两个分布的差异。**

我们在上一篇博客中提到一开始作者们并没有想到合适的方法模拟 Wasserstein Metric这个过程, 于是提出了使用KL散度做近似的想法。紧接着作者们又提出了更 "正统" 的算法QR-DQN，它继承了最开始的理论想法。 首先，我们要做的是改变 "分布" 的表现形式：
$Z(x, a)$ 是 $Z: \mathcal{X} \times \mathcal{A} \rightarrow \mathscr{P}(\mathbb{R})$ 的函数, 它的输出是一个分布 $\mathscr{P}(\mathbb{R}) $
我们一开始是用 $N$ 个atoms $\left\{z_{0}, z_{1}, \cdots, z_{N-1}\right\}$ 作为基准，再用 $N$ 个离散的分布 $\left\{p_{0}, p_{1}, \cdots, p_{N-1}\right\}$ 来描述这个分布。这种形式用来计算KL散度是极好的，但是不适合计算Wasserstein Metric度量，现在我们介绍另外一种，是用分位数描述的方法。其实也很直觉，就是按照这个分布的CDF的 $y$ 轴, 把它均等的分成 $N$ 分, 例如下面的是分布的 PDF的 $y$ 轴, 我们把它分成10等分：

![image-20210226103741126](https://raw.githubusercontent.com/Yunhui1998/markdown_image/main/RL/image-20210226103741126.png)

那么自然就会得到10个 $\hat{\tau},$ 这10个 $\hat{\tau}$ 就定义了10个分位数
                                                                              $\hat{\tau}_{i}=\frac{2(i-1)+1}{2 N}, \quad i=1, \ldots, N$
分位数是下图的小红点：

![3](https://raw.githubusercontent.com/Yunhui1998/markdown_image/main/RL/3.jpg)

于是，我们现在只需要记录 $N$ 个分位数的位置, 就可以描述整个分布了。
接下来，我们解决如何去学习出这 $N$ 个分位数这个问题：
我们设计一个神经网络 $Z$ ，它的输入是状态 $s,$ 输出是一个矩阵, 矩阵的每 一行代表一个动作的 $N$ 个概率, 分别是 $\left\{p_{0}, p_{1}, \cdots, p_{N-1}\right\}$ 。

在QR-DQN中，神经网络也是输出一个矩阵, 只不过每列不再是atoms对应的 $p_{i}$ 了，**而是atoms的位置**, 也就是 $z_{i},$ 因为在QR-DQN中atoms的概率是确定的, 都是 $\frac{1}{N} $。 现在让我们看一下训练的过程。
首先我们从Buffer中采样出 $\left(s, a, r, s^{\prime}\right),$ 接下来我们需要计算出 $a^{*},$ 和上一篇博客的想法一样, 我们依旧用 $Q(s, a)$ 来计算。
先算                                   $Q\left(s^{\prime}, a^{\prime}\right)：$$Q\left(s^{\prime}, a^{\prime}\right):=\sum_{j} q_{j} \theta_{j}\left(x^{\prime}, a^{\prime}\right)$
挑出最大的作为 $a^{*}$：$a^{*} \leftarrow \arg \max _{a^{\prime}} Q\left(x, a^{\prime}\right)$
根据这个 $a^{*}$ 计算出分布 $Z\left(s^{\prime}, a^{*}\right)$, 我们设这个分布的atoms的位置表示为 $\left\{\theta_{0}^{\prime}, \theta_{1}^{\prime}, \ldots, \theta_{N-1}^{\prime}\right\}$

那么目标分布表示为：$\mathcal{T} \theta_{j}^{\prime}=r+\gamma \theta_{j}^{\prime}, \quad i=0, \ldots, N-1$
**这里的好处是不用再对齐了，因为我们的atoms的位置是可以改变的**，而正是用这个变量来描述整个分布, 自然没有对齐之说。
最关键的是，我们要让分布 $Z(s, a)$ 和目标分布 $r+\gamma Z\left(s^{\prime}, a^{*}\right)$ 尽可能相似。
我们假设用 $\left\{\theta_{0}, \theta_{1}, \ldots, \theta_{N-1}\right\}$ 来描述分布 $Z(s, a),$ 这其实就是 $N$ 个分位数。
那么描述目标分布的 $\left\{r+\gamma \theta_{0}^{\prime}, r+\gamma \theta_{1}^{\prime}, \cdots, r+\gamma \theta_{N-1}^{\prime}\right\}$ 就可以当作ground truth, 也就是把他们看作 $L_{\tau}=\mathbb{E}\left[\rho_{\tau}^{1}\left(y_{i}-\xi\left(x_{i}, \beta_{\tau}\right)\right)\right]$ 中不同的 $y_{i}$ 。此外，我们并不是只有一个 $\tau$, 我们有 $N$ 个 $\tau$, 我们需要计算它们的损失函数的和, 也就是
$$
\begin{aligned}
L_{\beta} &=\sum_{i=1}^{N} \mathbb{E}_{Y}\left[\rho_{\tau_{i}}^{1}\left(Y-\xi(\beta)_{i}\right)\right] \\
&=\sum_{i=1}^{N} \mathbb{E}_{\mathcal{T} Z^{\prime}}\left[\rho_{\hat{\tau}_{i}}^{1}\left(\mathcal{T} Z^{\prime}-\theta_{i}\right)\right] \\
&=\frac{1}{N} \sum_{i=1}^{N} \sum_{j=1}^{N}\left[\rho_{\hat{\tau}_{i}}^{1}\left(\mathcal{T} \theta_{j}^{\prime}-\theta_{i}\right)\right]
\end{aligned}
$$
其中
$$
\mathcal{T} Z^{\prime}=r+\gamma Z\left(x^{\prime}, a^{*}\right)
$$
而 $\hat{\tau}_{i}$ 就是用来决定 $N$ 个分位数的值
$$
\hat{\tau}_{i}=\frac{2 i+1}{2 N}, \quad i=0, \ldots, N-1
$$
最终的算法如下：

![image-20210226104445132](https://raw.githubusercontent.com/Yunhui1998/markdown_image/main/RL/image-20210226104445132.png)

最终算法的效果如下：

![image-20210226104423027](https://raw.githubusercontent.com/Yunhui1998/markdown_image/main/RL/v2-3679d55e0ca88163392460721c8eb945_720w.jpg)

可见最终算法效果很好。

接下来我们来直观的感受一下QR-DQN做了什么，（下面的内容都来自于一个大佬的[博客](https://zhuanlan.zhihu.com/p/138091493)）假设我们现在可以画出 ![[公式]](https://www.zhihu.com/equation?tex=r+%2B+\gamma+Z(s^\prime%2Ca^*)) ，它就是下图的红线：

![4](https://raw.githubusercontent.com/Yunhui1998/markdown_image/main/RL/4.jpg)

那么蓝色的线就是初始状态的 ![[公式]](https://www.zhihu.com/equation?tex=Z(s%2Ca)) 。

从Distributional DQN定义的角度，我们希望什么呢？我们希望**青色的面积更小**。

那么QR-DQN希望什么呢？

它希望 ![[公式]](https://www.zhihu.com/equation?tex=\theta_0) 可以可以作为 ![[公式]](https://www.zhihu.com/equation?tex=\tau_0) 对应的分位数， ![[公式]](https://www.zhihu.com/equation?tex=\theta_1) 可以可以作为 ![[公式]](https://www.zhihu.com/equation?tex=\tau_1) 对应的分位数，以此类推。

如果做到了这一点，图像就会变成

![img](https://raw.githubusercontent.com/Yunhui1998/markdown_image/main/RL/v2-3679d55e0ca88163392460721c8eb945_720w.jpg)

Amazing啊，青色这不就变少了。

如果我们调节超参数 ![[公式]](https://www.zhihu.com/equation?tex=N) ，让QR-DQN的分布描述的更细致，会变成

![img](https://raw.githubusercontent.com/Yunhui1998/markdown_image/main/RL/v2-de46f31addd45ac85a6e2d2089c65f03_720w.jpg)

青色更少了。这就是DQ-DQN在做的事情。

##### Distributional DQN：Implicit Quantile Networks for Distributional Reinforcement Learning

我们在对比value-based和policy-based的方法的时候，都会提到说value-based的方法只能做出确定性的策略，而policy-based的方法可以做出动态的策略，这是因为我们用来评价每一个 $s, a$ 的信息只有 $Q(s, a)$,也就是 $\mathbb{E}[Z(s, a)],$ 连方差都没有, 自然也只能做出动态的策略。

而Risk-Sensitive强化学习指的是我们在针对相同的 $Z(s, a)$ 分布时, 根据不同的偏好, 应该做出不同的动作。我们把对待风险两种不同的态度成为risk-averse和risk-seeking, 接下来，我们用一种正式的数学 语言来描述它们。描述这种偏好的公理被成为独立性，它有两个版本。

- 版本一
  如果有两个随机变量 $X, Y$, 我们相比 $Y$ 更偏好 $X$, 写作 $X \succ Y$, 那么这代表对任何随 机变量 $Z, X$ 和 $Z$ 的混合都优于 $Y$ 和 $Z$ 的混合, 这种 “优于" 表示为
  $\alpha F_{X}+(1-\alpha) F_{Z} \geq \alpha F_{Y}+(1-\alpha) F_{Z}, \forall \alpha \in[0,1]$
  在这种情况下，我们可以找到一个效用函数 $U$ 来描述这种偏好，那么策略可以表示为
  $\pi(x)=\underset{a}{\arg \max } \underset{Z(x, a)}{\mathbb{E}}[U(z)]$

- 版本二

  如果有两个随机变量 $X, Y$, 我们相比 $Y$ 更偏好 $X$, 写作 $X \succ Y$, 那么这代表对任何随 机变量 $Z, X$ 和 $Z$ 的混合都优于 $Y$ 和 $Z$ 的混合, 这种 “优于" 表示为
  $\alpha F_{X}^{-1}+(1-\alpha) F_{Z}^{-1} \geq \alpha F_{Y}^{-1}+(1-\alpha) F_{Z}^{-1}, \forall \alpha \in[0,1]$
  在这种情况下，我们可以找到一个distortion risk measure $h$ 来描述这种偏好，那么策略可以表示为
  $\pi(x)=\underset{a}{\arg \max } \int_{-\infty}^{\infty} z \frac{\partial}{\partial z}\left(h \circ F_{Z(x, a)}\right)(z) d z$

可以证明, 这两种表示是可逆的, 也就是哪个方便按哪个来。
举个例子, 在第一个版本中, 如果 $U(x)=x,$ 那么策略就会变成
$$
\pi(x)=\underset{a}{\arg \max } \underset{Z(x, a)}{\mathbb{E}}[z]=\underset{a}{\arg \max } Q(x, a)
$$
在第二个版本中, 如果 $h(x)=x ，$ 那么策略就会变成
$$
\begin{aligned}
\pi(x) &=\underset{a}{\arg \max } \int_{-\infty}^{\infty} z \frac{\partial}{\partial z} F_{Z(x, a)}(z) d z \\
&=\underset{a}{\arg \max } \int_{-\infty}^{\infty} z P_{Z(x, a)}(z) d z \\
&=\underset{a}{\arg \max } \underset{Z(x, a)}{\mathbb{E}}[z] \\
&=\underset{a}{\arg \max } Q(x, a)
\end{aligned}
$$
不难证明, $\int_{0}^{1} F_{Z}^{-1}(\tau) d \beta(\tau)=\int_{-\infty}^{\infty} z \frac{\partial}{\partial z}\left(\beta \circ F_{Z}\right)(z) d z$
$$
\begin{array}{l}
\text { 令 } z=F_{Z}^{-1}(\tau) \\
\int_{0}^{1} F_{Z}^{-1}(\tau) d \beta(\tau) \stackrel{z=F_{Z}^{-1}(\tau)}{=} \int_{-\infty}^{\infty} z d \beta\left(F_{Z}(z)\right) \\
=\int_{-\infty}^{\infty} z \frac{\partial}{\partial z}\left(\beta \circ F_{Z}\right)(z) d z
\end{array}
$$
其中 $\beta$ 是一个 $[0,1] \rightarrow[0,1],$ **被称为distortion risk measure**, 我们定义基于 $\beta$ 的 distorted expectation
$$
Q_{\beta}(x, a):=\underset{\tau \sim U([0,1])}{\mathbb{E}}\left[Z_{\beta(\tau)}(x, a)\right.
$$
其中 $Z_{\tau}:=F_{Z}^{-1}(\tau),$ 显然
$$
Q_{\beta}(x, a):=\underset{\tau \sim U([0,1])}{\mathbb{E}}\left[Z_{\beta(\tau)}(x, a)\right]=\int_{0}^{1} F_{Z}^{-1}(\tau) d \beta(\tau)
$$
这就把 $Q_{\beta}(x, a)$ 和前面的风险偏好联系起来了。
最后，策略可以表示为：
$$
\begin{aligned}
\pi_{\beta}(x) &=\underset{a \in \mathcal{A}}{\arg \max } \int_{-\infty}^{\infty} z \frac{\partial}{\partial z}\left(\beta \circ F_{Z}\right)(z) d z \\
&=\underset{a \in \mathcal{A}}{\arg \max } \int_{0}^{1} F_{Z}^{-1}(\tau) d \beta(\tau) \\
&=\underset{a \in \mathcal{A}}{\arg \max }_{\tau \sim U([0,1])}\left[Z_{\beta(\tau)}(x, a)\right] \\
&=\underset{a \in \mathcal{A}}{\arg \max } Q_{\beta}(x, a)
\end{aligned}
$$
接下来再让我们看看不同的 $\beta$ 就起到什么不同的效果。
整理而言，当 $\beta$ 为凸函数时，偏好是risk-averse的，当 $\beta$ 为凹函数时，偏好是risk-seeking 的。
有一些现成的函数可以作为 $\beta$

- CPW函数：
  $\operatorname{CPW}(\eta, \tau)=\frac{\tau^{\eta}}{\left(\tau^{\eta}+(1-\tau)^{\eta}\right)^{\frac{1}{\eta}}}$

- Wang函数：

  (其中 $\Phi$ 是标准正态分布的CDF函数)
  $\operatorname{Wang}(\eta, \tau)=\Phi\left(\Phi^{-1}(\tau)+\eta\right)$

- Pow函数：
  $\operatorname{Pow}(\eta, \tau)=\left\{\begin{array}{ll}\tau^{\frac{1}{1+|\eta|}}, & \text { if } \eta \geq 0 \\ 1-(1-\tau)^{\frac{1}{1+|\eta|}}, & \text { otherwise }\end{array}\right.$

- conditional value-at-risk函数：
  $\mathrm{CVaR}(\eta, \tau)=\eta \tau$

  这些函数的 $\eta$ 都可以看作是超参数, 而 $\tau$ 则是自变量, 例如$\beta(\tau)=\operatorname{Wang}(.75, \tau)$

下面是有关这些函数的图像

![image-20210227153634361](C:\Users\yunhu\AppData\Roaming\Typora\typora-user-images\image-20210227153634361.png)

第二列的Neutral是原始的 $Z(s, a)$ 的分布, 而其他列的图像都是经过加工后的 $Z_{\beta}(s, a)$ 的图 像。
可以看到, 这些不同的 $\beta$ 有些对风险比较积极, 例如 $\operatorname{Wang}(.75)$, 而有些则很保守, 只集中 在原分布中值比较大的部分，例如 $\mathrm{CPW}(.71)$ 。
最后，让我们步入正题, 看看IQN是怎么训练的。
下面的图很好的描绘来DQN, C51，QR-DQN和IQN的区别。

![image-20210227153751854](https://raw.githubusercontent.com/Yunhui1998/markdown_image/main/RL/image-20210227153751854.png)

C51, QR-DQN和IQN都是想去学习一个分布, 但是它们的方式并不一样：

C51和QR-DQN是去找到了一种间接的方式去表示这个分布，也就是用atoms的方式。
而IQN更像是直接的学出了这个分布。

IQN的输入和输出是什么呢?
它是输入是状态 $s,$ 和采样 $\tau \sim U[0,1],$ 而输出和DQN很像, 是一个 $|\mathcal{A}|$ 维的向量。
区别在于, **DQN只能输出每个动作的期望, 而IQN可以根据输入的 $\tau$, 输出每个动作的 $\tau$ 分位数。**
这样看来, IQN和C51，QR-DQN的不同之处在于，它不在想办法表示这个分布，它直接就是这个 分布!
那这个前面提到的Risk-Sensitive强化学习有什么关系呢?
试想，如果我们可以学习出 $Z_{\tau}(x, a)$, 那么不就可以计算出对于任何 $\beta$ 的 $Q_{\beta}(x, a)$ 了吗? 这样我们就可以在作出决定的时候根据我们的偏好，而不是只能根据期望 $Q(x, a)$ 去计算。

 这个网络怎么训练呢? 

首先还是从Buffer中拿到采样 $\left(s, a, r, s^{\prime}\right)$ 。

接下来我们要根据 $s^{\prime}$ 选出最好的动作 $a^{*}$ 。

但是, 这里我们不再用 $Z(x, a)$ 算出 $Q(x, a)$ 选择了，而是应该加入偏好 $\beta,$ 当然, 如果没 有特殊的偏好, 令 $\beta(x)=x$ 即可。

我们需要事先设定一个超参数 $K$, 用来决定计算 $Q_{\beta}(x, a)$ 的采样次数, 于是

$Q_{\beta}\left(x^{\prime}, a^{\prime}\right)=\frac{1}{K} \sum_{k}^{K} Z_{\tilde{\tau}_{k}}\left(x^{\prime}, a^{\prime}\right)$
其中

$\tilde{\tau}_{k} \sim \beta(\cdot)$
于是

$a^{*} \leftarrow \arg \max _{a^{\prime}} \frac{1}{K} \sum_{k}^{K} Z_{\tilde{\tau}_{k}}\left(x^{\prime}, a^{\prime}\right), \quad \tilde{\tau}_{k} \sim \beta(\cdot)$
接下来我们要缩短 $Z(x, a)$ 和 $r+\gamma Z\left(x^{\prime}, a^{*}\right)$ 这两个分布之间的距离。

但是我们现在没有某种表示去表示这个分布了，我们的网络就是分布本身。
因此我们需要从网络中采样，来估计这两个分布。我们又引入两个超参数 $N, N^{\prime},$ 分别代表估 计这两个分布所需要的采样次数，于是有
$\tau_{i}, \tau_{j}^{\prime} \sim U([0,1]), \quad 1 \leq i \leq N, 1 \leq j \leq N^{\prime}$
对于两个单独的 $\tau_{i}, \tau_{j}^{\prime},$ 它们之间的差表示为
$\delta_{t}^{\tau_{i}, \tau_{j}^{\prime}}=r_{t}+\gamma Z_{\tau_{j}^{\prime}}\left(x_{t+1}, \pi_{\beta}\left(x_{t+1}\right)\right)-Z_{\tau_{i}}\left(x_{t}, a_{t}\right)$
那么总的差值就是
$\mathcal{L}\left(x_{t}, a_{t}, r_{t}, x_{t+1}\right)=\frac{1}{N^{\prime}} \sum_{i=1}^{N} \sum_{j=1}^{N^{\prime}} \rho_{\tau_{i}}^{\kappa}\left(\delta_{t}^{\tau_{i}, \tau_{j}^{\prime}}\right)$
我们没有用 $\left(\delta_{t}^{\tau_{i}, \tau_{j}^{\prime}}\right)^{2}$ 而是用 $\left|\delta_{t}^{\tau_{i}, \tau_{j}^{\prime}}\right|$ 这是因为我们本质上还是在做分位数回归, 而不是标准的回归。
最后, $\quad \rho_{\tau_{i}}^{\kappa}$ 表示的是绝对值函数的软化, 我们在上一篇博客中提到过：

最终的算法如下

![image-20210226110156088](https://raw.githubusercontent.com/Yunhui1998/markdown_image/main/RL/image-20210226110156088.png)

##### Noisy DQN

增强探索能力是强化学习中经常遇到的问题，在前面我们曾接触过 $\epsilon-$ greedy 的方 法，这个算法中以一定的概率 $\epsilon$ 随机执行行动，而在剩下的 $1-\epsilon$ 中执行最优行动，这相当于在执行策略的环节增加一定的噪声，使得模型具备一定的探索能力。

我们现在介绍另一种增强探索能力的方法: Noisy Network，来自论文 Noisy Networks for Exploration。不同于 $\epsilon$ -greedy 的方法，它使用了一种更平滑的手段增加探索能力。那么，算法是如何实现想要的效果的呢?我们以一个简单的函数为例，来看看它的
效果。这个函数的形式为
$$
y=w x+b
$$
其中 $x$ 表示输人、y 表示输出， $w$ 和 $b$ 是函数的参数。也就是说，如果输人是 $x,$ 那 么经过函数的变换，输出的结果就会变成 $y_{\circ}$ 这样，我们就可以使用这个函数表示自然界中一组 $x$ 和 $y$ 的关系。当然，自然界中存在着一定的噪声，我们无法直接使用这个函数进行表示，于是给函数加一个噪声项，于是函数变为
$$
y=w x+b+\epsilon
$$
其中 $\epsilon$ 服从均值为 0 , 方差为 $\sigma^{2}$ 的高斯分布
$$
\epsilon \sim N\left(0, \sigma^{2}\right)
$$
$\sigma$ 是一个固定值，表示噪声带来的方差。这样我们也可以认为 $y$ 服火如下的高斯分布：
$$
y \sim N\left(w x+b, \sigma^{2}\right)
$$
可以看出由于噪声的存在，我们可以从同一个 $x$ 映射到多个 $y$, 这相当于增加了输 出的不确定性。不确定性对于探索来说十分重要，由于不确定性的存在，我们可以选择确定行动之外的其他行动，因此我们发现噪声和探索存在某些类似的特性，我们可以利用噪声增加模型的探索能力。
**一种添加噪声的方法是在参数上增加噪声**。对于上面函数中的参数 $w$, 我们可以 定义参数来自均值为 $\mu_{w},$ 方差为 $\sigma_{w}$ 的高斯分布。同理，参数 $b$ 服从均值为 $\mu_{b},$ 方差 为 $\sigma_{b}$ 的高斯分布，这样函数就变成了下面的形式:

![image-20210307154351465](https://raw.githubusercontent.com/Yunhui1998/markdown_image/main/RL/image-20210307154351465.png)
$$
\begin{array}{l}
\tilde{w} \sim N\left(\mu_{w}, \sigma_{w}\right) \\
\tilde{b} \sim N\left(\mu_{b}, \sigma_{b}\right) \\
y=\tilde{w} x+\tilde{b}
\end{array}
$$
这个形式理解起来并不难，前向计算也并不困难，但是反向计算却有点困难，如何将得到的反向梯度传递到高斯分布中的分布参数呢?为了简化计算，我们需要将参数的表现形式做一定的变换，变成固定部分和随机部分的和，形式如下所示：
$$
\begin{array}{l}
\tilde{w}=\mu_{w}+\sigma_{w} \epsilon \\
\tilde{b}=\mu_{b}+\sigma_{b} \epsilon
\end{array}
$$

$$
\begin{array}{l}
\tilde{w}=\mu_{w}+\sigma_{w} \epsilon \\
\tilde{b}=\mu_{b}+\sigma_{b} \epsilon
\end{array}
$$

$\epsilon$代表参数中的随机部分，它不属于参数, 服从确定的统计分布，例如均值为 $0,$ 方差为 1 的高斯分布 $N(0,1)$ 。在完成采样后，它可以被当成一个常量对待，这样另外两 个参数就可以使用前向后向计算优化了。**当然，上面这种方法也只是为模型添加噪声的方法之一，如果利用这种方法增加噪声，那么如果函数原本有 $N$ 个参数，为了实现噪声的效果，我们需要把参数数量增加一倍，对于小型网络来说，使用这样的方法添加噪声是可行的，但对较大的网络来说，增加一倍的网络参数会给计算带来不小的负担。**
为了减少噪声参数的数量，我们还可以从函数参数的结构入手。在我们熟悉的全连接运算中，参数 $w$ 一般是一个二维的矩阵，假设它的维度为 $p \times q,$ 那么我们可以**只生成 $p+q$ 个噪声参数**，**也就是把对每个权重加噪声换成对每个神经元加噪声，**于是对于上述函数中每一个参数，可以用下面这种全新的形式 表示:
$$
\boldsymbol{w}_{i, j}=\mu_{w}[i, j]+f\left(\sigma_{p}[i]\right) f\left(\sigma_{q}[j]\right) \epsilon
$$

$$
b_{j}=\mu_{b}[j]+f\left(\sigma_{q}[j]\right) \epsilon
$$

其中参数数值的第一项保持不变，后面一项的表示有些变化。我们将添加的 $p+q$ 个参数分成两部分，一部分的维度为 $p, \sigma_{p}[i]$ 表示其中的第 $i$ 个噪声参数; 另一部分的维度 为 $q, \sigma_{q}[j]$ 表示其中第 $j$ 个噪声参数。这里的 $f(x)=\operatorname{sgn}(x) \sqrt{x}$样的设定，我 们在噪声效果和噪声参数数量两方面得到了很好的平衡。对于更复杂的模型，我们也 可以采用类似的方法添加噪声参数来实现对噪声信息的拟合。

完成了对添加噪声基本思想的介绍，下面就来介绍这个噪声的参数如何融入我们已经介绍的 DQN 算法，我们知道基于 Target Network 的 DQN 算法的目标函数公式为
$$
L(\theta)=E_{\left(s_{t}, a_{t}, \boldsymbol{r}_{t}, \boldsymbol{s}_{t+1}\right) \sim D}\left[\boldsymbol{r}_{t}+\gamma \max _{\boldsymbol{a} * \in \boldsymbol{A}} Q\left(\boldsymbol{s}_{t+1}, \boldsymbol{a}^{*} ; \theta^{-}\right)-Q\left(s_{t}, \boldsymbol{a}_{t} ; \theta\right)\right]^{2}
$$
其中 $\theta$ 表示 Behavior Network 的模型参数, $\theta^{-}$ 表示 Target Network 的模型参数。我们可以在值函数中加人一定的噪声，由于噪声会影响最终的价值输出，也会影响最终的行动，于是噪声的大小影响了模型的探索特性，噪声越小表示探索能力越小，噪声越大表示探索能力越大。我们可以为两个模型参数分别加入噪声随机变量 $\epsilon$ 和 $\epsilon^{-},$ 以及噪声参数 $\sigma$ 和 $\sigma^{-},$ 此时新的目标函数变为：
$$
\begin{aligned}
L(\theta)=& E_{\epsilon^{-}, \epsilon}\left[E_{\left(\boldsymbol{s}_{t}, \boldsymbol{a}_{t}, \boldsymbol{r}_{t}, \boldsymbol{s}_{t+1}\right) \sim D}\left[\boldsymbol{r}_{t}+\gamma \max _{\boldsymbol{a} * \in \boldsymbol{A}} Q\left(\boldsymbol{s}_{t+1}, \boldsymbol{a}^{*}, \epsilon^{-} ; \theta^{-}, \sigma^{-}\right)\right.\right.\\
&\left.\left.-Q\left(\boldsymbol{s}_{t}, \boldsymbol{a}_{t}, \epsilon ; \theta, \sigma\right)\right]^{2}\right]
\end{aligned}
$$
在原本的目标函数中噪声项并不存在，因此此时噪声的加入使得目标函数产生了偏差。为了消除这个偏差，我们可以对噪声求期望，由于噪声的期望值为 $0,$ 求解期望后目标函数不再有偏，但是模型依然拥有一定的探索能力。

最后，由于噪声的引人，我们需要考虑噪声参数的初始化。根据论文中的介绍，噪声参数为 $p+q$ 个时，令 $N=p \times q,$ 参数 $\mu$ 按照范围为 $\left[\frac{1}{\sqrt{N}},+\frac{1}{\sqrt{N}}\right]$ 的均匀分布进行初始化，参数 $\sigma$ 将初始化为常量 $\frac{0.4}{\sqrt{N}} $。

这里需要说明一下的是就是像$\epsilon-greedy$这种在动作空间上加噪声的方法和Noisy DQN这种在参数空间上加噪声的方法有什么差别呢？ $\epsilon-greedy$其实更像是一种随机地探索，也就是随机选一个动作，也就意味着下次碰到相同的状态的时候，$\epsilon-greedy$有可能会做出完全不一样的动作。但是Noisy DQN更像是一种有系统地探索，作者在论文里面提到了是一种“State-dependent Exploration”，也就是在同一个episode的情况下，noise是不变的，所以这种情况下下次遇到相同的动作就保证会做出相同的动作。

算法最终的结构如下：

![image-20210227155828839](https://raw.githubusercontent.com/Yunhui1998/markdown_image/main/RL/image-20210227155828839.png)

结果如下图所示：

![image-20210227155904294](https://raw.githubusercontent.com/Yunhui1998/markdown_image/main/RL/image-20210227155904294.png)

可见在DQN、Dueling、A3C中，加入noisy都有效提升了算法性能。

##### Q-learning with continuous actions

上面讲了DQN以及它的一些变体, 但是它们基本都立足于action space是discrete的，但是如果输出的action是continuous的，比如说控制机器人来控制机器人关节的角度。那么在求target value的时候取max操作就不太好进行了，这里其实有一个最直观的想法去解决这个问题，就是我们从动作空间中sample出n个动作，然后单独算出这n个动作的q值的大小，选择Q值最大的那个，这样看上去效率好像有点低，但是当我们可以并行地去处理这个问题时，其实速度并不会特别慢。在这里我们介绍三类解决这个问题的常见方法。

![image-20210226111950186](https://raw.githubusercontent.com/Yunhui1998/markdown_image/main/RL/image-20210226111950186.png)

- Optimizations
  首先是基于优化的方法，自然就会想到用SGD了，但是这个 max其实是一个内循环, 使用SGD实际上会显得有些慢。而考虑到action space通常都是低维的, 所以stochastic optimization似平成为一个不错的选择。
  最简单的思路是sampling的方法, 根据某种分布从连续值中sample出若干个点, 选择它们中的最大者, 这样只要选的点越多，那么结果也就越正确。它的好处是异常简单，而且容易并行, 但是由于计算量限制，采样点数往往不足，最终效果会略差，很容易陷入局部最优点，而且预算量很大。

  ![image-20210307194919722](https://raw.githubusercontent.com/Yunhui1998/markdown_image/main/RL/image-20210307194919722.png)

  而另外还有cross-entropy method以及CMA-ES两种iterative stochastic optimization方法, 最多可以在动作空间维数为40的情况下运行。

- Easily optimize function class
  第二大类则是引入一些容易取max的Q-function。例如在NAF(Normalized Advantage Functions)中，将网络的输出分成三部分，一个向量、一个矩阵、一个常数：

![img](https://raw.githubusercontent.com/Yunhui1998/markdown_image/main/RL/v2-18de211f35c0eab6b1f2de559635b846_720w.jpg)

此时max操作和argmax操作都可以用其中的head来表示：
$$
\begin{array}{c}
\arg \max _{a} Q_{\phi}(\mathbf{s}, \mathbf{a})=\mu_{\phi}(\mathbf{s}) \\
\max _{\mathbf{a}} Q_{\phi}(\mathbf{s}, \mathbf{a})=V_{\phi}(\mathbf{s})
\end{array}
$$
而一般的Q值则是也可以通过它们组合得到，也就是说我们先通过一个s得到这三个值，然后通过下面这个公式将这三个值转化成Q函数：
$$
Q_{\phi}(\mathbf{s}, \mathbf{a})=-\frac{1}{2}\left(\mathbf{a}-\mu_{\phi}(\mathbf{s})\right)^{T} P_{\phi}(\mathbf{s})\left(\mathbf{a}-\mu_{\phi}(\mathbf{s})\right)+V_{\phi}(\mathbf{s})
$$
这种方法的优点就是不对算法本身作出改动，没有添加inner loop的计算量, 效率保持一致。但是由于网络需要输出多个head，表达更多语意，会降低表达能力，需要更大网络。

![image-20210226112152637](https://raw.githubusercontent.com/Yunhui1998/markdown_image/main/RL/image-20210226112152637.png)

- Approximate maximizer
  第三种方式则是DDPG的方法, 单独train一个network去fitted argmax操作：

![image-20210226112255838](https://raw.githubusercontent.com/Yunhui1998/markdown_image/main/RL/image-20210226112255838.png)

![image-20210226112324810](https://raw.githubusercontent.com/Yunhui1998/markdown_image/main/RL/image-20210226112324810.png)

##### Rainbow

Rainbow一个集多长所长的模型，来自论文 Rainbow: Combining Improvements in Deep Reinforcement Learning，他主要包括我们前面提到的一下几个改进，我们这里稍微再回顾一下：

###### Double Q-Learning

Double Q-Learning 构建了两个结构相同但是参数数值不同的模型：Behavior Net-work 和 Target Network。在模型更新时，首先由 Behavior Network 选出 $t+1$ 时刻的最优行动，然后由 Target Network 得到估计 $t+1$ 时刻最优行动的目标价值估计。通过将这两个步骤解耦，我们可以减少 Q-Learning 方法对价值过高估计带来的影响，其中的核心公式为
$$
L(\theta)=\left(r_{t+1}+\gamma q_{\theta^{-}}\left(s_{t+1}, \operatorname{argmax}_{a^{\prime}} q_{\theta}\left(s_{t+1}, a^{\prime}\right)\right)-q_{\theta}\left(s_{t}, a_{t}\right)\right)^{2}
$$

###### Prioritized Replay Buffer

Prioritized Replay Buffer 通过对 Replay Buffer 中不同的样本赋子不同的权重，使得 模型有更高的概率训练对自己有更多提升的样本上, 同时以较低的概率训练对自己提升有限的样本。样本出现的权重和样本采样时的 TD-Error有关，样本的学习率和更新权重还可以根据参数进行调整。

###### Dueling Networks

Dueling Networks 将状态行动价值模型 $q(s, a)$ 分解成状态价值 $v(s)$ 和价值优势 $A(s, a)$ 两个部分，分解后的两部分具有明确的含义，而这样的分解也同时降低了训练的难度，其中的核心公式为

$$
q\left(s_{t}, a_{t}\right)=v\left(s_{t}\right)+\left(A\left(s_{t}, a_{t}\right)-\frac{1}{|A|} \sum_{a^{\prime}} A\left(s_{t}, a_{t}^{\prime}\right)\right)
$$

###### Multi-step Learning

前面介绍的 Q-Learning 大多通过下一时刻的回报和价值估计得到目标价值，这种方法在前期具有学习速度较慢的弱点，为了克服这个弱点，Multi-step Learning 使用了更多步的回报，这样在训练前期目标价值可以估计得更准确，从而加快训练速度，其中的核心公式为
$$
q^{\prime}\left(s_{t}, \boldsymbol{a}_{t}\right)=r_{t+1}+\gamma \boldsymbol{r}_{t+2}+\cdots+\gamma^{n-1} r_{t+n}+\max _{a} \gamma^{n} q\left(s_{t+n+1}, \boldsymbol{a}\right)
$$

###### Distributional Network

DQN 网络只输出了期望形式的价值，而对价值缺少更细致的刻画。Distributional Network 的模型结构可以输出价值的分布形式。我们可以设定价值模型可能的输出范围$\left[V_{\mathrm{MIN}}, V_{\mathrm{MAX}}\right],$ 并在范围内以直方图的形式表示价值对应的出现概率，这使模型的表现能力有了很大的提升，分布形式模型的表示形式如下所示：
$$
\begin{array}{l}
\left\{z_{i}=V_{\mathrm{MIN}}+i \Delta z: 0 \leqslant i<N\right\} \\
p_{i}(x, a)=\frac{\mathrm{e}^{\theta_{i}(\boldsymbol{x}, a)}}{\sum_{j} \mathrm{e}^{\theta_{j}(\boldsymbol{x}, \boldsymbol{a})}}
\end{array}
$$

###### Noisy Network

模型的探索能力一直是一个需要提高的方面。为了更优雅、灵活地提高模型的探索能力， Noisy Network 为模型参数加入了一定的噪声，通过噪声的随机性改变参数的 数值，进而改变模型输出的数值，对应的更新公式为
$$
\begin{aligned}
L(\theta)=& E_{\epsilon^{-}, \epsilon}\left[E_{\left(s_{t}, a_{t}, r_{t}, s_{t+1}\right) \sim D}\left[r_{t}+\gamma \max _{a * \in A} Q\left(s_{t+1}, a^{*}, \epsilon^{-} ; \theta^{-}, \sigma^{-}\right)\right.\right.\\
&\left.\left.-Q\left(s_{t}, a_{t}, \epsilon ; \theta, \sigma\right)\right]^{2}\right]
\end{aligned}
$$

###### 如何结合:

首先我们把TD的一步自举换成n步自举, 目的分布变成了
$d_{t}^{(n)}=\left(R_{t}^{(n)}+\gamma_{t}^{(n)} z, \quad \boldsymbol{p}_{\bar{\theta}}\left(S_{t+n}, a_{t+n}^{*}\right)\right)$
于是损失函数变成了
$D_{\mathrm{KL}}\left(\Phi_{z} d_{t}^{(n)} \| d_{t}\right)$
注意Rainbow是2017年提出的, 那时候人们还在用C51呢, 所以用KL散度衡量两个分布之间的差异不足为奇。
接下来, Prioritized experience replay排序的顺序也需要调整, 因为之前我们是依据TD-error排 序的, 现在因为不再使用Q值函数了, TD-error应该进化为KL散度了, 于是
$p_{t} \propto\left(D_{\mathrm{KL}}\left(\Phi_{z} d_{t}^{(n)} \| d_{t}\right)\right)^{\omega}$
最后是把Dueling DQN的结构和Distributional DQN的结构结合起来, 因为它们都是对网络结构进行了调整。
我们从Dueling DQN的结构出发, 原先是输出一个状态值函数 $V(s)$ 和一組动作的优势函鼓 $A(s, a),$ 现在我们需要输出分布了。
别忘了C51用的是固定的atoms位置, 可变的atoms概率表示分布。atoms的个数为 $N$
那么value stream $v_{\eta}$ 的输出应该是一个 $N$ 维的向量, 而advantage stream $a_{\psi}$ 的输出应该 是一个 $N \times|\mathcal{A}|$ 的矩阵, 最后每个atoms对应的概率可以表示头
$p_{\theta}^{i}(s, a)=\frac{\exp \left(v_{\eta}^{i}(\phi)+a_{\psi}^{i}(\phi, a)-\bar{a}_{\psi}^{i}(s)\right)}{\sum_{j} \exp \left(v_{\eta}^{j}(\phi)+a_{\psi}^{j}(\phi, a)-\bar{a}_{\psi}^{j}(s)\right)}$

最后，作者还比较了如果从Rainbow中去掉某个因素，模型的性能: 可以看出Prioritized replay和multi-step learning是最关键的两项技术，特别是multi-step learning。



![image-20210226125035964](https://raw.githubusercontent.com/Yunhui1998/markdown_image/main/RL/image-20210226125035964.png)

##### Practical tips for DQN

一些基本的tips：

- Q-learning对稳定性要求比较高，在编写算法的时候先在简单的环境上测试。

  ![image-20210226112448254](https://raw.githubusercontent.com/Yunhui1998/markdown_image/main/RL/image-20210226112448254.png)

- 探索的幅度在前期要大，随着训练进行，要慢慢降低。

- Replay Buffer越大，越有助于提高稳定性。

- 训练需要一定的时间，需要耐心。

一些选择上的tips：

- 由于Q的误差影响比较大，所以Bellman error会很大，进行gradient clip或者Huber loss都会有一定的帮助。

  ![image-20210226112508208](https://raw.githubusercontent.com/Yunhui1998/markdown_image/main/RL/image-20210226112508208.png)

- 这里Huber loss的系数也需要一定的调节，在DQN相关的作业中可以调整一下，初始值是1.0，不过调整下1.3似乎是一个更好的结果。

- Double Q-learning帮助很大，N-steps帮助也很大，但是会带来一些坏处。

- Adam Optimizer，Schedule exploration，Learning rates都会有帮助。

- 多个随机种子可能会带来惊喜。