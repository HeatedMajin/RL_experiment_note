[TOC]



## Q-learning

### 存在Q network和target Q network

- Q network 输出V[s]
- target Q network 输出V[s']
```
计算advantage的值：Advantage = r + gamma * V[s'] -V[s]
也可理解为regression问题：V[s] <-> r + gamma * V[s'] 
```
## 方案1：policy和value公用网络层，两个loss加起来

​		（见models/Dense_agent.py）

训练到最后，loss越来越大？难道说，这里loss越大，效果越好吗？

```angular2
Episode 10	 TOP@5's mean reward of the batch: 24.60	234.3179168701172
Episode 20	 TOP@5's mean reward of the batch: 26.40	239.0711669921875
Episode 30	 TOP@5's mean reward of the batch: 24.80	217.67605590820312
Episode 40	 TOP@5's mean reward of the batch: 41.20	383.6646728515625
Episode 50	 TOP@5's mean reward of the batch: 28.40	255.86346435546875
Episode 60	 TOP@5's mean reward of the batch: 27.80	268.3043518066406
......
Episode 2240	 TOP@5's mean reward of the batch: 10.20	144.5248565673828
Episode 2250	 TOP@5's mean reward of the batch: 9.80	103.40729522705078
Episode 2260	 TOP@5's mean reward of the batch: 9.80	112.94951629638672
Episode 2270	 TOP@5's mean reward of the batch: 10.40	106.0592269897461
Episode 2280	 TOP@5's mean reward of the batch: 10.20	117.04953002929688
Episode 2290	 TOP@5's mean reward of the batch: 10.00	91.89229583740234
Episode 2300	 TOP@5's mean reward of the batch: 10.20	118.0108642578125
Episode 2310	 TOP@5's mean reward of the batch: 10.40	124.95484924316406
Episode 2320	 TOP@5's mean reward of the batch: 11.00	157.35809326171875
Episode 2330	 TOP@5's mean reward of the batch: 9.60	74.11231994628906
Episode 2340	 TOP@5's mean reward of the batch: 10.00	162.55018615722656
Episode 2350	 TOP@5's mean reward of the batch: 10.00	112.04328155517578
Episode 2360	 TOP@5's mean reward of the batch: 10.00	135.82704162597656
......
Episode 34870	 TOP@5's mean reward of the batch: 155.00	20046.10546875
Episode 34880	 TOP@5's mean reward of the batch: 157.40	19974.64453125
Episode 34890	 TOP@5's mean reward of the batch: 139.40	19771.158203125
Episode 34900	 TOP@5's mean reward of the batch: 193.00	31595.37890625
Episode 34910	 TOP@5's mean reward of the batch: 176.00	27283.21875
Episode 34920	 TOP@5's mean reward of the batch: 184.60	30796.171875
Episode 34930	 TOP@5's mean reward of the batch: 195.60Solved! Episode reward is 195.6, the last episode runs to 200 time steps!

```

## 方案2：policy和value公用网络层，但梯度分别传递

​		（见models/Dense_agent2.py）

```
# 保存policy网路的所有变量梯度(name-str:grad-tensor)
self.policy_grads = {}

# 保存policy网路的所有变量（name-str:param-tensor）
self.policy_params = {}
for name,p in self.named_parameters():
	if "dense" in name or "action" in name:
        self.policy_params[name] = p
```

```
# 梯度清除
self.optimizer.zero_grad()

# 计算policy上的loss，计算梯度，保存变量的梯度
policy_loss_sum = torch.stack(policy_loss).sum()
policy_loss_sum.backward(retain_graph=True)
for name,param in self.policy_params.items():
	self.policy_grads[name] =param.grad

# 梯度清除
self.optimizer.zero_grad()

# 计算value上的loss，计算梯度，加上先前保存变量的梯度
value_loss_sum.backward()
for name,param in self.named_parameters():
	if name in self.policy_grads:
    	param.grad = param.grad + self.policy_grads[name]

# 反向传播，更新变量
self.optimizer.step()
```

### 出现问题：train不起来！！！！！！！！！！！！！！！！！

## 方案3：policy和value不共用网络

​		（见models/Dense_agent2.py）