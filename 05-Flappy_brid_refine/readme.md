## 模型结构
- 使用teacher model和student model两个model
- 当student model更新50次时，teacher model copy一次

## 目录结构
```
assetes/    ---- 游戏资源文件夹
checkpoint/ ---- 训练模型保存
game/       ---- 游戏文件夹
utils/      ---- 一些无关紧要的

Agent.py    ---- 定义Agent
Bird_DQN.py ---- 模型定义
main.py     ---- 入口函数 + 功能函数：包括模型train/play、模型评估、模型保存
```


## state 和 action
#### state：
- 连续三帧构成
- 初始时是全0的帧
#### action:
- [1,0] do nothing
- [0,1] 往上飞

## epsilon的变化曲线
`epsilon_by_epsilon = lambda epsilon_idx: epsilon_final + (epsilon_start - epsilon_final) * math.exp(
    -1. * epsilon_idx / epsilon_decay)`
## <img src=".\epsilon_curve.png" alt="epsilon_curve" style="zoom: 60%;" />

## LOSS 的计算
- 输入：s , a , r , s_
- 输出：loss
### 使用target model计算当前状态下Q的估计
    current_q_values = self.model(state_batch).gather(1, torch.argmax(action_batch, dim=1).unsqueeze(dim=1))

### 使用behavior model 计算下一个状态下Q的估计
    max_next_q_values = self.target_model(next_state_batch).max(1)[0]
    expected_q_values = reward_batch.unsqueeze(dim=1) + (self.gamma * max_next_q_values.unsqueeze(dim=1))

### 使用Huber loss计算，current_q_values和excepted_q_values之间的差距
    
# 更多的DQN模型
[here](https://github.com/qfettes/DeepRL-Tutorials)