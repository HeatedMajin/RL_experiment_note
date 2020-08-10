## 模型结构
使用单一的model,（不区分teacher model和student model）

## 目录结构
```
assetes/    ---- 游戏资源文件夹
checkpoint/ ---- 训练模型保存
game/       ---- 游戏文件夹
utils/      ---- 一些无关紧要的
main.py     ---- 入口函数
MSIC.py     ---- 功能函数：包括模型train/play、模型评估、模型保存
Bird_DQN.py ---- 模型定义
```

## state 和 action
state：
- 连续三帧构成
- 初始时是全0的帧
action:
- [1,0] do nothing
- [0,1] 往上飞

## epsilon的变化曲线
## <img src=".\epsilon_curve.png" alt="epsilon_curve" style="zoom: 60%;" />


## LOSS 的计算
- 输入：s , a , r , s_
- 输出：loss


### Bellman optimal equation : (使用下一个状态估算当前状态的value)

V[s] = max_a( Q[s,a] )
     = max_a( reward[s,a] + gamma * V[s_] )
     = max_a( reward[s,a] + gamma * max_a_( Q[s_,a_] ) )

### Value calc from Q table : 使用当前状态的q 估算

V[s] = sum_a ( p(a|s) * q(s,a) )    
