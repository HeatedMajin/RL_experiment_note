# The model-free algrithoms to solve CLIFF problem
## environment
```
python=3.7
numpy=1.19
matpolitlib=3.2.2
```

## File Explaination
```
cliff_grid.py   --- game environment
Q_learning.py   --- Q learning to solve
Sarsa.py        --- SARSA to solve
visual.py       --- the utils to visualize result 
```

## How to use the files
```
python Q_learning.py 
python Sarsa.py  
```

# Experiment Result
 * SARSA        -- **保守**的策略
 * Q learning   -- **激进**的策略
 
###一种解释：激进与否与[策略的熵](https://www.zhihu.com/question/329124024/answer/734041012) 有关,与on-policy/off-policy无关

熵：**greedy < e-greedy < 随机**

SRASA  | Q-learning|
--------- | --------|
熵大  | 熵小 |
robust好  | robust差 |
收益可能会少 | 收益可能大|

###另一种解释：**死亡记忆**

SRASA  | Q-learning|
--------- | --------|
评估value依据真实走过的轨迹  | 评估value依据下一步的最大值 |
死亡，导致悬崖周围的value变低  | 死亡记忆被MAX函数抹去了，看不到过去在这里死亡没有 |



# To be continued
但是，把e-greedy改成greedy后,SARSA算法和Q-learning算法都是激进的策略