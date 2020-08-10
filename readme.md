# 关于项目结构
```
00-something magical    ---- 比较了Value based的四种方法：model-based 和 model-free

value based：
01-Model-based          ---- Model-based(policy iteration、value iteration) for gridWorld game
02-Model-free           ---- Model-free(SARSA、Q-learning) for gridworld、cliff、mountainCar games   

value based with value function approximation：
03-Semi-gradient_SARSA  ---- SARSA for scaling up RL(linear
04-Flappy_bird_sigle    ---- DQN for Flappy bird game(but not fixed target)
05-Flappy_bird_refine   ---- DQN for Flappy bird game
                            (fixed target + replay memory)

policy based：
06-CEM                  ---- cross-entropy method(对于不可导或者难以求导的policy)
07-REINFORCE            ---- Monte-Carlo policy gradient(actual return)

Actor-critic：
08-Actor-cirtic         ---- 有点结合value-based和policy-based的味道
09-PPO                  ---- proximal policy optimization
                            (target policy + replay memory，减少了采样时间，一次采样可以多次训练)

```

# 大总结
## 一、value based：learn value function
### model-based
- policy iteration: Bellman equation
- value iteration: Bellman optimal equation

### model-free
- SARSA
- Q-learning

### other:value function approximation
- intention：too many actions or states, too slow to learn a value
    - linear feature representations
    - neural networks
- SARSA(VFA version,linear feature):semi-gradient Sarsa
- Q-learning(neural networks):DQN

## 二、policy based：learn policy
- CEM：cross-entropy method(对于不可导或者难以求导的policy)
- REINFORCE:Monte-Carlo policy gradient(actual return)

## 三、actor-critic：learn policy and value function
- Actor-critic: state->value , state->actions' prob
- PPO: proximal policy optimization(target policy即actor-critic + replay memory)

