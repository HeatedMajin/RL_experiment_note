## LOSS 的计算
输入：s , a , r , s_
输出：loss


### Bellman optimal equation : (使用下一个状态估算当前状态的value)

V[s] = max_a( Q[s,a] )
     = max_a( reward[s,a] + gamma * V[s_] )
     = max_a( reward[s,a] + gamma * max_a_( Q[s_,a_] ) )

### Value calc from Q table : 使用当前状态的q 估算

V[s] = sum_a ( p(a|s) * q(s,a) )    
