#encoding:UTF-8
#!/usr/bin/env python3

import random

#状态
states = ["1", "2"]

#动作
actions = ["a", "b"]

# 奖励的折扣因子
gama = 0.99

""" 状态值  v_value 
v_value={
"1":0,
"2":0
}"""
v_value = {}
for state in states:
    v_value[state] = 0

# 状态动作值 ("1", "a"):0
q_value = {}


def p_state_reward(state, action):
    # 输入当前状态，及行为
    # return 跳转概率，下一状态, 奖励
    if state == "1":
        if action == "a":
            return ((1.0 / 3, "1", 0),
                    (2.0 / 3, "2", 1))
        else:
            return ((2.0 / 3, "1", 0),
                    (1.0 / 3, "2", 1))
    if state == "2":
        if action == "a":
            return ((1.0 / 3, "1", 0),
                    (2.0 / 3, "2", 1))
        else:
            return ((2.0 / 3, "1", 0),
                    (1.0 / 3, "2", 1))


###########################################      根据state value计算Q value        #####################################
def q_value_fun():
    q_value.clear()
    for state in states:
        for action in actions:
            temp = 0
            for prob,nextstate,reward in p_state_reward(state, action):
                temp += prob * (reward + gama * v_value[nextstate])
            q_value[(state, action)] = temp


# q_value 初始值
"""q_value={
("1", "a"):(2/3),
("1", "b"):(1/3),
("2", "a"):(2/3),
("2", "b"):(1/3)
}"""

#q_value初始化
q_value_fun()

# 值迭代方法
def value_iteration():
    global v_value
    flag=True
    v_value_new={}

    for state in states:
        v_value_new[state]=max(q_value[(state, action)] for action in actions)

        if abs(v_value_new[state]-v_value[state])>0.0001:
            flag=False

    if flag==False:
        v_value=v_value_new
        q_value_fun()
    return flag


if __name__ == "__main__":
    i = 1
    flag = value_iteration()
    while flag != True:
        i += 1
        flag = value_iteration()

    #策略 pi
    pi = {}
    for state in states:
        act = max((q_value[(state, action)], action) for action in actions)[-1]
        temp = {}
        for action in actions:
            if action == act:
                temp[action] = 1.0
            else:
                temp[action] = 0.0

        pi[state] = temp

    print("*" * 30 + "\n")
    print("总共运行次数:" + str(i) + "\n")
    print("状态值为：")
    print(v_value)
    print("")
    print("行为值为：")
    print(q_value)
    print("策略为：")
    print(pi)