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

####################################################    策略 pi    ##############################################
#初始化   "1":{"a":0.5, "b":0.5}
pi = {}
for state in states:
    temp = {}
    for action in actions:
        temp[action] = 1.0 / len(actions)
    pi[state] = temp


################################################ policy evaluate ############################################
#策略评估 得出 v_value 值
def policy_evalue():
    global v_value
    v_value_new = {}

    def v_update():
        nonlocal v_value_new
        v_value_new = {}
        for state in states:
            temp = 0
            for action, p in pi[state].items():
                temp += p * q_value[(state, action)]
            v_value_new[state] = temp

    def stop_judge():
        flag = True
        for state, value in v_value.items():
            if abs(v_value_new[state] - value) > 0.0001:
                flag = False
        return flag

    # 计算 v_value_new
    v_update()

    while stop_judge() != True:
        # 更新 v_value
        v_value = v_value_new
        # 更新 q_value
        q_value_fun()
        # 再次迭代 计算v_value_new
        v_update()

###############################################    policy improve    ##############################################
#策略改进 max
def policy_improve():
    flag = True
    for state in states:
        action = max((q_value[state, action], action) for action in actions)[-1]

        for k in pi[state]:
            if k == action:
                if pi[state][k] != 1.0:
                    pi[state][k] = 1.0
                    flag = False
            else:
                pi[state][k] = 0.0
    return flag


if __name__ == "__main__":
    policy_evalue()
    flag = policy_improve()
    i = 1
    while flag != True:
        i += 1
        policy_evalue()
        flag = policy_improve()

    print("*" * 30 + "\n")
    print("总共运行次数:" + str(i) + "\n")
    print("状态值为：")
    print(v_value)
    print("")
    print("行为值为：")
    print(q_value)
    print("策略为：")
    print(pi)