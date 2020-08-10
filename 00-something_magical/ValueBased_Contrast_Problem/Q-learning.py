#encoding:UTF-8
#!/usr/bin/env python3

import random

random.seed(1)
#动作
actions = ["a", "b"]

#状态
states = ["1", "2"]


#构建环境   s, a    r s a
def next_state_reward(state, action):
    alfa = random.random()

    if state == "1":
        if action == "a":
            if alfa <= 1.0 / 3:
                new_state = "1"
                reward = 0
            else:
                new_state = "2"
                reward = 1
        else:
            if alfa > 1.0 / 3:
                new_state = "1"
                reward = 0
            else:
                new_state = "2"
                reward = 1
    else:
        if action == "a":
            if alfa <= 1.0 / 3:
                new_state = "1"
                reward = 0
            else:
                new_state = "2"
                reward = 1
        else:
            if alfa > 1.0 / 3:
                new_state = "1"
                reward = 0
            else:
                new_state = "2"
                reward = 1
    return new_state, reward


# q_value    state:{ action:0 }
q_value = {}
for state in states:
    temp = {}
    for action in actions:
        temp[action] = 0.0
    q_value[state] = temp


def action_max(state):
    temp = list(q_value[state].items())
    return max(temp, key=lambda p: p[-1])[0]


def action_greedy(state):
    if random.random() < epsilon:
        return random.choice(actions)
    else:
        return action_max(state)


epsilon = 0.4
gama = 0.99
learning_rate = 0.1


def q_learning():
    state = random.choice(states)
    action = action_greedy(state)

    next_state, reward = next_state_reward(state, action)
    next_action = action_max(next_state)

    q_estimate = reward + gama * q_value[next_state][next_action]

    td_error = q_estimate - q_value[state][action]

    q_value[state][action] += learning_rate * td_error


if __name__ == "__main__":
    for episode in range(10 ** 6):
        q_learning()

    epsilon = 0.01
    for episode in range(10 ** 5):
         q_learning()

    print(q_value)
