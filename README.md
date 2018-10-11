# DRL-navigation
Train an agent to navigate in a complex environment and collect bananas. The deep reinforcement learning algorithm is based on value-based method (DQN)

The Environment (determinist)
 + State: 
 The state space has 37 dimensions and contains:
    - 7 rays projecting from the agent at the following angles: [20, 90, 160, 45, 135, 70, 110] # 90 is directly in front of the agent
    - Each ray is projected into the scene. If it encounters one of four detectable objects the value at that position in the array is set to 1. Finally there is a distance measure which is a fraction of the ray length: [Banana, Wall, BadBanana, Agent, Distance]
    - the agent's velocity: Left/right velocity (usually near 0) and Forward/backward velocity (0-11.2)
 + Actions:
 The action space has 4 discrete action and contains:
    - 0: move forward
    - 1: move backward
    - 2: turn left
    - 3: turn righ
 + Reward strategy:
    - +1 for collecting a yellow banana
    - -1 for collecting a blue banana
 + Solved Requirements:
Considered solved when the average reward is greater than or equal to +13 over 100 consecutive trials.
