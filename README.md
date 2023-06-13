# Rocket Trajectory Optimization with Deep Reinforcement Learning

## Introduction

Model-free reinforcement learning has been applied to many challenging problems, especially when the environment is complex and hard to be modeled. Adapting deep learning techniques to the field of reinforcement learning introduces more alternatives while designing distinct policies and value functions. This project uses the DQN, Double DQN, and Dueling DQN models to solve the classic Rocket Trajectory Optimization problem. The environment is simulated by the LunarLander-v2 developed by [OpenAI Gym](https://www.gymlibrary.dev/environments/box2d/lunar_lander/). 

This project trained a DQN as the baseline model for optimizing the trajectory. Followed by implementing the Double DQN and Dueling DQN on top of the baseline model. The simulation results showed no significant differences between these models in terms of their final score and converging time, but the Dueling DON has the most stable performance among other models.


![](https://github.com/marswon0/rl_test/blob/main/assets/images/lunar_lander.gif)


## Reference

For model training and hyperparameter tuning processes, please check out the [report paper written for this project](https://github.com/marswon0/rocket_trajectory_optimization/blob/main/assets/paper/Solving%20the%20Lunar%20Lander%20with%20Deep%20Reinforcement%20Learning.pdf).


## DQN

Q-Learning iteratively updates the Q-Table to improve the performance of the learning agent. In a given state, the agent has a specific set of actions that are available. Upon taking an action, a reward will be assigned, the environment will change accordingly, and the agent will enter a new state. The value function measures how successful the action is, and saves the result to the Q-Table.

DQN uses a neural network named **Q network** to approximate the value function. The state-action pair **(s, a)** can be evaluated through the value function, and calculates the result **Q(s, a)**. The input of the neural network is the state **s** and corresponding observations. The output assigns **Q(s, a)**  for each action.

<img src="/assets/images/DQN.JPG" width="800" height="600">

DQN uses the epsilon-greedy strategy to output the action. The relationship between the **Q network** and the epsilon-greedy strategy is shown below. Step 1 to 5 will be repeated until an effective value function network is trained for the DQN model.

1. The environment gives an observation
2. The agent obtains all Q(s, a) about the observation according to the value function network
3. Using epsilon-greedy to select an action and make a decision
4. The environment gives a reward and the next observation after receiving the current action
5. Updating the parameters of the value function network according to the reward, and then go to the next state

The stability of the DQN model can be greatly improved by adding a separate neural network called **target network**. The **target network** has the exactly the same configurations as the **Q network**. The difference between these two networks would be the update frequency. The **Q network** updates after every iteration, whereas the **target network** updates after a fix number of iterations.

<img src="/assets/images/result_dqn.JPG" width="600" height="400">

Momentum is calculated in the Adam optimizer while performing the gradient descent. Momentum allows the optimizer to build inertia in the direction that the gradient descent approaches, therefore the optimizer can overcome the noises encountered while descending the gradient. By incresing the momentum decay from 0.99 to 0.5, the DQN model converged to a final policy that scored around 270 in 800 games of training.

## DDQN

Double DQN was implemented based on the DQN model. The difference between these two **target functions** is the optimal action selection. DDQN is based on the parameter θ of the **Q network** currently being updated, whereas the optimal action for DQN is based on the parameter θ in **target network**.

#### Target function for DQN:

<img src="/assets/images/eq1.JPG" width="400" height="55">

#### Target function for DDQN:

<img src="/assets/images/eq2.JPG" width="600" height="45">

The final result of the Double DQN model is shown below, the model converged in 200 games and achieved a final score of 260.

<img src="/assets/images/result_ddqn.JPG" width="600" height="400">

## Dueling DQN

In the DQN model, the last layer is a fully connected layer. After this layer, **Q values** (one for each selectable actions) are the outputs. Dueling DQN does not directly generate those **Q values**. It obtains the **V (state value)** and **A (action advantage)** through training, and then concatenates the two variables to obtain the final **Q value**.

#### State Value:

<img src="/assets/images/eq3.JPG" width="300" height="40">

#### Action Advantage:

<img src="/assets/images/eq4.JPG" width="320" height="40">

V represents the average expectation of the **Q value** in the current state **s** (considering all optional actions). **A** represents how much the **Q value** exceeds the expected value when choosing an action. Adding the state value **V** and the action advantage **A** together leads to the actual **Q(s, a)**.

<img src="/assets/images/dueling.JPG">

A generic structure of Dueling DQN is shown above. In this project, the Dueling DQN model uses fully-connected (FC) layers instead of convolution layers.

<img src="/assets/images/result_dueling.JPG" width="600" height="400">

The Dueling DQN model converged in ~350 games of training, and reached an average score of 270.

## Conclusion

This project proposes to use the DQN, Double DQN, and Dueling DQN model to solve the LunarLander-v2 environment. All three models converged into a final policy. Both the DQN and Dueling DQN scored near 280, and the Double DQN model scored around 260. Once the models converged to an optimal policy, the Dueling DQN model had a much more consistent performance compared to the other models.

