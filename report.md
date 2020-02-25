<h2>Learning Algorithm</h2>
<p>
DDPG was adapted to train multiple agents. Though it is possible to get both agents use the same actor network to select actions, and add the experience to a shared replay buffer, I implemented the more simple option. As a future idea, however, it is definitely worth trying to build a more complex scenario.
</p>
<p>
Hyperparameters were selected as follows:<br>
BUFFER_SIZE = int(1e6)  # replay buffer size<br>
BATCH_SIZE = 128        # minibatch size<br>
GAMMA = 0.99            # discount factor<br>
TAU = 1e-3              # for soft update of target parameters<br>
LR_ACTOR = 1e-4         # learning rate of the actor <br>
LR_CRITIC = 1e-3        # learning rate of the critic<br>
WEIGHT_DECAY = 0        # L2 weight decay<br>
</p>
<p>
The models applied are contained in the file named model.py
Both actor and critic weights are represented in neural networks. These two use 3 fully connected layers with a batch normalizer after the input layer. Regarding inputs, these networks employ a number of input neurons matching the number of state descriptors (8). In line with the general format of DDPG implementations, these neural networks are instantiated in the file named ddpg_agent.py, creating two of each type, e.g. actor and critic networks both exist in local and target "versions", realizing the DDPG "mechanics".
</p>
<h2>Plot of Rewards</h2>
List of scores for the final episodes were seen as below:<br>
![ListScores](https://github.com/palyi/DRLCollaborationAndCompetition/blob/master/scores_list.JPG)<br>
<br>
The overall plot of scores I've included under the below link:<br>
![PlotScores](https://github.com/palyi/DRLCollaborationAndCompetition/blob/master/scores_plot.JPG)<br>
<h2>Ideas for Future Work</h2>
As mentioned above, this is a rather simple - yet working - first implementation. There are further, more complex ways to implement multiagent DRL, which I would try as next idea. Additionally, my aim is to build some real-world scenario utilizing such learning algorithms (based on some prototyping electronics, servos, sensors and such).
