# deep_rl#

##Homework 1##

*Steps:*
###Behavioral Cloning###
* python run_expert.py experts/Reacher-v2.pkl Reacher-v2 --render --num_rollouts 20 --save_data True
* python behavior_cloning.py expert_data/Reacher-v2.pkl Reacher-v2 --render

###Dagger###
* python dagger.py experts/Reacher-v2.pkl Reacher-v2 --expert_data expert_data/Reacher-v2.pkl --render --num_rollouts 20

* Use tab to change camera view in simulation



