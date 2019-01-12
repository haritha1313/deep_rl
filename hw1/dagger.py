from __future__ import absolute_import, division, print_function

import os
import pickle
import tensorflow as tf
import numpy as np
import tf_util
import gym
import load_policy

def normalize(observation, observations_buffer):
    epsilon=1e-12
    obss_range = np.array(observations_buffer).ptp(axis=0)
    obss_range = np.array([max(epsilon, vr) for vr in obss_range]).astype(np.float32)
    obss_min = np.array(observations_buffer).min(axis=0)
    normalized_observation = (observation-obss_min)/obss_range

    return normalized_observation

if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('expert_policy_file', type=str)
    parser.add_argument('envname', type=str)
    parser.add_argument('--expert_data', type=str)
    parser.add_argument('--render',action='store_true')
    parser.add_argument('--max_timesteps', type=int)
    parser.add_argument('--num_rollouts', type=int, default=20)
    args = parser.parse_args()

    print("Loading and building expert policy")
    expert_policy_fn = load_policy.load_policy(args.expert_policy_file)
    print("Expert policy loaded and built")

    agg_observations=[]
    agg_actions=[]
    print("Loading expert policy generated data: obs and action pair")
    with open(args.expert_data, "rb") as f:
        expert_data = pickle.load(f)

    assert(expert_data["observations"].shape[0]==expert_data["actions"].shape[0])
    for i in range(expert_data["observations"].shape[0]):
        agg_observations.append(expert_data["observations"][i])
        agg_actions.append(expert_data["actions"][i,-1])

    learning_rate = 1e-3
    dag_model = tf.keras.Sequential([
        tf.keras.layers.Dense(32, activation=tf.nn.relu, input_shape=(expert_data["observations"].shape[1],)),
        tf.keras.layers.Dense(32, activation=tf.nn.relu),
        tf.keras.layers.Dense(expert_data["actions"],shape[-1])
    ])

    loss=tf_util.loss(dag_model, next_obss, next_acts)
    optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss)
    model_path="./dag_model/model.ckpt"
    saver = tf.train.Saver()
    normobs_placeholder = tf.placeholder(tf.float32, shape=(None, expert_data["observations"].shape[-1]))
    learned_action = dag_model(normobs_placeholder)

    env = gym.make(args.envname)
    max_steps = args.max_timesteps or env.spec.timestep_limit
    episodic_rewards=[]
    for rollout in range(args.num_rollouts):
        for episode in range(2):
            obs = env.reset()
            normed_obs = normalize(obs, agg_observations)
            done=False
            total_reward = 0
            step = 0
            with tf.Session() as sess:
                saver.restore(sess, model_path)
                while not done:
                    action = sess.run(learned_action, feed_dict={normobs_placeholder: normed_obs[None, :]})
                    expert_action = expert_policy_fn(obs[None, :])
                    agg_observations.append(obs)
                    agg_actions.append(expert_action)
                    obs, reward, done, _ = env.step(action)
                    normed_obs = normalize(obs, agg_observations)
                    total_reward +=reward
                    step+=1
                    if args.render():
                        env.render()
                    if not step%10:
                        print("Episode: {}, Step: {} of {}, total reward: {}".format(episode, step, max_steps, total_reward))
                    if step >= max_steps:
                        break
            episodic_rewards.append(total_reward)
            agg_dataset= tf.util.create_dataset(
                input_features = normalized_expert_obss,
                output_labels = expert_acts,
                batch_size = batch_size,
                num_epochs = num_epochs
            )
