from __future__ import absolute_import, division, print_function
import argparse
import os
import pickle
import tensorflow as tf
import numpy as np
import tf_util
import gym

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('expert_data_file', type=str)
    parser.add_argument('envname', type=str)
    parser.add_argument('--render', action ='store_true')
    parser.add_argument("--max_timesteps", type=int)
    parser.add_argument("--num_rollouts", type=int,default=20, help='Number of expert roll outs')

    args=parser.parse_args()
#Expert data
    with open(args.expert_data_file, "rb") as f:
        expert_data = pickle.load(f)

    expert_obss = expert_data["observations"].astype(np.float32)
    expert_acts = expert_data["actions"].reshape(expert_obss.shape[0], -1).astype(np.float32)

#Clone

#Normalize
    epsilon = 1e-12
    expert_obss_range = expert_obss.ptp(axis=0)
    expert_obss_range = np.array([max(epsilon, vr) for vr in expert_obss_range]).astype(np.float32)
    expert_obss_min = expert_obss.min(axis=0)
    normalized_obss = (expert_obss-expert_obss_min)/expert_obss_range

    batch_size = 40000
    num_epochs = 400
    learning_rate = 1e-3

    expert_dataset = tf_util.create_dataset(
        input_features = normalized_obss,
        output_labels = expert_acts,
        batch_size = batch_size,
        num_epochs = num_epochs
    )
    iterator = expert_dataset.make_one_shot_iterator()
    next_obss, next_acts = iterator.get_next()

    #model
    bc_model = tf.keras.Sequential([
        tf.keras.layers.Dense(32, activation=tf.nn.relu, input_shape=(expert_obss.shape[1],)),
        tf.keras.layers.Dense(32, activation=tf.nn.relu),
        tf.keras.layers.Dense(expert_acts.shape[-1])
    ])
    obs_placeholder = tf.placeholder(tf.float32, shape=(None, expert_obss.shape[-1]))
    cloned_action = bc_model(obs_placeholder)

    #training
    loss = tf_util.loss(bc_model, next_obss, next_acts)
    optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss)

    #train new policy
    init = tf.global_variables_initializer()
    saver=tf.train.Saver()
    with tf.Session() as sess:
        sess.run(init)
        i=0
        while True:
            try:
                _, loss_value = sess.run([train_op, loss])
                if not i%10:
                    print("Iteration: {}, Loss: {:.3f}".format(i, loss_value))
            except tf.errors.OutOfRangeError:
                break
            i+=1

            save_path = saver.save(sess, "./bc_model/model.ckpt")
            print("Model saved in path : {}".format(save_path))

            env = gym.make(args.envname)
            max_steps = args.max_timesteps or env.spec.timestep_limit
            episodic_returns = []
            observations = []
            actions = []
            for episode in range(args.num_rollouts):
                obs = env.reset()
                obs_norm = (obs-expert_obss_min)/expert_obss_range
                done=False
                total_reward=0
                step=0
                while not done:
                    action = sess.run(cloned_action, feed_dict={obs_placeholder: obs_norm[None,:]})
                    obs, reward, done, _ = env.step(action)
                    obs_norm = (obs-expert_obss_min)/expert_obss_range
                    total_reward+=reward
                    step+=1
                    if args.render:
                        env.render()
                    print("Episode: {}, Step: {} of {}, reward: {}".format(episode, step, max_steps, reward))
                    if step>=max_steps:
                        break
                episodic_returns.append(total_reward)
            print(
                "\nEpisodic returns: {}".format(episodic_returns),
                "\nAverage of the returns: {}".format(np.mean(episodic_returns)),
                "\nStandard deviation of the returns: {}".format(np.std(episodic_returns))
            )
