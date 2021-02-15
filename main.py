from rl.agents import DDPGAgent
from rl.memory import SequentialMemory
from rl.random import OrnsteinUhlenbeckProcess

import wandb
wandb.init(project = input("Enter your wandb project name: "), entity = input("Enter your wandb entity name: "))

from tensorflow.keras.optimizers import Adam
import tensorflow as tf

import numpy as np
import time
import gym
import os

from replay_buffer import ReplayBuffer
from action_noise import OUActionNoise
from frame_stack import FrameStack
from slither import Slitherio
from model import *
from utils import *

start = time.time()
os.mkdir(f'./models/{start}/')

max_memory_size = 1000000
minibatch_size = 16
gamma = 0.99
tau = 0.001
std_dev = 0.2
noise_generator = OUActionNoise(mean=np.zeros(1), std_deviation=float(std_dev) * np.ones(1))

actor = make_actor()
critic = make_critic()

actor.summary()
critic.summary()

actor_target = make_actor()
critic_target = make_critic()

actor_target.set_weights(actor.get_weights())
critic_target.set_weights(critic.get_weights())

actor_vars = actor.trainable_variables
actor_target_vars = actor_target.trainable_variables

critic_vars = critic.trainable_variables
critic_target_vars = critic_target.trainable_variables

actor_lr = 0.0001
critic_lr = 0.001

actor_optimizer = Adam(actor_lr)
critic_optimizer = Adam(critic_lr)

replay_memory = ReplayBuffer(max_memory_size)

def act(observation, length, noise_generator):
    sampled_actions = tf.squeeze(actor([tf.expand_dims(observation, axis = 0), tf.expand_dims(length, axis = 0)]))
    noise = noise_generator()
    sampled_actions = sampled_actions.numpy() + noise
    return sampled_actions

env = Slitherio(nickname = "Bot")
env = FrameStack(env)
env.start()

updates = 0
episodes = 0
highscore = 0
longest_duration = 0

while True:
    observation = env.reset()
    length = env.score
    terminal = False
    mean_actor_loss = 0
    mean_critic_loss = 0
    mean_total_loss = 0
    timestep = 0
    while not terminal:
        action = act(observation, length, noise_generator)
        angle, acceleration = action
        acceleration = int(acceleration > 0)
        next_observation, reward, terminal, info = env.step(angle, acceleration)
        next_length = env.score
        replay_memory.store(observation, length, action, reward, next_observation, next_length)
        observation = next_observation
        length = next_length
        states, lengths, actions, rewards, next_states, next_lengths = replay_memory.sample(minibatch_size)

        with tf.GradientTape() as tape:
            target_actions = actor_target([next_states, next_lengths], training = True)
            target_values = rewards + gamma * critic_target([next_states, next_lengths, target_actions], training = True)
            pred_values = critic([states, lengths, actions], training = True)
            critic_loss = tf.math.reduce_mean(tf.math.square(target_values - pred_values))
        
        critic_grads = tape.gradient(critic_loss, critic_vars)
        critic_optimizer.apply_gradients(zip(critic_grads, critic_vars))

        with tf.GradientTape() as tape:
            actions = actor([states, lengths], training = True)
            pred_values = critic([states, lengths, actions], training = True)
            actor_loss = -tf.math.reduce_mean(pred_values)

        actor_grads = tape.gradient(actor_loss, actor_vars)
        actor_optimizer.apply_gradients(zip(actor_grads, actor_vars))

        total_loss = actor_loss + critic_loss

        update_target(actor_target_vars, actor_vars, tau)
        update_target(critic_target_vars, critic_vars, tau)

        timestep += 1
        updates += 1

        mean_critic_loss += (critic_loss - mean_critic_loss) / timestep
        mean_actor_loss += (actor_loss - mean_actor_loss) / timestep
        mean_total_loss += (total_loss - mean_total_loss) / timestep

        wandb.log({
            'actor_loss': actor_loss,
            'critic_loss': critic_loss,
            'total_loss': total_loss
        })

        # env.render()

    episodes += 1
    score = env.score
    highscore = max(highscore, score)
    longest_duration = max(longest_duration, timestep)

    print ("""
    Episode:          %d
    Score:            %d
    High Score:       %d
    Updates:          %d
    Duration:         %d
    Longest Survival: %d
    Actor Loss:       %.6f
    Critic Loss:      %.6f
    Total Loss:       %.6f
    """ % (episodes, score, highscore, updates, timestep, longest_duration, mean_actor_loss, mean_critic_loss, mean_total_loss))

    wandb.log({
        'score': score,
        'highscore': highscore,
        'updates': updates,
        'duration': timestep,
        'longest_duration': longest_duration,
        'mean_actor_loss': mean_actor_loss,
        'mean_critic_loss': mean_critic_loss,
        'mean_total_loss': mean_total_loss
    })

    actor.save_weights(f'./models/{start}/actor.h5')
    critic.save_weights(f'./models/{start}/critic.h5')
    actor_target.save_weights(f'./models/{start}/actor_target.h5')
    critic_target.save_weights(f'./models/{start}/critic_target.h5')
    
env.close()
