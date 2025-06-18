# -*- coding: utf-8 -*-
"""
Created on Tue Jun 10 10:06:27 2025

@author: AmitTzahar
"""

import gym
import numpy as np
from gym import spaces
import random
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import os
#%% Create Wordle Environment:
class WordleEnv(gym.Env):
    """
    Environment for the Wordle game.
    The agent must guess a 5 letter word in maximum 6 tries.
    Action space: indexes of words in a predefined word list.
    Observation space: Encoded feedback (green/yellow/gray).
    """

    def __init__(self, word_list, max_steps=6):
        super(WordleEnv, self).__init__()
        self.word_list = word_list
        self.word_to_idx = {word: i for i, word in enumerate(self.word_list)}
        self.idx_to_word = {i: word for word, i in self.word_to_idx.items()}
        self.word_length = 5
        self.max_steps = max_steps

        self.action_space = spaces.Discrete(len(self.word_list))
        self.observation_space = spaces.Box(
            low=0, high=2, shape=(self.max_steps, self.word_length), dtype=np.int8)
        self.target_word = None
        self.state = None
        self.current_step = 0
        self.done = False

    def reset(self):
        self.target_word = random.choice(self.word_list)
        self.state = np.full((self.max_steps, self.word_length), -1, dtype=np.int8)
        self.current_step = 0
        self.done = False
        return self.state

    def step(self, action):
        assert self.action_space.contains(action), "Invalid action."
        if isinstance(action, np.ndarray):
            action = int(action.item())  
        else:
            action = int(action)
        guess_word = self.idx_to_word[action]
        feedback = self._get_feedback(guess_word)
        self.state[self.current_step] = feedback
        self.current_step += 1

        reward = 1.0 if guess_word == self.target_word else 0.0
        self.done = (guess_word == self.target_word) or (self.current_step >= self.max_steps)
        return self.state, reward, self.done, {}

    def _get_feedback(self, guess):
        feedback = np.zeros(self.word_length, dtype=np.int8)
        for i in range(self.word_length):
            if guess[i] == self.target_word[i]:
                feedback[i] = 2  # Green (correct position)
            elif guess[i] in self.target_word:
                feedback[i] = 1  # Yellow (wrong position)
            else:
                feedback[i] = 0  # Gray (not in word)
        return feedback

    def render(self, mode='human'):
        print("Wordle board:")
        for i in range(self.max_steps):
            if i < self.current_step:
                feedback = self.state[i]
                feedback_symbols = ""
                for val in feedback:
                    if val == 2:
                        feedback_symbols += "🟩"
                    elif val == 1:
                        feedback_symbols += "🟨"
                    else:
                        feedback_symbols += "⬜"
                print(f"Guess {i+1}: {feedback_symbols}")
            else:
                print(f"Guess {i+1}: ⬛⬛⬛⬛⬛")  # Empty row

#%% Words list:
WORD_LIST = [
    "apple", "grape", "mango", "peach", "melon",
    "berry", "lemon", "chess", "flame", "tiger",
    "chair", "plant", "smart", "stone", "crane"
]

def get_word_list():
    """Return the list of valid 5-letter words."""
    return WORD_LIST


def get_random_word():
    """Return a random word from the list."""
    return random.choice(WORD_LIST)


def word_to_index_map():
    return {word: i for i, word in enumerate(WORD_LIST)}


def index_to_word_map():
    return {i: word for i, word in enumerate(WORD_LIST)}

#%% Create PPO agent:
class PPOAgent:
    def __init__(self, env, agent_id="agent", save_dir="models", total_timesteps=10000):
        self.env = env
        self.agent_id = agent_id
        self.save_dir = save_dir
        self.total_timesteps = total_timesteps
        self.model = PPO("MlpPolicy", env, verbose=0)

    def train(self):
        self.model.learn(total_timesteps=self.total_timesteps)

    def save(self):
        path = os.path.join(self.save_dir, f"{self.agent_id}.zip")
        os.makedirs(self.save_dir, exist_ok=True)
        self.model.save(path)

    def load(self, path):
        self.model = PPO.load(path, env=self.env)

    def predict(self, obs, deterministic=True):
        action, _ = self.model.predict(obs, deterministic=deterministic)
        return action


def evaluate_agent(agent, env, num_episodes=10):
    success_count = 0
    steps_list = []  # <-- Track number of steps per episode

    for episode in range(num_episodes):
        obs = env.reset()
        done = False
        step_counter = 0
        print(f"\nEpisode {episode + 1} — Target word: {env.target_word}")
        guesses = []

        while not done:
            action = agent.predict(obs)
            guess_word = env.idx_to_word[int(action)]
            guesses.append(guess_word)
            obs, reward, done, info = env.step(action)
            step_counter += 1

        steps_list.append(step_counter)

        # Show guesses
        for i, guess in enumerate(guesses):
            print(f"{i+1}. {guess}")

        env.render()

        if reward == 1.0:
            print("Agent guessed the word correctly!")
            success_count += 1
        else:
            print(f"Agent failed. The correct word was: {env.target_word}")

    success_rate = success_count / num_episodes
    avg_steps = np.mean(steps_list)
    print(f"\nSummary over {num_episodes} episodes:")
    print(f"Success Rate: {success_rate:.2f}")
    print(f"Average Steps to Solve: {avg_steps:.2f}")
    return success_rate, avg_steps
    
#%% run:

def single_experiment(total_timesteps=5000, num_eval_episodes=10):
    word_list = get_word_list()
    env = WordleEnv(word_list)

    agent = PPOAgent(env=env, agent_id="ppo_wordle_test", total_timesteps=total_timesteps)
    agent.train()
    agent.save()

    success_rate, avg_steps = evaluate_agent(agent, env, num_eval_episodes)
    print(f"Success rate over {num_eval_episodes} episodes: {success_rate:.2f}")
    print(f"Average steps to solve: {avg_steps:.2f}")
    return success_rate, avg_steps


def run_multiple_experiments(num_experiments=20, total_timesteps=5000, num_eval_episodes=10):
  
    success_rates = []
    avg_steps_list = []
    
    for i in range(num_experiments):
        print(f"\n--- Running Experiment {i+1}/{num_experiments} ---")
        success_rate, avg_steps = single_experiment(total_timesteps, num_eval_episodes)
        success_rates.append(success_rate)
        avg_steps_list.append(avg_steps)
    
    success_rates = np.array(success_rates)
    avg_steps_list = np.array(avg_steps_list)
    
    mean_success = np.mean(success_rates)
    std_success = np.std(success_rates)
    mean_avg_steps = np.mean(avg_steps_list)
    std_avg_steps = np.std(avg_steps_list)
    
    print("\n--- Final Results ---")
    print(f"Average Success Rate: {mean_success:.4f}")
    print(f"Standard Deviation: {std_success:.4f}")
    print(f"Average Steps to Solve: {mean_avg_steps:.2f}")
    print(f"Steps Std Dev: {std_avg_steps:.2f}")

if __name__ == "__main__":
    run_multiple_experiments(num_experiments=20)
    