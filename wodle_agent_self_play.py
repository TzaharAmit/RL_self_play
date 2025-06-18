# -*- coding: utf-8 -*-
"""
Created on Tue Jun 10 10:33:35 2025

@author: AmitTzahar
"""

import gym
import numpy as np
from gym import spaces
import random
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import os
import shutil
import stat
import csv
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
    "chair", "plant", "smart", "stone", "crane"]

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

#%% Self Play:
NUM_GENERATIONS = 5
NUM_EPISODES_PER_EVAL = 10
AGENT_POOL_DIR = r""

def evaluate_match(agent_a, agent_b, env, num_episodes=10):
    """Return the win rate of agent_a over agent_b on the same episodes."""
    wins = 0
    for _ in range(num_episodes):
        target_word = random.choice(env.word_list)
        env.target_word = target_word
        obs_a, obs_b = env.reset(), env.reset()

        # Agent A plays
        done_a = False
        env.target_word = target_word
        while not done_a:
            action_a = agent_a.predict(obs_a)
            obs_a, reward_a, done_a, _ = env.step(action_a)

        # Agent B plays
        env.target_word = target_word
        obs_b = env.reset()
        done_b = False
        while not done_b:
            action_b = agent_b.predict(obs_b)
            obs_b, reward_b, done_b, _ = env.step(action_b)

        if reward_a > reward_b:
            wins += 1
        elif reward_a == reward_b:
            # Tiebreaker: fewer steps wins
            if env.current_step < env.max_steps:
                wins += 1
    return wins / num_episodes

def promote_agent(new_agent_path, gen):
    shutil.copy(new_agent_path, os.path.join(AGENT_POOL_DIR, f"gen_{gen}.zip"))

def evaluate_agent(agent, env, num_episodes=10):
    success_count = 0
    steps_list = []

    for episode in range(num_episodes):
        obs = env.reset()
        done = False
        step_counter = 0
        print(f"\n Episode {episode + 1} — Target word: {env.target_word}")
        guesses = []

        while not done:
            action = agent.predict(obs)
            guess_word = env.idx_to_word[int(action)]
            guesses.append(guess_word)
            obs, reward, done, info = env.step(action)
            step_counter += 1

        steps_list.append(step_counter)

        # Print all guesses
        print(f" Guesses ({len(guesses)} steps):")
        for i, guess in enumerate(guesses):
            print(f"  {i+1}. {guess}")

        env.render()

        if reward == 1.0:
            print(" Agent guessed the word correctly!")
            success_count += 1
        else:
            print(f" Agent failed. The correct word was: {env.target_word}")

    success_rate = success_count / num_episodes
    avg_steps = np.mean(steps_list)
    print(f"\n Total Successes: {success_count}/{num_episodes}")
    print(f"Average Steps to Solve: {avg_steps:.2f}")
    return success_rate, avg_steps

def self_play_loop():
    word_list = get_word_list()
    env = WordleEnv(word_list)

    os.makedirs(AGENT_POOL_DIR, exist_ok=True)
    base_agent = PPOAgent(env, agent_id="gen_0", save_dir="models", total_timesteps=5000)
    base_agent.train()
    base_agent.save()
    promote_agent(os.path.join(r"C:\Users\AmitTzahar\OneDrive - Tyto Care Ltd\Documents\afeka\Copmuter Vision\results\RL\models", "gen_0.zip"), gen=0)

    for gen in range(1, NUM_GENERATIONS + 1):
        print(f"\n Training Generation {gen}")
        new_agent = PPOAgent(env, agent_id=f"gen_{gen}", save_dir="models", total_timesteps=5000)

        new_agent.train()
        new_agent.save()

        # Load previous best agent
        prev_agent = PPOAgent(env)
        prev_agent.load(os.path.join("models", f"gen_{gen-1}"))  

        # Evaluate
        win_rate = evaluate_match(new_agent, prev_agent, env, NUM_EPISODES_PER_EVAL)
        print(f" New agent win rate vs. previous: {win_rate:.2f}")

        if win_rate >= 0.6:
            print(" Promoting new agent to pool.")
            promote_agent(os.path.join("models", f"gen_{gen}.zip"), gen)

        else:
            print(" New agent not promoted. Keeping previous best.")
    # Evaluate best agent after training
    print("\n Final Evaluation of Best Agent")
    best_gen = max([
        int(f.split('_')[1].split('.')[0])
        for f in os.listdir(AGENT_POOL_DIR)
        if f.endswith(".zip")])
    best_agent_path = os.path.join(AGENT_POOL_DIR, f"gen_{best_gen}")
    
    best_agent = PPOAgent(env)
    best_agent.load(best_agent_path)
    success_rate, avg_steps = evaluate_agent(best_agent, env, num_episodes=10)
    print(f"Final success rate of best agent (gen_{best_gen}): {success_rate:.2f}")
    print(f"Average steps to solve: {avg_steps:.2f}")
    return success_rate, avg_steps

if __name__ == "__main__":
    self_play_loop()

#%% run:

EXPERIMENT_RESULTS_DIR = r""

def handle_remove_readonly(func, path, exc):
    excvalue = exc[1]
    if func in (os.unlink, os.remove, os.rmdir):
        os.chmod(path, stat.S_IWRITE)
        func(path)
    else:
        raise

def clean_agent_pool():
    """Clean up agent pool directory before each experiment"""
    if os.path.exists(AGENT_POOL_DIR):
        shutil.rmtree(AGENT_POOL_DIR, onerror=handle_remove_readonly)
    os.makedirs(AGENT_POOL_DIR) 

def run_single_experiment():
    clean_agent_pool()
    success_rate, avg_steps = self_play_loop()
    return success_rate, avg_steps

def run_multiple_self_play_experiments(num_experiments=10):
    success_rates = []
    avg_steps_list = []

    for i in range(num_experiments):
        print(f"\n======= Starting Experiment {i+1}/{num_experiments} =======")
        success_rate, avg_steps = run_single_experiment()
        success_rates.append(success_rate)
        avg_steps_list.append(avg_steps)

    success_rates = np.array(success_rates)
    avg_steps_list = np.array(avg_steps_list)

    # Summary:
    mean_success = np.mean(success_rates)
    std_success = np.std(success_rates)
    mean_avg_steps = np.mean(avg_steps_list)
    std_avg_steps = np.std(avg_steps_list)
    min_success = np.min(success_rates)
    max_success = np.max(success_rates)

    print("\n======= Overall Results =======")
    print(f"Mean Success Rate: {mean_success:.4f}")
    print(f"Std Dev: {std_success:.4f}")
    print(f"Mean Avg Steps: {mean_avg_steps:.2f}")
    print(f"Avg Steps Std Dev: {std_avg_steps:.2f}")
    print(f"Max Success: {max_success:.4f}")
    print(f"Min Success: {min_success:.4f}")

    # Save full log to CSV
    csv_path = os.path.join(EXPERIMENT_RESULTS_DIR, "self_play_results.csv")
    with open(csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Experiment#", "BestAgentSuccessRate", "AvgSteps"])
        for i, (s, steps) in enumerate(zip(success_rates, avg_steps_list)):
            writer.writerow([i+1, s, steps])
        writer.writerow([])
        writer.writerow(["OverallMeanSuccess", mean_success])
        writer.writerow(["OverallStdSuccess", std_success])
        writer.writerow(["OverallMeanAvgSteps", mean_avg_steps])
        writer.writerow(["OverallStdAvgSteps", std_avg_steps])
        writer.writerow(["OverallMinSuccess", min_success])
        writer.writerow(["OverallMaxSuccess", max_success])

if __name__ == "__main__":
    run_multiple_self_play_experiments(num_experiments=20)
