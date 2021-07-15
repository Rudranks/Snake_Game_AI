import torch
import random
import numpy as np
from collections import deque  # Deque is a double ended queue which i will use to store memory
from game import snakegameAI, Point, Direction
from model import Liner_QNet, QTrainer
from helper import plot


MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001  # Learning rate is a value between 0 and 1 which indicates how quick will the agent abandon a previous Q valee for a new one


class Agent:
    def __init__(self):
        self.n_games = 0  # No of games
        self.epsilon = 0  # Controls randomness, ie, controls exploration/exploitation
        self.gamma = 0.9  # Discount rate must be < 1
        self.memory = deque(maxlen=MAX_MEMORY)  # popleft() is called if it exceeds memory limitation
        self.model = Liner_QNet(11, 256, 3)  # size of initial is the 11 states and output is 3 for direction
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)


    def get_state(self, game):
        # Takes the head of the snake and creates some points around it in all directions
        head = game.snake[0]
        point_l = Point(head.x - 20, head.y)  # using 20 as its the block size i used in game.py
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)

        # Checks current game direction using bools
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            # This all depends on the current direction
            # If the danger is straight ahead
            (dir_r and game.is_collision(point_r)) or
            (dir_l and game.is_collision(point_l)) or
            (dir_u and game.is_collision(point_u)) or
            (dir_d and game.is_collision(point_d)),

            # If the danger is right
            (dir_u and game.is_collision(point_r)) or
            (dir_d and game.is_collision(point_l)) or
            (dir_l and game.is_collision(point_u)) or
            (dir_r and game.is_collision(point_d)),

            # If the danger is  left
            (dir_d and game.is_collision(point_r)) or
            (dir_u and game.is_collision(point_l)) or
            (dir_r and game.is_collision(point_u)) or
            (dir_l and game.is_collision(point_d)),

            # Move direction (only one of them is true and the others false)
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            # Food location
            game.food.x < game.head.x,  # food is on the left
            game.food.x > game.head.x,  # food is on the right
            game.food.y < game.head.y,  # food is above
            game.food.y > game.head.y  # food is below
        ]

        return np.array(state, dtype=int)  # Converts the bools to 0 or 1

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))  # popleft if MAX_MEMORY is reached

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)  # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # Random moves in the start: trading off exploration/exploitation
        # The more games we play the smaller epsilon gets, and the smaller epsilon gets, the less random the move are
        # This is called an epsilon greedy strategy
        self.epsilon = 80 - self.n_games
        final_move = [0,0,0]
        if random.randint(0, 200) < self.epsilon:
            # If r < epsilon exploration is preferred over exploitation
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            # as it can give u a raw input, we take the maximum and convert it all to 1 and 0
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move


def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = snakegameAI()

    # Training loop
    while True:
        # get old state
        state_old = agent.get_state(game)

        # get move
        final_move = agent.get_action(state_old)

        # Perform move and get new state
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        # Train short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # remember and store in memory
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            # Train long memory or replay memory, plot result
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save()
            print('Game', agent.n_games, 'Score:', score, 'Record:', record)

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)



if __name__ == '__main__':
    train()
