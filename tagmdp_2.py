import numpy as np
import random
import time

class GameMap:
    def __init__(self, num_rows, num_columns, wall_probability):
        self.num_rows = num_rows
        self.num_columns = num_columns
        self.wall_probability = wall_probability
        self.map = self.generate_random_map()
    
    def generate_random_map(self):
        game_map = []
        for i in range(self.num_rows):
            row = []
            for j in range(self.num_columns):
                if i == 0 or i == self.num_rows - 1 or j == 0 or j == self.num_columns - 1:
                    row.append("#")
                elif random.random() < self.wall_probability:
                    row.append("#")
                else:
                    row.append(" ")
            game_map.append(row)
        self.place_agents(game_map)
        return game_map

    def place_agents(self, game_map):
        r_row, r_col = self.get_random_position()
        game_map[r_row][r_col] = "R"
        t_row, t_col = self.get_random_position()
        while (t_row, t_col) == (r_row, r_col) or abs(t_row - r_row) <= 2 or abs(t_col - r_col) <= 2:
            t_row, t_col = self.get_random_position()
        game_map[t_row][t_col] = "T"

    def get_random_position(self):
        return random.randint(1, self.num_rows - 2), random.randint(1, self.num_columns - 2)

    def find_agent_location(self, agent):
        for i in range(self.num_rows):
            for j in range(self.num_columns):
                if self.map[i][j] == agent:
                    return (i, j)
        return None

    def is_wall(self, y, x):
        return self.map[y][x] == '#'

    def print_map(self, turn_counter=10):
        if turn_counter > 0:
            for row in self.map:
                print(' '.join(row))


class Agent:
    def __init__(self, name, game_map, game):
        self.name = name
        self.game_map = game_map
        self.game = game
        self.previous_action = None
        self.q_sa = np.zeros((self.game_map.num_rows, self.game_map.num_columns))

    def act(self, action):
        y, x = self.game_map.find_agent_location(self.name)
        new_y, new_x = self.get_new_position(y, x, action)
        if new_y is not None and not self.game_map.is_wall(new_y, new_x):
            self.game_map.map[y][x] = " "
            self.game_map.map[new_y][new_x] = self.name
            return True
        return False

    def get_new_position(self, y, x, action):
        new_y, new_x = None, None
        if action == "up" and self.previous_action != "down" and y > 0:
            new_y, new_x = y - 1, x
        elif action == "down" and self.previous_action != "up" and y < self.game_map.num_rows - 1:
            new_y, new_x = y + 1, x
        elif action == "right" and self.previous_action != "left" and x < self.game_map.num_columns - 1:
            new_y, new_x = y, x + 1
        elif action == "left" and self.previous_action != "right" and x > 0:
            new_y, new_x = y, x - 1
        
        if new_y is not None and not self.game_map.is_wall(new_y, new_x):
            return new_y, new_x
        return None, None

    def reward_function(self):
        raise NotImplementedError

    def q_value_update(self, iterations, gamma=1):
        raise NotImplementedError

    def best_action(self, epsilon=0.05, alpha=0.95):
        y, x = self.game_map.find_agent_location(self.name)
        best_move = None
        top_q = -999999
        actions = ["up", "down", "left", "right"]

        if np.random.uniform() > alpha:
            print(f'The agent {self.name} has chosen an awaiting move: {best_move}, Turn = {self.game.turn_counter}')
            return best_move

        if np.random.uniform() < epsilon:
            best_move = random.choice(actions)
            if self.get_new_position(y, x, best_move) != (None, None):
                print(f'The agent {self.name} has chosen a random exploration move: {best_move}, Turn = {self.game.turn_counter}')
                return best_move

        for action in actions:
            new_y, new_x = self.get_new_position(y, x, action)
            if new_y is not None:
                current_q = self.q_sa[new_y][new_x]
                if current_q >= top_q:
                    top_q = current_q
                    best_move = action

        print(f'The agent {self.name} has chosen the best move: {best_move}, Turn = {self.game.turn_counter}, Q_sa = {top_q:.3f}')
        return best_move


class Runner(Agent):
    def reward_function(self):
        reward_list = np.zeros((self.game_map.num_rows, self.game_map.num_columns))
        y, x = self.game_map.find_agent_location("T")
        max_distance = (self.game_map.num_rows + self.game_map.num_columns - 2)
        for i in range(self.game_map.num_rows):
            for j in range(self.game_map.num_columns):
                distance = abs(i - y) + abs(j - x)
                reward_list[i][j] = -(max_distance-distance)  # Runner wants to maximize the distance, so we use negative distance
        return reward_list

    def q_value_update(self, iterations, gamma=1):
        reward_list = self.reward_function()
        for _ in range(iterations):
            for i in range(1, self.game_map.num_rows - 1):
                for j in range(1, self.game_map.num_columns - 1):
                    if self.game_map.map[i][j] == "R":
                        self.q_sa[i][j] = 0
                    else:
                        neighbors_q_values = []
                        if i > 0:
                            neighbors_q_values.append(self.q_sa[i-1][j])
                        if i < self.game_map.num_rows - 1:
                            neighbors_q_values.append(self.q_sa[i+1][j])
                        if j > 0:
                            neighbors_q_values.append(self.q_sa[i][j-1])
                        if j < self.game_map.num_columns - 1:
                            neighbors_q_values.append(self.q_sa[i][j+1])

                        if neighbors_q_values:
                            self.q_sa[i][j] = reward_list[i][j] + gamma * max(neighbors_q_values)
                        else:
                            self.q_sa[i][j] = reward_list[i][j]

    def best_action(self, epsilon=0.05, alpha=0.95):
        y, x = self.game_map.find_agent_location(self.name)
        best_move = None
        top_q = -np.inf
        actions = ["up", "down", "left", "right"]

        if np.random.uniform() < epsilon:
            best_move = random.choice(actions)
            new_y, new_x = self.get_new_position(y, x, best_move)
            if new_y is not None:
                print(f'The agent {self.name} has chosen a random exploration move: {best_move}, Turn = {self.game.turn_counter}')
                return best_move

        for action in actions:
            new_y, new_x = self.get_new_position(y, x, action)
            if new_y is not None:
                current_q = self.q_sa[new_y][new_x]
                if current_q > top_q:
                    top_q = current_q
                    best_move = action

        print(f'The agent {self.name} has chosen the best move: {best_move}, Turn = {self.game.turn_counter}, Q_sa = {top_q:.3f}')
        return best_move

class Tagger(Agent):
    def reward_function(self):
        reward_list = np.zeros((self.game_map.num_rows, self.game_map.num_columns))
        y, x = self.game_map.find_agent_location("R")
        reward = 1

        for i in range(self.game_map.num_rows):
            for j in range(self.game_map.num_columns):
                distance = abs(i - y) + abs(j - x)
                if distance == 1:
                    reward_list[i][j] = reward - 1
                elif distance == 0:
                    reward_list[i][j] = reward
                else:
                    reward_list[i][j] = -1
                    if self.game_map.is_wall(i, j):
                        reward_list[i][j] = -3

        return reward_list

    def q_value_update(self, iterations, gamma=1):
        for _ in range(iterations):
            for i in range(self.game_map.num_rows-1):
                for j in range(self.game_map.num_columns-1):
                    if self.game_map.map[i][j] == "T":
                        self.q_sa[i][j] = 0
                    else:
                        self.q_sa[i][j] = self.reward_function()[i][j] + gamma * max(
                            self.q_sa[i - 1][j], self.q_sa[i + 1][j], self.q_sa[i][j - 1], self.q_sa[i][j + 1]
                        )


class Game:
    def __init__(self, max_turns, iterations, map_height, map_width, wall_prob):
        self.max_turns = max_turns
        self.iterations = iterations
        self.turn_counter = 0
        self.game_map = GameMap(map_height, map_width, wall_prob)
        self.runner = Runner("R", self.game_map, self)
        self.tagger = Tagger("T", self.game_map, self)

    def terminal(self):
        y_r, x_r = self.game_map.find_agent_location("R")
        y_t, x_t = self.game_map.find_agent_location("T")
        if abs(y_r - y_t) + abs(x_r - x_t) == 1:
            print("Player Tag has won the game!")
            return True
        return False

    def play_turn(self):
        if self.turn_counter % 2 == 0:
            self.runner.q_value_update(self.iterations)
            best_act = self.runner.best_action()
            self.runner.act(best_act)
            self.runner.previous_action = best_act
        else:
            self.tagger.q_value_update(self.iterations)
            best_act = self.tagger.best_action()
            self.tagger.act(best_act)
            self.tagger.previous_action = best_act
        self.turn_counter += 1
        print("-" * (self.game_map.num_columns * 2))
        self.game_map.print_map(self.turn_counter)

    def run(self):
        while not self.terminal() and self.turn_counter < self.max_turns:
            self.play_turn()
        if self.turn_counter >= self.max_turns:
            print("Player Run has won the game!")
            return 1
        return -1


def main():
    print("Enter the number of maximum turns the Runner gets before winning the game")
    max_turns = int(input("maxTurns = "))
    print("Enter number of iterations the Bellman equation is allowed (1-100000)")
    iterations = int(input("iterations = "))
    print("Please enter the dimensions of your map:")
    map_height = int(input("Height = "))
    map_width = int(input("Width = "))
    wall_prob = float(input("Wall probability (0 to 1) = "))

    game = Game(max_turns, iterations, map_height, map_width, wall_prob)
    print("This will be your map for the game:")
    print("-------------------------------------")
    game.game_map.print_map(turn_counter=10)
    print("-------------------------------------")
    print("Do you wish to regenerate the map?")
    if input("Y/N? ") == "Y":
        game = Game(max_turns, iterations, map_height, map_width, wall_prob)

    start_time = time.time()
    game.run()
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f'Elapsed time: {elapsed_time:.3f} seconds, iterations = {iterations}, total turns = {game.turn_counter}, maxTurns = {max_turns}, wallProbability = {wall_prob}')
    if input("Would you like to simulate again? (Y/N)? ").lower() == "y":
        main()


if __name__ == "__main__":
    main()
