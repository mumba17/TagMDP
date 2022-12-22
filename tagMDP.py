import numpy as np
import random
import time

def load_game_map(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
    return [[c for c in line.strip()] for line in lines]

def generate_random_map(num_rows, num_columns, wall_probability):
    game_map = []
    for i in range(num_rows):
        row = []
        for j in range(num_columns):
            if i == 0 or i == num_rows - 1 or j == 0 or j == num_columns - 1:
                # Place a wall on the edges of the map
                row.append("#")
            elif random.random() < wall_probability:
                # Place a wall with probability wall_probability
                row.append("#")
            else:
                row.append(" ")
        game_map.append(row)
    # Place agent R at a random location
    r_row = random.randint(1, num_rows - 2)
    r_column = random.randint(1, num_columns - 2)
    game_map[r_row][r_column] = "R"
    # Place agent T at a random location, avoiding the location of agent R
    # and locations that are next to agent R
    t_row = random.randint(1, num_rows - 2)
    t_column = random.randint(1, num_columns - 2)
    while (t_row, t_column) == (r_row, r_column) or abs(t_row - r_row) <= 1 or abs(t_column - r_column) <= 1:
        t_row = random.randint(1, num_rows - 2)
        t_column = random.randint(1, num_columns - 2)
    game_map[t_row][t_column] = "T"
    return game_map

# game_map = load_game_map('map.txt')
taggerLocation = (0,0)
list_of_actions = ["up","down","left","right"]
TurnCounter = 0

def find_agent_location(agent):
    for i in range(num_rows):
        for j in range(num_columns):
            if game_map[i][j] == agent:
                return (i, j)
    return None

def isWall(currentY, currentX):
    if game_map[currentY][currentX] == '#':
        return True
    else:
        return False

def Act(action,agent,previous):
    locationy,locationx = find_agent_location(agent)
    if action == "up" and previous != "down":
        if locationy < len(game_map) and not isWall(locationy+1,locationx):
            game_map[locationy][locationx] = " "
            game_map[locationy+1][locationx] = agent
            return True
    if action == "down" and previous != "up":
        if locationy > 0 and not isWall(locationy-1,locationx):
            game_map[locationy][locationx] = " "
            game_map[locationy-1][locationx] = agent
            return True
    if action == "right" and previous != "left":
        if locationx < len(game_map[0]) and not isWall(locationy,locationx+1):
            game_map[locationy][locationx] = " "
            game_map[locationy][locationx+1] = agent
            return True
    if action == "left" and previous != "right":
        if locationx > 0 and not isWall(locationy,locationx-1):
            game_map[locationy][locationx] = " "
            game_map[locationy][locationx-1] = agent
            return True
    return False

def ActCords(action,agent,previous):
    locationy,locationx = find_agent_location(agent)
    if action == "up" and previous != "down":
        if locationy < len(game_map) and not isWall(locationy+1,locationx):
            return (locationy+1,locationx)
    if action == "down" and previous != "up":
        if locationy > 0 and not isWall(locationy-1,locationx):
            return (locationy-1,locationx)
    if action == "right" and previous != "left":
        if locationx < len(game_map[0]) and not isWall(locationy,locationx+1):
            return (locationy,locationx+1)
    if action == "left" and previous != "right":
        if locationx > 0 and not isWall(locationy,locationx-1):
            return (locationy,locationx-1)
    return False


def rewardFunctionTag():
    global rewardListTagger
    rewardListTagger = np.zeros((num_rows,num_columns))
    locationy,locationx = find_agent_location("R")
    reward = 50
    if locationy > 0 and not isWall(locationy-1, locationx):
        rewardListTagger[locationy-1, locationx] = reward
    if locationy < num_rows-1 and not isWall(locationy+1, locationx):
        rewardListTagger[locationy+1, locationx] = reward
    if locationx > 0 and not isWall(locationy, locationx-1):
        rewardListTagger[locationy, locationx-1] = reward
    if locationx < num_columns-1 and not isWall(locationy, locationx+1):
        rewardListTagger[locationy, locationx+1] = reward
    for i in range(num_rows):
        for j in range(num_columns):
            if rewardListTagger[i][j] != reward and find_agent_location("R") != (i,j) and not isWall(i,j):
                rewardListTagger[i][j] = -1

def rewardFunctionRun():
    global rewardListRun
    rewardListRun = np.zeros((num_rows,num_columns))
    locationy,locationx = find_agent_location("T")
    reward = -100
    for i in range(num_rows):
        for j in range(num_columns):
            if isWall(i,j):
                rewardListRun[i][j] = -50
            else:
                # Encourage the AI to stay away from the Tagger
                distance_from_tagger = abs(i - locationy) + abs(j - locationx)
                if distance_from_tagger > 0:
                    rewardListRun[i][j] = reward / distance_from_tagger
                else:
                    rewardListRun[i][j] = reward
    return rewardListRun

def Terminal():
    locationyR, locationxR = find_agent_location("R")
    locationyT, locationxT = find_agent_location("T")
    if abs(locationyR - locationyT) + abs(locationxR - locationxT) == 1:
        print("Player Tag has won the game!")
        return True


def Q_value_Run(z,gamma=1):
    Q_sa = np.zeros([num_rows, num_columns])
    for i in range(z):
        for i in range(num_rows):
            for j in range(num_columns):
                if game_map[i][j] == '#':
                    continue
                Q_sa[i][j] = rewardListRun[i][j] + gamma * max(Q_sa[i-1][j], Q_sa[i+1][j], Q_sa[i][j-1], Q_sa[i][j+1])
    return Q_sa


def Q_value_Tag(z,gamma=1):
    Q_sa = np.zeros([num_rows, num_columns])
    for i in range(z):
        for i in range(num_rows):
            for j in range(num_columns):
                if game_map[i][j] == '#':
                    continue
                Q_sa[i][j] = rewardListTagger[i][j] + gamma * max(Q_sa[i-1][j], Q_sa[i+1][j], Q_sa[i][j-1], Q_sa[i][j+1])
    return Q_sa

def bestAction(agent,Q_sa,previous, epsilon=0.1):
    i,j = find_agent_location(agent)
    Top = -999999
    bestMove = 0
    cantRandom = True
    if np.random.uniform() < epsilon:
        i = np.random.randint(len(list_of_actions))
        if ActCords(list_of_actions[i],agent,previous) != False:
            bestMove = list_of_actions[i]
            cantRandom = False
            print(f'The agent {agent} has chosen a random exploration move: {bestMove}, Turn = {TurnCounter}')
    if cantRandom:
        for act in list_of_actions:
            if ActCords(act,agent,previous) != False:
                y,x = ActCords(act,agent,previous)
                current = Q_sa[y][x]
                if current > Top:
                    Top = current
                    bestMove = act
        print(f'The agent {agent} has chosen the best move: {bestMove}, Turn = {TurnCounter}, Q_sa = {Top:.3f}')
    return bestMove

def printMap(game):
    for row in game:
        print(' '.join(row))

def main():
    global num_columns,num_rows,game_map,TurnCounter,previousActR,previousActT
    previousActR = 0
    previousActT = 0
    TurnCounter = 0
    print("Enter the number of maximum turns the Runner gets before winning the game")
    maxTurns = input("maxTurns = ")
    maxTurns = int(maxTurns)
    print("Enter number of iterations the Bellman equation is allowed (1-100000)")
    z = input("z = ")
    z = int(z)
    print("Please enter the dimensions of your map:")
    m = input("Height = ")
    m = int(m)
    n = input("Width = ")
    n = int(n)
    az = input("Wall probability (0 to 1) = ")
    wallprob = float(az)
    game_map = generate_random_map(m,n,wallprob)
    print("This will be your map for the game:")
    print("-------------------------------------")
    printMap(game_map)
    print("-------------------------------------")
    print("Do you wish to regenerate the map?")
    regen = input("Y/N? ")
    if regen == "Y":
        game_map = generate_random_map(m,n,wallprob)

    num_rows = len(game_map)
    num_columns = len(game_map[0])
    start_time = time.time()
    while not Terminal():
        if TurnCounter % 2 == 0:
            rewardFunctionRun()
            Q_run = Q_value_Run(z)
            agent = "R"
            bestAct = bestAction(agent,Q_run,previousActR)
            Act(bestAct,agent,previousActR)
            previousActR = bestAct
        if TurnCounter % 2 == 1:
            rewardFunctionTag()
            Q_tag = Q_value_Tag(z)
            agent = "T"
            bestAct = bestAction(agent,Q_tag,previousActT)
            Act(bestAct,agent,previousActT)
            previousActT = bestAct
        TurnCounter += 1
        print("-"*(n*2))
        printMap(game_map)
        if TurnCounter == maxTurns:
            print("Player Run has won the game!")
            break
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f'Elapsed time: {elapsed_time:.3f} seconds, z = {z:.2f}, total turns = {TurnCounter}, maxTurns = {maxTurns}, wallProbability = {wallprob}')
    agane = input("Would you like to simulate again? (Y/N)? ")
    if agane == "Y" or agane == "y":
        main()

main()