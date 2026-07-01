from collections import deque
import random, os, time

WALL = -1
GRASS = 0
APPLE = 1
SNAKE = 2

#          N E S W
ACTIONS = [0,1,2,3]
ACTION_VECTORS = ((-1,0), (0,1), (1,0), (0,-1))
ADJ_VECTORS = ((-1,-1), (0,-1), (1,-1), (-1,0), (1,0), (-1, 1), (0, 1), (1, 1))

class VectorCalc:
    @staticmethod
    def add(t0: tuple, t1: tuple):
        return tuple(t0[i] + t1[i] for i in range(min(len(t0), len(t1))))

    @staticmethod
    def multiply(t0: tuple, a: int):
        return tuple(t0[i] * a for i in range(len(t0)))


class SnakeEnvironment:
    def __init__(self):
        # width, height
        self.board_dims = (10, 10)


    def reset(self):
        # snake is stored in a queue, FIFO
        # this ensures that the head is always being dequeued and the tail enqueued
        self.snake = deque()
        self.grass = set()
        self.walls = set()
        self.apple = None

        # state facing north
        self.curr_dir = 0

        # snake starting position is always in the middle of the board
        snake_head = (self.board_dims[0]//2, self.board_dims[1]//2)
        self.snake.appendleft(snake_head)

        # fill in the walls in a ring around the outside
        for y in range(self.board_dims[0]):
            self.walls.add((y, 0))
            self.walls.add((y, self.board_dims[1]-1))
        for x in range(self.board_dims[1]):
            self.walls.add((0, x))
            self.walls.add((self.board_dims[0]-1, x))

        # fill in the grass
        for y in range(self.board_dims[0]):
            for x in range(self.board_dims[1]):
                if (y, x) not in self.walls:
                    self.grass.add((y, x))
        self.grass.remove(self.snake[0])

        # spawn the apple in random grass position
        self.add_apple()

    
    def random_grass_pos(self):
        # choose a random grass position from the grass set
        grass = tuple(self.grass)
        if len(grass) <= 0: 
            return (-1, -1) # position outside the board
        return random.choice(grass)


    def add_apple(self):
        pos = self.random_grass_pos()
        self.apple = pos
        if pos != (-1, -1):
            self.grass.remove(pos)


    def entity_at_pos(self, pos):
        # return the entity at the position the snake head is trying to move to
        if pos in self.walls:
            return WALL
        elif pos in self.grass:
            return GRASS
        elif pos == self.apple:
            return APPLE
        else:
            return SNAKE


    def is_dangerous_pos(self, pos):
        entity = self.entity_at_pos(pos)
        return entity == WALL or entity == SNAKE


    def encode_state_for_mlp(self):
        # encode whether each adjacent position is dangerous
        dangers = [0 for _ in range(len(ADJ_VECTORS))]
        for i, adj_vector in enumerate(ADJ_VECTORS):
            pos = VectorCalc.add(self.snake[0], adj_vector)
            dangers[i] = int(self.is_dangerous_pos(pos))
        
        # encode snakes current direction (one hot)
        direction = [0 for _ in range(len(ACTIONS))]
        direction[self.curr_dir] = 1

        # encode the direction from snake head to apple
        to_apple = VectorCalc.add(self.apple, VectorCalc.multiply(self.snake[0], -1))
        north = int(to_apple[0] < 0)
        south = int(to_apple[0] > 0)
        east = int(to_apple[1] > 0)
        west = int(to_apple[1] < 0)

        # 16 feature state vector
        return dangers + direction + [north, east, south, west]


    def encode_state_for_cnn(self):
        walls = [[0 for _ in range(self.board_dims[1])] for _ in range(self.board_dims[0])]
        snakeHead = [[0 for _ in range(self.board_dims[0])] for _ in range(self.board_dims[1])]
        snakeBody = [[0 for _ in range(self.board_dims[0])] for _ in range(self.board_dims[1])]
        apple = [[0 for _ in range(self.board_dims[0])] for _ in range(self.board_dims[1])]

        for y in range(self.board_dims[0]):
            for x in range(self.board_dims[1]):
                if (y, x) in self.grass:
                    continue
                if (y, x) in self.walls:
                    walls[y][x] = 1
                elif (y, x) == self.apple:
                    apple[y][x] = 1
                elif (y, x) == self.snake[0]:
                    snakeHead[y][x] = 1
                else:
                    snakeBody[y][x] = 1
        
        return [walls, snakeHead, snakeBody, apple]


    def step(self, action: int):
        # perform the environment step - returns the reward, and whether the episode terminated
        move_vector = ACTION_VECTORS[action]
        new_snake_head = VectorCalc.add(self.snake[0], move_vector)
        entity_at_new_head = self.entity_at_pos(new_snake_head)
        
        if self.is_dangerous_pos(new_snake_head):
            return -50, True # died

        # update environment state
        self.snake.appendleft(new_snake_head)
        self.currentDirection = action      

        if entity_at_new_head == GRASS:
            self.grass.remove(new_snake_head)
            tail = self.snake.pop()
            self.grass.add(tail)
            return -0.01, False # alive but didn't eat apple
        
        # else entity at head is apple
        # don't remove tail, need to make the snake longer
        self.add_apple()

        if self.apple == (-1,-1):
            return 100, True # ate apple, filled board (won)
        
        return 5, False # ate apple


    def printBoard(self):
        # print the board to the terminal to watch an episode and view learning progress
        os.system("cls")
        board = [["" for _ in range(self.board_dims[1])] for _ in range(self.board_dims[0])]

        for y in range(self.board_dims[0]):
            for x in range(self.board_dims[1]):
                if (y, x) in self.walls:
                    board[y][x] = "@@"
                elif (y, x) in self.grass:
                    board[y][x] = "  "
                elif (y, x) == self.apple:
                    board[y][x] = "**"
                elif (y, x) == self.snake[0]:
                    board[y][x] = "[]"
                else:
                    board[y][x] = "<>"
        
        boardstr = "\n".join(["".join(row) for row in board])
        print(boardstr)
        time.sleep(1/10)
