from collections import deque
import random, os, torch, time
import numpy as np

OUT_OF_BOUNDS = -1
GRASS = 0
APPLE = 1
SNAKE = 2

ADJ = ((-1,-1), (0,-1), (1,-1), (-1,0), (1,0), (-1, 1), (0, 1), (1, 1))
ACTION_VECTORS = ((-1,0), (0,1), (1,0), (0,-1))
ACTIONS = [0,1,2,3]

class VectorCalc:
    @staticmethod
    def add(t0: tuple, t1: tuple):
        return tuple(t0[i] + t1[i] for i in range(min(len(t0), len(t1))))

    @staticmethod
    def multiply(t0: tuple, a: int):
        return tuple(t0[i] * a for i in range(len(t0)))


class SnakeEnvironment:

    def __init__(self):
        # rows, cols
        self.boardDims = (10, 10) 

    # reset the environment for a new episode
    def reset(self):
        # snake is stored in a queue, FIFO
        # this ensures that the head is always being dequeued and the tail enqueued
        self.snake = deque()
        self.grass = set()
        self.apple = None

        # state facing north
        self.currentDirection = 1 

        # snake starting position is always in the middle of the board
        startSnakeHead = (self.boardDims[0]//2, self.boardDims[1]//2)
        self.snake.appendleft(startSnakeHead)

        # fill the board with grass except from the snake head
        for y in range(self.boardDims[0]):
            for x in range(self.boardDims[1]):
                pos = (y, x)
                self.grass.add(pos)
        self.grass.remove(self.snake[0])

        # spawn the apple in random grass position
        self._addApple()

    # choose a random grass position from the grass set
    def _randomGrassPos(self):
        grass = tuple(self.grass)
        if len(grass) <= 0: 
            return (-1, -1) # position outside the board
        return random.choice(grass)

    def _addApple(self):
        pos = self._randomGrassPos()
        self.apple = pos
        if pos != (-1, -1):
            self.grass.remove(pos)

    def _isOutOfBounds(self, pos):
        return pos[0] < 0 or pos[0] >= self.boardDims[0] or pos[1] < 0 or pos[1] >= self.boardDims[1]

    def _entityAtPos(self, pos):
        if self._isOutOfBounds(pos):
            return OUT_OF_BOUNDS
        elif pos in self.grass:
            return GRASS
        elif pos == self.apple:
            return APPLE
        else:
            return SNAKE

    def _isDangerousPos(self, pos):
        entity = self._entityAtPos(pos)
        return entity == OUT_OF_BOUNDS or entity == SNAKE

    def _encodeDangers(self):
        dangers = [0 for _ in range(8)]
        for i, adjVector in enumerate(ADJ):
            adjPos = VectorCalc.add(self.snake[0], adjVector)
            dangers[i] = int(self._isDangerousPos(adjPos))
        return dangers

    def _encodeSnakeDirection(self):
        direction = [0 for _ in range(4)]
        direction[self.currentDirection] = 1
        return direction

    def _encodeAppleDirection(self):
        # self.apple - self.snake[0]
        toApple = VectorCalc.add(self.apple, VectorCalc.multiply(self.snake[0], -1))
        north = toApple[0] < 0
        south = toApple[0] > 0
        east = toApple[1] > 0
        west = toApple[1] < 0
        return [north, east, south, west]

    # encode environment into a single flat feature vector for passing into MLP
    def _encodeState(self):
        return self._encodeDangers() + self._encodeSnakeDirection() + self._encodeAppleDirection()

    def startState(self):
        return self._encodeState()

    # perform the environment step. returns the next state, reward, and whether the episode terminated
    def step(self, action: int):
        moveVector = ACTION_VECTORS[action]
        newSnakeHead = VectorCalc.add(self.snake[0], moveVector)
        entityAtHead = self._entityAtPos(newSnakeHead)
        
        # check if episode terminated 
        if entityAtHead == SNAKE or entityAtHead == OUT_OF_BOUNDS:
            return self._encodeState(), -50, True  # terminated with death

        # update environment state
        self.snake.appendleft(newSnakeHead)
        self.currentDirection = action      

        if entityAtHead == GRASS:
            self.grass.remove(newSnakeHead)
            tail = self.snake.pop()
            self.grass.add(tail)
            return self._encodeState(), -0.01, False # stayed alive without eating apple
        
        # else entity at head is apple
        # don't remove tail, need to make the snake longer
        self._addApple()

        # the snake fills the entire board, so the apple was placed outside the board
        if self.apple == (-1,-1):
            return self._encodeState(), 100, True
        
        # snake ate apple and does not fill board
        return self._encodeState(), 5, False 
    
    # print the board to the terminal to watch an episode and view learning progress
    def printBoard(self):
        os.system("cls")
        display = "@@" * (self.boardDims[1] + 2) + "\n"
        for y in range(self.boardDims[0]):
            row = "@@"
            for x in range(self.boardDims[1]):
                pos = (y, x)
                if pos in self.grass:
                    char = "  "
                elif pos == self.apple:
                    char = "**"
                elif pos == self.snake[0]:
                    char = "[]"
                else:
                    char = "<>"
                row += char
            row += "@@" + "\n"
            display += row
        display += "@@" * (self.boardDims[1] + 2)
        print(display)
        time.sleep(1/10)
