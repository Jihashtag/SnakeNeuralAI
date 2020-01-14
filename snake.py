#!/usr/bin/env python3

import argparse
import curses
from curses import KEY_RIGHT, KEY_LEFT, KEY_UP, KEY_DOWN
from multiprocessing import Pool
import numpy as np
from time import sleep
from random import randint, random, seed

DEFAULT_VALUES = (KEY_RIGHT, 0, 0)
HEIGHT = 19
WIDTH = 59
KEY = KEY_RIGHT


class NeuralNetwork:
    def __init__(self, weights):
        self.weights = weights

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def play(self, horizon):
        return self.sigmoid(np.dot(np.array(horizon), self.weights))


class Game:
    def __init__(self, best=False, turns=None):
        if best:
            curses.initscr()
            self.win = curses.newwin(20, 60, 0, 0)
            self.win.keypad(1)
            curses.noecho()
            curses.curs_set(0)
            self.win.border(0)
            self.win.nodelay(1)

        self.best = best
        self.turns = turns
        self.key, self.score, self.turn = DEFAULT_VALUES
        self.snake = [[10, 30], [10, 29], [10, 28]]
        self.positions = []
        self.place_food()

    def __iter__(self):
        return self

    def place_food(self):
        food = []
        while food == []:
            food = [randint(1, HEIGHT - 1), randint(1, WIDTH - 1)]
            if food in self.snake:
                food = []
        if self.best:
            self.win.addch(food[0], food[1], '*')
        self.food = food

    def __next__(self):
        global KEY
        if self.key == 27 or (self.turns and self.turn == self.turns):
            raise StopIteration
        self.turn += 1

        if self.best:
            self.win.border(0)
            self.win.addstr(0, 2, 'Score : ' + str(int(self.score)) + ' ')
            self.win.addstr(0, 27, ' SNAKE ')
            self.win.timeout(150 - round(len(self.snake) / 5 + len(self.snake) / 10) % 120)

        prevKey = self.key
        event = self.win.getch() if self.best else KEY
        if event != -1:
            self.key = event
        else:
            self.key = KEY

        if self.key not in [KEY_LEFT, KEY_RIGHT, KEY_UP, KEY_DOWN, 27]:
            self.key = prevKey

        # Calculates the new coordinates of the head of the snake.
        self.snake.insert(0, [
            self.snake[0][0] + (self.key == KEY_DOWN and 1) + (self.key == KEY_UP and -1),
            self.snake[0][1] + (self.key == KEY_LEFT and -1) + (self.key == KEY_RIGHT and 1)
        ])

        if self.snake[0][0] == 0 or self.snake[0][1] == 0 \
                or self.snake[0][0] == HEIGHT or self.snake[0][1] == WIDTH \
                or self.snake[0] in self.snake[1:]:
            raise StopIteration

        if self.snake[0] == self.food:
            self.score += 10
            self.place_food()
            self.positions = []
        else:
            last = self.snake.pop()
            if self.turns and last in self.positions:
                raise StopIteration
            self.positions.append(last)
            if self.best:
                self.win.addch(last[0], last[1], ' ')
        if self.best:
            self.win.addch(self.snake[0][0], self.snake[0][1], '#')

        return self.snake, self.food, self.score, self.turn


def get_map(snake, food, horizon=1):
    head = snake[0]
    square_size = horizon + horizon + 1
    area = square_size * square_size
    len_map = area - 1
    pseudo_map = [0] * len_map

    i = 0
    for x in range(-horizon, horizon + 1):
        for y in range(-horizon, horizon + 1):
            # Border
            if head[0] + x <= 0 or head[1] + y <= 0 or head[0] + x >= HEIGHT or head[1] + y >= WIDTH:
                pseudo_map[i] = -1
            # Food
            elif food == [head[0] + x, head[1] + y]:
                pseudo_map[i] = 1
            # Self
            else:
                for v in snake[1:]:
                    if v == [head[0] + x, head[1] + y]:
                        pseudo_map[i] = -1
                        break
            if x != 0 or y != 0:
                i += 1

    pseudo_map.append((head[0] - food[0]) / HEIGHT)
    pseudo_map.append((head[1] - food[1]) / WIDTH)
    pseudo_map.append((food[0] - head[0]) / HEIGHT)
    pseudo_map.append((food[1] - head[1]) / WIDTH)

    return pseudo_map


def game_loop(data):
    global KEY
    KEY = KEY_RIGHT
    if not data["keepSeed"]:
        data["seed"] = random()
    seed(data["seed"])
    game = iter(Game(data["best"], data["turns"]))
    IA = NeuralNetwork(data["weights"])
    score = 0
    turn = 0
    used = []
    for snake, food, score, turn in game:
        pseudo_map = get_map(snake, food, data["horizon"])
        keys = IA.play(pseudo_map)
        if max(keys) == keys[0]:
            KEY = KEY_UP
        elif max(keys) == keys[1]:
            KEY = KEY_DOWN
        elif max(keys) == keys[2]:
            KEY = KEY_LEFT
        elif max(keys) == keys[3]:
            KEY = KEY_RIGHT
        else:
            raise Exception("This should not happen")
        if KEY not in used:
            used.append(KEY)

    if data["best"]:
        curses.endwin()
    data["score"] = round(score)
    data["turn"] = turn
    data["keys"] = len(used)
    return data


def train(args):
    global LBESTS
    data_list = []
    nbest = args.nbest
    ncopy = round(args.snakes / nbest) - 1

    square_size = args.horizon + args.horizon + 1
    area = square_size * square_size
    len_map = area - 1

    for _ in range(args.snakes):
        data_list.append({
            "weights": np.random.rand(len_map + 4, 4),
            "horizon": args.horizon,
            "turns": args.fstep,
            "best": False,
            "keepSeed": False
        })

    pool = Pool(8)
    for gen in range(1, args.gens + 1):
        ret = pool.map(game_loop, data_list)
        # ret = list(map(game_loop, data_list))
        ret.sort(key=lambda x: -(x['keys'] * 5) - x["score"] + (x['turns'] - (x['turn'] + args.step)))
        bests = ret[:nbest]
        print("Gen", gen, ":", *list((x["score"], "{}/{}".format(x["turn"], x["turns"])) for x in bests[:10]))

        LBESTS = bests[:3]
        if gen == args.gens:
            continue

        for data in bests:
            if data["turn"] > data["turns"] - args.step:
                data["turns"] += args.step if data['turns'] <= args.step * 100 else args.step * 10

            data["best"] = False
            data["keepSeed"] = False

        based_on_bests = []
        for best in bests:
            for _ in range(ncopy):
                copy = best.copy()
                if random() > 0.5:
                    copy["weights"] = best["weights"] + np.dot(np.random.rand(len_map + 4, 4), 0.1)
                else:
                    copy["weights"] = best["weights"] - np.dot(np.random.rand(len_map + 4, 4), 0.1)
                based_on_bests.append(copy)
        best = bests[0].copy()
        best["keepSeed"] = True
        best["best"] = args.ncurse

        data_list = [best, *bests, *based_on_bests]

    return LBESTS


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Snake Neural AI")
    parser.add_argument('-s', '--seed', type=float, dest='seed', help='Seed to use', default=random())
    parser.add_argument('-sn', '--snakes', type=int, dest='snakes', help='Number of snakes per gen', default=1000)
    parser.add_argument('-g', '--gen', type=int, dest='gens', help='Number of gen', default=1000)
    parser.add_argument('-hb', '--hide-best', dest='ncurse', action='store_false', help='Hide best of each gen', default=True)
    parser.add_argument('-it', '--init-turns', type=int, dest='fstep', help='Start with x step', default=20)
    parser.add_argument('-step', '--step', type=int, dest='step', help='Number of steps', default=10)
    parser.add_argument('-nb', '--num-best', type=int, dest='nbest', help='Number of bests duplicated', default=10)
    parser.add_argument('-sh', '--snake-horizon', type=int, dest='horizon', help='Vision of the snake', default=1)

    args = parser.parse_args()

    seed(args.seed)

    LBESTS = None
    try:
        bests = train(args)
    except KeyboardInterrupt:
        bests = LBESTS
        print('Leaving')
        sleep(1)
        input('Continue ? (press enter)')

    for best in bests:
        best["best"] = True
        best["turns"] = None
        game_loop(best)
        print(best["score"], best["turn"])
    print(args.seed)
    answer = input('Redo without Seed ? (Y/n) ')
    if answer and answer[0] == 'n':
        exit()
    for best in bests:
        best['keepSeed'] = False
        best["best"] = True
        best["turns"] = None
        game_loop(best)
        print(best["score"], best["turn"])
