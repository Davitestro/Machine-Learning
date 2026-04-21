import pygame
import numpy as np
import random
import sys
from collections import deque
import QLearning

# =========================
# CONFIG
# =========================

GRID_SIZE = 8
CELL_SIZE = 80
WIDTH = GRID_SIZE * CELL_SIZE
HEIGHT = GRID_SIZE * CELL_SIZE

EPISODES = 3000
MAX_STEPS = 100
FPS = 8


# =========================
# PATH CHECK (IMPORTANT)
# =========================

def is_solvable(walls, size, start, goal):
    q = deque([start])
    visited = set([start])

    moves = [(1,0),(-1,0),(0,1),(0,-1)]

    while q:
        x, y = q.popleft()

        if (x, y) == goal:
            return True

        for dx, dy in moves:
            nx, ny = x + dx, y + dy

            if 0 <= nx < size and 0 <= ny < size:
                if (nx, ny) not in walls and (nx, ny) not in visited:
                    visited.add((nx, ny))
                    q.append((nx, ny))

    return False


# =========================
# MAZE GENERATION (DFS)
# =========================

def generate_map(size):
    maze = [[1 for _ in range(size)] for _ in range(size)]

    start = (0, 0)
    maze[start[0]][start[1]] = 0

    walls = []

    def add_walls(x, y):
        dirs = [(1,0),(-1,0),(0,1),(0,-1)]
        for dx, dy in dirs:
            nx, ny = x + dx, y + dy
            if 0 <= nx < size and 0 <= ny < size:
                if maze[nx][ny] == 1:
                    walls.append((nx, ny, x, y))  # wall + origin

    add_walls(*start)

    while walls:
        random.shuffle(walls)
        wx, wy, px, py = walls.pop()

        if maze[wx][wy] == 1:
            visited = 0
            for dx, dy in [(1,0),(-1,0),(0,1),(0,-1)]:
                nx, ny = wx + dx, wy + dy
                if 0 <= nx < size and 0 <= ny < size:
                    if maze[nx][ny] == 0:
                        visited += 1

            if visited <= 1:
                maze[wx][wy] = 0
                add_walls(wx, wy)

    walls_set = set()
    for i in range(size):
        for j in range(size):
            if maze[i][j] == 1:
                walls_set.add((i, j))

    return walls_set


# =========================
# ENVIRONMENT
# =========================

class GridWorld:
    def __init__(self, size):
        self.size = size
        self.start = (0, 0)
        self.goal = (size - 1, size - 1)

        # regenerate until solvable
        while True:
            walls = generate_map(size)
            if is_solvable(walls, size, self.start, self.goal):
                self.walls = walls
                break

        self.walls.discard(self.start)
        self.walls.discard(self.goal)

        self.reset()

    def reset(self):
        self.state = self.start
        return self.state

    def step(self, action):
        x, y = self.state

        moves = [(-1,0),(1,0),(0,-1),(0,1)]
        dx, dy = moves[action]

        nx, ny = x + dx, y + dy

        if nx < 0 or nx >= self.size or ny < 0 or ny >= self.size:
            nx, ny = x, y

        if (nx, ny) in self.walls:
            nx, ny = x, y

        self.state = (nx, ny)

        if self.state == self.goal:
            return self.state, 10, True

        return self.state, -1, False



# =========================
# TRAINING
# =========================

def train(env, agent):
    for ep in range(EPISODES):
        state = env.reset()

        for _ in range(MAX_STEPS):
            action = agent.act(state)
            next_state, reward, done = env.step(action)

            agent.learn(state, action, reward, next_state, done)

            state = next_state
            if done:
                break

        agent.epsilon *= 0.995
        agent.epsilon = max(agent.epsilon, 0.01)

        if ep % 300 == 0:
            print(f"Episode {ep} | epsilon={agent.epsilon:.3f}")


# =========================
# RENDER
# =========================

def draw(screen, env, path):
    colors = {
        "bg": (30,30,30),
        "grid": (60,60,60),
        "wall": (20,20,20),
        "agent": (255,80,80),
        "goal": (80,255,80),
        "path": (80,80,255)
    }

    screen.fill(colors["bg"])

    for x in range(env.size):
        for y in range(env.size):
            rect = pygame.Rect(y*CELL_SIZE, x*CELL_SIZE, CELL_SIZE, CELL_SIZE)

            if (x, y) in env.walls:
                pygame.draw.rect(screen, colors["wall"], rect)

            pygame.draw.rect(screen, colors["grid"], rect, 1)

    gx, gy = env.goal
    pygame.draw.rect(screen, colors["goal"],
                     (gy*CELL_SIZE, gx*CELL_SIZE, CELL_SIZE, CELL_SIZE))

    for (x, y) in path:
        pygame.draw.circle(
            screen,
            colors["path"],
            (y*CELL_SIZE + CELL_SIZE//2, x*CELL_SIZE + CELL_SIZE//2),
            CELL_SIZE//6
        )

    x, y = env.state
    pygame.draw.circle(
        screen,
        colors["agent"],
        (y*CELL_SIZE + CELL_SIZE//2, x*CELL_SIZE + CELL_SIZE//2),
        CELL_SIZE//3
    )


# =========================
# RUN VISUAL TEST
# =========================

def run(env, agent):
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("RL GridWorld - SOLVABLE RANDOM MAZE")
    clock = pygame.time.Clock()

    state = env.reset()
    path = [state]

    done = False

    while True:
        clock.tick(FPS)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        if not done:
            action = agent.act(state)
            next_state, reward, done = env.step(action)

            agent.learn(state, action, reward, next_state, done)

            state = next_state
            path.append(state)

        draw(screen, env, path)
        pygame.display.flip()


# =========================
# MAIN
# =========================

if __name__ == "__main__":
    env = GridWorld(GRID_SIZE)
    agent = QLearning.QAgent(GRID_SIZE)

    print("🧠 Training...")
    train(env, agent)

    print("🎮 Running...")
    agent.epsilon = 0

    run(env, agent)