import random
import pygame
import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import os

# Get the absolute path of the directory containing this script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Initialize Pygame
pygame.init()
pygame.mixer.init()
music_path = os.path.join(BASE_DIR, 'tetris_music.mp3')
pygame.mixer.music.load(music_path)
pygame.mixer.music.set_volume(0.5)

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
CYAN = (0, 255, 255)
YELLOW = (255, 255, 0)
MAGENTA = (255, 0, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
ORANGE = (255, 165, 0)
GRAY = (40, 40, 40)

# Game dimensions
BLOCK_SIZE = 30
GRID_WIDTH = 10
GRID_HEIGHT = 20
SCREEN_WIDTH = BLOCK_SIZE * (GRID_WIDTH + 6)
SCREEN_HEIGHT = BLOCK_SIZE * GRID_HEIGHT

# Tetromino shapes
SHAPES = [
    [[1, 1, 1, 1]],
    [[1, 1], [1, 1]],
    [[1, 1, 1], [0, 1, 0]],
    [[1, 1, 1], [1, 0, 0]],
    [[1, 1, 1], [0, 0, 1]],
    [[1, 1, 0], [0, 1, 1]],
    [[0, 1, 1], [1, 1, 0]]
]

SHAPE_COLORS = [CYAN, YELLOW, MAGENTA, RED, GREEN, BLUE, ORANGE]

# AI move delay (in seconds)
AI_MOVE_DELAY = 0.2

# Neural Network Architecture
class TetrisNN(nn.Module):
    def __init__(self):
        super(TetrisNN, self).__init__()
        self.fc1 = nn.Linear(5, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class Tetris:
    def __init__(self):
        self.reset_game()

    def reset_game(self):
        self.grid = np.zeros((GRID_HEIGHT, GRID_WIDTH), dtype=int)
        self.color_grid = [[None for _ in range(GRID_WIDTH)] for _ in range(GRID_HEIGHT)]
        self.current_piece = self.new_piece()
        self.next_piece = self.new_piece()
        self.score = 0
        self.lines_cleared = 0
        self.level = 1
        self.fall_speed = 0.5

    def new_piece(self):
        shape = random.choice(SHAPES)
        return {
            'shape': shape,
            'color': SHAPE_COLORS[SHAPES.index(shape)],
            'x': GRID_WIDTH // 2 - len(shape[0]) // 2,
            'y': 0
        }

    def valid_move(self, piece, x, y):
        for i, row in enumerate(piece['shape']):
            for j, cell in enumerate(row):
                if cell:
                    if (x + j < 0 or x + j >= GRID_WIDTH or
                        y + i >= GRID_HEIGHT or
                        (y + i >= 0 and self.grid[y + i][x + j])):
                        return False
        return True

    def place_piece(self):
        for i, row in enumerate(self.current_piece['shape']):
            for j, cell in enumerate(row):
                if cell:
                    y = self.current_piece['y'] + i
                    x = self.current_piece['x'] + j
                    if 0 <= y < GRID_HEIGHT and 0 <= x < GRID_WIDTH:
                        self.grid[y][x] = 1
                        self.color_grid[y][x] = self.current_piece['color']
                    else:
                        return False
        self.clear_lines()
        self.current_piece = self.next_piece
        self.next_piece = self.new_piece()
        if not self.valid_move(self.current_piece, self.current_piece['x'], self.current_piece['y']):
            return False
        return True

    def clear_lines(self):
        lines_cleared = 0
        for i in range(GRID_HEIGHT):
            if all(self.grid[i]):
                lines_cleared += 1
                for y in range(i, 0, -1):
                    self.grid[y] = self.grid[y-1]
                    self.color_grid[y] = self.color_grid[y-1]
                self.grid[0] = [0] * GRID_WIDTH
                self.color_grid[0] = [None] * GRID_WIDTH
        
        self.lines_cleared += lines_cleared
        if lines_cleared == 1:
            self.score += 1
        elif lines_cleared == 2:
            self.score += 3
        elif lines_cleared == 3:
            self.score += 6
        elif lines_cleared == 4:
            self.score += 15
        
        if self.lines_cleared // 10 > (self.lines_cleared - lines_cleared) // 10:
            self.level += 1
            self.fall_speed *= 0.97

    def rotate_piece(self):
        rotated = list(zip(*self.current_piece['shape'][::-1]))
        if self.valid_move({'shape': rotated, 'x': self.current_piece['x'], 'y': self.current_piece['y']}, self.current_piece['x'], self.current_piece['y']):
            self.current_piece['shape'] = rotated

    def move(self, dx, dy):
        if self.valid_move(self.current_piece, self.current_piece['x'] + dx, self.current_piece['y'] + dy):
            self.current_piece['x'] += dx
            self.current_piece['y'] += dy
            return True
        return False

    def drop_piece(self):
        while self.move(0, 1):
            pass
        return self.place_piece()

    def draw(self, screen):
        screen.fill(BLACK)
        pygame.draw.rect(screen, GRAY, (GRID_WIDTH * BLOCK_SIZE, 0, SCREEN_WIDTH - GRID_WIDTH * BLOCK_SIZE, SCREEN_HEIGHT))

        for i, row in enumerate(self.grid):
            for j, cell in enumerate(row):
                if cell:
                    color = self.color_grid[i][j] or WHITE
                    pygame.draw.rect(screen, color, (j * BLOCK_SIZE, i * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE), 0)

        for i, row in enumerate(self.current_piece['shape']):
            for j, cell in enumerate(row):
                if cell:
                    pygame.draw.rect(screen, self.current_piece['color'],
                                     ((self.current_piece['x'] + j) * BLOCK_SIZE,
                                      (self.current_piece['y'] + i) * BLOCK_SIZE,
                                      BLOCK_SIZE, BLOCK_SIZE), 0)

        for i, row in enumerate(self.next_piece['shape']):
            for j, cell in enumerate(row):
                if cell:
                    pygame.draw.rect(screen, self.next_piece['color'],
                                     ((GRID_WIDTH + 1 + j) * BLOCK_SIZE,
                                      (1 + i) * BLOCK_SIZE,
                                      BLOCK_SIZE, BLOCK_SIZE), 0)

        font = pygame.font.Font(None, 36)
        score_text = font.render(f"Score: {self.score}", True, WHITE)
        level_text = font.render(f"Level: {self.level}", True, WHITE)
        screen.blit(score_text, (GRID_WIDTH * BLOCK_SIZE + 10, 200))
        screen.blit(level_text, (GRID_WIDTH * BLOCK_SIZE + 10, 240))

        pygame.display.flip()

def draw_game_over(screen, score):
    screen.fill(BLACK)
    font = pygame.font.Font(None, 64)
    game_over_text = font.render("GAME OVER", True, WHITE)
    score_text = font.render(f"Final Score: {score}", True, WHITE)
    restart_text = font.render("Press R to Restart", True, WHITE)
    quit_text = font.render("Press Q to Quit", True, WHITE)

    screen.blit(game_over_text, (SCREEN_WIDTH // 2 - game_over_text.get_width() // 2, SCREEN_HEIGHT // 2 - 150))
    screen.blit(score_text, (SCREEN_WIDTH // 2 - score_text.get_width() // 2, SCREEN_HEIGHT // 2 - 50))
    screen.blit(restart_text, (SCREEN_WIDTH // 2 - restart_text.get_width() // 2, SCREEN_HEIGHT // 2 + 50))
    screen.blit(quit_text, (SCREEN_WIDTH // 2 - quit_text.get_width() // 2, SCREEN_HEIGHT // 2 + 100))

    pygame.display.flip()
    pygame.mixer.music.stop()

def draw_start_menu(screen):
    screen.fill(BLACK)
    font = pygame.font.Font(None, 64)
    title_text = font.render("Tetris", True, WHITE)
    player_text = font.render("PLAYER", True, WHITE)
    ai_text = font.render("AI", True, WHITE)

    screen.blit(title_text, (SCREEN_WIDTH // 2 - title_text.get_width() // 2, SCREEN_HEIGHT // 2 - 150))
    
    player_rect = player_text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2))
    ai_rect = ai_text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 + 100))
    
    pygame.draw.rect(screen, WHITE, player_rect.inflate(20, 10), 2)
    pygame.draw.rect(screen, WHITE, ai_rect.inflate(20, 10), 2)
    
    screen.blit(player_text, player_rect)
    screen.blit(ai_text, ai_rect)

    pygame.display.flip()
    return player_rect, ai_rect

def calculate_features(grid):
    heights = [0] * GRID_WIDTH
    for col in range(GRID_WIDTH):
        for row in range(GRID_HEIGHT):
            if grid[row][col]:
                heights[col] = GRID_HEIGHT - row
                break
    
    bumpiness = sum(abs(heights[i] - heights[i+1]) for i in range(GRID_WIDTH-1))
    holes = sum(1 for col in range(GRID_WIDTH) for row in range(GRID_HEIGHT-1, GRID_HEIGHT-heights[col]-1, -1) if not grid[row][col])
    max_height = max(heights)
    min_height = min(heights)
    
    return [bumpiness, holes, max_height, min_height]

def get_possible_moves(game):
    moves = []
    original_piece = game.current_piece.copy()
    
    rotations = 1 if game.current_piece['shape'] in [SHAPES[1]] else \
                2 if game.current_piece['shape'] in [SHAPES[0], SHAPES[5], SHAPES[6]] else 4

    for rotation in range(rotations):
        for x in range(GRID_WIDTH):
            game.current_piece = original_piece.copy()
            game.current_piece['x'] = x
            
            for _ in range(rotation):
                game.rotate_piece()
            
            if game.valid_move(game.current_piece, x, game.current_piece['y']):
                test_grid = game.grid.copy()
                test_piece = game.current_piece.copy()
                
                while game.valid_move(test_piece, test_piece['x'], test_piece['y'] + 1):
                    test_piece['y'] += 1
                
                for i, row in enumerate(test_piece['shape']):
                    for j, cell in enumerate(row):
                        if cell:
                            test_grid[test_piece['y'] + i][test_piece['x'] + j] = 1
                
                lines_cleared = sum(all(row) for row in test_grid)
                features = calculate_features(test_grid)
                moves.append((rotation, x, features, lines_cleared))
    
    game.current_piece = original_piece
    return moves

def ai_move(game, model):
    moves = get_possible_moves(game)
    if not moves:
        return False

    device = next(model.parameters()).device
    best_move = max(moves, key=lambda m: model(torch.tensor(m[2] + [m[3]], dtype=torch.float32).to(device)).item())
    rotation, x, _, _ = best_move

    for _ in range(rotation):
        game.rotate_piece()
    game.current_piece['x'] = x
    game.drop_piece()
    return True

def main():
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Tetris")
    clock = pygame.time.Clock()
    game = Tetris()
    
    # Load the trained model
    model = TetrisNN()
    model_path = os.path.join(BASE_DIR, 'tetris_model_1345.pth')
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    def reset_game_state():
        nonlocal fall_time, fast_fall, last_ai_move_time
        game.reset_game()
        fall_time = 0
        fast_fall = False
        last_ai_move_time = 0

    fall_time = 0
    fast_fall = False
    game_over = False
    running = True
    ai_mode = False
    last_ai_move_time = 0
    
    # Start with the menu
    in_menu = True
    player_rect, ai_rect = draw_start_menu(screen)

    while running:
        if in_menu:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.MOUSEBUTTONDOWN:
                    if player_rect.collidepoint(event.pos):
                        in_menu = False
                        ai_mode = False
                        pygame.mixer.music.play(-1)
                    elif ai_rect.collidepoint(event.pos):
                        in_menu = False
                        ai_mode = True
                        last_ai_move_time = time.time()
                        pygame.mixer.music.play(-1)
            continue

        if not game_over:
            current_time = time.time()
            current_fall_speed = game.fall_speed / 3 if fast_fall else game.fall_speed
            fall_time += clock.get_rawtime()
            clock.tick()

            if fall_time / 1000 > current_fall_speed:
                if not game.move(0, 1):
                    if not game.place_piece():
                        game_over = True
                fall_time = 0

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if not ai_mode and event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_LEFT:
                        game.move(-1, 0)
                    if event.key == pygame.K_RIGHT:
                        game.move(1, 0)
                    if event.key == pygame.K_UP:
                        game.rotate_piece()
                    if event.key == pygame.K_DOWN:
                        fast_fall = True
                    if event.key == pygame.K_SPACE:
                        game.drop_piece()
                if not ai_mode and event.type == pygame.KEYUP:
                    if event.key == pygame.K_DOWN:
                        fast_fall = False

            if ai_mode and current_time - last_ai_move_time >= AI_MOVE_DELAY:
                if not ai_move(game, model):
                    game_over = True
                last_ai_move_time = current_time

            game.draw(screen)
        else:
            draw_game_over(screen, game.score)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_r:
                        game_over = False
                        reset_game_state()
                        in_menu = True
                        player_rect, ai_rect = draw_start_menu(screen)
                    if event.key == pygame.K_q:
                        running = False

    pygame.quit()

if __name__ == "__main__":
    main()