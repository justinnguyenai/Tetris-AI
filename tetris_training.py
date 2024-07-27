import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import copy
from tetris_game import Tetris, GRID_WIDTH, GRID_HEIGHT, SHAPES
import time
import pickle

# CUDA verification
def verify_cuda():
    if torch.cuda.is_available():
        print("CUDA is available!")
        print(f"CUDA version: {torch.version.cuda}")
        print(f"PyTorch CUDA version: {torch.version.cuda}")
        print(f"GPU device name: {torch.cuda.get_device_name(0)}")
        
        # Create a sample tensor and move it to GPU
        x = torch.rand(5, 3)
        print(f"Sample tensor created on: {x.device}")
        x = x.cuda()
        print(f"Sample tensor moved to: {x.device}")
        
        # Perform a simple operation to test CUDA
        y = x * 2
        print(f"Operation result tensor on: {y.device}")
        
        print("CUDA is set up correctly and can be used for computations.")
    else:
        print("CUDA is not available. The script will run on CPU.")

# Set up CUDA for GPU acceleration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

# Function to calculate features
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

# Function to get all possible moves
def get_possible_moves(game):
    moves = []
    original_piece = copy.deepcopy(game.current_piece)
    
    rotations = 1 if game.current_piece['shape'] in [SHAPES[1]] else \
                2 if game.current_piece['shape'] in [SHAPES[0], SHAPES[5], SHAPES[6]] else 4

    for rotation in range(rotations):
        for x in range(GRID_WIDTH):
            game.current_piece = copy.deepcopy(original_piece)
            game.current_piece['x'] = x
            
            for _ in range(rotation):
                game.rotate_piece()
            
            if game.valid_move(game.current_piece, x, game.current_piece['y']):
                test_grid = game.grid.copy()
                test_piece = copy.deepcopy(game.current_piece)
                
                while game.valid_move(test_piece, test_piece['x'], test_piece['y'] + 1):
                    test_piece['y'] += 1
                
                # Check if the piece is within bounds
                if test_piece['y'] + len(test_piece['shape']) > GRID_HEIGHT:
                    continue

                for i, row in enumerate(test_piece['shape']):
                    for j, cell in enumerate(row):
                        if cell:
                            if 0 <= test_piece['y'] + i < GRID_HEIGHT and 0 <= test_piece['x'] + j < GRID_WIDTH:
                                test_grid[test_piece['y'] + i][test_piece['x'] + j] = 1
                            else:
                                # Skip this move if it's out of bounds
                                break
                    else:
                        continue
                    break
                else:
                    lines_cleared = sum(all(row) for row in test_grid)
                    features = calculate_features(test_grid)
                    moves.append((rotation, x, features, lines_cleared))
    
    game.current_piece = original_piece
    return moves

# Evolutionary Algorithm
def evolve_population(population, elite_size, mutation_rate):
    sorted_population = sorted(population, key=lambda x: x[1], reverse=True)
    elite = sorted_population[:elite_size]
    
    # Tournament Selection
    def tournament_select(k=3):
        selection = random.sample(sorted_population, k)
        return max(selection, key=lambda x: x[1])[0]
    
    children = []
    while len(children) < len(population) - elite_size:
        parent1 = tournament_select()
        parent2 = tournament_select()
        child = TetrisNN().to(device)
        
        # Crossover
        for param1, param2, param_child in zip(parent1.parameters(), parent2.parameters(), child.parameters()):
            param_child.data.copy_(torch.where(torch.rand(param1.shape, device=device) > 0.5, param1, param2))
        
        # Mutation
        for param in child.parameters():
            if random.random() < mutation_rate:
                param.data += torch.randn(param.shape, device=device) * 0.1
        
        children.append(child)
    
    return [model for model, _ in elite] + children

# Play game with neural network
def play_game(model):
    game = Tetris()
    score = 0
    moves_without_scoring = 0
    max_moves_without_scoring = 50  # Adjust this value as needed

    while moves_without_scoring < max_moves_without_scoring:
        moves = get_possible_moves(game)
        if not moves:
            break
        
        best_move = max(moves, key=lambda m: model(torch.tensor(m[2] + [m[3]], dtype=torch.float32, device=device)).item())
        rotation, x, _, lines_cleared = best_move
        
        for _ in range(rotation):
            game.rotate_piece()
        game.current_piece['x'] = x
        
        if not game.drop_piece():
            break  # Game over

        new_score = game.score
        if new_score > score:
            score = new_score
            moves_without_scoring = 0
        else:
            moves_without_scoring += 1

    return score

# Training loop
def train(population_size, generations, elite_size, mutation_rate):
    population = [(TetrisNN().to(device), 0) for _ in range(population_size)]
    best_score = 0
    
    for gen in range(generations):
        for i, (model, _) in enumerate(population):
            score = play_game(model)
            population[i] = (model, score)
            
            if score > best_score:
                best_score = score
                torch.save(model.state_dict(), f'best_model_score_{best_score}.pth')
                print(f"New best score: {best_score}")
        
        avg_score = sum(score for _, score in population) / population_size
        print(f"Generation {gen + 1}: Best Score = {best_score}, Average Score = {avg_score:.2f}")
        
        population = [(model, 0) for model in evolve_population(population, elite_size, mutation_rate)]

if __name__ == "__main__":
    # Hyperparameters
    POPULATION_SIZE = 50
    GENERATIONS = 100
    ELITE_SIZE = 5
    MUTATION_RATE = 0.1
    
    train(POPULATION_SIZE, GENERATIONS, ELITE_SIZE, MUTATION_RATE)