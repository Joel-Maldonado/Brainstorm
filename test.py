import sys
import time
import pygame
from pygame.locals import *

# Initialize PyGame
pygame.init()

# Setup the game window
screen_width = 640
screen_height = 480
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption("Snake Game")

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)

# Snake settings
snake_speed = 10
snake_body = [(int(screen_width / 2), int(screen_height / 2))]
snake_direction = "right"

# Food settings
food_position = (random.randrange(screen_width - 10), random.randrange(screen_height - 10))
food_size = 10

def draw_grid():
    # Draw grid lines
    for x in range(0, screen_width, 10):
        pygame.draw.line(screen, BLACK, (x, 0), (x, screen_height))
    for y in range(0, screen_height, 10):
        pygame.draw.line(screen, BLACK, (0, y), (screen_width, y))

def draw_snake():
    # Draw snake body
    for segment in snake_body:
        pygame.draw.rect(screen, GREEN, (segment[0], segment[1], 10, 10))

def move_snake():
    global snake_direction
    new_head = list(snake_body[-1])
    if snake_direction == "left":
        new_head[0] -= 10
    elif snake_direction == "right":
        new_head[0] += 10
    elif snake_direction == "down":
        new_head[1] += 10
    else:
        new_head[1] -= 10
    snake_body.append(new_head)
    del snake_body[0]

def check_collision():
    # Check collision with wall
    if snake_body[0][0] < 0 or snake_body[0][0] >= screen_width or \
       snake_body[0][1] < 0 or snake_body[0][1] >= screen_height:
        return True
    
    # Check collision with self
    for i in range(len(snake_body)):
        if i != 0 and snake_body[i] == snake_body[0]:
            return True
    
    return False

while True:
    # Handle events
    for event in pygame.event.get():
        if event.type == QUIT:
            pygame.quit()
            sys.exit()
        elif event.type == KEYDOWN:
            if event.key == K_LEFT:
                snake_direction = "left"
            elif event.key == K_RIGHT:
                snake_direction = "right"
            elif event.key == K_UP:
                snake_direction = "up"
            elif event.key == K_DOWN:
                snake_direction = "down"

    # Move snake
    move_snake()

    # Check for collisions
    if check_collision():
        print("Collision detected!")
        break

    # Redraw screen
    screen.fill(BLACK)
    draw_grid()
    draw_snake()
    pygame.display.flip()
    clock.tick(60)