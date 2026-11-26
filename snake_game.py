# Snake game in Python

import random
import time
import pygame

WIDTH = 20
HEIGHT = 20

snake = [(WIDTH//2, HEIGHT//2)]
food = (random.randint(0, WIDTH-1), random.randint(0, HEIGHT-1))

while True:
    screen.fill(BLACK)

    for pos in snake:
        pygame.draw.rect(screen, GREEN, pygame.Rect(pos[0]*CELL_SIZE, pos[1]*CELL_SIZE, CELL_SIZE, CELL_SIZE))

    pygame.draw.rect(screen, RED, pygame.Rect(food[0]*CELL_SIZE, food[1]*CELL_SIZE, CELL_SIZE, CELL_SIZE))

    new_head = (snake[0][0] + dx, snake[0][1] + dy)

    if new_head in snake or not (0 <= new_head[0] < WIDTH and 0 <= new_head[1] < HEIGHT):
        game_over = True

    snake.insert(0, new_head)

    if new_head == food:
        food = None
        while food is None:
            food = (random.randint(0, WIDTH-1), random.randint(0, HEIGHT-1))
            if food not in snake:
                break

        score += 1

    screen.blit(font.render(f'Score: {score}', True, WHITE), [10, 10])

    pygame.display.flip()

    time.sleep(0.1)

    if game_over:
        game_over_screen = pygame.Surface((WIDTH*CELL_SIZE, HEIGHT*CELL_SIZE))
        game_over_screen.fill(BLACK)

        game_over_message = font.render('Game Over', True, RED)
        game_over_screen.blit(game_over_message, (WIDTH*CELL_SIZE//2 - game_over_message.get_width()//2, HEIGHT*CELL_SIZE//2 - game_over_message.get_height()//2))

        pygame.display.flip()

        time.sleep(2)

        running = False

pygame.quit()
