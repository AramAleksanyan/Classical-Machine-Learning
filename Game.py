import pygame

SCREEN_WIDTH, SCREEN_HEIGHT = 1000, 700
PLAYER_RADIUS = 40
pygame.init()
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("First Game Launch ")
clock = pygame.time.Clock()
running = True

dt = .0
player_pos = pygame.Vector2(screen.get_width() / 2, screen.get_height() / 2)

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    screen.fill('#678991')
    pygame.draw.circle(screen, "#033660", player_pos, PLAYER_RADIUS)
    keys = pygame.key.get_pressed()
    if keys[pygame.K_UP] or keys[pygame.K_w]:
        player_pos.y -= 300 * dt
    if keys[pygame.K_DOWN] or keys[pygame.K_s]:
        player_pos.y += 300 * dt
    if keys[pygame.K_LEFT] or keys[pygame.K_a]:
        player_pos.x -= 300 * dt
    if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
        player_pos.x += 300 * dt

    player_pos.x = max(PLAYER_RADIUS, min(screen.get_width() - PLAYER_RADIUS, player_pos.x))
    player_pos.y = max(PLAYER_RADIUS, min(screen.get_height() - PLAYER_RADIUS, player_pos.y))

    pygame.display.flip()
    dt = clock.tick(60) / 1000

pygame.quit()
