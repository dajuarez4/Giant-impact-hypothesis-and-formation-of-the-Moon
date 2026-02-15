import math
import pygame
from collections import deque

# -----------------------------
# Pygame: Giant Impact Toy Model (2D gravity + adjustable theta)
# -----------------------------

WIDTH, HEIGHT = 1100, 650
FPS = 60

# "Physics" in toy units (not real SI units)
G = 1200.0          # gravitational constant in toy units
DT = 1.0 / FPS      # seconds per frame (toy)
SOFTEN = 5.0        # softening to avoid singular acceleration

# Bodies (toy masses/radii)
M_EARTH = 4000.0
M_THEIA = 600.0
R_EARTH = 28
R_THEIA = 16

# Initial conditions (screen coordinates)
EARTH_POS0 = (WIDTH * 0.35, HEIGHT * 0.55)
THEIA_POS0 = (WIDTH * 0.85, HEIGHT * 0.30)

# Trail settings
TRAIL_MAX = 900

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Giant Impact Hypothesis (Toy 2D) — Adjust θ and watch the collision")
clock = pygame.time.Clock()
font = pygame.font.SysFont("Arial", 18)
big = pygame.font.SysFont("Arial", 22, bold=True)

def vec_add(a, b): return (a[0] + b[0], a[1] + b[1])
def vec_sub(a, b): return (a[0] - b[0], a[1] - b[1])
def vec_mul(a, s): return (a[0] * s, a[1] * s)
def vec_len(a): return math.hypot(a[0], a[1])

def accel_due_to(m_other, r_self, r_other):
    # a = G m * r / |r|^3 (softened)
    rx, ry = r_other[0] - r_self[0], r_other[1] - r_self[1]
    dist2 = rx*rx + ry*ry + SOFTEN*SOFTEN
    dist = math.sqrt(dist2)
    inv3 = 1.0 / (dist2 * dist + 1e-9)
    ax = G * m_other * rx * inv3
    ay = G * m_other * ry * inv3
    return (ax, ay)

class Body:
    def __init__(self, pos, vel, mass, radius, color, name):
        self.pos = list(pos)
        self.vel = list(vel)
        self.mass = mass
        self.radius = radius
        self.color = color
        self.name = name

    def step(self, a, dt):
        # Semi-implicit Euler (stable enough for a toy demo)
        self.vel[0] += a[0] * dt
        self.vel[1] += a[1] * dt
        self.pos[0] += self.vel[0] * dt
        self.pos[1] += self.vel[1] * dt

def reset(theta_deg, v0):
    # Earth starts at rest; Theia launched toward Earth with tilt theta
    earth = Body(EARTH_POS0, (0.0, 0.0), M_EARTH, R_EARTH, (70, 120, 255), "Proto-Earth")
    theia = Body(THEIA_POS0, (0.0, 0.0), M_THEIA, R_THEIA, (255, 120, 70), "Theia")

    # Direction from Theia to Earth
    dx, dy = earth.pos[0] - theia.pos[0], earth.pos[1] - theia.pos[1]
    base_ang = math.atan2(dy, dx)  # angle pointing at Earth
    theta = math.radians(theta_deg)

    # Rotate by +theta (counterclockwise) relative to direct-hit direction
    launch_ang = base_ang + theta

    theia.vel[0] = v0 * math.cos(launch_ang)
    theia.vel[1] = v0 * math.sin(launch_ang)

    trail_e = deque(maxlen=TRAIL_MAX)
    trail_t = deque(maxlen=TRAIL_MAX)
    return earth, theia, trail_e, trail_t

def draw_body(b):
    pygame.draw.circle(screen, b.color, (int(b.pos[0]), int(b.pos[1])), b.radius)
    label = font.render(b.name, True, (20, 20, 20))
    screen.blit(label, (b.pos[0] + b.radius + 6, b.pos[1] - 10))

def draw_trail(trail, color):
    if len(trail) < 2:
        return
    pts = [(int(x), int(y)) for (x, y) in trail]
    pygame.draw.lines(screen, color, False, pts, 2)

def clamp(x, lo, hi): return max(lo, min(hi, x))

theta_deg = 25.0   # impact angle control
v0 = 260.0         # initial speed control
paused = False
show_trail = True

earth, theia, trail_e, trail_t = reset(theta_deg, v0)
collided = False

running = True
while running:
    dt = DT
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False

            elif event.key == pygame.K_SPACE:
                paused = not paused

            elif event.key == pygame.K_t:
                show_trail = not show_trail

            elif event.key == pygame.K_r:
                earth, theia, trail_e, trail_t = reset(theta_deg, v0)
                collided = False
                paused = False

    keys = pygame.key.get_pressed()
    if keys[pygame.K_LEFT]:
        theta_deg -= 0.5
    if keys[pygame.K_RIGHT]:
        theta_deg += 0.5
    theta_deg = clamp(theta_deg, -85.0, 85.0)

    if keys[pygame.K_UP]:
        v0 += 1.5
    if keys[pygame.K_DOWN]:
        v0 -= 1.5
    v0 = clamp(v0, 20.0, 700.0)

    if not paused and not collided:
        # accelerations
        a_e = accel_due_to(theia.mass, earth.pos, theia.pos)
        a_t = accel_due_to(earth.mass, theia.pos, earth.pos)

        earth.step(a_e, dt)
        theia.step(a_t, dt)

        # trails
        trail_e.append(tuple(earth.pos))
        trail_t.append(tuple(theia.pos))

        # collision check
        d = vec_len(vec_sub(earth.pos, theia.pos))
        if d <= (earth.radius + theia.radius):
            collided = True

    # draw
    screen.fill((255, 255, 255))

    # Trails first
    if show_trail:
        draw_trail(trail_e, (120, 170, 255))
        draw_trail(trail_t, (255, 170, 130))

    # Bodies
    draw_body(earth)
    draw_body(theia)

    # HUD
    title = big.render("Giant Impact Toy Simulation (2D)", True, (10, 10, 10))
    screen.blit(title, (16, 12))

    info1 = font.render(f"θ = {theta_deg:.1f}°   v0 = {v0:.1f}   (←/→ for θ, ↑/↓ for v0, R reset, SPACE pause)", True, (10, 10, 10))
    screen.blit(info1, (16, 40))

    if collided:
        msg = big.render("COLLISION! Press R to reset with current θ and v0.", True, (180, 0, 0))
        screen.blit(msg, (16, 68))
    elif paused:
        msg = big.render("PAUSED (SPACE to resume)", True, (0, 0, 140))
        screen.blit(msg, (16, 68))

    pygame.display.flip()
    clock.tick(FPS)

pygame.quit()

