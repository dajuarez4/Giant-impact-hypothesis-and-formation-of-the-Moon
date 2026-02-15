import math
import random
import pygame
import numpy as np

# -----------------------------
# More realistic Giant Impact (2-body + debris disk)
# - Physical-ish parameters (kg, km, s), but displayed with scaling
# - Controls: impact parameter b, speed factor f relative to mutual escape speed
# -----------------------------

# ---------- Pygame ----------
WIDTH, HEIGHT = 1200, 700
FPS = 60

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Giant Impact (more realistic) — b controls angle, v controls speed")
clock = pygame.time.Clock()
font = pygame.font.SysFont("Arial", 18)
big = pygame.font.SysFont("Arial", 22, bold=True)

# ---------- Physics (SI) ----------
G = 6.67430e-11

# Earth
M1 = 5.972e24          # kg
R1 = 6_371_000.0        # m

# Theia ~ Mars-ish (you can tune)
M2 = 6.39e23            # kg (Mars mass)
R2 = 3_390_000.0        # m (Mars radius)

# Display scaling: meters -> pixels
# 1 pixel ~ SCALE meters
SCALE = 2.5e5  # 250 km / px (tune if you want zoom)
CENTER = np.array([WIDTH * 0.42, HEIGHT * 0.55], dtype=float)  # Earth starts near left-center

# Time scaling: we simulate with dt_phys each frame
dt_phys = 30.0  # seconds per physics step (tune for stability/speed)

# Softening to avoid singular acceleration (meters)
SOFTEN = 2.0e5

# ---------- Debris ----------
MAX_PARTICLES = 1600
debris_active = True

# We ignore mutual gravity among debris for speed: particles feel only Earth (post-impact merged mass)
# ---------- Controls ----------
# b_frac = impact parameter as fraction of (R1+R2): 0 = head-on; 1 = grazing-ish
b_frac = 0.60

# v_factor relative to mutual escape speed at contact
v_factor = 1.10  # typical giant-impact range ~ 1.0–1.3

paused = False
show_trails = True

# ---------- Utility ----------
def to_px(r_m):
    """meters -> pixels"""
    return r_m / SCALE

def from_px(p):
    """pixels -> meters (vector)"""
    return p * SCALE

def norm(v):
    return math.hypot(v[0], v[1])

def unit(v):
    d = norm(v) + 1e-30
    return (v[0]/d, v[1]/d)

def accel(m_other, r_self, r_other):
    # softened Newtonian gravity
    rx = r_other[0] - r_self[0]
    ry = r_other[1] - r_self[1]
    dist2 = rx*rx + ry*ry + SOFTEN*SOFTEN
    dist = math.sqrt(dist2)
    inv3 = 1.0 / (dist2 * dist + 1e-40)
    ax = G * m_other * rx * inv3
    ay = G * m_other * ry * inv3
    return (ax, ay)

def mutual_escape_speed():
    # escape speed at contact distance (R1+R2)
    return math.sqrt(2.0 * G * (M1 + M2) / (R1 + R2))

def impact_angle_deg(r_rel, v_rel):
    # angle between incoming direction (-v) and line of centers (r)
    rr = np.array(r_rel, dtype=float)
    vv = np.array(v_rel, dtype=float)
    rr /= (np.linalg.norm(rr) + 1e-30)
    vv /= (np.linalg.norm(vv) + 1e-30)
    cosang = np.clip(np.dot(-vv, rr), -1.0, 1.0)
    return float(np.degrees(np.arccos(cosang)))

# ---------- Simulation state ----------
def reset():
    global earth, theia, trails, collided, t_since, debris, M_merged

    vesc = mutual_escape_speed()
    v0 = v_factor * vesc

    # Initial separation (meters): start far enough away
    r0 = 25.0 * (R1 + R2)
    b = b_frac * (R1 + R2)

    # Place Earth at origin (meters), Theia at (+r0, +b)
    r1 = np.array([0.0, 0.0], dtype=float)
    r2 = np.array([r0, b], dtype=float)

    # Velocity aimed toward Earth center (simple) — gravity will focus trajectory
    direction = -r2
    direction /= (np.linalg.norm(direction) + 1e-30)
    v2 = v0 * direction

    # Center-of-mass frame (optional but nicer): give Earth recoil so total momentum = 0
    v1 = -(M2 / M1) * v2

    earth = {"r": r1, "v": v1, "m": M1, "R": R1, "color": (70, 120, 255), "name": "Proto-Earth"}
    theia = {"r": r2, "v": v2, "m": M2, "R": R2, "color": (255, 120, 70), "name": "Theia"}

    trails = {"earth": [], "theia": []}
    collided = False
    t_since = 0.0
    debris = None

    # After impact, we "merge" masses to set central gravity for debris
    M_merged = M1 + M2

reset()

# ---------- Debris spawner ----------
def spawn_debris(impact_point_m, n=1200):
    # Create particles in a disk around Earth with near-orbital velocities.
    # This is simplified but gives a believable debris ring.
    global debris

    n = min(n, MAX_PARTICLES)
    pos = np.zeros((n, 2), dtype=float)
    vel = np.zeros((n, 2), dtype=float)

    # Disk radii (meters) — around a few Earth radii
    rmin = 1.2 * R1
    rmax = 6.0 * R1

    for i in range(n):
        # random radius biased toward mid-range
        u = random.random()
        r = rmin * (1 - u) + rmax * u
        ang = random.random() * 2 * math.pi

        # position around Earth
        pos[i, 0] = r * math.cos(ang)
        pos[i, 1] = r * math.sin(ang)

        # circular speed around merged Earth
        vc = math.sqrt(G * M_merged / (r + 1e-30))

        # tangential direction
        tx = -math.sin(ang)
        ty =  math.cos(ang)

        # add randomness + some "hot" ejecta
        jitter = 0.25 + 0.75 * random.random()
        kick = (random.random() - 0.5) * 0.25 * vc

        vel[i, 0] = (vc * jitter) * tx + kick * math.cos(ang)
        vel[i, 1] = (vc * jitter) * ty + kick * math.sin(ang)

    debris = {"pos": pos, "vel": vel}

# ---------- Draw ----------
def draw_body(body, offset_px):
    r_px = offset_px + np.array([to_px(body["r"][0]), to_px(body["r"][1])])
    pygame.draw.circle(screen, body["color"], (int(r_px[0]), int(r_px[1])), max(3, int(to_px(body["R"]))))
    label = font.render(body["name"], True, (20, 20, 20))
    screen.blit(label, (r_px[0] + 12, r_px[1] - 10))

def draw_trail(tr, color):
    if len(tr) < 2:
        return
    pygame.draw.lines(screen, color, False, tr, 2)

def draw_debris(offset_px):
    if debris is None:
        return
    pts_m = debris["pos"]
    # downsample for drawing speed if needed
    step = 1 if len(pts_m) <= 1800 else 2
    for i in range(0, len(pts_m), step):
        x = offset_px[0] + to_px(pts_m[i, 0])
        y = offset_px[1] + to_px(pts_m[i, 1])
        if 0 <= x < WIDTH and 0 <= y < HEIGHT:
            screen.set_at((int(x), int(y)), (120, 120, 120))

# ---------- Main loop ----------
running = True
last_angle = None
last_speed = None

while running:
    clock.tick(FPS)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False
            elif event.key == pygame.K_SPACE:
                paused = not paused
            elif event.key == pygame.K_r:
                reset()
            elif event.key == pygame.K_t:
                show_trails = not show_trails
            elif event.key == pygame.K_d:
                debris_active = not debris_active

    keys = pygame.key.get_pressed()

    # Adjust parameters live (then press R to apply)
    if keys[pygame.K_LEFT]:
        b_frac -= 0.005
    if keys[pygame.K_RIGHT]:
        b_frac += 0.005
    b_frac = max(0.0, min(1.2, b_frac))

    if keys[pygame.K_DOWN]:
        v_factor -= 0.005
    if keys[pygame.K_UP]:
        v_factor += 0.005
    v_factor = max(0.70, min(2.00, v_factor))

    if not paused:
        if not collided:
            # Accelerations
            a1 = np.array(accel(theia["m"], earth["r"], theia["r"]))
            a2 = np.array(accel(earth["m"], theia["r"], earth["r"]))

            # Semi-implicit Euler
            earth["v"] += a1 * dt_phys
            theia["v"] += a2 * dt_phys
            earth["r"] += earth["v"] * dt_phys
            theia["r"] += theia["v"] * dt_phys

            # Trails (in pixels)
            if show_trails:
                off = CENTER
                e_px = off + np.array([to_px(earth["r"][0]), to_px(earth["r"][1])])
                t_px = off + np.array([to_px(theia["r"][0]), to_px(theia["r"][1])])
                trails["earth"].append((int(e_px[0]), int(e_px[1])))
                trails["theia"].append((int(t_px[0]), int(t_px[1])))
                if len(trails["earth"]) > 800:
                    trails["earth"].pop(0)
                if len(trails["theia"]) > 800:
                    trails["theia"].pop(0)

            # Collision check
            r_rel = theia["r"] - earth["r"]
            dist = np.linalg.norm(r_rel)
            if dist <= (R1 + R2):
                v_rel = theia["v"] - earth["v"]
                last_angle = impact_angle_deg(r_rel, v_rel)
                last_speed = float(np.linalg.norm(v_rel))

                collided = True

                # Spawn debris disk centered on Earth (post-impact)
                if debris_active:
                    spawn_debris(impact_point_m=earth["r"].copy(), n=1400)

        else:
            # Evolve debris after impact (particles orbit Earth)
            if debris is not None:
                pos = debris["pos"]
                vel = debris["vel"]

                # acceleration from merged Earth at origin
                # a = -GM r / (r^2 + soft^2)^(3/2)
                rx = pos[:, 0]
                ry = pos[:, 1]
                dist2 = rx*rx + ry*ry + SOFTEN*SOFTEN
                dist = np.sqrt(dist2)
                inv3 = 1.0 / (dist2 * dist + 1e-40)
                ax = -G * M_merged * rx * inv3
                ay = -G * M_merged * ry * inv3

                vel[:, 0] += ax * dt_phys
                vel[:, 1] += ay * dt_phys
                pos[:, 0] += vel[:, 0] * dt_phys
                pos[:, 1] += vel[:, 1] * dt_phys

    # ---------- Render ----------
    screen.fill((255, 255, 255))

    # Trails
    if show_trails:
        draw_trail(trails["earth"], (140, 190, 255))
        draw_trail(trails["theia"], (255, 180, 140))

    # Bodies / debris
    draw_body(earth, CENTER)
    draw_body(theia, CENTER)
    draw_debris(CENTER)

    # HUD
    vesc = mutual_escape_speed()
    hud1 = big.render("Giant Impact (more realistic) — adjust b and v, press R to reset", True, (10, 10, 10))
    screen.blit(hud1, (16, 12))

    hud2 = font.render(
        f"b = {b_frac:.3f} × (R1+R2)   v = {v_factor:.3f} × v_esc   v_esc ≈ {vesc/1000:.2f} km/s   (←/→ b, ↑/↓ v, R reset, SPACE pause, T trails, D debris)",
        True, (10, 10, 10)
    )
    screen.blit(hud2, (16, 42))

    if collided:
        ang_txt = f"{last_angle:.1f}°" if last_angle is not None else "?"
        spd_txt = f"{(last_speed/1000):.2f} km/s" if last_speed is not None else "?"
        hud3 = big.render(f"IMPACT!  contact angle ≈ {ang_txt}   relative speed ≈ {spd_txt}", True, (160, 0, 0))
        screen.blit(hud3, (16, 70))

    pygame.display.flip()

pygame.quit()

