import math
import random
import pygame
import numpy as np

# -----------------------------
# Giant Impact (2-body + debris disk + Moon accretion)
# - Physical-ish parameters (kg, m, s) with screen scaling
# - Disk forms after collision; particles outside Roche limit can accrete
# - A Moon seed grows by sweeping up debris
# -----------------------------

# ---------- Pygame ----------
WIDTH, HEIGHT = 1200, 700
FPS = 60

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Giant Impact hypothesis: Earth + Theia → debris disk → Moon accretion (toy model)")
clock = pygame.time.Clock()
font = pygame.font.SysFont("Arial", 18)
big = pygame.font.SysFont("Arial", 22, bold=True)

# ---------- Physics (SI-ish) ----------
G = 6.67430e-11

# Earth
M1 = 5.972e24
R1 = 6_371_000.0

# Theia (Mars-ish)
M2 = 6.39e23
R2 = 3_390_000.0

# Display scale (meters per pixel)
SCALE = 2.5e5  # 250 km/px
CENTER = np.array([WIDTH * 0.42, HEIGHT * 0.55], dtype=float)

dt_phys = 30.0  # seconds per frame
SOFTEN = 2.0e5

# ---------- Controls ----------
show_trails = True
show_debris = True
show_ui = True      # <--- hide/show HUD text
show_labels = True  # <--- hide/show object labels (Proto-Earth, Roche limit, etc.)
b_frac = 0.60        # impact parameter fraction of (R1+R2)
v_factor = 1.10      # relative to mutual escape speed
paused = False
show_trails = True
show_debris = True

# ---------- Debris / Moon ----------
MAX_PARTICLES = 2200
N_SPAWN = 1600

# Roche limit (fluid satellite) approx: ~2.44 R_planet*(rho_planet/rho_sat)^(1/3)
# We'll use a simple constant ~2.9 R_earth for rocky-ish satellite: good for a demo.
ROCHE = 2.9 * R1

# Spawn disk between:
RMIN = 1.2 * R1
RMAX = 7.0 * R1

# Moon seed initial placement just outside Roche limit
MOON_SEED_R = 3.3 * R1

# Moon "sweep" tuning
SWEEP_FACTOR = 2.5     # bigger = accretes more easily
EJECT_FRACTION = 0.03  # small fraction of collisions eject particles (optional variability)

# ---------- Utility ----------
def to_px(m): return m / SCALE

def accel_point_mass(M, r):
    """Acceleration at position r (vector) toward origin from mass M with softening."""
    rx, ry = r[0], r[1]
    dist2 = rx*rx + ry*ry + SOFTEN*SOFTEN
    dist = math.sqrt(dist2)
    inv3 = 1.0 / (dist2 * dist + 1e-40)
    ax = -G * M * rx * inv3
    ay = -G * M * ry * inv3
    return np.array([ax, ay], dtype=float)

def mutual_escape_speed():
    return math.sqrt(2.0 * G * (M1 + M2) / (R1 + R2))

def impact_angle_deg(r_rel, v_rel):
    rr = r_rel / (np.linalg.norm(r_rel) + 1e-30)
    vv = v_rel / (np.linalg.norm(v_rel) + 1e-30)
    cosang = float(np.clip(np.dot(-vv, rr), -1.0, 1.0))
    return float(np.degrees(np.arccos(cosang)))

# def draw_body(pos_m, radius_m, color, name=None):
#     p = CENTER + np.array([to_px(pos_m[0]), to_px(pos_m[1])])
#     pygame.draw.circle(screen, color, (int(p[0]), int(p[1])), max(3, int(to_px(radius_m))))
#     if name:
#         lbl = font.render(name, True, (20, 20, 20))
#         screen.blit(lbl, (p[0] + 12, p[1] - 10))
def draw_body(pos_m, radius_m, color, name=None):
    p = CENTER + np.array([to_px(pos_m[0]), to_px(pos_m[1])])
    pygame.draw.circle(screen, color, (int(p[0]), int(p[1])), max(3, int(to_px(radius_m))))
    if name and show_labels:
        lbl = font.render(name, True, (20, 20, 20))
        screen.blit(lbl, (p[0] + 12, p[1] - 10))

def draw_trail(tr, color):
    if len(tr) > 1:
        pygame.draw.lines(screen, color, False, tr, 2)

def clamp(x, lo, hi): return max(lo, min(hi, x))

# ---------- State ----------
def reset():
    global earth, theia, trails, collided, debris, moon, last_angle, last_speed, M_merged

    vesc = mutual_escape_speed()
    v0 = v_factor * vesc

    r0 = 25.0 * (R1 + R2)
    b = b_frac * (R1 + R2)

    r1 = np.array([0.0, 0.0], dtype=float)
    r2 = np.array([r0, b], dtype=float)

    direction = -r2 / (np.linalg.norm(r2) + 1e-30)
    v2 = v0 * direction
    v1 = -(M2 / M1) * v2  # COM frame-ish

    earth = {"r": r1, "v": v1, "m": M1, "R": R1}
    theia = {"r": r2, "v": v2, "m": M2, "R": R2}

    trails = {"earth": [], "theia": []}
    collided = False

    debris = None
    moon = None

    last_angle = None
    last_speed = None

    M_merged = M1 + M2

def spawn_debris(n=N_SPAWN):
    """Spawn debris disk around Earth at origin; velocities near orbital."""
    n = min(n, MAX_PARTICLES)
    pos = np.zeros((n, 2), dtype=float)
    vel = np.zeros((n, 2), dtype=float)

    for i in range(n):
        # radius + angle
        u = random.random()
        r = RMIN * (1 - u) + RMAX * u
        ang = random.random() * 2 * math.pi

        pos[i, 0] = r * math.cos(ang)
        pos[i, 1] = r * math.sin(ang)

        # circular speed
        vc = math.sqrt(G * M_merged / (r + 1e-30))
        # tangential dir
        tx, ty = -math.sin(ang), math.cos(ang)

        # jitter for hotter debris
        jitter = 0.65 + 0.70 * random.random()
        kick = (random.random() - 0.5) * 0.20 * vc

        vel[i, 0] = vc * jitter * tx + kick * math.cos(ang)
        vel[i, 1] = vc * jitter * ty + kick * math.sin(ang)

    return {"pos": pos, "vel": vel, "alive": np.ones(n, dtype=bool)}

def init_moon_seed():
    """Create a small moon seed just outside Roche limit with near-circular orbit."""
    ang = 0.35 * math.pi
    r = MOON_SEED_R
    x, y = r * math.cos(ang), r * math.sin(ang)

    vc = math.sqrt(G * M_merged / r)
    # tangential direction
    tx, ty = -math.sin(ang), math.cos(ang)

    return {
        "r": np.array([x, y], dtype=float),
        "v": np.array([vc * tx, vc * ty], dtype=float),
        "m": 7.35e22 * 0.05,          # start at 5% of Moon mass (toy)
        "R": 1.74e6 * 0.35            # start small
    }

def moon_radius_from_mass(m):
    # Keep density ~ constant: R ~ m^(1/3)
    # Use Moon reference radius/mass
    M_moon = 7.35e22
    R_moon = 1.74e6
    return R_moon * (m / M_moon) ** (1/3)

# ---------- Init ----------
reset()

# ---------- Main loop ----------
running = True
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
                show_debris = not show_debris
            elif event.key == pygame.K_h:      
                show_ui = not show_ui
            elif event.key == pygame.K_l:      # L = toggle labels only (without hiding HUD)
                show_labels = not show_labels
            elif event.key == pygame.K_p:      # P = save screenshot
                pygame.image.save(screen, "screenshot.png")
                print("Saved screenshot.png")

    keys = pygame.key.get_pressed()

    # Adjust parameters live; press R to apply
    if keys[pygame.K_LEFT]:
        b_frac -= 0.005
    if keys[pygame.K_RIGHT]:
        b_frac += 0.005
    b_frac = clamp(b_frac, 0.0, 1.2)

    if keys[pygame.K_DOWN]:
        v_factor -= 0.005
    if keys[pygame.K_UP]:
        v_factor += 0.005
    v_factor = clamp(v_factor, 0.70, 2.00)

    if not paused:
        if not collided:
            # 2-body grav (Earth + Theia), softened
            r_rel = theia["r"] - earth["r"]
            # accel on Earth from Theia
            a1 = (G * theia["m"] * r_rel) / ((np.linalg.norm(r_rel)**3) + SOFTEN**3 + 1e-30)
            # accel on Theia from Earth
            a2 = (G * earth["m"] * (-r_rel)) / ((np.linalg.norm(r_rel)**3) + SOFTEN**3 + 1e-30)

            earth["v"] += a1 * dt_phys
            theia["v"] += a2 * dt_phys
            earth["r"] += earth["v"] * dt_phys
            theia["r"] += theia["v"] * dt_phys

            if show_trails:
                e_px = CENTER + np.array([to_px(earth["r"][0]), to_px(earth["r"][1])])
                t_px = CENTER + np.array([to_px(theia["r"][0]), to_px(theia["r"][1])])
                trails["earth"].append((int(e_px[0]), int(e_px[1])))
                trails["theia"].append((int(t_px[0]), int(t_px[1])))
                if len(trails["earth"]) > 800: trails["earth"].pop(0)
                if len(trails["theia"]) > 800: trails["theia"].pop(0)

            # collision
            dist = np.linalg.norm(theia["r"] - earth["r"])
            if dist <= (R1 + R2):
                v_rel = theia["v"] - earth["v"]
                last_angle = impact_angle_deg(theia["r"] - earth["r"], v_rel)
                last_speed = float(np.linalg.norm(v_rel))

                collided = True
                debris = spawn_debris()
                moon = init_moon_seed()

        else:
            # After impact: treat Earth+Theia as merged at origin for disk+moon dynamics
            # Update debris under gravity of merged Earth
            if debris is not None:
                pos = debris["pos"]
                vel = debris["vel"]
                alive = debris["alive"]

                # Gravity from merged Earth at origin
                a = np.zeros_like(pos)
                a[alive] = np.array([accel_point_mass(M_merged, pos[i]) for i in np.where(alive)[0]])
                vel[alive] += a[alive] * dt_phys
                pos[alive] += vel[alive] * dt_phys

                # Remove debris that falls back to Earth (inside Earth radius)
                rmag = np.sqrt(pos[:,0]**2 + pos[:,1]**2)
                alive &= (rmag > 1.05 * R1)
                debris["alive"] = alive

            # Update moon seed orbit (gravity)
            if moon is not None:
                moon["v"] += accel_point_mass(M_merged, moon["r"]) * dt_phys
                moon["r"] += moon["v"] * dt_phys

                # Accretion: sweep up debris particles that come within a capture radius,
                # but ONLY if outside Roche limit (so clumps can form).
                if debris is not None:
                    pos = debris["pos"]
                    vel = debris["vel"]
                    alive = debris["alive"]

                    rmag = np.sqrt(pos[:,0]**2 + pos[:,1]**2)
                    outside_roche = rmag > ROCHE

                    # capture radius grows with moon radius
                    capture_R = SWEEP_FACTOR * moon["R"]

                    # find candidates near moon
                    dr = pos - moon["r"][None, :]
                    d2 = dr[:,0]**2 + dr[:,1]**2
                    hit = alive & outside_roche & (d2 < capture_R**2)

                    idx = np.where(hit)[0]
                    if idx.size > 0:
                        # Each particle represents some small mass budget (toy)
                        # Total disk mass in SPH studies might be a few lunar masses; choose a small per-particle mass.
                        m_part = 2.0e20  # kg (tune)
                        gained = m_part * idx.size

                        # Momentum-conserving merge (moon + captured particles)
                        # v_new = (m v + sum(m_part v_i)) / (m + gained)
                        p_moon = moon["m"] * moon["v"]
                        p_parts = m_part * np.sum(vel[idx], axis=0)
                        moon["m"] += gained
                        moon["v"] = (p_moon + p_parts) / moon["m"]

                        # Update radius by constant-density scaling
                        moon["R"] = moon_radius_from_mass(moon["m"])

                        # Sometimes eject a few instead of accrete (optional realism spice)
                        if EJECT_FRACTION > 0 and idx.size > 10:
                            k = int(EJECT_FRACTION * idx.size)
                            if k > 0:
                                kick_idx = np.random.choice(idx, size=k, replace=False)
                                vel[kick_idx] *= 1.15  # mild speed-up
                                # remaining are removed (accreted)
                                idx = np.setdiff1d(idx, kick_idx)

                        # remove accreted debris
                        alive[idx] = False
                        debris["alive"] = alive

    # ---------- Draw ----------
    screen.fill((255, 255, 255))

    # Trails
    if show_trails and not collided:
        draw_trail(trails["earth"], (140, 190, 255))
        draw_trail(trails["theia"], (255, 180, 140))

    # Bodies
    if not collided:
        draw_body(earth["r"], earth["R"], (70, 120, 255), "Proto-Earth")
        draw_body(theia["r"], theia["R"], (255, 120, 70), "Theia")
    else:
        # Earth at origin (merged) — draw as Earth
        draw_body(np.array([0.0, 0.0]), R1, (70, 120, 255), "                                           Earth (post-impact)")

        # Roche limit ring
        roche_px = int(to_px(ROCHE))
        pygame.draw.circle(screen, (210, 210, 210), (int(CENTER[0]), int(CENTER[1])), roche_px, 1)
        # lbl = font.render("Roche limit", True, (120, 120, 120))
        # screen.blit(lbl, (CENTER[0] + roche_px + 8, CENTER[1] - 10))
        if show_labels:
            lbl = font.render("Roche limit", True, (120, 120, 120))
            screen.blit(lbl, (CENTER[0] + roche_px + 8, CENTER[1] - 10))

        # Debris
        if show_debris and debris is not None:
            pos = debris["pos"]
            alive = debris["alive"]
            # draw downsample for speed
            idx = np.where(alive)[0]
            step = 1 if idx.size < 1400 else 2
            for j in idx[::step]:
                x = CENTER[0] + to_px(pos[j,0])
                y = CENTER[1] + to_px(pos[j,1])
                if 0 <= x < WIDTH and 0 <= y < HEIGHT:
                    screen.set_at((int(x), int(y)), (130, 130, 130))

        # Moon
        if moon is not None:
            draw_body(moon["r"], moon["R"], (110, 110, 110), "Moon (growing)")

    # HUD
    
    vesc = mutual_escape_speed()
    # title = big.render("Giant Impact hypothesis", True, (10, 10, 10))
    # screen.blit(title, (16, 12))
    # hud = font.render(
    #     f"b = {b_frac:.3f} (R1+R2)   v = {v_factor:.3f} v_esc   v_esc ≈ {vesc/1000:.2f} km/s    Controls: (←/→ impact parameter, ↑/↓ velocity, R reset, SPACE pause, T trails, D debris)",
    #     True, (10, 10, 10)
    # )
    # screen.blit(hud, (16, 42))

    if show_ui:
        title = big.render("Giant Impact hypothesis", True, (10, 10, 10))
        screen.blit(title, (16, 12))
        hud = font.render(
            f"b = {b_frac:.3f} (R1+R2)   v = {v_factor:.3f} v_esc   v_esc ≈ {vesc/1000:.2f} km/s    Controls: (←/→ impact parameter, ↑/↓ velocity, R reset, SPACE pause, T trails, D debris)",
            True, (10, 10, 10)
        )
        screen.blit(hud, (16, 42))

        if collided:
            ang_txt = f"{last_angle:.1f}°" if last_angle is not None else "?"
            spd_txt = f"{(last_speed/1000):.2f} km/s" if last_speed is not None else "?"
            screen.blit(big.render(f"IMPACT!  angle≈{ang_txt}  speed≈{spd_txt}", True, (160, 0, 0)), (16, 70))
            if moon is not None:
                screen.blit(font.render(f"Moon mass ≈ {moon['m']:.2e} kg", True, (0,0,0)), (16, 98))
                screen.blit(font.render(f"Moon radius ≈ {moon['R']/1000:.0f} km", True, (0,0,0)), (16, 120))


    # if collided:
    #     ang_txt = f"{last_angle:.1f}°" if last_angle is not None else "?"
    #     spd_txt = f"{(last_speed/1000):.2f} km/s" if last_speed is not None else "?"
    #     screen.blit(big.render(f"IMPACT!  angle≈{ang_txt}  speed≈{spd_txt}", True, (160, 0, 0)), (16, 70))
    #     if moon is not None:
    #         screen.blit(font.render(f"Moon mass ≈ {moon['m']:.2e} kg", True, (0,0,0)), (16, 98))
    #         screen.blit(font.render(f"Moon radius ≈ {moon['R']/1000:.0f} km", True, (0,0,0)), (16, 120))


    pygame.display.flip()

pygame.quit()

