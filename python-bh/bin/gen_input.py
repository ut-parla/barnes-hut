#!/usr/bin/env python3
import argparse
import random

MIN_PARTICLE_MASS = 0.5
MAX_PARTICLE_MASS = 5
DOMAIN_SIZE = 500

class Particle:
    def __init__(self, x, y, mass, xVel, yVel):
        self.x = x
        self.y = y
        self.xVel = xVel
        self.yVel = yVel
        self.mass = mass

    def to_string(self):
        return "{}, {}, {}, {}, {}\n".format(
            self.x,
            self.y,
            self.mass,
            self.xVel,
            self.yVel
        )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(dest='distribution', help="Need a distribution: normal, gauss or splash2")
    parser.add_argument(dest='num_particles',
                        help="Number of particles to generate")
    parser.add_argument(dest='fout', help="Name of file to write out")
    args = parser.parse_args()

    distribution = str(args.distribution)
    print(f"Using distribution {distribution}")
    num_particles = int(args.num_particles)

    particles = generateParticles(num_particles, distribution)

    with open(args.fout, "w") as fp:
        fp.write(f"{num_particles}\n")
        for p in particles:
            fp.write(p.to_string())


def generateParticles(num_particles, distribution):
    max_coord = DOMAIN_SIZE-1
    particles = []

    for _ in range(num_particles):
        if distribution == "normal":
            x = random.random() * max_coord
            y = random.random() * max_coord
        elif distribution == "gauss":
            x = random.gauss(max_coord/2, max_coord/12)
            y = random.gauss(max_coord/2, max_coord/12)

        xVel = (random.random()-0.5) * 2
        yVel = (random.random()-0.5) * 2
        #angle = random.gauss(math.pi*2, math.pi/2)
        # xVel = sun.pos.dist(pos) * math.sin(angle)
        # yVel = sun.pos.dist(pos) * math.cos(angle)

        # random mass
        mass = MIN_PARTICLE_MASS + random.random() * (MAX_PARTICLE_MASS-MIN_PARTICLE_MASS)
        particles.append(Particle(x, y, mass, xVel, yVel))

    return particles


if __name__ == "__main__":
    main()
