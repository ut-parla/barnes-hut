#!/usr/bin/env python3
import argparse
import random

MAX_PARTICLE_MASS = 5

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
    parser.add_argument(dest='max_coord', help="Max coord a point will have")
    parser.add_argument(dest='num_particles',
                        help="Number of particles to generate")
    parser.add_argument(dest='fout', help="Name of file to write out")
    args = parser.parse_args()

    max_coord = int(args.max_coord)
    num_particles = int(args.num_particles)

    particles = generateParticles(max_coord, num_particles)

    with open(args.fout, "w") as fp:
        fp.write(f"{max_coord}\n")
        fp.write(f"{num_particles}\n")
        for p in particles:
            fp.write(p.to_string())


def generateParticles(max_coord, num_particles):
    w = h = max_coord
    particles = []

    for _ in range(num_particles):
        # x = (self.width - 150) + random.random()*100
        # y = (self.height - 150) + random.random()*100

        x = random.gauss(w/2, w/12)
        y = random.gauss(h/2, h/12)

        xVel = (random.random()-0.5) * 2
        yVel = (random.random()-0.5) * 2
        #angle = random.gauss(math.pi*2, math.pi/2)
        # xVel = sun.pos.dist(pos) * math.sin(angle)
        # yVel = sun.pos.dist(pos) * math.cos(angle)

        # random mass
        mass = random.random() * MAX_PARTICLE_MASS
        particles.append(Particle(x, y, mass, xVel, yVel))

    return particles


if __name__ == "__main__":
    main()
