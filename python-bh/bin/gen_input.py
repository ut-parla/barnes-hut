#!/usr/bin/env python3
import argparse
from struct import pack
from numpy.random import default_rng
rng = default_rng()

MIN_PARTICLE_MASS = 0.5
MAX_PARTICLE_MASS = 5
DOMAIN_SIZE = 500
CHUNK = 100000

def particle_to_bytes(x, y, mass, xVel, yVel):
    return pack('ddddd', x, y, mass, xVel, yVel)

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

    with open(args.fout, "wb") as fp:
        fp.write(pack("i", num_particles))
        while num_particles != 0:
            left = min(num_particles, CHUNK)
            num_particles -= left
            particles_bytes = generateParticles(left, distribution)
            fp.write(particles_bytes)

def generateParticles(num_particles, distribution):
    max_coord = DOMAIN_SIZE-1
    particles_bytes = b''

    for _ in range(num_particles):
        if distribution == "normal":
            x = rng.random() * max_coord
            y = rng.random() * max_coord
        elif distribution == "gauss":
            x = random.gauss(max_coord/2, max_coord/12)
            y = random.gauss(max_coord/2, max_coord/12)

        xVel = (rng.random()-0.5) * 2
        yVel = (rng.random()-0.5) * 2

        # random mass
        mass = MIN_PARTICLE_MASS + rng.random() * (MAX_PARTICLE_MASS-MIN_PARTICLE_MASS)
        particles_bytes += particle_to_bytes(x, y, mass, xVel, yVel)

    return particles_bytes


if __name__ == "__main__":
    main()
