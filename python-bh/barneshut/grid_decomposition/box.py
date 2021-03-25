from barneshut.internals import Particle, Cloud
from barneshut.internals.config import Config
from collections.abc import Iterable
import numpy as np
import logging

class Box:

    def __init__(self, bottom_left, top_right, grav_kernel=None):
        self.bottom_left = bottom_left
        self.top_right = top_right
        self.cloud = Cloud(grav_kernel)
        self.is_COMs = False

    @staticmethod
    def from_list_of_boxes(boxes, is_COMs=False):
        # TODO: fix this passing around of kernels. we need a singleton or something
        grav_kernel = boxes[0].cloud.grav_kernel

        b = Box((-1,-1), (-1,-1))
        b.cloud = Cloud(grav_kernel, pre_alloc=len(boxes))
        b.is_COMs = is_COMs

        #logging.debug(f"concatenating {len(boxes)} boxes")
        for box in boxes:
            b.cloud.add_particles(box.get_COM().particles)
        return b

    def tick(self):
        self.cloud.tick_particles()

    def add_particle_slice(self, pslice):
        self.cloud.add_particle_slice(pslice)

    def add_particle(self, new_particle):
        # if we are leaf and have space, add to our set
        if self.cloud.is_full():
            print("adding to a full leaf, something is wrong")
        self.cloud.add_particle(new_particle)

    def get_COM(self):
        return self.cloud.get_COM()

    def apply_force(self, other_box):
        # don't update other box if both are the same box or if
        # other is a list of COMs
        update_other = self is not other_box and not other_box.is_COMs
        self.cloud.apply_force(other_box.cloud, update_other)
