from barneshut.internals import Particle, Cloud
from barneshut.internals.config import Config
from collections.abc import Iterable
import numpy as np
import logging

class Box:

    def __init__(self, bottom_left, top_right):
        self.bottom_left = bottom_left
        self.top_right = top_right
        self.cloud = Cloud()
        self.is_COMs = False

    @staticmethod
    def from_list_of_boxes(boxes, is_COMs=False):
        # TODO: fix this passing around of kernels. we need a singleton or something

        b = Box((-1,-1), (-1,-1))
        b.cloud = Cloud(pre_alloc=len(boxes))
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

    def apply_force(self, other_box, update_other=False):
        self.cloud.apply_force(other_box.cloud, update_other)
