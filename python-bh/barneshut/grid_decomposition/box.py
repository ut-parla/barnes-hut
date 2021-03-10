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

    @staticmethod
    def from_list_of_boxes(boxes):
        b = Box((-1,-1), (-1,-1))
        b.cloud = Cloud(pre_alloc=len(boxes))

        logging.debug(f"concatenating {len(boxes)} boxes")
        for box in boxes:
            b.cloud.add_particles(box.get_COM().particles)
        return b

    def tick(self):
        self.cloud.tick_particles()

    def add_particle(self, new_particle):
        # if we are leaf and have space, add to our set
        if self.cloud.is_full():
            print("adding to a full leaf, something is wrong")
        self.cloud.add_particle(new_particle)

    def get_COM(self):
        return self.cloud.get_COM()

    def apply_force(self, other_box):
        self.cloud.apply_force(other_box.cloud)
        #if self.cloud.is_empty() or other_box.cloud.is_empty():
        #    return
        #use_COM = self.approximation_distance(other_box)
        #self.cloud.apply_force(other_box.cloud, use_COM)

    # # this checks if nodes are neighbors
    # def approximation_distance(self, other_box):
    #     corners1 = self.get_corners()
    #     corners2 = other_box.get_corners()

    #     # there's gotta be a better way to do this
    #     for x in corners1[0]:
    #         for y in corners1[1]:
    #             px1, px2 = corners2[0]
    #             py1, py2 = corners2[1]
    #             if (
    #                 ((x >= px1 and x <= px2) or (x >= px2 and x <= px1)) and
    #                 ((y >= py1 and y <= py2) or (y >= py2 and y <= py1)) 
    #                ):
    #                 return False
    #     for x in corners2[0]:
    #         for y in corners2[1]:
    #             px1, px2 = corners1[0]
    #             py1, py2 = corners1[1]
    #             if (
    #                 ((x >= px1 and x <= px2) or (x >= px2 and x <= px1)) and
    #                 ((y >= py1 and y <= py2) or (y >= py2 and y <= py1)) 
    #                ):
    #                 return False

    #     return True

    # def get_corners(self):
    #     x1, x2 = self.bottom_left[0], self.top_right[0]
    #     y1, y2 = self.bottom_left[1], self.top_right[1]
    #     return ((x1, x2), (y1, y2))
