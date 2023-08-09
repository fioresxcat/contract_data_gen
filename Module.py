import numpy as np
from SubModule import SubModule
from common import *
import pdb

class Module:
    def __init__(self):
        self.submodules = []
        self.canvas = None
        self.fields = []

    def get_shape(self):
        if self.canvas is None:
            print('Module does not build')
        return self.canvas.shape[:2]
    
    def get_fields(self):
        fields = []
        for submodule in self.submodules:
            fields += list(submodule['submodule'].fields)
        self.fields = fields
        return fields

    def __paste__(self, submodule, position):
        '''
        This function not actually "paste"
        It only compute the cordinate of submodule in the module space
        '''
        submodule_shape = submodule.get_shape()
        x, y = position #top-left
        x2 = x + submodule_shape[1]
        y2 = y + submodule_shape[0]
        # Paste submodule and correct the box coordinate
        boxes = [text['box'] for text in submodule.fields]
        boxes = mapping(boxes, position)
        for i in range(len(boxes)):
            submodule.fields[i]['box'] = boxes[i]
        submodule.in_module_position = position
        self.submodules.append({'box':[x, y, x2, y2], 'submodule':submodule})
    
    def __call__(self, submodules:list):
        '''
        Paste all submodules to module
        '''

    def resize(self, new_shape):
        self.get_fields()
        boxes = [text['box'] for text in self.fields]
        self.canvas, boxes = resize(new_shape, self.canvas, boxes)
        for i in range(len(boxes)):
            self.fields[i]['box'] = boxes[i]
        self.shape = self.canvas.shape[:2]
