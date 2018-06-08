# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 20:24:33 2018

@author: rahul.ghosh
"""

import data
import config

data = data.DATA()
data.read(config.TRAIN_FILENAME)
ax, ay = data.generate_batch()
bx, by = data.generate_batch()
cx, cy = data.generate_batch()
dx, dy = data.generate_batch()
ex, ey = data.generate_batch()
fx, fy = data.generate_batch()
gx, gy = data.generate_batch()
hx, hy = data.generate_batch()
ix, iy = data.generate_batch()
jx, jy = data.generate_batch()
kx, ky = data.generate_batch()