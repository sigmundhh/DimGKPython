# -*- coding: utf-8 -*-
"""
Created on Sun Oct 21 16:38:14 2018

@author: bjohau
"""

# example exs5
# ----------------------------------------------------------------
# PURPOSE 
#    Analysis of a simply supported beam.
# ----------------------------------------------------------------

# REFERENCES
#     G"oran Sandberg 94-03-08 
#     Karl-Gunnar Olsson 95-09-28
#     Ola Dahlblom 2004-09-21
# ----------------------------------------------------------------

import numpy as np
import triangle_elements as tri

# ----- Topology -------------------------------------------------
ex = np.array([0.,1.,0.])
ey = np.array([0.,0.,1.])

th = 0.1
ep = [1,th]

E  = 2.1e11
nu = 0.3

D = np.mat([
        [ 1.0,  nu,  0.],
        [  nu, 1.0,  0.],
        [  0.,  0., (1.0-nu)/2.0]]) * E/(1.0-nu**2)

eq = [1.0, 3.0]

Ke, fe = tri.plante(ex,ey,ep,D,eq)

print('Stiffness matrix:\n', Ke)
print('Consistent forces:\n', fe)

