# -*- coding: utf-8 -*-
"""
Created on Sun Oct 21 08:15:51 2018

@author: bjohau
"""
import numpy as np


def plante(ex,ey,ep,D,eq=None):

    Dshape = D.shape
    if Dshape[0] != 3:
        raise NameError('Wrong constitutive dimension in plante')

    if ep[0] == 1 :
        return tri3e(ex,ey,D,ep[1],eq)
    else:
        Dinv = np.inv(D)
        return tri3e(ex,ey,Dinv,ep[1],eq)


def tri3e(ex,ey,D,th,eq=None):
    """
    Compute the stiffness matrix for a two dimensional beam element.
    
    :param list ex: element x coordinates [x1, x2, x3]
    :param list ey: element y coordinates [y1, y2, y3]
    :param list D : 2D constitutive matrix
    :param list th: element thickness
    :param list eq: distributed loads, local directions [bx, by]
    :return mat Ke: element stiffness matrix [6 x 6]
    :return mat fe: element stiffness matrix [6 x 1] (if eq!=None)
    """

    tmp = np.matrix([[1,ex[0],ey[0]],
                     [1,ex[1],ey[1]],
                     [1,ex[2],ey[2]]])

    A2 = np.linalg.det(tmp)  # Double of triangle area

    A = A2 / 2.0

    zi_px, zi_py = zeta_partials_x_and_y(ex, ey)         # Partial derivative with respect to y

    B = tri3_Bmat(zi_px, zi_py)

    Ke = tri3_Kmat(B, D, A, th)

    if eq is None:
        return Ke
    else:
        fx = A * th * eq[0]/ 3.0
        fy = A * th * eq[1]/ 3.0
        fe = np.mat([[fx],[fy],[fx],[fy],[fx],[fy]])
        return Ke, fe
    
def tri3_Kmat(B, D, A, th):
    return (B.T * D * B) * A * th

def tri3_Bmat(zi_px, zi_py): 
    B = np.matrix([
            [zi_px[0],        0, zi_px[1],        0, zi_px[2],       0],
            [       0, zi_py[0],        0, zi_py[1],        0,zi_py[2]],
            [zi_py[0], zi_px[0], zi_py[1], zi_px[1], zi_py[2], zi_px[2]]])
    return B
    

def zeta_partials_x_and_y(ex,ey):
    """
    Compute partials of area coordinates with respect to x and y.
    
    :param list ex: element x coordinates [x1, x2, x3]
    :param list ey: element y coordinates [y1, y2, y3]
    """

    tmp = np.matrix([[1,ex[0],ey[0]],
                     [1,ex[1],ey[1]],
                     [1,ex[2],ey[2]]])

    A2 = np.linalg.det(tmp)  # Double of triangle area

    cyclic_ijk = [0,1,2,0,1]      # Cyclic permutation of the nodes i,j,k
    
    zeta_px = np.zeros(3)           # Partial derivative with respect to x
    zeta_py = np.zeros(3)           # Partial derivative with respect to y

    # TODO: fill out missing parts (or reformulate completely)
    
    for i in range(3):
        j = cyclic_ijk[i+1]
        k = cyclic_ijk[i+2]
        zeta_px[i] = (ey[j] - ey[k]) / A2
        zeta_py[i] = (ex[k] - ex[j]) / A2

    return zeta_px, zeta_py

# Functions for 6 node triangle
    
def tri6_area(ex,ey):
        
    tmp = np.matrix([[1,ex[0],ey[0]],
                     [1,ex[1],ey[1]],
                     [1,ex[2],ey[2]]])
    
    A = np.linalg.det(tmp) / 2
    
    return A


def tri6_shape_functions(zeta):
    
    cyclic_ijk = [0,1,2,0,1]      # Cyclic permutation of the nodes i,j,k

    N6 = np.zeros(6)

    # TODO: fill out missing parts (or reformulate completely)

    return N6


def tri6_shape_function_partials_x_and_y(zeta,ex,ey):
    
    zeta_px, zeta_py = zeta_partials_x_and_y(ex,ey)
    
    N6_px = np.zeros(6)
    N6_py = np.zeros(6)
    
    cyclic_ijk = [0,1,2,0,1]      # Cyclic permutation of the nodes i,j,k

    # TODO: fill out missing parts (or reformulate completely)

    return N6_px, N6_py


def tri6_Bmatrix(zeta,ex,ey):
    
    nx,ny = tri6_shape_function_partials_x_and_y(zeta, ex, ey)

    Bmatrix = np.matrix(np.zeros((3,12)))

    # TODO: fill out missing parts (or reformulate completely)

    return Bmatrix


def tri6_Kmatrix(ex,ey,D,th,eq=None):
    
    zetaInt = np.array([[0.5,0.5,0.0],
                        [0.0,0.5,0.5],
                        [0.5,0.0,0.5]])
    
    wInt = np.array([1.0/3.0,1.0/3.0,1.0/3.0])

    A    = tri6_area(ex,ey)
    
    Ke = np.matrix(np.zeros((12,12)))

    # TODO: fill out missing parts (or reformulate completely)

    if eq is None:
        return Ke
    else:
        fe = np.matrix(np.zeros((12,1)))

        # TODO: fill out missing parts (or reformulate completely)

        return Ke, fe

def tri6e(ex,ey,D,th,eq=None):
    return tri6_Kmatrix(ex,ey,D,th,eq)





  