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

    N6[0] = zeta[0] * (zeta[0] - 0.5) * 2.0
    N6[1] = zeta[1] * (zeta[1] - 0.5) * 2.0
    N6[2] = zeta[2] * (zeta[2] - 0.5) * 2.0
    N6[3] = zeta[0] * zeta[1] * 4.0
    N6[4] = zeta[1] * zeta[2] * 4.0
    N6[5] = zeta[0] * zeta[2] * 4.0

    return N6


def tri6_shape_function_partials_x_and_y(zeta,ex,ey):
    
    zeta_px, zeta_py = zeta_partials_x_and_y(ex,ey)
    
    N6_px = np.zeros(6)
    N6_py = np.zeros(6)

    cyclic_ijk = [0, 1, 2, 0, 1] # Cyclic permutation of the nodes i,j,k

    for i in range(3):
        j = cyclic_ijk[i+1]
        N6_px[i] = (4 * zeta[i] - 1) * zeta_px[i]
        N6_py[i] = (4 * zeta[i] - 1) * zeta_py[i]
        N6_px[i+3] = 4 * zeta[j] * zeta_px[i] + 4 * zeta[i] * zeta_px[j]
        N6_py[i+3] = 4 * zeta[j] * zeta_py[i] + 4 * zeta[i] * zeta_py[j]

    # TODO: fill out missing parts (or reformulate completely)

    return N6_px, N6_py


def tri6_Bmatrix(zeta,ex,ey):
    
    nx, ny = tri6_shape_function_partials_x_and_y(zeta, ex, ey)

    Bmatrix = np.matrix([
        [nx[0], 0, nx[1], 0, nx[2], 0, nx[3], 0, nx[4], 0, nx[5], 0],
        [0, ny[0], 0, ny[1], 0, ny[2], 0, ny[3], 0, ny[4], 0, ny[5]],
        [ny[0], nx[0], ny[1], nx[1], ny[2], nx[2], ny[3], nx[3], ny[4], nx[4], ny[5], nx[5]]])

    return Bmatrix


def tri6_Kmatrix(ex,ey,D,th,eq=None):
    
    zetaInt = np.array([[0.5, 0.5, 0.0],
                        [0.0, 0.5, 0.5],
                        [0.5, 0.0, 0.5]])
    
    wInt = np.array([1.0/3.0, 1.0/3.0, 1.0/3.0])

    A = tri6_area(ex, ey)
    
    Ke = np.zeros((12, 12))
    fe = np.zeros((12, 1))

    for iG in range(3):
        zeta = zetaInt[iG]

        B = tri6_Bmatrix(zeta, ex, ey)
        Ke += (B.T @ D @ B) * A * th * wInt[iG]

        if eq is not None:
            fvec = np.array([[eq[0]], [eq[1]]])
            N6 = tri6_shape_functions(zeta)
            N2mat = np.zeros((2, 12))

        for i in range(6):
            N2mat[0, i * 2] = N6[i]
            N2mat[1, 1 + i * 2] = N6[i]

            fe += N2mat.T @ fvec * A * wInt[iG]

        if eq is None:
            return Ke
    else:
        return Ke, fe

def tri6e(ex,ey,D,th,eq=None):
    return tri6_Kmatrix(ex,ey,D,th,eq)




  