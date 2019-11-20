# -*- coding: utf-8 -*-
"""
Created on Sun Oct 21 08:15:51 2018

@author: bjohau
"""
import numpy as np
import sys

def gauss_points(iRule):
    """
    Returns gauss coordinates and weight given integration number

    Parameters:

        iRule = number of integration points

    Returns:

        gp : row-vector containing gauss coordinates
        gw : row-vector containing gauss weight for integration point

    """
    gauss_position = [[ 0.000000000],
                      [-0.577350269,  0.577350269],
                      [-0.774596669,  0.000000000,  0.774596669],
                      [-0.8611363116, -0.3399810436, 0.3399810436, 0.8611363116],
                      [-0.9061798459, -0.5384693101, 0.0000000000, 0.5384693101, 0.9061798459]]
    gauss_weight   = [[2.000000000],
                      [1.000000000,   1.000000000],
                      [0.555555556,   0.888888889,  0.555555556],
                      [0.3478548451,  0.6521451549, 0.6521451549, 0.3478548451],
                      [0.2369268850,  0.4786286705, 0.5688888889, 0.4786286705, 0.2369268850]]


    if iRule < 1 and iRule > 5:
        sys.exit("Invalid number of integration points.")

    idx = iRule - 1
    return gauss_position[idx], gauss_weight[idx]


def quad4_shapefuncs(xsi, eta):
    """
    Calculates shape functions evaluated at xsi, eta
    :param float xsi : The xsi coordinate we're interested in
    :param float eta : The eta coordinate we're interested in
    :return list N: The four shapefuncions evaluated in the point xsi and eta
    """
    # ----- Shape functions -----
    # 
    N = np.zeros(4)
    N[0] = 0.25*(1+xsi)*(1+eta)
    N[1] = 0.25*(1-xsi)*(1+eta)
    N[2] = 0.25*(1-xsi)*(1-eta)
    N[3] = 0.25*(1+xsi)*(1-eta)

    return N

def quad4_shapefuncs_grad_xsi(xsi, eta):
    """
    Calculates derivatives of shape functions wrt. xsi
    
    :param float xsi : The xsi coordinate we're interested in
    :param float eta : The eta coordinate we're interested in
    :return list Ndeta: The four shapefuncions derivatives with 
                        respect to xsi in the point (xsi, eta)
    """  
    Ndxi = np.zeros(4)
    Ndxi[0] = 0.25*(1+eta)
    Ndxi[1] = -0.25*(1+eta)
    Ndxi[2] = -0.25*(1-eta)
    Ndxi[3] = 0.25*(1-eta)

    return Ndxi


def quad4_shapefuncs_grad_eta(xsi, eta):
    """
    Calculates derivatives of shape functions wrt. eta
    :param float xsi : The xsi coordinate we're interested in
    :param float eta : The eta coordinate we're interested in
    :return list Ndeta: The four shapefuncions derivatives with 
                        respect to eta in the point (xsi, eta)
    """
    Ndeta = np.zeros(4)
    Ndeta[0] = 0.25*(1+xsi)
    Ndeta[1] = 0.25*(1-xsi)
    Ndeta[2] = -0.25*(1-xsi)
    Ndeta[3] = -0.25*(1+xsi)
    return Ndeta


def quad4e(ex, ey, D, thickness, eq=None):
    """
    Calculates the stiffness matrix for a 8 node isoparametric element in plane stress

    :param list ex : [x1 ... x4] The element x-coordinates of the four corners
    :param list ey : [y1 ... y4] The element y-coordinates of the four corners
    :param mat D : Constitutive matrix. 3x3 Matrix describing the relation between stress and strain. 
    :param float thickness : The thickness of the element
    :param list eq : [bx, by]   Equidistributed loads on the element in x and y. Has a unit of force pr unit area 
        bx:     Distributed force in x direction
        by:     Distributed force in y direction

    :return mat Ke : Stiffness matrix for the element (8 x 8)
    :return mat fe : equivalent nodal forces (8 x 1)

    """
    t = thickness

    # Make the load vector ready for intergration
    if eq is 0:
        f = np.zeros((2,1))  # Create zero matrix for load if load is zero
    else:
        f = np.array([eq]).T  # Convert load to 2x1 matrix

    Ke = np.zeros((8,8))        # Create zero matrix for stiffness matrix
    fe = np.zeros((8,1))        # Create zero matrix for distributed load

    numGaussPoints = 2  # Number of integration points
    gp, gw = gauss_points(numGaussPoints)  # Get integration points and -weight

    for iGauss in range(numGaussPoints):  # Solves for K and fe at all integration points
        for jGauss in range(numGaussPoints):
            
            # The current postition in (xsi, eta) space
            xsi = gp[iGauss]
            eta = gp[jGauss]
            
            # Determine the values of the shapefunctions and their derivatives at the current position
            # wrt. xsi and eta
            Ndxsi = quad4_shapefuncs_grad_xsi(xsi, eta)
            Ndeta = quad4_shapefuncs_grad_eta(xsi, eta)
            N1    = quad4_shapefuncs(xsi, eta)  # Collect shape functions evaluated at xsi and eta

            # Matrix H and G defined according to page 52 of Waløens notes
            H = np.transpose([ex, ey])    # Collect global x- and y coordinates in one matrix
            G = np.array([Ndxsi, Ndeta])  # Collect gradients of shape function evaluated at xsi and eta

            # The Jacobian matrix is obtained by a matrix multiplacation between G and H
            J = np.matmul(G,H)
            
            # Determine the inverse of the Jacobian and its determinant. 
            # We need this to find the the derivative of the shape functions wrt. x and y
            invJ = np.linalg.inv(J)  # Inverse of Jacobian
            detJ = np.linalg.det(J)  # Determinant of Jacobian

            dN = invJ @ G            # Derivatives of shape functions with respect to x and y
            dNdx = dN[0]             # Make an own variable with the shapefunction's derivatives wrt. x
            dNdy = dN[1]             # Make an own variable with the shapefunction's derivatives wrt. x

            #Strain displacement matrix at current xsi and eta. 
            # We will fill in this matrix as described in theory section of the delivered report
            B  = np.zeros((3,8))
            
            #Displacement interpolation xsi and eta
            N2 = np.zeros((2,8))
            
            for k in range(8):
                if(k%2 == 0):
                    B[0][k] = dNdx[k//2]
                    N2[0][k] = N1[k//2]
                    B[2][k] = dNdy[k//2]
                else:
                    B[0][k] = 0
            for k in range(8):
                if(k%2 == 1):
                    B[1][k] = dNdy[k//2]
                    N2[1][k] = N1[k//2]
                    B[2][k] = dNdx[k//2]
                else:
                    B[1][k] = 0
            
            # Evaluates integrand at current integration points and adds to final solution
            Ke += (B.T) @ D @ B * detJ * t * gw[iGauss] * gw[jGauss]
            fe += (N2.T) @ f    * detJ * t * gw[iGauss] * gw[jGauss]

    return Ke, fe  # Returns stiffness matrix and nodal force vector


def quad9e(ex,ey,D,th,eq=None):
    """
    Compute the stiffness matrix for a four node membrane element.

    :param list ex: element x coordinates [x1, x2, x3]
    :param list ey: element y coordinates [y1, y2, y3]
    :param list D : 2D constitutive matrix
    :param list th: element thickness
    :param list eq: distributed loads, local directions [bx, by]
    :return mat Ke: element stiffness matrix [6 x 6]
    :return mat fe: consistent load vector [6 x 1] (if eq!=None)
    """

    t = th

    if eq is 0:
        f = np.zeros((2, 1))  # Create zero matrix for load if load is zero
    else:
        f = np.array([eq]).T  # Convert load to 2x1 matrix

    Ke = np.zeros((18, 18))        # Create zero matrix for stiffness matrix
    fe = np.zeros((18, 1))        # Create zero matrix for distributed load

    numGaussPoints = 3  # Number of integration points
    gp, gw = gauss_points(numGaussPoints)  # Get integration points and -weight

    for iGauss in range(numGaussPoints):  # Solves for K and fe at all integration points
        for jGauss in range(numGaussPoints):

            xsi = gp[iGauss]
            eta = gp[jGauss]

            Ndxsi = quad9_shapefuncs_grad_xsi(xsi, eta)
            Ndeta = quad9_shapefuncs_grad_eta(xsi, eta)
            N1    = quad9_shapefuncs(xsi, eta)  # Collect shape functions evaluated at xsi and eta

            # Matrix H and G defined according to page 52 of Waløens notes
            H = np.transpose([ex, ey])    # Collect global x- and y coordinates in one matrix
            G = np.array([Ndxsi, Ndeta])  # Collect gradients of shape function evaluated at xsi and eta
            #print("G: ", G)
            #print("H: ", H)
            J = np.matmul(G,H)
            N_dxsi_and_deta = np.zeros(18)
            N_dxsi_and_deta[0:9] = Ndxsi
            N_dxsi_and_deta[9:18] = Ndeta
            
            #print("J: ", J)
            invJ = np.linalg.inv(J)  # Inverse of Jacobian
            detJ = np.linalg.det(J)  # Determinant of Jacobian

            dN = invJ @ G  # Derivatives of shape functions with respect to x and y
            dNdx = dN[0]
            dNdy = dN[1]

            # Strain displacement matrix calculated at position xsi, eta

            #TODO: Fill out correct values for strain displacement matrix at current xsi and eta
            B  = np.zeros((3,18))
            # Flyll inn Nd i B, har dette i notatbok. Fort gjort
            """B[0][0:4] = dNdx
            B[0][4:8] = np.zeros(4)
            B[1][0:4] = np.zeros(4)
            B[1][4:8] = dNdy  
            B[2][0:4] = dNdx
            B[2][4:8] = dNdy"""
            
            N2 = np.zeros((2,18))
            for k in range(18):
                if(k%2 == 0):
                    B[0][k] = dNdx[k//2]
                    N2[0][k] = N1[k//2]
                    B[2][k] = dNdy[k//2]
                else:
                    B[0][k] = 0
            for k in range(18):
                if(k%2 == 1):
                    B[1][k] = dNdy[k//2]
                    N2[1][k] = N1[k//2]
                    B[2][k] = dNdx[k//2]
                else:
                    B[1][k] = 0
                    
            #B[2][0:9] = dNdx
            #B[2][9:18] = dNdy
            print("N2: ", N2)
            print("B: ", B)
            
            

            #TODO: Fill out correct values for displacement interpolation xsi and eta
            # Aner ikke hvordan jeg gjør dette...

            """N2 = np.zeros((2,8))
            N2[0][0:4] = N1
            N2[0][4:8] = np.zeros(4)
            N2[1][0:4] = np.zeros(4)
            N2[1][4:8] = N1"""

            # Evaluates integrand at current integration points and adds to final solution
            Ke += (B.T) @ D @ B * detJ * t * gw[iGauss] * gw[jGauss]
            fe += (N2.T) @ f    * detJ * t * gw[iGauss] * gw[jGauss]

    return Ke, fe  # Returns stiffness matrix and nodal force vector

def quad9_shapefuncs(xsi, eta):
    """
    Calculates shape functions evaluated at xi, eta
    """
    # ----- Shape functions -----
    N = np.zeros(9)
    N[0] = 0.25*(1+xsi)*(1+eta) * xsi * eta
    N[1] = -0.25*(1-xsi)*(1+eta) * xsi * eta
    N[2] = 0.25*(1-xsi)*(1-eta) * xsi * eta
    N[3] = -0.25*(1+xsi)*(1-eta) * xsi * eta
    N[4] = 0.5*(1+xsi)*(1-xsi)*(1+eta) * eta
    N[5] = -0.5*(1+eta)*(1-eta)*(1-xsi) * xsi
    N[6] = -0.5*(1+xsi)*(1-xsi)*(1-eta) * eta
    N[7] = 0.5*(1+eta)*(1-eta)*(1+xsi) * xsi
    N[8] = (1+eta)*(1-eta)*(1+xsi)*(1-xsi)

    return N

def quad9_shapefuncs_grad_eta(xsi, eta):
    """
    Calculates derivatives of shape functions wrt. eta
    
    :param float xsi : 
    :param float eta : 
    :return mat Ndeta: 
    """
    # ----- Derivatives of shape functions with respect to eta -----

    Ndeta = np.zeros(9)
    Ndeta[0] = 0.25*(1+xsi)*(1+2*eta)*xsi
    Ndeta[1] = -0.25*(1-xsi)*(1+2*eta)*xsi
    Ndeta[2] = 0.25*(1-xsi)*(1-2*eta)*xsi
    Ndeta[3] = -0.25*(1+xsi)*(1-2*eta)*xsi
    Ndeta[4] = 0.5*(1+xsi)*(1-xsi)*(1+2*eta)
    Ndeta[5] = eta*(1-xsi)*xsi
    Ndeta[6] = -0.5*(1+xsi)*(1-xsi)*(1-2*eta)
    Ndeta[7] = -eta*(1+xsi)*xsi
    Ndeta[8] = (-2*eta)*(1-xsi**2)

    return Ndeta


def quad9_shapefuncs_grad_xsi(xsi, eta):
    """
    Calculates derivatives of shape functions wrt. xsi
    """
    # ----- Derivatives of shape functions with respect to xsi -----    
    Ndxi = np.zeros(9)
    Ndxi[0] = 0.25*(1+2*xsi)*(1+eta)*eta
    Ndxi[1] = -0.25*(1-2*xsi)*(1+eta)*eta
    Ndxi[2] = 0.25*(1-2*xsi)*(1-eta)*eta
    Ndxi[3] = -0.25*(1+2*xsi)*(1-eta)*eta
    Ndxi[4] = -xsi*(1+eta)*eta
    Ndxi[5] = -0.5*(1-2*xsi)*(1+eta)*(1-eta)
    Ndxi[6] = xsi*(1-eta)*eta
    Ndxi[7] = 0.5*(1-eta)*(1+eta)*(1+2*xsi)
    Ndxi[8] = (-2*xsi)*(1-eta**2)

    return Ndxi
