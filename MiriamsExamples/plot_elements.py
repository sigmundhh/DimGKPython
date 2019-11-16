# -*- coding: utf-8 -*-
"""
Created on Sun Oct 21 16:38:14 2018

@author: andreas arnholm
"""
import numpy as np
import calfem.core as cfc
import triangle_elements as tri
import four_node_element as quads
import calfem.vis as cfv
import matplotlib.pyplot as plt
import time


# Cantilever with dimensions H x L x thickness
H         = 2.5
L         = 12.5
thickness = 0.1

# Properties
E = 2.1e11
nu = 0

# Distributed load in x and y
eq = np.array([0,1.0e3])

# Exact solution
wy = (eq[1]*H*thickness*L**4)/(8*E*1/12*H**3*thickness)

def y_deformation_eq_plan3(number_of_elements, height, length, thickness, E, nu, eq):
    nElx = number_of_elements
    nEly = number_of_elements
    
    Dmat = np.mat([
            [ 1.0,  nu,  0.],
            [  nu, 1.0,  0.],
            [  0.,  0., (1.0-nu)/2.0]]) * E/(1.0-nu**2)


    nEls = nElx * nEly * 2
    nNodx = nElx +1
    nNody = nEly +1
    nNods = nNodx * nNody
    L_elx = length / nElx
    L_ely = height / nEly
    nelnod = 3


    coords = np.zeros((nNods,2))
    dofs   = np.zeros((nNods,2),int)       #Dofs is starting on 1 on first dof

    inod = 0 # The coords table starts numbering on 0
    idof = 1 # The values of the dofs start on 1
    ndofs = nNods *2

    # Set the node coordinates and node dofs

    for i in range(nNodx):
        for j in range(nNody):
            coords[inod,0] = L_elx * i
            coords[inod,1] = L_ely * j
            dofs[inod,0] = idof
            dofs[inod,1] = idof+1
            idof += 2
            inod += 1

    # Set the element connectivites and element dofs

    elnods = np.zeros((nEls,3),int)
    eldofs = np.zeros((nEls,3*2),int)

    iel = 0
    for i in range(nElx):
        for j in range(nEly):
            # 0 based node numbers
            i1 =     i*nNody + j
            i2 = (i+1)*nNody + j
            i3 = (i+1)*nNody + (j+1)
            i4 =     i*nNody + (j+1)
            # 1 based dof numbers
            idof1 = i1*2 +1  # 1 based dof number for the element
            idof2 = i2*2 +1
            idof3 = i3*2 +1
            idof4 = i4*2 +1

            elnods[iel,:] = [i1, i2, i3]
            eldofs[iel,:] = [idof1,idof1+1, idof2,idof2+1, idof3,idof3+1]
            iel += 1
            elnods[iel,:] = [i3,i4,i1]
            eldofs[iel,:] = [idof3,idof3+1, idof4,idof4+1, idof1,idof1+1]
            iel += 1

    # Set fixed boundary condition on left side, i.e. nodes 0-nNody
    bc = np.array(np.zeros(nNody*2),'i')
    idof = 1
    for i in range(nNody):
        idx = i*2
        bc[idx]   = idof
        bc[idx+1] = idof+1
        idof += 2

    # Assemble stiffness matrix

    # Extract element coordinates
    ex, ey = cfc.coordxtr(eldofs,coords,dofs)

    K = np.zeros((ndofs,ndofs))
    R = np.zeros((ndofs,1))
    #r = np.zeros( ndofs )

    for iel in range(nEls):
        K_el, f_el = tri.plan3(ex[iel],ey[iel],Dmat,thickness,eq)
        cfc.assem(eldofs[iel],K,K_el,R,f_el)

    r = cfc.solveq(K,R,bc)[0]

    yTR = r[-1,0]
    # Returning the deformation at the end in y-direction
    return yTR


def y_deformation_eq_plan6(number_of_elements, height, length, thickness, E, nu, eq):

    nElx = number_of_elements
    nEly = number_of_elements

    Dmat = np.mat([
        [ 1.0,  nu,  0.],
        [  nu, 1.0,  0.],
        [  0.,  0., (1.0-nu)/2.0]]) * E/(1.0-nu**2)

    nEls  = nElx * nEly * 2
    nNodx = nElx*2 +1
    nNody = nEly*2 +1
    nNods = nNodx * nNody

    L_elx = length / nElx / 2
    L_ely = height / nEly / 2

    nelnod = 6

    coords = np.zeros((nNods,2))
    dofs   = np.zeros((nNods,2),int)       #Dofs is starting on 1 on first dof
    #edofs  = np.zeros((nEls,nelnod*2),int) #edofs are also starting on 1 based dof


    inod = 0 # The coords table starts numbering on 0
    idof = 1 # The values of the dofs start on 1
    ndofs = nNods *2

    # Set the node coordinates and node dofs

    for i in range(nNodx):
        for j in range(nNody):
            coords[inod,0] = L_elx * i
            coords[inod,1] = L_ely * j
            dofs[inod,0] = idof
            dofs[inod,1] = idof+1
            idof += 2
            inod += 1

    # Set the element connectivites and element dofs
    elnods = np.zeros((nEls,6),int)
    eldofs = np.zeros((nEls,6*2),int)

    iel = 0
    for i in range(nElx):
        ii = i*2
        for j in range(nEly):
            jj = j*2
            # 0 based node numbers, 9 nodes of a 3x3 patch
            nod9 = np.array([
                (ii  )*nNody + (jj  ),
                (ii+1)*nNody + (jj  ),
                (ii+2)*nNody + (jj  ),
                (ii  )*nNody + (jj+1),
                (ii+1)*nNody + (jj+1),
                (ii+2)*nNody + (jj+1),
                (ii  )*nNody + (jj+2),
                (ii+1)*nNody + (jj+2),
                (ii+2)*nNody + (jj+2)],'i')

            elnods[iel,:] = [nod9[0],nod9[2],nod9[8],nod9[1],nod9[5],nod9[4]]
            eldofs[iel, ::2] = elnods[iel,:] * 2 + 1
            eldofs[iel,1::2] = elnods[iel,:] * 2 + 2
            iel += 1
            elnods[iel,:] = [nod9[8],nod9[6],nod9[0],nod9[7],nod9[3],nod9[4]]
            eldofs[iel, ::2] = elnods[iel,:] * 2 + 1
            eldofs[iel,1::2] = elnods[iel,:] * 2 + 2
            iel += 1

    # Extract element coordinates
    ex, ey = cfc.coordxtr(eldofs,coords,dofs)

    # Set fixed boundary condition on left side, i.e. nodes 0-nNody
    bc = np.array(np.zeros(nNody*2),'i')
    idof = 1
    for i in range(nNody):
        idx = i*2
        bc[idx]   = idof
        bc[idx+1] = idof+1
        idof += 2

    # Assemble stiffness matrix

    K = np.zeros((ndofs,ndofs))
    R = np.zeros((ndofs,1))
    #r = np.zeros( ndofs )

    for iel in range(nEls):
        K_el, B, f_el = tri.plan6(ex[iel],ey[iel],Dmat,thickness,eq)
        cfc.assem(eldofs[iel],K,K_el,R,f_el)

    r = cfc.solveq(K,R,bc)[0]

    yTR = r[-1,0]
    # Returning the deformation at the end in y-direction
    return yTR

def y_deformation_eq_quad4(number_of_elements, height, length, thickness, E, nu, eq):
    nElx = number_of_elements
    nEly = number_of_elements

    eqTotal = eq * length * height * thickness

    # Material properties and thickness

    ep = [1,thickness]

    Dmat = np.mat([
            [ 1.0,  nu,  0.],
            [  nu, 1.0,  0.],
            [  0.,  0., (1.0-nu)/2.0]]) * E/(1.0-nu**2)

    nEls = nElx * nEly
    nNodx = nElx +1
    nNody = nEly +1
    nNods = nNodx * nNody

    L_elx = length / nElx
    L_ely = height / nEly

    nelnod = 4

    coords = np.zeros((nNods,2))
    dofs   = np.zeros((nNods,2),int)       #Dofs is starting on 1 on first dof
    edofs  = np.zeros((nEls,nelnod*2),int) #edofs are also starting on 1 based dof

    inod = 0 # The coords table starts numbering on 0
    idof = 1 # The values of the dofs start on 1
    ndofs = nNods *2

    # Set the node coordinates and node dofs

    for i in range(nNodx):
        for j in range(nNody):
            coords[inod,0] = L_elx * i
            coords[inod,1] = L_ely * j
            dofs[inod,0] = idof
            dofs[inod,1] = idof+1
            idof += 2
            inod += 1

    # Set the element connectivites and element dofs

    elnods = np.zeros((nEls,nelnod),int)
    eldofs = np.zeros((nEls,nelnod*2),int)

    iel = 0
    for i in range(nElx):
        for j in range(nEly):
            # 0 based node numbers
            i1 =     i*nNody + j
            i2 = (i+1)*nNody + j
            i3 = (i+1)*nNody + (j+1)
            i4 =     i*nNody + (j+1)
            # 1 based dof numbers
            idof1 = i1*2 +1  # 1 based dof number for the element
            idof2 = i2*2 +1
            idof3 = i3*2 +1
            idof4 = i4*2 +1

            elnods[iel,:] = [i1, i2, i3, i4]
            eldofs[iel,:] = [idof1,idof1+1, idof2,idof2+1, idof3,idof3+1, idof4,idof4+1]
            iel += 1

    # Set fixed boundary condition on left side, i.e. nodes 0-nNody
    bc = np.array(np.zeros(nNody*2),'i')
    idof = 1
    for i in range(nNody):
        idx = i*2
        bc[idx]   = idof
        bc[idx+1] = idof+1
        idof += 2

    # Assemble stiffness matrix

    # Extract element coordinates
    ex, ey = cfc.coordxtr(eldofs,coords,dofs)

    K = np.zeros((ndofs,ndofs))
    R = np.zeros((ndofs,1))
  
    for iel in range(nEls):
        K_el, B, f_el = quads.plani4e(ex[iel],ey[iel],Dmat,thickness,eq)
        cfc.assem(eldofs[iel],K,K_el,R,f_el)

    r = cfc.solveq(K,R,bc)[0]
    yTR = r[-1,0]

    # Returning the deformation at the end in y-direction
    return yTR


# Defining the the amounts of elements that are going to be plotted
number_of_elements = range(6,50,2)

# Creating lists for the different elements
y_def_plan3 = []
y_def_plan6 = []
y_def_quad4 = []

# Looping through the elements and storing the results in lists
for numEl in number_of_elements:
    y_def_plan3.append(y_deformation_eq_plan3(numEl, H, L, thickness, E, 0, eq))
    y_def_plan6.append(y_deformation_eq_plan6(numEl, H, L, thickness, E, 0, eq))
    y_def_quad4.append(y_deformation_eq_quad4(numEl, H, L, thickness, E, 0, eq))


# Plotting the results using matplotlib
plt.plot(number_of_elements, y_def_plan3, number_of_elements, y_def_plan6, number_of_elements, y_def_quad4)
plt.xlabel('Number of elements')
plt.ylabel('Derformation [y]')
plt.legend(['3-nodes Triangle', '6-nodes Triangle', '4-nodes Isoparametric'])
plt.show()