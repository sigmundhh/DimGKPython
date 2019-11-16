import numpy as np

def plani(ex,ey,ep,D,eq=None):
    colD = np.shape(D)[0]
    if colD>3:
        # --------- plane stress --------
        if ep[0] == 1:
            Cm=np.linalg.inv(D)
            Dm=np.linalg.inv(Cm[ np.ix_([0,1,3],[0,1,3]) ])
        elif ep[0]==2:
            Dm = D[np.ix_([0,1,3],[0,1,3])]
        else:
            cfinfo("Error! Check first argument, ptype=1 or 2 allowed")
    else:
        Dm=D
    return plani4e(ex,ey,Dm,ep[1],eq)

def plani4e(x,y,D,t,eq=None):
    if eq == None:
        q = np.zeros((2,1))
    else:
        q = np.reshape(eq, (2,1))
    
    # For linear deformations we use one gauss point. Then follows:
    # w = 2, a = 0 --> xsi = eta = 0
    gp = np.mat([.0,.0])
    w = 2.

    xsi = gp[:,0]
    eta = gp[:,1]

    # All values for shape functions will be 1/4, for example N1 = (1-eta)(1-xsi)/4
    N = 0.25
    dNr = np.mat([             #Uttrykk for translasjonmatrise
        [-N, N, N, -N],
        [-N, -N, N, N]
    ])

    JT = dNr*np.mat([x,y]).T 

    indx = np.array([1,2])
    detJ = np.linalg.det(JT[indx-1,:])
    if detJ < 10*np.finfo(float).eps:
        cfinfo("Jacobideterminant equal or less than zero!")

    JTinv = np.linalg.inv(JT[indx-1,:])  
    dNx=JTinv*dNr[indx-1,:]

    B=np.matrix(np.zeros((3,8)))
    N2=np.matrix(np.zeros((2,8)))
    counter=0    
    for index in [0,2,4,6]:
        B[0,index] = dNx[0,counter]
        B[2,index] = dNx[1,counter]
        N2[0,index]=N
        counter=counter+1

    counter=0    
    for index in [1,3,5,7]:
        B[1,index]   = dNx[1,counter]
        B[2,index]   = dNx[0,counter]
        N2[1,index]  = N
        counter=counter+1

    Ke = B.T * D * B * detJ * w*w * t
    fe = N2.T*q*detJ* w*w * t
    return Ke,fe