import numpy as np

def plante(ex,ey,ep,D,eq=None):
    
    Dshape = D.shape
    if Dshape[0] != 3:
        raise NameError('Wrong constitutive dimension in plante')
        
    if ep[0] == 1 :
        return plan3(ex,ey,D,ep[1],eq)
    else:
        Dinv = np.inv(D)
        return plan3(ex,ey,Dinv,ep[1],eq)

def plan3(x,y,D,t,eq=None):

    A = 0.5*np.linalg.det(np.mat([      #Regner arealet til trekanten
        [1, x[0], y[0]],
        [1, x[1], y[1]],
        [1, x[2], y[2]]
        ]))
    N = np.mat([                        #Uttrykk for translasjonmatrise
        [1, x[0], y[0], 0, 0, 0],
        [0, 0, 0, 1, x[0], y[0]],
        [1, x[1], y[1], 0, 0, 0],
        [0, 0, 0, 1, x[1], y[1]],
        [1, x[2], y[2], 0, 0, 0],
        [0, 0, 0, 1, x[2], y[2]]
    ])
    Epsilon = np.mat([                  #Tøyning-forskyvningsmatrise
        [0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1],
        [0, 0, 1, 0, 1, 0]
    ])
    B = np.dot(Epsilon, np.linalg.inv(N))        #Regner ut B-matrise

    Ke = B.T * D * B * A * t
    if eq is None:
    	return Ke
    else:                                       #De fordelte lastene er like store i hver node
        fx = A * t * eq[0] / 3.0
        fy = A * t * eq[1] / 3.0
        fe = np.mat([[fx],[fy],[fx],[fy],[fx],[fy]])
        return Ke,fe

def plante6(ex,ey,ep,D,eq=None):
    
    Dshape = D.shape
    if Dshape[0] != 3:
        raise NameError('Wrong constitutive dimension in plante')
        
    if ep[0] == 1 :
        return plan6(ex,ey,D,ep[1],eq)
    else:
        Dinv = np.inv(D)
        return plan6(ex,ey,Dinv,ep[1],eq)

def b6(node,x,y):                       #Funksjon for å regne ut B-matrise for et gausspunkt til et 6-noders element
    A = 0.5*np.linalg.det(np.mat([      #Regner arealet til trekanten
        [1, x[0], y[0]],
        [1, x[1], y[1]],
        [1, x[2], y[2]]
        ]))
    
    cyclic_ijk = [0,1,2,0,1]            #Cyclic for å slippe utstrakt %-bruk
    zi_px = np.zeros(3)                 #Regner ut de deriverte av zheta_i med hensyn på x og y
    zi_py = np.zeros(3)
    for i in range(3):
        j = cyclic_ijk[i+1]
        k = cyclic_ijk[i+2]
        zi_px[i] = (y[j] - y[k]) / (A*2)
        zi_py[i] = (x[k] - x[j]) / (A*2)
    
    B = np.zeros((3,12))
    Ni_px = np.zeros(6)                 #Regner ut den deriverte av N med hensyn på x og y            
    Ni_py = np.zeros(6)
    for i in range(3):
        j = cyclic_ijk[i]
        k = cyclic_ijk[j+1]
        Ni_px[i] = (4*node[0,i] - 1) * zi_px[i]
        Ni_py[i] = (4*node[0,i] - 1) * zi_py[i]
        Ni_px[i+3] = 4*node[0,j]*zi_px[k] + 4*node[0,k]*zi_px[j]
        Ni_py[i+3] = 4*node[0,j]*zi_py[k] + 4*node[0,k]*zi_py[j]

    for i in range(6):                  #Fyller opp B-matrisen med riktige verdier
        B[0,i*2] += Ni_px[i]
        B[1, i*2 + 1] += Ni_py[i]
        B[2, i*2] += Ni_py[i]
        B[2, i*2 + 1] += Ni_px[i]
            
    return B




def plan6(x,y,D,t,eq=None):
    
    A = 0.5*np.linalg.det(np.mat([      #Regner arealet til trekanten
        [1, x[0], y[0]],
        [1, x[1], y[1]],
        [1, x[2], y[2]]
        ]))
    cyclic_ijk = [0,1,2,0,1]            #Cyclic for å slippe utstrakt %-bruk
    K = np.zeros((12,12))

    

    zeta_nodes = np.mat([[2/3, 1/6, 1/6], #Verdiene av zheta i de forskjellige gausspunktene
                         [1/6, 2/3, 1/6],
                         [1/6, 1/6, 2/3]])
    
    '''Alternative zheta nodes
    
    zeta_nodes = np.mat([[0.5, 0.5, 0.0],
                         [0.5, 0.0, 0.5],
                         [0.0, 0.5, 0.5]])
    '''

        
    B_ret = []

    for node in zeta_nodes:                #Gjør gaussintegrasjon for å regne ut stivhetsmatrisen
        B = b6(node,x,y)
        B_ret.append(B)

        K += (B.T * D * B) * A * t /3

    if eq is None:
    	return K, B_ret
    else:
        # Integrerer N-funksjonene over volumet for å få riktig fordelte laster
        # Trenger bare å integrere N, fordi Areal(A),tykkelse(t), og vekt(3) er
        # konstante så de trenger bare å regnes inn en gang
        IntN = np.zeros(6)
        for node in zeta_nodes:
            z0 = node[0,0]
            z1 = node[0,1]
            z2 = node[0,2]
            IntN[0] += z0*(z0-1/2)*2
            IntN[1] += z1*(z1-1/2)*2
            IntN[2] += z2*(z2-1/2)*2
            IntN[3] += z0*z1*4
            IntN[4] += z1*z2*4
            IntN[5] += z2*z0*4
            
        fe = np.zeros((12,1))
        for i in range(6):
            fe[i*2, 0] += IntN[i]*eq[0]*t/3*A
            fe[i*2+1, 0] += IntN[i]*eq[1]*t/3*A
        return K, B_ret, fe
