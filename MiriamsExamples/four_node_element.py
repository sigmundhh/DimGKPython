import numpy as np

def plani4e(x,y,t,D,eq=None):
    """
    Denne funksjonen kalkulerer stivhetsmatrisen til et 4-noders firkant element i plan tøyning eller plan spenning.

    Parametre:
    x  = [x1 ...   x4]
    y  = [y1 ...   y4]
    t  = tykkelse
    D  = D-matrisen
    eq = [bx; by]       bx: last i x-retning
                        by: last i y-retning
                        Alle arrays med 2 elementer er aksepterte

    Returnerer:
    Ke                  element stivhetsmatrisen (4x4)
    B                   B-matrisen
    fe                  fordelte laster
    """

    # Regner ut arealet
    width = abs(x[0]-x[1])
    height = abs(y[0]-y[2])
    A = width*height

    # For lineære deformasjoner bruker vi n=1 gauss punkt, altså er a = 0 og H = 2
    # Alle formfunksjonene N vil da bli 1/4, for eksempel N1 = (1-eta)(1-xsi)/4, der eta = xsi = a = 0
    H = 2.
    N = 0.25
    dNr = np.array([             # Uttrykk for den deriverte av translasjonmatrisen, mhp xsi rad en og mhp eta rad 2
        [-N, N, N, -N],
        [-N, -N, N, N]
    ])
    
    # Jacobi-matrisen: J = G*H
    J = dNr@np.array([x,y]).T 
    detJ = np.linalg.det(J)         # Jacobi-determinanten
    Jinv = np.linalg.inv(J)         # Invers av Jacobi-matrisen
    dNx=Jinv@dNr                    # Uttrykk for den deriverte av translasjonmatrisen, mhp x rad en og mhp y rad 2

    B=np.array(np.zeros((3,8)))
    N2=np.array(np.zeros((2,8)))
    
    # Fyller inn B-matrisen  
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
    print(B.T)
    print(B)
    #print(D)
    Ke = (B.T @ D @ B * detJ) * H*H * t
    
    # Regner ut fordelte laster
    if eq is not None:
        fx = eq[0]*A*t/4
        fy = eq[1]*A*t/4
        fe = np.zeros((8,1))
        for i in range(4):
            fe[i*2,0]=fx
            fe[i*2+1,0]=fy
        return Ke, B, fe
    return Ke, B