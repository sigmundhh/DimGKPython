import numpy as np

def nine_node_element(ex,ey,ep,D,eq=None):
    colD = np.shape(D)[0]
    if colD>3:
        if ep[0] == 1:
            Cm=np.linalg.inv(D)
            Dm=np.linalg.inv(Cm[ np.ix_([0,1,3],[0,1,3]) ])
        elif ep[0]==2:
            Dm = D[np.ix_([0,1,3],[0,1,3])]
        else:
            cfinfo("Error! Check first argument, ptype=1 or 2 allowed")
    else:
        Dm=D
    return plani9e(ex,ey,Dm,ep[0],eq)


def plani9e(x,y,D,t,eq=None):
	"""
    Denne funksjonen kalkulerer stivhetsmatrisen til et 9-noders firkant element.

    Parametre:
    x  = [x1 ...   x9]
    y  = [y1 ...   y9]
    t  = tykkelse
    D  = D-matrisen
    eq = [bx; by]       Ytre laster

    Returnerer:
    Ke 					element stivhetsmatrisen (9x9)
    """

	if eq == None:
	    q = 0
	else:
	    q = eq

	g1 = 0.774596669241483
	g2 = 0.
	w1 = 0.555555555555555
	w2 = 0.888888888888888
	gp = np.mat([
	    [-g1,-g1],
	    [-g2,-g1],
	    [ g1,-g1],
	    [-g1, g2],
	    [ g2, g2],
	    [ g1, g2],
	    [-g1, g1],
	    [ g2, g1],
	    [ g1, g1]
	])
	w = np.mat([
	    [ w1, w1],
	    [ w2, w1],
	    [ w1, w1],
	    [ w1, w2],
	    [ w2, w2],
	    [ w1, w2],
	    [ w1, w1],
	    [ w2, w1],
	    [ w1, w1]
	])

	wp = np.multiply(w[:,0],w[:,1])

	xsi = gp[:,0]
	eta = gp[:,1]

	# Antall Gauss-punkt: ir = 3
	ir = 3
	ngp = ir*ir
	r2 = ngp*2

    # 9x9 matrise med nullinjer av gausspunktene
	N = np.multiply(np.multiply(np.multiply(xsi,eta),(1+xsi)),(1+eta))/4.
	N = np.append(N,np.multiply(np.multiply(np.multiply(xsi,eta),-(1-xsi)),(1+eta))/4.,axis=1)
	N = np.append(N,np.multiply(np.multiply(np.multiply(xsi,eta),(1-xsi)),(1-eta))/4.,axis=1)
	N = np.append(N,np.multiply(np.multiply(np.multiply(xsi,eta),-(1+xsi)),(1-eta))/4.,axis=1)
	N = np.append(N,np.multiply(np.multiply(np.multiply(eta,(1+eta)),(1+xsi)),(1-xsi))/2.,axis=1)
	N = np.append(N,np.multiply(np.multiply(np.multiply(xsi,-(1-xsi)),(1+eta)),(1-eta))/2.,axis=1)
	N = np.append(N,np.multiply(np.multiply(np.multiply(eta,-(1-eta)),(1+xsi)),(1-xsi))/2.,axis=1)
	N = np.append(N,np.multiply(np.multiply(np.multiply(xsi,(1+xsi)),(1+eta)),(1-eta))/2.,axis=1)
	N = np.append(N,np.multiply((1-np.multiply(xsi,xsi)),(1-np.multiply(eta,eta))),axis=1)

	dNr = np.mat(np.zeros((r2,9)))
	# Derivert med hensyn på xsi
	dNr[0:r2:2,0] = (np.multiply(np.multiply(eta,(1+eta)),(1+xsi+xsi)))/4.
	dNr[0:r2:2,1] = (np.multiply(np.multiply(eta,(1+eta)),(xsi+xsi-1)))/4.
	dNr[0:r2:2,2] = (np.multiply(np.multiply(eta,(1-eta)),(1-xsi-xsi)))/4.
	dNr[0:r2:2,3] = (np.multiply(np.multiply(eta,(1-eta)),(-1-xsi-xsi)))/4.
	dNr[0:r2:2,4] = (np.multiply(np.multiply(-xsi,eta),(1+eta)))
	dNr[0:r2:2,5] = (np.multiply((1-np.multiply(eta,eta)),(xsi+xsi-1)))/2.
	dNr[0:r2:2,6] = (np.multiply(np.multiply(xsi,eta),(1-eta)))
	dNr[0:r2:2,7] = (np.multiply(1-np.multiply(-eta,eta),(1+xsi+xsi)))/2.
	dNr[0:r2:2,8] = (np.multiply(1-np.multiply(-eta,eta),(-xsi-xsi)))

	# Derivert med hensyn på eta
	dNr[1:r2+1:2,0] = (np.multiply(np.multiply(xsi,(1+xsi)),(1+eta+eta)))/4.
	dNr[1:r2+1:2,1] = (np.multiply(np.multiply(xsi,(1-xsi)),(-1-eta-eta)))/4.
	dNr[1:r2+1:2,2] = (np.multiply(np.multiply(xsi,(1-xsi)),(1-eta-eta)))/4.
	dNr[1:r2+1:2,3] = (np.multiply(np.multiply(xsi,(1+xsi)),(eta+eta-1)))/4.
	dNr[1:r2+1:2,4] = (np.multiply(1-np.multiply(-xsi,xsi),(1+eta+eta)))/2.
	dNr[1:r2+1:2,5] = (np.multiply(np.multiply(xsi,eta),(1-xsi)))
	dNr[1:r2+1:2,6] = (np.multiply((1-np.multiply(xsi,xsi)),(eta+eta-1)))/2.
	dNr[1:r2+1:2,7] = (np.multiply(np.multiply(-eta,xsi),(1+xsi)))
	dNr[1:r2+1:2,8] = (np.multiply(1-np.multiply(-xsi,xsi),(-eta-eta)))
	print('dNr: ',dNr.shape)
	print('N:',N.shape)
	Ke1 = np.mat(np.zeros((9,9)))
	fe1 = np.mat(np.zeros((9,1)))
	JT = dNr*np.mat([x,y]).T

	for i in range(ngp):
	    indx = np.array([2*(i+1)-1,2*(i+1)])
	    detJ = np.linalg.det(JT[indx-1,:])
	    JTinv = np.linalg.inv(JT[indx-1,:])
	    print('Jaco-inv:',JTinv.shape)
	    B = JTinv*dNr[indx-1,:]
	    print('Rar:',dNr[indx-1,:].shape)
	    print('B:',B.shape)
	    Ke1 += B.T*D*B*detJ*np.asscalar(wp[i])
	    fe1 = fe1+N[i,:].T*detJ*wp[i]

	if eq != None:
	    return Ke1*t,fe1*t*q
	else:
	    return Ke1*t