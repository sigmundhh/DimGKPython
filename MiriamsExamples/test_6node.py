import numpy as np
import triangle_elements as tri
import itertools

# ----- Topologi -------------------------------------------------
# Definerer geometri for trekanten
ex = np.array([0.0, 1.0, 0.0, 0.5, 0.5, 0.0])
ey = np.array([0.0, 0.0, 1.0, 0.0, 0.5, 0.5])

# Tykkelse og analysetype
th = 0.1
ep = [1,th]

# E-module og tverrkontraksjonstall
E  = 2.1e11
nu = 0.3

# "Constitutive matrix"
D = np.mat([
        [ 1.0,  nu,  0.],
        [  nu, 1.0,  0.],
        [  0.,  0., (1.0-nu)/2.0]]) * E/(1.0-nu**2)

# Fordelte laster
eq = [1.0, 3.0]

# Beregner stivhetsmatrise, B-matrise og laster
Ke, B_ret, fe = tri.plante6(ex,ey,ep,D, eq) 

B = B_ret[0] + B_ret[1] + B_ret[2]

# Etablerer matriser for orskjellige test-deformasjoner

# Stivlegeme enhetstranslasjoner og rotasjoner
v_x = np.mat([1,0,1,0,1,0,1,0,1,0,1,0])
v_y = np.mat([0,1,0,1,0,1,0,1,0,1,0,1])
v_rot = np.mat([ey[0],-ex[0],ey[1],-ex[1],ey[2],-ex[2],ey[3],-ex[3],
ey[4],-ex[4],ey[5],-ex[5]])

# Setter deformasjon u = x, og v = y slik at du/dx = 1 og dv/dy = 1
v_epsilonx = np.mat([ex[0],0,ex[1],0,ex[2],0,ex[3],0,ex[4],0,ex[5],0])
v_epsilony = np.mat([0,ey[0],0,ey[1],0,ey[2],0,ey[3],0,ey[4],0,ey[5]])
v_gamma1 = np.mat([0,ex[0],0,ex[1],0,ex[2],0,ex[3],0,ex[4],0,ex[5]])
v_gamma2 = np.mat([ey[0],0,ey[1],0,ey[2],0,ey[3],0,ey[4],0,ey[5],0])

#Regner ut
S_x = Ke*v_x.T                      #Krefter ved enhetsdeformasjon x-retning(=0)
S_y = Ke*v_y.T                      #Krefter ved enhetsdeformasjon y-retning(=0)             
S_rot = Ke*v_rot.T                  #Krefter ved rotasjon(=0)

e0_x = B_ret[0]*v_x.T               #Tøyninger i node 4 ved enhetsdeformasjon x-retning(=0)
e1_x = B_ret[1]*v_x.T               #Tøyninger i node 5 ved enhetsdeformasjon x-retning(=0)
e2_x = B_ret[2]*v_x.T               #Tøyninger i node 6 ved enhetsdeformasjon x-retning(=0)

e0_y = B_ret[0]*v_y.T               #Tøyninger i node 4 ved enhetsdeformasjon y-retning(=0)
e1_y = B_ret[1]*v_y.T               #Tøyninger i node 5 ved enhetsdeformasjon y-retning(=0)
e2_y = B_ret[2]*v_y.T               #Tøyninger i node 6 ved enhetsdeformasjon y-retning(=0)

e0_r = B_ret[0]*v_rot.T             #Tøyninger i node 4 ved rotasjon(=0)  
e1_r = B_ret[1]*v_rot.T             #Tøyninger i node 5 ved rotasjon(=0)
e2_r = B_ret[2]*v_rot.T             #Tøyninger i node 6 ved rotasjon(=0)


#Sjekker for deformasjon u = x, slik at du/dx blir 1
epsilon_x0 = B_ret[0]*v_epsilonx.T  #Konstant tøyning x-retning i node 4 (=[1,0,0])
epsilon_x1 = B_ret[1]*v_epsilonx.T  #Konstant tøyning x-retning i node 5 (=[1,0,0])
epsilon_x2 = B_ret[2]*v_epsilonx.T  #Konstant tøyning x-retning i node 6 (=[1,0,0])

#Sjekker for deformasjon v = y, slik at dv/dy blir 1
epsilon_y0 = B_ret[0]*v_epsilony.T  #Konstant tøyning y-retning i node 4 (=[0,1,0])
epsilon_y1 = B_ret[2]*v_epsilony.T  #Konstant tøyning y-retning i node 6 (=[0,1,0])  
epsilon_y2 = B_ret[1]*v_epsilony.T  #Konstant tøyning y-retning i node 5 (=[0,1,0])

#Sjekker for deformasjon u = y, slik at du/dy blir 1
gamma_x0 = B_ret[0]*v_gamma2.T      #Konstant skjærvinkel x-retning i node 4 (=[0,0,1])
gamma_x1 = B_ret[1]*v_gamma2.T      #Konstant skjærvinkel x-retning i node 5 (=[0,0,1])
gamma_x2 = B_ret[2]*v_gamma2.T      #Konstant skjærvinkel x-retning i node 6 (=[0,0,1])

#Sjekker for deformasjon v = x, slik at dv/dx blir 1
gamma_y0 = B_ret[0]*v_gamma1.T      #Konstant skjærvinkel x-retning i node 4 (=[0,0,1])
gamma_y1 = B_ret[1]*v_gamma1.T      #Konstant skjærvinkel x-retning i node 5 (=[0,0,1])
gamma_y2 = B_ret[2]*v_gamma1.T      #Konstant skjærvinkel x-retning i node 6 (=[0,0,1])


# Stivhetsmatrise
def print_stivhetsmatrise():
        print('Stivhetsmatrise:\n', Ke)

# Laster
def print_fordeltelaster():
        print('Fordelte laster:\n', fe)

#  Sjekk stivlegemebevegelser på tøyning-forskyvingsmatrisen
def test_stf(print_results):
        if print_results:
                print('Stivlegemebevegelse x node 4:\n', e0_x)
                print('Stivlegemebevegelse x node 5:\n', e1_x)
                print('Stivlegemebevegelse x node 6:\n', e2_x)
                print('Stivlegemebevegelse y node 4:\n', e0_y)
                print('Stivlegemebevegelse y node 5:\n', e1_y)
                print('Stivlegemebevegelse y node 6:\n', e2_y)
                print('Stivlegemebevegelse rotasjon node 4:\n', e0_r)
                print('Stivlegemebevegelse rotasjon node 5:\n', e1_r)
                print('Stivlegemebevegelse rotasjon node 6:\n', e2_r)
        for s in itertools.chain(e0_x, e1_x, e2_x, e0_y, e1_y, e2_y, e0_r, e1_r, e2_r):
                if not isZero(s):
                        return False
        return True

#  Sjekk konstant tøyning på tøynings-forskyvingsmatrisen
def test_kttf(print_results):
        if print_results:
                print('Konstant tøyning i x-retning node 4:\n', epsilon_x0)
                print('Konstant tøyning i x-retning node 5:\n', epsilon_x1)
                print('Konstant tøyning i x-retning node 6:\n', epsilon_x2)
                print('Konstant tøyning i y-retning node 4:\n', epsilon_y0)
                print('Konstant tøyning i y-retning node 5:\n', epsilon_y1)
                print('Konstant tøyning i y-retning node 6:\n', epsilon_y2)
                print('Konstant skjærvinkel fra u i node 4:\n', gamma_x0)
                print('Konstant skjærvinkel fra u i node 5:\n', gamma_x1)
                print('Konstant skjærvinkel fra u i node 6:\n', gamma_x2)
                print('Konstant skjærvinkel fra v i node 4:\n', gamma_y0)
                print('Konstant skjærvinkel fra v i node 5:\n', gamma_y1)
                print('Konstant skjærvinkel fra v i node 6:\n', gamma_y2)
        
        Ex = [epsilon_x0, epsilon_x1, epsilon_x2]
        Ey = [epsilon_y0, epsilon_y1, epsilon_y2]
        Gx = [gamma_x0, gamma_x1, gamma_x2]
        Gy = [gamma_y0, gamma_y1, gamma_y2]

        for E in Ex:
                if np.array_equal(E, [1, 0, 0]):
                        return False
        for E in Ey:
                if np.array_equal(E, [0, 1, 0]):
                        return False
        for G in itertools.chain(Gx, Gy):
                if np.array_equal(G, [0, 0, 1]):
                        return False
        return True

#  Sjekk stivlegemebevegelser på stivhetsmatrisen
def test_ss(print_results):
        if print_results:
                print('Total stivlegemebevegelse x-translasjon:\n', S_x)
                print('Total stivlegemebevegelse y-translasjon:\n', S_y)
                print('Total stivlegemebeveglse rotasjon:\n', S_rot)
        for s in itertools.chain(S_x, S_y, S_rot):
                if not isZero(s):
                        return False
        return True
        

# Funksjon for å sjekke om et tall er tilnærmet null
def isZero(num):
        if np.abs(num)< 1e-5:
                return True
        return False

def hovedtest(print_results):
        boolean = True
        if not test_stf(print_results):
                print("Ikke stivlegemebevegelser på tøyning-forskyvingsmatrisen")
                boolean = False
        if not test_kttf(print_results):
                print("Ikke konstant tøyning på tøynings-forskyvingsmatrisen")
                boolean = False
        if not test_ss(print_results):
                print("Ikke stivlegemebevegelser på stivhetsmatrisen")
                boolean = False
        if boolean:
                print("Alle testene er bestått!")
        return boolean                


# Kjør print(hovedtest(print_results=True)) for å printe ut resultatene, ((print_results=False)) for å bare se om testene er riktig
print(hovedtest(print_results=False))
'''
Teste punktene inviduelt:

For å teste stivlegemebevegelser på tøyning-forskyvingsmatrisen
- test_stf(print_results=True)

For å teste konstant tøyning på tøynings-forskyvingsmatrisen
- test_kttf(print_results=True)

For å testestivlegemebevegelser på stivhetsmatrisen
- test_ss(print_results=True)
'''