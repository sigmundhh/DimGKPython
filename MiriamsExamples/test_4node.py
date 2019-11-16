import numpy as np
import four_node_element as our_element
import itertools

# ----- Topology -------------------------------------------------
ex = np.array([0.,1.,1.,0.])
ey = np.array([0.,0.,1.,1.])

thickness = 0.1

E  = 2.1e11
nu = 0.3

D = np.mat([
        [ 1.0,  nu,  0.],
        [  nu, 1.0,  0.],
        [  0.,  0., (1.0-nu)/2.0]]) * E/(1.0-nu**2)

eq = [1.0, 3.0]

# Printer for vaar implementasjon
Ke, B_ret, fe = our_element.plani4e(ex,ey,thickness,D,eq)


# Stivlegeme enhetstranslasjoner og rotasjoner
v_x = np.mat([1,0,1,0,1,0,1,0])
v_y = np.mat([0,1,0,1,0,1,0,1])
v_rot = np.mat([ey[0],-ex[0],ey[1],-ex[1],ey[2],-ex[2],ey[3],-ex[3]])

# Setter deformasjon u = x, og v = y slik at du/dx = 1 og dv/dy = 1
v_epsilonx = np.mat([ex[0],0,ex[1],0,ex[2],0,ex[3],0])
v_epsilony = np.mat([0,ey[0],0,ey[1],0,ey[2],0,ey[3]])
v_gamma1 = np.mat([0,ex[0],0,ex[1],0,ex[2],0,ex[3]])
v_gamma2 = np.mat([ey[0],0,ey[1],0,ey[2],0,ey[3],0])

#Regner ut
S_x = Ke*v_x.T                      #Krefter ved enhetsdeformasjon x-retning(=0)
S_y = Ke*v_y.T                      #Krefter ved enhetsdeformasjon y-retning(=0)             
S_rot = Ke*v_rot.T                  #Krefter ved rotasjon(=0)

e_x = B_ret*v_x.T                   #Tøyninger ved enhetsdeformasjon x-retning(=0)
e_y = B_ret*v_y.T                   #Tøyninger ved enhetsdeformasjon y-retning(=0)
e_r = B_ret*v_rot.T                 #Tøyninger ved rotasjon(=0)  

#Sjekker for deformasjon u = x, slik at du/dx blir 1
epsilon_x = B_ret*v_epsilonx.T      #Konstant tøyning x-retning (=[1,0,0])

#Sjekker for deformasjon v = y, slik at dv/dy blir 1
epsilon_y = B_ret*v_epsilony.T      #Konstant tøyning y-retning (=[0,1,0])

#Sjekker for deformasjon u = y, slik at du/dy blir 1
gamma_x = B_ret*v_gamma2.T          #Konstant skjærvinkel x-retning (=[0,0,1])

#Sjekker for deformasjon v = x, slik at dv/dx blir 1
gamma_y = B_ret*v_gamma1.T          #Konstant skjærvinkel x-retning (=[0,0,1])

# Stivhetsmatrise
def print_stivhetsmatrise():
        print('Stivhetsmatrise:\n', Ke)

# Laster
def print_fordeltelaster():
        print('Fordelte laster:\n', fe)

#  Sjekk stivlegemebevegelser på tøyning-forskyvingsmatrisen
def test_stf(print_results):
        if print_results:
                print('Stivlegemebevegelse x:\n', e_x)
                print('Stivlegemebevegelse y:\n', e_y)
                print('Stivlegemebevegelse rotasjon:\n', e_r)
        for s in itertools.chain(e_x, e_y, e_r):
                if not isZero(s):
                        return False
        return True

#  Sjekk konstant tøyning på tøynings-forskyvingsmatrisen
def test_kttf(print_results):
        if print_results:
                print('Konstant tøyning i x-retning:\n', epsilon_x)
                print('Konstant tøyning i y-retning:\n', epsilon_y)
                print('Konstant skjærvinkel fra u:\n', gamma_x)
                print('Konstant skjærvinkel fra v:\n', gamma_y)
        
        if np.array_equal(epsilon_x, [1, 0, 0]):
                 return 
                 
        if np.array_equal(epsilon_y, [0, 1, 0]):
                return False
                
        for G in itertools.chain(gamma_x, gamma_y):
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