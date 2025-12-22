import numpy as np
from gekko import GEKKO

m2 = GEKKO ()

m2.options.IMODE =6

n_points = 201
m2.time = np.linspace (0 , 1 , n_points )

x = m2.Var ( value =1)

m2.fix_initial (x , val =1) # x (0) =1
m2.fix_final (x , val =3) # x (1) =3

J = m2.Var ( value =0)
m2.fix_initial (J , val =0)

t = m2.Param ( value = m2.time )

integrand = 24* x * t + 2* x.dt () **2 - 4* t # x . dt to x prim

m2.Equation ( J.dt () == integrand )

J_f = m2.FV ()
J_f.STATUS = 1

m2.Connection ( J_f , J , pos2 = 'end')

m2.Obj ( J_f )

m2.options.SOLVER =3

m2.solve ( disp = False )

x_res = np.array ( x.value )
J_res = J_f.value [0]

print(f'minimalna wartosc calki = {J_res :.6f}')