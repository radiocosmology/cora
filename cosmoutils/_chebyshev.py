#--------------------------------------------------------------------
#
# startup 
#
#--------------------------------------------------------------------
import math
import numpy as np

#--------------------------------------------------------------------
# List of weights and abscissae for each level of refinement 'p'
#
# wi[p]: will give an array with the weights for all points 
#        corresponding to refinement level 'p'
# xi[p]: will contain the new abscissae to be added
#--------------------------------------------------------------------
Chebyshev_xi_List = []
Chebyshev_wi_List = []
Chebyshev_pmax=18


#--------------------------------------------------------------------
#
# coordinate transformation of integrand
#
#--------------------------------------------------------------------
def Chebyshev_f1(f, mu, vars) :

    dx_dmu=vars[2]/pow(vars[0]-mu, 2)
    x=vars[3]*(vars[1]+mu)/(vars[0]-mu)
    return dx_dmu*f(x) 

#--------------------------------------------------------------------
def Chebyshev_f2(f, mu, vars) :

    dx_dmu=vars[0]
    x=vars[0]*mu+vars[1]
    return dx_dmu*f(x) 


#--------------------------------------------------------------------
#
# performs the sums for weights
#
#--------------------------------------------------------------------
def sum_Chebyshev_weights(nmax, theta_k) :

    mvector=np.arange(1, (nmax+1)/2+1)
    theta_k_m = theta_k * (2.0*mvector-1.0) 
    
    fac_m = 1.0/( 2.0*mvector-1.0 )
    st    = np.sin(theta_k_m)
    
    r=np.add.reduce(st * fac_m)

    return r*4.0*np.sin(theta_k)/(nmax+1)


#--------------------------------------------------------------------
#
# create weights and abscissae
#
#--------------------------------------------------------------------
def create_Chebyshev_weights(p) :
   
    if len(Chebyshev_xi_List)>=p-1 : return
   
    #----------------------------------------------------------------
    print " creating Chebyshev weights and abscissae for refinement level p= ", p      

    #----------------------------------------------------------------
    # actual computation
    #----------------------------------------------------------------
    for pl in range(len(Chebyshev_xi_List)+2, p+1) :

        #------------------------------------------------------------
        # define how many abscissae there are for given p
        #------------------------------------------------------------
        nmax=2**pl-1
        
        #------------------------------------------------------------
        # compute new abscissae
        #------------------------------------------------------------
        thetas = np.pi*np.arange(1, nmax/2+1)/(nmax+1)
        xi = np.cos(thetas[::2])
        Chebyshev_xi_List.append(xi)

        #------------------------------------------------------------
        # compute all weights
        #------------------------------------------------------------
        wi = np.zeros(thetas.size+1)

        for k in range(0, thetas.size) : 
            wi[k]=sum_Chebyshev_weights(nmax, thetas[k])
            
        wi[thetas.size]=sum_Chebyshev_weights(nmax, np.pi*(nmax/2+1)/(nmax+1))
        
        Chebyshev_wi_List.append(wi)
    
    #----------------------------------------------------------------
    # show results
    #----------------------------------------------------------------
    #print " xi-List: ", Chebyshev_xi_List
    #print " wi-List: ", Chebyshev_wi_List
            
    return 0
        
def pregen_Chebyshev_weights(p):
    import os
    from os.path import join, dirname, exists
    import nputil

    print "Generating Chebyshev weights (p=%i)" % p

    create_Chebyshev_weights(p)

    chebdir = join(dirname(__file__), ".chebyshev")
    if not exists(chebdir):
        os.mkdir(chebdir)

    nputil.save_ndarray_list(join(chebdir, "xi_list.npz"), Chebyshev_xi_List)
    nputil.save_ndarray_list(join(chebdir, "wi_list.npz"), Chebyshev_wi_List)


def load_Chebyshev_weights():
    global Chebyshev_xi_List, Chebyshev_wi_List
    
    import nputil
    import os
    from os.path import join, dirname, exists

    chebdir = join(dirname(__file__), ".chebyshev")
    if exists(chebdir):
        xl = nputil.load_ndarray_list(join(chebdir, "xi_list.npz"))
        wl = nputil.load_ndarray_list(join(chebdir, "wi_list.npz"))

        if len(xl) != len(wl):
            raise Exception("Chebyshev cache is strange.")

        Chebyshev_xi_List = xl
        Chebyshev_wi_List = wl
    else:
        pregen_Chebyshev_weights(18)

#--------------------------------------------------------------------
#
# compute integral
#
#--------------------------------------------------------------------
def do_integration(f, fC, p, fvals, vars) :

    #----------------------------------------------------------------
    # set number of evaluations
    #----------------------------------------------------------------
    neval=2**p-1
    ploc=p-2
    create_Chebyshev_weights(p)
        
    #----------------------------------------------------------------
    # only add new points
    #----------------------------------------------------------------
    index_f=0
    nxi = Chebyshev_xi_List[ploc].size
    loc_fvals=np.zeros(2*nxi)

    #----------------------------------------------------------------
    # copy first point
    #----------------------------------------------------------------
    mu =Chebyshev_xi_List[ploc][0]
    loc_fvals[0]=fC(f, mu, vars)+fC(f,-mu, vars)

    #----------------------------------------------------------------
    # interal points
    #----------------------------------------------------------------
    for k in range(1, nxi) :
        
        #------------------------------------------------------------
        # copy old point
        #------------------------------------------------------------
        loc_fvals[2*k-1]=fvals[index_f]
        index_f = index_f + 1

        #------------------------------------------------------------
        # calc new point
        #------------------------------------------------------------
        mu=Chebyshev_xi_List[ploc][k]
        loc_fvals[2*k]=fC(f, mu, vars)+fC(f,-mu, vars)
    
    #----------------------------------------------------------------
    # finish last point (computed f(mu=0) outside)
    #----------------------------------------------------------------
    loc_fvals[2*nxi-1]=fvals[index_f]
    
    #----------------------------------------------------------------
    # compute integral
    #----------------------------------------------------------------
    r=np.add.reduce(loc_fvals * Chebyshev_wi_List[ploc])
    
    return r, neval, loc_fvals
   
   
#--------------------------------------------------------------------
#
# driver for integrator
#
#--------------------------------------------------------------------
def chebyshev(f, a, b, epsrel = 1e-6, epsabs = 1e-16) :

    #----------------------------------------------------------------
    r=0.0 
    neval=0
    if a>=b : return r, neval, 2
    
    #----------------------------------------------------------------
    # parameters for transformation formula
    #----------------------------------------------------------------
    vars = np.zeros(4)
    
    #----------------------------------------------------------------
    vars[0]=(b-a)*0.5
    vars[1]=(b+a)*0.5

    #----------------------------------------------------------------
    #x0=(a+b)/2.0*(1.0-0.1)
    #if x0<=a or x0>=(a+b)/2.0*(1.0-1.0e-2) : x0=(3.0*a+b)/4.0
    #
    #vars[0]=(b-a)/(a+b-2.0*x0)
    #vars[1]=x0*(b-a)/((a+b)*x0-2.0*a*b)
    #vars[3]=((a+b)*x0-2.0*a*b)/(a+b-2.0*x0)
    #vars[2]=vars[3]*(vars[0]+vars[1])
    
    #----------------------------------------------------------------
    fvals=np.zeros(1)
    fvals[0]=Chebyshev_f2(f, 0.0, vars)

    #----------------------------------------------------------------
    # Integral for 1-point formula
    #----------------------------------------------------------------
    r=2.0*fvals[0]
    neval=1    
    r1=0.0
    
    #----------------------------------------------------------------
    # higher order quadrature rules
    #----------------------------------------------------------------
    for it in range(2, Chebyshev_pmax+1) :

        r1, neval, fvals = do_integration(f, Chebyshev_f2, it, fvals, vars)
        Dr=r-r1
        r=r1
        
        #------------------------------------------------------------
        # check error
        #------------------------------------------------------------
        if np.abs(Dr)<=max(epsabs, np.abs(r)*epsrel) :
        
            return r, neval, 0
            
    #----------------------------------------------------------------
    # reaching below here means the intgration failed
    #----------------------------------------------------------------
    print " Chebyshev integration failed for pmax=", Chebyshev_pmax, "."
    print " Returning. "        
    
    return r, neval, 1
    

#--------------------------------------------------------------------
#
# compute integral for vectorizable function f
#
#--------------------------------------------------------------------
def do_integration_vec(f, p, fvals, vars, args=()) :

    #----------------------------------------------------------------
    # set number of evaluations
    #----------------------------------------------------------------
    neval=2**p-1
    ploc=p-2
    create_Chebyshev_weights(p)
        
    #----------------------------------------------------------------
    # compute x(mu)
    #----------------------------------------------------------------
    mu  = Chebyshev_xi_List[ploc]
    xap =  vars[0]*mu + vars[1]
    xam = -vars[0]*mu + vars[1]

    #----------------------------------------------------------------
    # compute all f(x)
    #----------------------------------------------------------------
    fvalnew = f(xap, *args) + f(xam, *args)    

    #----------------------------------------------------------------
    # copy points
    #----------------------------------------------------------------
    nxi = Chebyshev_xi_List[ploc].size
    loc_fvals=np.zeros(2*nxi)
    
    loc_fvals[0::2]=fvalnew[::]
    loc_fvals[1::2]=fvals  [::]
        
    #----------------------------------------------------------------
    # compute integral
    #----------------------------------------------------------------
    r=np.add.reduce(loc_fvals * Chebyshev_wi_List[ploc])
    
    return r, neval, loc_fvals
   
   
#--------------------------------------------------------------------
#
# Epsilon algorithm (does not work so far)
#
#--------------------------------------------------------------------
def update_epsilon_table(it, epstable, fac) :

    #----------------------------------------------------------------
    if it>0 :

        Del=epstable[0,it]-epstable[0,it-1]+1.0e-300      
        epstable[1,it-1]=1.0/Del

        for m in range(1, it) :

            j=it-1-m
            Del=epstable[m,j+1]-epstable[m,j]        
            epstable[m+1,j]=epstable[m-1,j+1]+1.0/Del
    
    #----------------------------------------------------------------
    for j in range(0, it+2) :
    
        print " j= ", j, " || ", it, " "
        print fac*epstable[0:it+2-j:1,j]
        
        print "\n"

    #----------------------------------------------------------------

    return 

#--------------------------------------------------------------------
#
# driver for integrator for vectorizable function f
#
#--------------------------------------------------------------------
def chebyshev_vec(f, a, b, epsrel = 1e-6, epsabs = 1e-16, args =()) :

    #----------------------------------------------------------------
    r=0.0 
    neval=0
    if a>=b : return r, neval, 2
    
    #----------------------------------------------------------------
    # parameters for transformation formula
    #----------------------------------------------------------------
    vars = np.zeros(2)
    vars[0]=(b-a)*0.5
    vars[1]=(b+a)*0.5

    #----------------------------------------------------------------
    # Integral for 1-point formula
    #----------------------------------------------------------------
    fvals=np.zeros(1)
    fvals[0]=f(vars[1], *args)
    
    r=2.0*fvals[0]
    neval=1    
    r1=0.0
    
    #epstable=np.zeros((10, 10))
    #epstable[0,0]=r

    #----------------------------------------------------------------
    # higher order quadrature rules
    #----------------------------------------------------------------
    for it in range(2, Chebyshev_pmax+1) :
        r1, neval, fvals = do_integration_vec(f, it, fvals, vars, args=args)
        Dr=r-r1
        r=r1
        
        #epstable[0,it-1]=r1
        #update_epsilon_table(it-1, epstable, vars[0])

        #------------------------------------------------------------
        # check convergence
        #------------------------------------------------------------
        if np.abs(Dr)<=max(epsabs, np.abs(r)*epsrel) :
        
            return vars[0]*r, neval, 0

        


            
    #----------------------------------------------------------------
    # reaching below here means the intgration failed
    #----------------------------------------------------------------
    print " Chebyshev integration failed for pmax=", Chebyshev_pmax, "."
    print " Returning. "        
    xv = vars[0]*Chebyshev_xi_List[it-2] + vars[1], -vars[0]
    
    return vars[0]*r, neval, 1




load_Chebyshev_weights()
