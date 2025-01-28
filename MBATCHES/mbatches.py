import numpy as np
import math
from numpy.lib.function_base import median
import pandas as pd
from scipy import stats
import seaborn as sns
import random as rd
import os
import matplotlib
import time
#from interval import interval, inf

""" matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
}) """
import matplotlib.pyplot as plt
from matplotlib import patches
import matplotlib.ticker as ticker
from matplotlib.patches import Polygon

np.random.seed(0)
a=-1; b=1
prec=10**(-16)

def g(z,typeR):
    if(typeR=="polyTwo"):
        return z**2-1
    elif(typeR=="polyThree"):
        return 2*z**3-3*z**2+5
    elif(typeR=="polyFour"):
        return z**4-2*z**2+3
    elif(typeR=="polyFive"):
        return z**5-4*z**4+2*z**3+8*z**2-11*z-12
    
def gp(z,typeR):
    if(typeR=="polyTwo"):
        return 2*z
    elif(typeR=="polyThree"):
        return 6*(z**2-z)
    elif(typeR=="polyFour"):
        return 4*z*(z**2-1)
    elif(typeR=="polyFive"):
        return 5*z**4-16*z**3+6*z**2+16*z-11

def gpp(z,typeR):
    if(typeR=="polyTwo"):
        return 2
    elif(typeR=="polyThree"):
        return 6*(2*z-1)
    elif(typeR=="polyFour"):
        return 4*(3*z**2-1)
    elif(typeR=="polyFive"):
        return 20*z**3-48*z**2+12*z+16

#ex1 et 2 sur [-1,1]
def R(x,i,type='ex1'):
    if(type=="ex1"):
        if(i==0):
            return x[0]**2+1
        else:
            return (2*x[0]-1)**2+1
    elif(type=="ex2"):
        if(i==0):
            return 2*x[0]**4-x[0]**2+1 # par défaut +1
        else:
            return 12*x[0]**4+4*x[0]**3-9*x[0]**2+4
    elif(type=="ex3"):
        if(i==0):
            return 12*x[0]**10-4*x[0]**9+5*x[0]**8+x[0]**6-3*x[0]**5-2*x[0]**4-x[0]**3+x[0]**2-x[0]+1
        else:
            return x[0]**6-x[0]**5-x[0]**3+x[0]+1
    elif(type=="ex4"):
        if(i==0):
            return x[0]**6+x[0]**5+x[0]**3-2*x[0]**2+9
        else:
            return x[0]**6-x[0]**5-x[0]**3+x[0]+1
    elif(type=="ex5"):
        if(i==0):
            return x[0]**2+1
        elif(i==1):
            return (2*x[0]-1)**2+1
        else:
            return (2*x[0]+3)**2+1
    elif(type=="ex6"):
        if(i==0):
            return 2*x[0]**4-x[0]**2+1
        elif(i==1):
            return 12*x[0]**4+4*x[0]**3-9*x[0]**2+4
        else:
            return 4*x[0]**6+14/5*x[0]**5-7/4*x[0]**4-x[0]**3+1
    elif(type=="ex7"):
        if(i==0):
            return x[0]**6+x[0]**5+x[0]**3-2*x[0]**2+9
        elif(i==1):
            return x[0]**6-x[0]**5-x[0]**3+x[0]+1
        else:
            return x[0]**6+0.5*x[0]**5+2*x[0]**3+2
    elif(type=="ex8"):
        if(i==0):
            return x[0]**6+x[0]**5+x[0]**3-2*x[0]**2+9
        elif(i==1):
            return x[0]**6-x[0]**5-x[0]**3+x[0]+1
        elif(i==2):
            return x[0]**6+0.5*x[0]**5+2*x[0]**3+2
        else:
            return x[0]**3+x[0]**2-2*x[0]+1
    elif(type=="ex9"):
        if(i==0):
            return x[0]**6+x[0]**5+x[0]**3-2*x[0]**2+9
        elif(i==1):
            return x[0]**6-x[0]**5-x[0]**3+x[0]+1
        elif(i==2):
            return x[0]**6+0.5*x[0]**5+2*x[0]**3+2
        elif(i==3):
            return x[0]**3+x[0]**2-2*x[0]+1
        elif(i==4):
            return 12*x[0]**10-4*x[0]**9+5*x[0]**8+x[0]**6-3*x[0]**5-2*x[0]**4-x[0]**3+x[0]**2-x[0]+1
        elif(i==5):
            return (2*x[0]-1)**2+1
        elif(i==6):
            return x[0]**2+1
        else:
            return 4*x[0]**6+14/5*x[0]**5-7/4*x[0]**4-x[0]**3+1
    elif(type=="polyTwo" or type=="polyThree" or type=="polyFour" or type=="polyFive"):
        if(i==0):
            return 0.5*g(x[1],typeR)**2
        else:
            return 0.5*g(x[0]+x[1],typeR)**2

def gradR(x,i,type='ex1'):
    if(type=="ex1"):
        if(i==0):
            return 2*x[0]
        else:
            return 4*(2*x[0]-1)
    elif(type=="ex2"):
        if(i==0):
            return 8*x[0]**3-2*x[0]
        else:
            return 48*x[0]**3+12*x[0]**2-18*x[0] 
    elif(type=="ex3"):
        if(i==0):
            return 120*x[0]**9-36*x[0]**8+40*x[0]**7+6*x[0]**5-15*x[0]**4-8*x[0]**3-3*x[0]**2+2*x[0]-1
        else:
            return 6*x[0]**5-5*x[0]**4-3*x[0]**2+1
    elif(type=="ex4"):
        if(i==0):
            return 6*x[0]**5+5*x[0]**4+3*x[0]**2-4*x[0]
        else:
            return 6*x[0]**5-5*x[0]**4-3*x[0]**2+1
    elif(type=="ex5"):
        if(i==0):
            return 2*x[0]
        elif(i==1):
            return 4*(2*x[0]-1)
        else:
            return 4*(2*x[0]+3)
    elif(type=="ex6"):
        if(i==0):
            return 8*x[0]**3-2*x[0]
        elif(i==1):
            return 48*x[0]**3+12*x[0]**2-18*x[0] 
        else:
            return 24*x[0]**5+14*x[0]**4-7*x[0]**3-3*x[0]**2
    elif(type=="ex7"):
        if(i==0):
            return 6*x[0]**5+5*x[0]**4+3*x[0]**2-4*x[0]
        elif(i==1):
            return 6*x[0]**5-5*x[0]**4-3*x[0]**2+1 
        else:
            return 6*x[0]**5+5/2*x[0]**4+6*x[0]**2
    elif(type=="ex8"):
        if(i==0):
            return 6*x[0]**5+5*x[0]**4+3*x[0]**2-4*x[0]
        elif(i==1):
            return 6*x[0]**5-5*x[0]**4-3*x[0]**2+1 
        elif(i==2):
            return 6*x[0]**5+5/2*x[0]**4+6*x[0]**2
        else:
            return 3*x[0]**2+2*x[0]-2
    elif(type=="ex9"):
        if(i==0):
            return 6*x[0]**5+5*x[0]**4+3*x[0]**2-4*x[0]
        elif(i==1):
            return 6*x[0]**5-5*x[0]**4-3*x[0]**2+1 
        elif(i==2):
            return 6*x[0]**5+5/2*x[0]**4+6*x[0]**2
        elif(i==3):
            return 3*x[0]**2+2*x[0]-2
        elif(i==4):
            return 120*x[0]**9-36*x[0]**8+40*x[0]**7+6*x[0]**5-15*x[0]**4-8*x[0]**3-3*x[0]**2+2*x[0]-1
        elif(i==5):
            return 4*(2*x[0]-1)
        elif(i==6):
            return 2*x[0]
        else:
            return 24*x[0]**5+14*x[0]**4-7*x[0]**3-3*x[0]**2
    elif(type=="polyTwo" or type=="polyThree" or type=="polyFour" or type=="polyFive"):
        grad=np.ones(2)
        if(i==0):
            grad[0]=0
            grad[1]=g(x[1],type)*gp(x[1],type)
        else:
            grad[0]=g(x[0]+x[1],type)*gp(x[0]+x[1],type)
            grad[1]=g(x[0]+x[1],type)*gp(x[0]+x[1],type)
        return grad

def grad_sum(x,typeR,m):
    g=np.zeros(N)
    for i in range(m):
        g+=gradR(x,i,typeR)
    return g

def suivant(i,m):
    if(i==m-1):
        return 0
    else:
        return i+1
        
def u0(x,typeCI="uniform"):
    if(typeCI=="uniform"):
        if(x[0]>a and x[0]<b):
            return 1/(b-a)
        else:
            return 0
    elif(typeCI=="normal"):
        mu=0; sigma=1
        return 1/(sigma*np.sqrt(2*np.pi))*np.exp(-0.5*((x[0]-mu)/sigma)**2)
    
def tire_saut(a,b,i):
    if(i==0):
        if(a<10**(-16)):
            return np.inf
        else:
            return np.random.exponential(1/a)
    else:
        if(b<10**(-16)):
            return np.inf
        else:
            return np.random.exponential(1/b)
        
def change(i):
    if(i==0):
        return 1
    else:
        return 0
    
def tire_prive(i,m):
    L=[]
    for j in range(m):
        if(j!=i):
            L.append(j)
    return np.random.choice(np.array(L))

#Dans nos examples, P=m
def RAG(x0,m,lr_init,eps,maxEpoch,typeR):
    epoch=0; P=m
    x=x0
    f1=30; f2=10000; lamb=0.5
    eta=lr_init; eta0=lr_init; eta1=lr_init; eta_start=lr_init; 
    cost_prec=0; cost=0; Rv=0; Rv_prec=0
    grad_square=0; gNorm=1000; dist=0; L_local=0; 
    grad_tab=[]; R_tab=[]; Ls_effectif=np.zeros(m); diffs=np.zeros(m)
    g = np.zeros(N); grad = np.zeros(N)

    gauche=0; droite=0; milieu=0; m_best=0; nLoops=2; last_pass=False
    iterLoop=0; total_iterLoop=0

    coeff_max=2*m-1; heuris_max=2
    coeff=coeff_max; heuris=heuris_max

    x_prec=x0
    for i in range(m):
            cost_prec=R(x_prec,i,typeR); R_tab.append(cost_prec); Rv+=cost_prec
            grad=gradR(x,i,typeR); grad_square=np.linalg.norm(grad)**2
            g+=grad; grad_tab.append(grad)
            condition=(grad_square>eps**2)
            while(condition):
                x=x_prec-eta*grad
                cost=R(x,i,typeR)
                condition=(cost-cost_prec>-lamb*eta*grad_square)
                if(condition):
                    eta/=f1
            if(grad_square<eps**2 or eta<prec):
                Ls_effectif[i]=0
            else:
                Ls_effectif[i]=(2*(1-lamb))/eta
            x=x_prec; eta=lr_init
    gNorm=np.linalg.norm(g)
    if(gNorm/P>eps):
        L_local=np.sum(Ls_effectif)
        if(L_local<prec):
            eta=lr_init
        else:
            eta=(2*(1-lamb))/L_local
        x=x_prec-eta*g

    while((gNorm/P>eps or dist/P>eps) and epoch<=maxEpoch):

        Rv_prec=Rv
        for i in range(m):
            x_prec=x

            cost_prec = R(x,i,typeR)
            Rv-=R_tab[i]; Rv+=cost_prec; R_tab[i]=cost_prec

            grad = gradR(x,i,typeR); grad_square=np.linalg.norm(grad)**2
            g-=grad_tab[i]; g+=grad; grad_tab[i]=grad
            prod=np.dot(grad,g); gNorm=np.linalg.norm(g)

            eta0=eta_start; eta1=eta_start

            #linesearch on the batch i
            iterLoop=0
            condition=(grad_square>eps**2)
            while(condition):
                x=x_prec-eta0*grad
                cost=R(x,i,typeR)
                condition=(cost-cost_prec>-lamb*eta0*grad_square)
                if(condition):
                    eta0/=f1
                iterLoop+=1
            if(iterLoop>1 and eta0>prec):
                gauche = np.log10(eta0); droite = np.log10(f1*eta0)
                for k in range(nLoops):
                    milieu=(gauche+droite)/2; eta0=10**milieu
                    x=x_prec-eta0*grad
                    cost=R(x,i,typeR)
                    if(cost-cost_prec>-lamb*eta0*grad_square):
                        m_best=gauche
                        droite=milieu
                        last_pass=False
                    else:
                        gauche=milieu
                        last_pass=True
                    iterLoop+=1
                if(not last_pass):
                    eta0=10**(m_best)
            total_iterLoop+=iterLoop

            imax=np.argmax(Ls_effectif)

            #linesearch on the dot product direction
            if(prod>prec and i==imax and prod<gNorm**2):
                iterLoop=0
                condition=True
                while(condition):
                    x=x_prec-eta1*g
                    cost=R(x,i,typeR)
                    condition=(cost-cost_prec>-lamb*eta1*prod)
                    if(condition):
                        eta1/=f1
                if(iterLoop>1 and eta1>prec):
                    gauche = np.log10(eta1); droite = np.log10(f1*eta1)
                    for k in range(nLoops):
                        milieu=(gauche+droite)/2; eta1=10**milieu
                        x=x_prec-eta1*g
                        cost=R(x,i,typeR)
                        if(cost-cost_prec>-lamb*eta1*prod):
                            m_best=gauche
                            droite=milieu
                            last_pass=False
                        else:
                            gauche=milieu
                            last_pass=True
                        iterLoop+=1
                if(not last_pass):
                    eta1=10**(m_best)
                total_iterLoop+=iterLoop
                eta=max(eta0,eta1)
            else:
                eta=eta0 

            if(grad_square<eps**2 or eta<prec):
                Ls_effectif[i]=0
            else:
                Ls_effectif[i]=(2*(1-lamb))/eta

            L_local=np.sum(Ls_effectif)
            if(L_local<prec):
                eta=lr_init; eta_start=lr_init
            else:
                eta_start=f2*(2*(1-lamb))/np.max(Ls_effectif)
                if(coeff==coeff_max):
                    eta=2/(coeff*L_local)
                else:
                    eta=2/(heuris*coeff*L_local)
            
            x = x_prec-eta*g

            dist-=diffs[i]
            if(grad_square>eps**2 and eta0>prec):
                diffs[i]=((2*(1-lamb))/eta0)*eta*gNorm
            else:
                diffs[i]=0
            dist+=diffs[i]
            
            if(gNorm/P<eps and dist/P<eps):
                break
        
        if(dist<gNorm):
            heuris=(heuris+heuris_max)/2
        else:
            heuris=(1+heuris)/2
        if(Rv-Rv_prec<0):
            coeff/=heuris
        else:
            coeff=coeff_max

        epoch+=1

    return x, epoch

def RAG_L(x0,m,lr_init,eps,maxEpoch,typeR):
    epoch=0; x=x0; P=m
    f1=30; f2=10000; lamb=0.5
    eta=lr_init; eta0=lr_init; eta1=lr_init; eta_start=lr_init; 
    cost_prec=0; cost=0; Rv=0; Rv_prec=0
    grad_square=0; gNorm=1000; dist=0; L_local=0; 
    R_tab=[]; Ls_effectif=np.zeros(m)
    g = np.zeros(N); grad = np.zeros(N); grad_suiv=np.zeros(N); grad_sum=np.zeros(N)

    gauche=0; droite=0; milieu=0; m_best=0; nLoops=2; last_pass=False
    iterLoop=0; total_iterLoop=0

    coeff_max=4*m-1; heuris_max=2
    coeff=coeff_max; heuris=heuris_max

    x_prec=x0; x_inter=x0
    for i in range(m):
            cost_prec=R(x_prec,i,typeR); R_tab.append(cost_prec)
            grad=gradR(x,i,typeR); grad_square=np.linalg.norm(grad)**2
            g+=grad
            condition=(grad_square>eps**2)
            while(condition):
                x=x_prec-eta*grad
                cost=R(x,i,typeR)
                condition=(cost-cost_prec>-lamb*eta*grad_square)
                if(condition):
                    eta/=f1
            if(grad_square<eps**2 or eta<prec):
                Ls_effectif[i]=0
            else:
                Ls_effectif[i]=(2*(1-lamb))/eta
            x=x_prec; eta=lr_init
    gNorm=np.linalg.norm(g)
    if(gNorm/P>eps):
        L_local=np.average(Ls_effectif)
        if(L_local<prec):
            eta=lr_init
        else:
            eta=(2*(1-lamb))/L_local
        x=x_prec-eta*g
    
    while((gNorm/P>eps or dist/P>eps) and epoch<=maxEpoch):
        dist=0; grad_sum=np.zeros(N)
        Rv_prec=Rv

        for i in range(m):
            cost_prec = R(x,i,typeR)
            Rv-=R_tab[i]; Rv+=cost_prec; R_tab[i]=cost_prec

            if(i<m-1):
                grad_suiv=gradR(x_inter,i,typeR)
            x_prec=x
            grad = gradR(x,i,typeR); grad_sum+=grad

            if(i<m-1):
                g+=grad; g-=grad_suiv
            else:
                g=grad_sum
            
            prod=np.dot(grad,g); gNorm=np.linalg.norm(g); grad_square=np.linalg.norm(grad)**2

            eta0=eta_start; eta1=eta_start

            #linesearch on the batch i
            iterLoop=0
            condition=(grad_square>eps**2)
            while(condition):
                x=x_prec-eta0*grad
                cost=R(x,i,typeR)
                condition=(cost-cost_prec>-lamb*eta0*grad_square)
                if(condition):
                    eta0/=f1
                iterLoop+=1
            if(iterLoop>1 and eta0>prec):
                gauche = np.log10(eta0); droite = np.log10(f1*eta0)
                for k in range(nLoops):
                    milieu=(gauche+droite)/2; eta0=10**milieu
                    x=x_prec-eta0*grad
                    cost=R(x,i,typeR)
                    if(cost-cost_prec>-lamb*eta0*grad_square):
                        m_best=gauche
                        droite=milieu
                        last_pass=False
                    else:
                        gauche=milieu
                        last_pass=True
                    iterLoop+=1
                if(not last_pass):
                    eta0=10**(m_best)
            total_iterLoop+=iterLoop

            imax=np.argmax(Ls_effectif)

            #linesearch on the dot product direction
            if(prod>prec and i==imax and prod<gNorm**2):
                iterLoop=0
                condition=True
                while(condition):
                    x=x_prec-eta1*g
                    cost=R(x,i,typeR)
                    condition=(cost-cost_prec>-lamb*eta1*prod)
                    if(condition):
                        eta1/=f1
                if(iterLoop>1 and eta1>prec):
                    gauche = np.log10(eta1); droite = np.log10(f1*eta1)
                    for k in range(nLoops):
                        milieu=(gauche+droite)/2; eta1=10**milieu
                        x=x_prec-eta1*g
                        cost=R(x,i,typeR)
                        if(cost-cost_prec>-lamb*eta1*prod):
                            m_best=gauche
                            droite=milieu
                            last_pass=False
                        else:
                            gauche=milieu
                            last_pass=True
                        iterLoop+=1
                    if(not last_pass):
                        eta1=10**(m_best)
                total_iterLoop+=iterLoop
                eta=max(eta0,eta1)
            else:
                eta=eta0 

            if(grad_square<eps**2 or eta<prec):
                Ls_effectif[i]=0
            else:
                Ls_effectif[i]=(2*(1-lamb))/eta

            L_local=np.sum(Ls_effectif)
            if(L_local<prec):
                eta=lr_init; eta_start=lr_init
            else:
                eta_start=f2*(2*(1-lamb))/np.max(Ls_effectif)
                if(coeff==coeff_max):
                    eta=2/(coeff*L_local)
                else:
                    eta=2/(heuris*coeff*L_local)   

            if(grad_square>eps**2 and eta0>prec):
                dist+=((2*(1-lamb))/eta0)*eta*gNorm
            
            if(i==0):
                LMax_now=Ls_effectif[0]; imax_now=0
            else:
                if(Ls_effectif[i]>LMax_now):
                    LMax_now=Ls_effectif[i]; imax_now=i
            
            if(i==imax_now):
                x_memory=x_prec
            
            x = x_prec-eta*g
        
        x_inter=x_memory
        g=grad_sum
        gNorm=np.linalg.norm(g)

        if(dist<gNorm):
            heuris=(heuris+heuris_max)/2
        else:
            heuris=(1+heuris)/2
        if(Rv-Rv_prec<0):
            coeff/=heuris
        else:
            coeff=coeff_max
        
        epoch+=1

    return x, epoch

#On utilise l'estimation par la somme
def GD_estim(x0,m,lr_init,eps,maxEpoch,typeR):
    epoch=1; x=x0; P=m
    f1=30; f2=10000; lamb=0.5
    eta=lr_init; eta0=lr_init; eta1=lr_init; eta_start=lr_init; 
    cost_prec=0; cost=0; Rv=0
    grad_square=0; gNorm=1000; L_local=0; 
    R_tab=[]; Ls_effectif=np.zeros(m); var_tab=np.zeros(m); moment_tab=np.zeros(m)
    g = np.zeros(N); grad = np.zeros(N)
    coeff=1; heuris=1

    gauche=0; droite=0; milieu=0; m_best=0; nLoops=2; last_pass=False
    iterLoop=0

    x_prec=x0
    for i in range(m):
            cost_prec=R(x_prec,i,typeR); R_tab.append(cost_prec); Rv+=cost_prec
            grad=gradR(x,i,typeR); grad_square=np.linalg.norm(grad)**2
            moment_tab[i]=grad_square
            g+=grad
            condition=(grad_square>eps**2)
            while(condition):
                x=x_prec-eta*grad
                cost=R(x,i,typeR)
                condition=(cost-cost_prec>-lamb*eta*grad_square)
                if(condition):
                    eta/=f1
            if(grad_square<eps**2 or eta<prec):
                Ls_effectif[i]=0
            else:
                Ls_effectif[i]=(2*(1-lamb))/eta
            x=x_prec; eta=lr_init
    gNorm=np.linalg.norm(g)
    for i in range(m):
        grad=gradR(x,i,typeR)
        var_tab[i]=np.linalg.norm(grad-g/m)**2
    
    if(gNorm/P>eps):
        L_local=np.sum(Ls_effectif)
        if(L_local<prec):
            eta=lr_init
        else:
            eta=(2*(1-lamb))/L_local
        x=x_prec-eta*g

    while(gNorm/P>eps and epoch<=maxEpoch):
        gSum=np.zeros(N)
        R_epoch=Rv

        for i in range(m):
            x_prec=x
            cost_prec = R(x,i,typeR); grad=gradR(x,i,typeR); gSum+=grad
            #estimateur parfait pour tester
            #g=gradR(x,0,typeR)+gradR(x,1,typeR); gNorm=np.linalg.norm(g)
            Rv-=R_tab[i]; Rv+=cost_prec; R_tab[i]=cost_prec
            grad_square=np.linalg.norm(grad)**2; diff_square=np.linalg.norm(grad-g/m)**2
            var_tab[i]=diff_square; moment_tab[i]=grad_square
            var=np.sum(var_tab); moment = var+gNorm**2
            prod=np.dot(g,grad)

            eta0=eta_start; eta1=eta_start

            #linesearch on the batch i
            iterLoop=0
            condition=(grad_square>eps**2)
            while(condition):
                x=x_prec-eta0*grad
                cost=R(x,i,typeR)
                condition=(cost-cost_prec>-lamb*eta0*grad_square)
                if(condition):
                    eta0/=f1
                iterLoop+=1
            if(iterLoop>1 and eta0>prec):
                gauche = np.log10(eta0); droite = np.log10(f1*eta0)
                for k in range(nLoops):
                    milieu=(gauche+droite)/2; eta0=10**milieu
                    x=x_prec-eta0*grad
                    cost=R(x,i,typeR)
                    if(cost-cost_prec>-lamb*eta0*grad_square):
                        m_best=gauche
                        droite=milieu
                        last_pass=False
                    else:
                        gauche=milieu
                        last_pass=True
                if(not last_pass):
                    eta0=10**(m_best)

            imax=np.argmax(Ls_effectif)

            #linesearch on the dot product direction
            if(prod>prec and i==imax and prod<gNorm**2):
                iterLoop=0
                condition=True
                while(condition):
                    x=x_prec-eta1*g
                    cost=R(x,i,typeR)
                    condition=(cost-cost_prec>-lamb*eta1*prod)
                    if(condition):
                        eta1/=f1
                if(iterLoop>1 and eta1>prec):
                    gauche = np.log10(eta1); droite = np.log10(f1*eta1)
                    for k in range(nLoops):
                        milieu=(gauche+droite)/2; eta1=10**milieu
                        x=x_prec-eta1*g
                        cost=R(x,i,typeR)
                        if(cost-cost_prec>-lamb*eta1*prod):
                            m_best=gauche
                            droite=milieu
                            last_pass=False
                        else:
                            gauche=milieu
                            last_pass=True
                        iterLoop+=1
                    if(not last_pass):
                        eta1=10**(m_best)
                eta=max(eta0,eta1)
            else:
                eta=eta0 

            if(grad_square<eps**2 or eta<prec):
                Ls_effectif[i]=0
            else:
                Ls_effectif[i]=(2*(1-lamb))/eta

            L_local=np.sum(Ls_effectif)
            if(L_local<prec):
                eta=lr_init; eta_start=lr_init
            else:
                eta_start=f2*(2*(1-lamb))/np.max(Ls_effectif)
                """ if(prod>0):
                    if(prod/grad_square>1):
                        eta0=1/L_local
                    else:
                        eta0=prod/(L_local*grad_square)
                else:
                    eta0=0 """
                eta=(2*(1-lamb)/(L_local))*(1-var/moment)  
            
            x = x_prec-coeff*eta*grad
        
        prod=np.dot(gSum,g)
        g=gSum
        if(L_local>prec and prod>0):
            x=x-1/L_local*g
        gNorm=np.linalg.norm(g)

        print("var: ", var)
        print("R: ", Rv/P)
        print("LTab: ", Ls_effectif)
        print("gNorm", gNorm/P)
        print("prod:grad:", prod/grad_square)
        print("eta", eta)

        
        epoch+=1

    return x, epoch

def Euler_exp(x0,i0,tf, lr_init,eps,maxEpoch,typeR):
    x=x0; eta=lr_init; epoch=0; t=0
    eta_start=lr_init
    lamb=0.5; f1=2; f2=10**(4)
    m=2
    Lip=np.zeros(m); g=np.zeros(N); grads=[]
    moment_tab=np.zeros(m)

    #initialisation of Lipschitz constant
    for i in range(m):
        eta=lr_init
        x_prec=x
        cost_prec=R(x,i,typeR)
        grad=gradR(x,i,typeR); g+=grad
        grads.append(grad)
        grad_square=np.linalg.norm(grad)**2
        moment_tab[i]=grad_square

        condition=(grad_square>eps**2)
        while(condition):
            x=x_prec-eta*grad
            cost=R(x,i,typeR)
            condition=(cost-cost_prec>-lamb*eta*grad_square)
            if(condition):
                eta/=f1
        Lip[i]=(2*(1-lamb))/eta


    i=i0
    moment=np.sum(moment_tab)
    while(epoch<maxEpoch):
        x_prec=x; eta=lr_init
        cost_prec=R(x,i,typeR)
        grad=gradR(x,i,typeR)
        g-=grads[i]; g+=grad; grads[i]=grad
        #g=gradR(x,0,typeR)+gradR(x,1,typeR)
        grad_square=np.linalg.norm(grad)**2; gNorm=np.linalg.norm(g)

        moment_tab[i]=grad_square
        moment=np.sum(moment_tab)

        #linesearch on the batch i
        iterLoop=0; eta=eta_start
        condition=(grad_square>eps**2)
        while(condition):
            x=x_prec-eta*grad
            cost=R(x,i,typeR)
            condition=(cost-cost_prec>-lamb*eta*grad_square)
            if(condition):
                eta/=f1
            iterLoop+=1
        Lip[i]=(2*(1-lamb))/eta
        L_local=np.sum(Lip)
        eta_start=f2*(2*(1-lamb))/np.max(Lip)

        eta=2*(1-lamb)/L_local
        
        tau=tire_saut(moment/gNorm**2, moment/gNorm**2, i)
        #tau = tire_saut(1/gNorm**2,1/gNorm**2,i)

        x=x_prec-min(eta,tau)*grad
        i = tire_prive(i,m)
        t+=min(tau,eta)
        epoch+=1/m
        print(tau)
    
    return x, i, epoch

def Euler_exp2(x0,i0,tf, lr_init,eps,maxEpoch,typeR):
    x=x0; eta=lr_init; epoch=0; t=0
    eta_start=lr_init
    lamb=0.5; f1=2; f2=10**(4)
    m=2
    Lip=np.zeros(m)
    k=0; moy_moment=0; moy_gradient=np.zeros(m)

    #initialisation of Lipschitz constant
    for i in range(m):
        eta=lr_init
        x_prec=x
        cost_prec=R(x,i,typeR)
        grad=gradR(x,i,typeR); moy_gradient+=grad
        grad_square=np.linalg.norm(grad)**2; moy_moment+=grad_square
        k+=1

        condition=(grad_square>eps**2)
        while(condition):
            x=x_prec-eta*grad
            cost=R(x,i,typeR)
            condition=(cost-cost_prec>-lamb*eta*grad_square)
            if(condition):
                eta/=f1
        Lip[i]=(2*(1-lamb))/eta


    i=i0
    while(epoch<maxEpoch):
        x_prec=x; eta=lr_init
        cost_prec=R(x,i,typeR)
        grad=gradR(x,i,typeR); moy_gradient+=grad
        grad_square=np.linalg.norm(grad)**2; moy_moment+=grad_square
        k+=1


        #linesearch on the batch i
        iterLoop=0; eta=eta_start
        condition=(grad_square>eps**2)
        while(condition):
            x=x_prec-eta*grad
            cost=R(x,i,typeR)
            condition=(cost-cost_prec>-lamb*eta*grad_square)
            if(condition):
                eta/=f1
            iterLoop+=1
        Lip[i]=(2*(1-lamb))/eta
        L_local=np.sum(Lip)
        eta=2*(1-lamb)/L_local
        if(np.max(Lip)>prec):
            eta_start=f2*(2*(1-lamb))/np.max(Lip)
        else:
            eta_start=lr_init

        a=(moy_moment/k)/np.linalg.norm(moy_gradient/k)**2
        #a=1/np.linalg.norm(moy_gradient/k)**2
        tau = tire_saut(a,a,i)
        x=x_prec-min(eta,tau)*grad
        i=tire_prive(i,m)
        
        t+=min(tau,eta)
        epoch+=1/m
    
    return x, i, epoch

#-------------------------------------- forward et backward général -----------------------------------------------------------------------------

#dimension 1
def solution_forward(nbPoints, nbParticules, m, lr_init, eps, maxEpoch, typeR, typeCI):
    count=0
    x_maille=np.linspace(a,b,num=nbPoints)
    u=np.zeros((nbPoints,1))
    t1=time.process_time()
    nc=0
    for p in range(nbParticules):
        x_unif=np.random.uniform(a,b); x0=x_unif*np.ones(N)
        
        #x,epoch = RAG(x0,m,lr_init,eps,maxEpoch,typeR)

        #x, epoch = RAG_L(x0,m,lr_init,eps,maxEpoch,typeR)
        x, epoch = GD_estim(x0,m,lr_init,eps,maxEpoch,typeR)

        if(epoch==maxEpoch):
            nc+=1
            print(x0)
        count+=1; 
        print(count)
        if(x>a and x<b):
            #u[int(nbPoints*(x-a)/(b-a)),0]+=1
            u[int(nbPoints*(x-a)/(b-a)),0]+=u0(x0,typeCI)
    t2=time.process_time()
    print("time: ", t2-t1)
    print("nc: ", nc/nbParticules)
    u/=nbParticules

    if(typeR=="ex1"):
        plt.axvline(x=0.4,ymin=0,ymax=1,color='green')
    elif(typeR=="ex2"):
        plt.axvline(x=0.5,ymin=0,ymax=1, color='blue')
        plt.axvline(x=-0.714285714,ymin=0,ymax=1, color="green")
    elif(typeR=="ex3"):
        plt.axvline(x=0,ymin=0,ymax=1, color='blue')
        plt.axvline(x=0.229395,ymin=0,ymax=1, color='blue')
        plt.axvline(x=0.7210246,ymin=0,ymax=1, color="green")
    elif(typeR=="ex4"):
        plt.axvline(x=0.6771,ymin=0,ymax=1, color='blue')
        plt.axvline(x=-0.812540,ymin=0,ymax=1, color="green")
    elif(typeR=="ex5"):
        plt.axvline(x=-0.444,ymin=0,ymax=1,color='green')
    elif(typeR=="ex6"):
        plt.axvline(x=0.5,ymin=0,ymax=1, color='blue')
        plt.axvline(x=-0.71806,ymin=0,ymax=1, color="green")
    elif(typeR=="ex7"):
        plt.axvline(x=-0.91228,ymin=0,ymax=1, color="green")
    elif(typeR=="ex8"):
        plt.axvline(x=0.413212,ymin=0,ymax=1, color='blue')
        plt.axvline(x=-0.870466,ymin=0,ymax=1, color="green")
    locs, labels = plt.xticks()
    plt.xticks(np.arange(a, b, step=0.1))
    plt.plot(x_maille, u, label="u", color="red", ms=5, marker='.')
    plt.legend()
    plt.show()

def solution_sto(nbPoints, nbParticules, typeR, typeCI, tf, eps, lr_init):
    count=0
    x_maille=np.linspace(a,b,num=nbPoints)
    #pos0=[]; pos1=[] 
    u_0=np.zeros((nbPoints,1)); u_1=np.zeros((nbPoints,1)); 
    for p in range(nbParticules):
        x_unif=np.random.uniform(a,b); x0=x_unif*np.ones(N)
        i0=np.random.randint(0,2)
        
        x,i,epoch = Euler_exp(x0,i0,tf,lr_init,eps,maxEpoch,typeR)
        #x,i,epoch = Euler_exp2(x0,i0,tf,lr_init,eps,maxEpoch,typeR)

        count+=1; print(count)
        if(x>a and x<b):
            if(i==0):
                u_0[int(nbPoints*(x-a)/(b-a)),0]+=1
                #u_0[int(nbPoints*(x-g)/(d-g)),0]+=u0(x0,i0,typeCI)
            else:
                u_1[int(nbPoints*(x-a)/(b-a)),0]+=1
                #u_1[int(nbPoints*(x-g)/(d-g)),0]+=u0(x0,i0,typeCI)
    u_0/=nbParticules; u_1/=nbParticules

    if(typeR=="ex1"):
        plt.axvline(x=0.4,ymin=0,ymax=1,color='green')
    elif(typeR=="ex2"):
        plt.axvline(x=0.5,ymin=0,ymax=1, color='blue')
        plt.axvline(x=-0.714285714,ymin=0,ymax=1, color="green")
    locs, labels = plt.xticks()
    plt.xticks(np.arange(a, b, step=0.1))
    plt.plot(x_maille, u_0, label="u_0", color="black", ms=5, marker='.')
    plt.plot(x_maille, u_1, label="u_1", color="brown", ms=5, marker='.')
    plt.plot(x_maille, (u_0+u_1)/2, label="tot", color="red", ms=5, marker='.')
    plt.legend()
    plt.show()
    
# dimension 1
typeCI="uniform"
typeR="ex8"; N=1; m=4
nbPoints=1000
nbParticules=1000
lr_init=0.1
tf=10
eps=10**(-4); maxEpoch=1000
#solution_forward(nbPoints,nbParticules,m,lr_init,eps,maxEpoch,typeR,typeCI)
#solution_sto(nbPoints,nbParticules,typeR,typeCI,tf,eps,lr_init)

# dimension 2 polys
""" w=7; b=7; x0=np.array([w,b]).reshape(2)
N=2; m=2
lr_init=0.1
eps=10**(-4); maxEpoch=10000
typeR="polyFive"
x, epoch = GD_estim(x0,m,lr_init,eps,maxEpoch,typeR)
print(x); print(epoch) """

""" typeR="ex8"
N=1; m=4
lr_init=0.1
eps=10**(-4); maxEpoch=10000
x_unif=-30
x0=x_unif*np.ones(N)
x,epoch = GD_estim(x0,m,lr_init,eps,maxEpoch,typeR)
#x,i,epoch = Euler_exp(x0,0,tf,lr_init,eps,maxEpoch,typeR)
print(x); print(epoch); print(0.5*np.linalg.norm(gradR(x,0,typeR)+gradR(x,1,typeR))) """

#---------------------------------------------------- graphiques chapitre 5 1D ---------------------------------------------------------------------

def exs(nbPoints, nbParticules, lr_init, eps, maxEpoch, typeCI):
    x_maille=np.linspace(a,b,num=nbPoints)
    u=np.zeros((nbPoints,1))

    fig = plt.figure(figsize=(10,10))
    plt.gcf().subplots_adjust(wspace=0.2,hspace=0.57)
    axes = fig.subplots(nrows=4,ncols=2)

    for (typeR,m) in [("ex1",2), ("ex2",2), ("ex3",2), ("ex4",2), ("ex5",3), ("ex6",3), ("ex7",3), ("ex8",4)]:
        print(typeR)
        for p in range(nbParticules):
            x_unif=np.random.uniform(a,b); x0=x_unif*np.ones(N)
            
            #x,epoch = RAG(x0,m,lr_init,eps,maxEpoch,typeR)
            #x, epoch = RAG_L(x0,m,lr_init,eps,maxEpoch,typeR)
            x, epoch = GD_estim(x0,m,lr_init,eps,maxEpoch,typeR)


            if(x>a and x<b):
                #print(x) 
                #print(u[int(nbPoints*(x[0]-a)/(b-a)),0])
                u[int(nbPoints*(x[0]-a)/(b-a)),0]+=u0(x0,typeCI)
        u/=nbParticules

        if(typeR=="ex1"):
            ax=axes[0,0]; titre="a)"
            ax.axvline(x=0.4,ymin=0,ymax=1.1,color='green')
        elif(typeR=="ex2"):
            ax=axes[0,1]; titre="b)"
            ax.axvline(x=0.5,ymin=0,ymax=1.1, color='blue')
            ax.axvline(x=-0.714285714,ymin=0,ymax=1.1, color="green")
        elif(typeR=="ex3"):
            ax=axes[1,0]; titre="c)"
            ax.axvline(x=0,ymin=0,ymax=1.1, color='blue')
            ax.axvline(x=0.7210246,ymin=0,ymax=1.1, color="green")
        elif(typeR=="ex4"):
            ax=axes[1,1]; titre="d)"
            ax.axvline(x=0.6771,ymin=0,ymax=1.1, color='blue')
            ax.axvline(x=-0.812540,ymin=0,ymax=1.1, color="green")
        elif(typeR=="ex5"):
            ax=axes[2,0]; titre="e)"
            ax.axvline(x=-0.444,ymin=0,ymax=1.1,color='green')
        elif(typeR=="ex6"):
            ax=axes[2,1]; titre="f)"
            ax.axvline(x=0.5,ymin=0,ymax=1.1, color='blue')
            ax.axvline(x=-0.71806,ymin=0,ymax=1.1, color="green")
        elif(typeR=="ex7"):
            ax=axes[3,0]; titre="g)"
            ax.axvline(x=-0.91228,ymin=0,ymax=1.1, color="green")
        elif(typeR=="ex8"):
            ax=axes[3,1]; titre="h)"
            ax.axvline(x=0.413212,ymin=0,ymax=1.1, color='blue')
            ax.axvline(x=-0.870466,ymin=0,ymax=1.1, color="green")

        ax.set_title(titre)
        ax.set_xlabel(r"$\theta$")
        #locs, labels = ax.set_xticks()
        ax.set_xticks(np.arange(a, b, step=0.25))
        ax.plot(x_maille, u, label=r"$u$", color="black", ms=5, marker='.')
        ax.legend()
    
    #fig.show()
    plt.show()
    plt.savefig('RAG_exs.pgf')

typeCI="uniform"
N=1
nbPoints=1000
nbParticules=10000
lr_init=0.1
eps=10**(-4); maxEpoch=10000

exs(nbPoints,nbParticules,lr_init,eps,maxEpoch,typeCI)
