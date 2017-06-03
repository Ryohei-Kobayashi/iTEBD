""" iTEBD code to find the ground state of 
the Kitaev model on an infinite Bathe lattice."""

import numpy as np
from scipy import integrate
from scipy.linalg import expm 

# First define the parameters of the model / simulation
J0=1.0; J1=0.25; J2=0.25; g=0.0; chi=10; d=2; delta=0.005; N=10000;

G=[]
for i in range(2):
    G.append(np.zeros([2,1,1,1])); G[-1][0,0,0,0]=1
s=[]
for i in range(4):
    s.append(np.ones([1]))

# Generate the two-site time evolution operator
H_bond = [np.array( [[2*g/3,0,0,-J0], [0,0,-J0,0], [0,-J0,0,0], [-J0,0,0,-2*g/3]] ),
          np.array( [[2*g/3,0,0,J1], [0,0,-J1,0], [0,-J1,0,0], [J1,0,0,-2*g/3]] ),
          np.array( [[-J2+2*g/3,0,0,0], [0,J2,0,0], [0,0,J2,0], [0,0,0,-J2-2*g/3]] )]

U=[]
for i in range(3): 
    U.append(np.reshape(expm(-delta*H_bond[i]),(2,2,2,2)))

# Perform the imaginary time evolution alternating on A, B and C bonds                                                                                                                              
for step in range(0, N):
    for i_bond in range(3):
        ia = 3-np.mod(-i_bond-1,3); ib = 3-np.mod(-i_bond-2,3); ic = 3-np.mod(-i_bond-3,3)
        chia = G[0].shape[ia]; chib = G[0].shape[ib]; chic = G[0].shape[ic]

        # Construct theta matrix and time evolution #                                                                                                                                            
        theta = np.tensordot(G[0],np.diag(s[ia]**(-1)),axes=(ia,1))
        theta = np.tensordot(theta,G[1],axes=(3,ia))
        theta = np.tensordot(U[np.mod(ia-1,3)],theta,axes=([2,3],[0,3]))
        theta = np.reshape(np.transpose(theta,(0,2,3,1,4,5)),(d*chib*chic,d*chib*chic))
        
        # Schmidt decomposition #                                                                                                       
        X, Y, Z = np.linalg.svd(theta,full_matrices=0)
        chi2 = np.min([np.sum(Y>10.**(-10)), chi])
        
        piv = np.zeros(len(Y), np.bool)
        piv[(np.argsort(Y)[::-1])[:chi2]] = True
        
        Y = Y[piv]; invsq = np.sqrt(sum(Y**2))
        X = X[:,piv]
        Z = Z[piv,:]
        
        # Obtain the new values for G and s #
        s[ia] = Y/invsq
        X = np.tensordot(X,np.diag(s[ia]),axes=(1,1))
        Z = np.tensordot(np.diag(s[ia]),Z,axes=(0,0))
        
        if ia == 1:
            X = np.reshape(X,(d,chib,chic,chi2))
            G[0] = np.transpose(X,(0,3,1,2))
            Z = np.reshape(Z,(chi2,d,chib,chic))
            G[1] = np.transpose(Z,(1,0,2,3))

        elif ia == 2:
            X = np.reshape(X,(d,chic,chib,chi2))
            G[0] = np.transpose(X,(0,1,3,2))
            Z = np.reshape(Z,(chi2,d,chic,chib))
            G[1] = np.transpose(Z,(1,2,0,3))

        elif ia == 3:
            X = np.reshape(X,(d,chib,chic,chi2))
            G[0] = np.transpose(X,(0,1,2,3))
            Z = np.reshape(Z,(chi2,d,chib,chic))
            G[1] = np.transpose(Z,(1,2,3,0))

# Get the bond energies
E=[]
for i in range(3):
    M = np.tensordot(G[0],np.diag(s[i+1]**(-1)),axes=(i+1,1))
    GM = np.tensordot(M,G[1],axes=(3,i+1))
    C = np.tensordot(U[np.mod(i,3)],GM,axes=([2,3],[0,3]))
    GM = np.conj(GM)
    E.append(np.squeeze(np.tensordot(GM,C,axes=([0,3,1,2,4,5],[0,1,2,3,4,5]))).item()) 
print "E_iTEBD =", np.mean(E)
