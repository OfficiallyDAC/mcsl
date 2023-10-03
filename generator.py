import numpy as np

from copy import deepcopy
from numpy import eye
from numpy.linalg import inv
from scipy.sparse.linalg import eigs
from scipy.stats import gennorm
from pywt import mra

def process_singlescale(N, T, beta):
    """
    This function generates T samples from a SVAR(1) process. This is instrumental for test.ipynb.

    INPUT
    =====
    N: int, number of dimensions;
    T: int, number of samples;
    beta: float, parameter of the generalized normal distribution (2. for Gaussian, 1. for Laplace, etc).

    OUTPUT
    ======
    Y: np.ndarray, (N,T) array of synthetic data;
    C: np.ndarray, (L,N,N) matrices of causal coefficients. Here we set L=1.
    """
    assert isinstance(T, int) and isinstance(N, int) and T>2**3 and N>0, 'N and T must be strictly positive integers.'
    assert isinstance(beta, (int, float)) and beta>0, 'Beta must be strictly positive scalar.'

    L=1
    added = L*10
    
    #define the matrices of causal coefficients
    C = np.zeros((L+1,N,N))
    M = np.zeros((N,N))
    #reduced form VAR(1)
    G = np.zeros_like(M)

    C[0] = -.7*eye(N, k=1) + .0*eye(N, k=-2) 
    C[1] = .5*eye(N)+.4*eye(N, k=-1)-.4*eye(N, k=1) 

    #generate the noise
    eps = gennorm.rvs(beta, size=(N,T+added))
    
    Y = deepcopy(eps)

    M+=inv(eye(N)-C[0])
    G+=M@C[1]

    #check eigenvalues for stability (since VAR(1), the companion matrix equals G)
    max_eig = np.absolute(eigs(G,k=1,which='LM', return_eigenvectors=False, maxiter=1e4))
    if max_eig>.99:
        print(max_eig)
        raise Exception('SVAR(1) is not stable.')
    
    for t in range(L,T+added):
        Y[:,t]=G@Y[:,t-L]+M@eps[:,t]

    return Y, C


def process_multiscale(N,T, beta, transform='dwt', wv='db1'):
    """
    This function generates T samples according to the multiscale stationary causal structure in Appendix F.

    INPUT
    =====
    N: int, number of dimensions;
    T: int, number of samples;
    beta: float, parameter of the generalized normal distribution (2. for Gaussian, 1. for Laplace, etc). 
    transform: str, either 'dwt' or 'swt',i.e., discrete or stationary wavelet transform;
    wavelet: str, wavelet to use, see pywt discrete wavelet families (only if multiscale=True).

    OUTPUT
    ======
    Y: np.ndarray, (N,T) array of synthetic data;
    Y_mra: np.ndarray, (J,N,T) array of ground truth multiscale representation of Y;
    eps_mra: np.ndarray, (J,N,T) array of ground truth multiscale noise;
    C: np.ndarray, (J,L,N,N) matrices of causal coefficients. Here we set J=3, L=1.
    """
    assert isinstance(T, int) and isinstance(N, int) and T>2**3 and N>0, 'N and T must be strictly positive integers.'
    assert isinstance(beta, (int, float)) and beta>0, 'Beta must be strictly positive scalar.'

    T = 2**int(np.log2(T))
    J = 3
    L= 1
    added = L*10

    #define the matrices of causal coefficients
    #j=0 is the smooth, where there aren't causal interactions
    #j=1 is the third scale level
    #j=3 is the finest
    C = np.zeros((J+1,L+1,N,N))
    M = np.zeros((J+1,N,N))
    #reduced form VAR(1)
    G = np.zeros_like(M)

    C[0,1] = -.7*eye(N) # W^4_1 in Appendix F
    
    C[1,1] = .5*eye(N)+.4*eye(N, k=-1)-.4*eye(N, k=1) # W^3_1

    C[2,0] = -.5*eye(N, k=-2) # W^2_0
    C[2,1] = -.5*eye(N)-.4*eye(N, k=-2)+.4*eye(N, k=2) # W^2_1

    C[3,0] = .6*eye(N, k=1) # W^1_0
    C[3,1] = -.6*eye(N)+.3*eye(N, k=-1)+.3*eye(N, k=1) # W^1_1
    
    #generate multiscale noise
    eps = gennorm.rvs(beta, size=(N,T+added))

    eps_mra = np.asarray(mra(eps, wavelet=wv, transform=transform, level=J))
    Y_mra = deepcopy(eps_mra)


    for j in range(J+1):
        M[j]+=inv(eye(N)-C[j,0])
        G[j]+=M[j]@C[j,1]

        #check eigenvalues for stability (since VAR(1), the companion matrix equals G)
        max_eig = np.absolute(eigs(G[j],k=1,which='LM', return_eigenvectors=False, maxiter=1e4))
        if max_eig>.99:
            print(max_eig)
            raise Exception('SVAR(1) at scale {} is not stable.'.format(J+1-j))
        
        for t in range(L,T+added):
            Y_mra[j,:,t]=G[j]@Y_mra[j,:,t-L]+M[j]@eps_mra[j,:,t]
    
    Y = Y_mra.sum(axis=0) 

    return Y[:,-T:], Y_mra[..., -T:], eps_mra[..., -T:], C