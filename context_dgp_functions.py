
import torch
from torch.special import expit
import numpy as np
from scipy.stats import beta as Beta
from util import parse_bool

def generate_data_beta_context_product(D, N, cnts=5, binaryX=False):
    '''
    Z: shape (D,1)
    X: shape (D,N,1)
    Y: shape (D,N)
    click_rate: shape (D,N)
    '''

    Z = torch.rand( (D,1) )    # 2 dimensional Z1
    X = torch.rand( (D,N,1) )     # 2 dimensional X
    if binaryX:
        X = (X > 0.5).float()

    XZ = torch.einsum('di,dti->dt', [Z,X])
    XZ = torch.clip(XZ, 1e-10)
    f = lambda z: z-z*torch.log(z)
    alpha = f(XZ)*cnts + 1
    beta = (1-f(XZ))*cnts + 1
    
    success_p = torch.distributions.beta.Beta(alpha, beta).sample()
    Y = torch.bernoulli(success_p)
    res = {}
    res['Z'] = Z
    res['X'] = X
    res['Y'] = Y
    res['click_rate'] = success_p


    return res

def generate_data_beta_context_sum(D, N, cnts=5, binaryX=False, clip=1e-10):
    '''
    Z: shape (D,1)
    X: shape (D,N,1)
    Y: shape (D,N)
    click_rate: shape (D,N)
    '''

    Z = torch.rand( (D,1) )    # 2 dimensional Z1
    X = torch.rand( (D,N,1) )     # 2 dimensional X
    if binaryX:
        X = (X > 0.5).float()

    X1 = X[:,:,0]
    Z1 = Z.repeat((1,N))
        
    f = lambda x,z: (x + z)/2
    alpha = f(X1,Z1)*cnts + 1
    beta = (1-f(X1,Z1))*cnts + 1
    
    success_p = torch.distributions.beta.Beta(alpha, beta).sample()
    Y = torch.bernoulli(success_p)
    res = {}
    res['Z'] = Z
    res['X'] = X
    res['Y'] = Y
    res['click_rate'] = success_p

    return res


def generate_data_beta_context_sum_dumb(D, N, binaryX=False, num_Us=0, one_X_per_col=False, generator=None, ave_U=False):
    '''
    Z: shape (D,1)
    X: shape (D,N,1)
    Y: shape (D,N)
    click_rate: shape (D,N)
    '''
    if generator is None:
        generator = torch.Generator()

    Z = torch.rand( (D,1), generator=generator)    
    U = torch.rand( (D,1), generator=generator)
    if one_X_per_col:
        X = torch.rand( (1,N,1), generator=generator ).repeat( (D,1,1) )  
    else:
        X = torch.rand( (D,N,1), generator=generator )  

    if binaryX:
        X = (X > 0.5).float()

    X1 = X[:,:,0]
    Z1 = Z.repeat((1,N))
    U1 = U.repeat((1,N))
    
    success_p = (Z1+U1+X1)/3
    Y = torch.bernoulli(success_p, generator=generator)
    
    res = {}
    res['Z'] = Z
    res['X'] = X
    res['Y'] = Y
    res['click_rate'] = success_p
    
    more_success_ps = []
    if num_Us > 0:
        for _ in range(num_Us):
            U = torch.rand( (D,1), generator=generator ) 
            U1 = U.repeat((1,N))
            more_success_ps.append((Z1+U1+X1)/3)
        res['all_click_rate'] = torch.concatenate([x.unsqueeze(0) for x in more_success_ps])    
    if ave_U:
        res['click_rate_ave_U'] = (Z1+0.5+X1)/3
    return res

def generate_data_beta_context_sum_dumb_z_ux(D, N, binaryX=False, num_Us=0, one_X_per_col=False, generator=None, ave_U=False):
    '''
    Z: shape (D,1)
    X: shape (D,N,1)
    Y: shape (D,N)
    click_rate: shape (D,N)
    '''
    if generator is None:
        generator = torch.Generator()

    Z = torch.rand( (D,1), generator=generator)    
    U = torch.rand( (D,1), generator=generator)
    if one_X_per_col:
        X = torch.rand( (1,N,1), generator=generator ).repeat( (D,1,1) )  
    else:
        X = torch.rand( (D,N,1), generator=generator )  

    if binaryX:
        X = (X > 0.5).float()

    X1 = X[:,:,0]
    Z1 = Z.repeat((1,N))
    U1 = U.repeat((1,N))
    
    success_p = (Z1+2*U1*X1)/3
    Y = torch.bernoulli(success_p, generator=generator)
    
    res = {}
    res['Z'] = Z
    res['X'] = X
    res['Y'] = Y
    res['click_rate'] = success_p
    
    more_success_ps = []
    if num_Us > 0:
        for _ in range(num_Us):
            U = torch.rand( (D,1), generator=generator ) 
            U1 = U.repeat((1,N))
            more_success_ps.append((Z1+2*U1*X1)/3)
        res['all_click_rate'] = torch.concatenate([x.unsqueeze(0) for x in more_success_ps])    
    if ave_U:
        res['click_rate_ave_U'] = (Z1+X1)/3        
    return res


def generate_data_logistic_context(D, N, binaryX=False, num_Us=1):
    '''
    Z: shape (D,1)
    X: shape (D,N,1)
    Y: shape (D,N)
    click_rate: shape (D,N)
    '''

    Z = torch.rand( (D,1) )    # 2 dimensional Z1
    X = torch.rand( (D,N,1) )     # 2 dimensional X
    if binaryX:
        X = (X > 0.5).float()
    U = torch.normal( torch.zeros((D,N,4)) )

    XZ = torch.einsum('di,dti->dt', [Z,X])
    X1 = torch.einsum('dti->dt', [X])
    Z1 = torch.einsum('di->d', [Z]).unsqueeze(1).repeat((1,N))

    W = XZ * U[:,:,0] + X1 * U[:,:,1] + Z1 * U[:,:,2] + U[:,:,3]

    success_p = expit(W)
    Y = torch.bernoulli(success_p)
    res = {}
    res['Z'] = Z
    res['X'] = X
    res['Y'] = Y
    res['W'] = W
    res['click_rate'] = success_p

    if num_Us > 1:
        all_Us = torch.normal( torch.zeros((num_Us, D,N,4)) )
        all_W = XZ * all_Us[:,:,:,0] + X1 * all_Us[:,:,:,1] + Z1 * all_Us[:,:,:,2] + all_Us[:,:,:,3]
        all_success_p = expit(all_W)
        res['all_click_rate'] = all_success_p
    
    return res



def generate_data_logistic_context_simple(D, N, dimX=1, binaryX=False, ave_U=False, one_X_per_col=True, num_Us=1, generator=None):
    '''
    Z: shape (D,1)
    X: shape (D,N,dimX)
    Y: shape (D,N)
    click_rate: shape (D,N)
    '''

    Z = torch.normal( torch.zeros((D,1)), generator=generator )    # 2 dimensional Z1
    if one_X_per_col:
        X = torch.normal( torch.zeros((1,N,dimX)), generator=generator ).repeat( (D,1,1) )  
    else:
        X = torch.normal( torch.zeros((D,N,dimX)), generator=generator )  
    if binaryX:
        X = (X > 0.5).float()
    U = torch.normal( torch.zeros((D,1)), generator=generator )

    #W = U + 0.1*Z + 10*X.sum(2)
    W = 2*(U-0.5) * ( Z + 3 * X.sum(2)/np.sqrt(dimX) )
    success_p = expit(W)
    Y = torch.bernoulli(success_p, generator=generator)
    res = {}
    res['Z'] = Z
    res['X'] = X
    res['Y'] = Y
    res['W'] = W
    res['click_rate'] = success_p

    if ave_U:
        num_gen = 1001
        Umany = torch.normal( torch.zeros((D,1,num_gen)), generator=generator )
        Wmany = Umany + X.sum(2).unsqueeze(2) + Z.unsqueeze(2)
        success_p_many = expit(Wmany)
        res['click_rate_ave_U'] = success_p_many.mean(2)        
    
    return res


def generate_data_logistic_context_simple2(D, N, dimX=1, binaryX=False, ave_U=False, one_X_per_col=True, num_Us=1, generator=None):
    '''
    Z: shape (D,1)
    X: shape (D,N,dimX)
    Y: shape (D,N)
    click_rate: shape (D,N)
    '''

    Z = torch.normal( torch.zeros((D,1)), generator=generator )    # 2 dimensional Z1
    if one_X_per_col:
        X = torch.normal( torch.zeros((1,N,dimX)), generator=generator ).repeat( (D,1,1) )  
    else:
        X = torch.normal( torch.zeros((D,N,dimX)), generator=generator )  
    if binaryX:
        X = (X > 0.5).float()
    U = torch.normal( torch.zeros((D,1)), generator=generator )

    #W = U + 0.1*Z + 10*X.sum(2)
    W = 2*(U-0.5) * ( Z + X.sum(2)/np.sqrt(dimX) )
    success_p = expit(W)
    Y = torch.bernoulli(success_p, generator=generator)
    res = {}
    res['Z'] = Z
    res['X'] = X
    res['Y'] = Y
    res['W'] = W
    res['click_rate'] = success_p

    if ave_U:
        num_gen = 1001
        Umany = torch.normal( torch.zeros((D,1,num_gen)), generator=generator )
        Wmany = Umany + X.sum(2).unsqueeze(2) + Z.unsqueeze(2)
        success_p_many = expit(Wmany)
        res['click_rate_ave_U'] = success_p_many.mean(2)        
    
    return res


def generate_data_logistic_context_simple3(D, N, dimX=1, binaryX=False, ave_U=False, one_X_per_col=True, num_Us=1, generator=None, xfactor=5):
    '''
    Z: shape (D,1)
    X: shape (D,N,dimX)
    Y: shape (D,N)
    click_rate: shape (D,N)
    '''

    Z = torch.normal( torch.zeros((D,1)), generator=generator )    # 2 dimensional Z1
    if one_X_per_col:
        X = torch.normal( torch.zeros((1,N,dimX)), generator=generator ).repeat( (D,1,1) )  
    else:
        X = torch.normal( torch.zeros((D,N,dimX)), generator=generator )  
    if binaryX:
        X = (X > 0.5).float()
    U = torch.normal( torch.zeros((D,1)), generator=generator )

    W = 2*U * ( Z + xfactor * X.sum(2)/np.sqrt(dimX) )
    success_p = expit(W)
    Y = torch.bernoulli(success_p, generator=generator)
    res = {}
    res['Z'] = Z
    res['X'] = X
    res['Y'] = Y
    res['W'] = W
    res['click_rate'] = success_p

    if ave_U:
        num_gen = 1001
        Umany = torch.normal( torch.zeros((D,1,num_gen)), generator=generator )
        Wmany = Umany + X.sum(2).unsqueeze(2) + Z.unsqueeze(2)
        success_p_many = expit(Wmany)
        res['click_rate_ave_U'] = success_p_many.mean(2)        
    
    return res



from scipy.stats import beta as Beta

def generate_data_beta_context_shareU(D, N, dimX=1, cnts=5, binaryX=False, one_X_per_col=True, 
    num_Us=0, generator=None, ave_U=False, verbose=False, uniform=False, discU=None):
    '''
    Z: shape (D,1)
    X: shape (D,N,1)
    Y: shape (D,N)
    click_rate: shape (D,N)
    '''
    
    if one_X_per_col:
        X = torch.rand( (1,N,dimX), generator=generator ).repeat( (D,1,1) )  
    else:
        X = torch.rand( (D,N,dimX), generator=generator )  
    if binaryX:
        X = (X > 0.5).float()
    
    if uniform:
        if dimX > 1:
            raise ValueError("not implemented")
            
        U = torch.rand( (D,1), generator=generator ) 
        Z = torch.rand( (D,1), generator=generator )   
        if discU is not None:
            U = torch.round(U * discU) / discU    
    
        X1 = X[:,:,0]
        Z1 = Z.repeat((1,N))
            
        f = lambda x,z: (x + z)/2
        alpha = f(X1,Z1)*cnts + 1
        beta = (1-f(X1,Z1))*cnts + 1
    else:
        U = torch.rand( (D,1), generator=generator )
        Z = torch.rand( (D,1), generator=generator )
        #Z = torch.normal( torch.zeros((D,1)), torch.ones((D,1)) )    
        if discU is not None:
            U = torch.round(U * discU) / discU

        if dimX == 1:
            X1 = X[:,:,0]
            Z1 = Z.repeat((1,N))
    
            #f = lambda x,z: torch.sigmoid( 50*(x-0.5) + 50*(z-0.5) )
            f = lambda x,z: torch.sigmoid( 100*(x-0.5) + 100*(z-0.5) )
            #f = lambda x,z: torch.sigmoid( 10*(x-0.5) + 0.1*(z-0.5) )
            #f = lambda x,z: torch.sigmoid( (x-0.5) )
            #f = lambda x,z: torch.sigmoid( 50*(x-0.5) ) # + 0.01*(z-0.5)
            alpha = f(X1,Z1)*cnts + 1
            beta = (1-f(X1,Z1))*cnts + 1
        else:
            Z1 = Z.repeat((1,N))
            f = lambda x,z: torch.sigmoid( 100*(x-0.5).sum(2) + 100*(z-0.5) )
            #multvec = torch.arange(dimX)/(dimX**2)+1; multvec = dimX * multvec / multvec.sum()
            #f = lambda x,z: torch.sigmoid( 200*( multvec*(X-0.5)).sum(2) + 200*(z-0.5) )   
            alpha = f(X,Z1)*cnts + 1
            beta = (1-f(X,Z1))*cnts + 1

    import time
    start_time = time.time()
    success_p_np = Beta.ppf(U.repeat(1, alpha.shape[1]), alpha, beta)
    if verbose:
        print("--- %s seconds ---" % (time.time() - start_time))

    success_p = torch.Tensor(success_p_np)
    Y = torch.bernoulli(success_p, generator=generator)
    res = {}
    res['Z'] = Z
    res['X'] = X
    res['Y'] = Y
    res['U'] = U
    res['click_rate'] = success_p

    if ave_U:
        new_U = U*0+0.5
        success_p_np = Beta.ppf(new_U.repeat(1, alpha.shape[1]), alpha, beta)
        success_p = torch.Tensor(success_p_np)
        res['click_rate_ave_U'] = success_p

    assert num_Us == 0
    
    return res
    

def generate_data_beta_context_shareU_bimodal(D, N, cnts1=5, cnts2=5, binaryX=False, one_X_per_col=True, clip=1e-10, num_Us=100, generator=None, ave_U=False, verbose=False, uniform = False, dimX=1):
    '''
    Z: shape (D,1)
    X: shape (D,N,1)
    Y: shape (D,N)
    click_rate: shape (D,N)
    '''
    
    if generator is None:
        generator = torch.Generator()

    Z = torch.rand( (D, 2) )    # two dimensional Z
     
    if one_X_per_col:
        X = torch.rand( (1,N,dimX), generator=generator ).repeat( (D,1,1) )  
    else:
        X = torch.rand( (D,N,dimX), generator=generator )  
    if binaryX:
        X = (X > 0.5).float()

    mode1_ind = torch.bernoulli(torch.ones(D)*0.5)
    mode1_ind = mode1_ind.unsqueeze(1).repeat((1,N))
    U = torch.rand( (D,1) )

    assert dimX == 1 # not implemented yet for higher dimX

    #mean1 = Z[:,0]*0.25; mean2 = 0.75+Z[:,1]*0.25
    f = lambda x,z: torch.sigmoid( 50*(x-0.5) + 50*(z-0.5) )
    #f = lambda x,z: torch.sigmoid( 100*(x-0.5) + (z-0.5) )
    Z0 = Z[:,[0]].repeat((1,N)); Z1 = Z[:,[1]].repeat((1,N))
    mean1 = f(Z[:,0], X[:,:,0])*0.5; mean2 = 0.5+f(Z[:,1], X[:,:,0])*0.5
    #mean1 = f(Z0, X[:,:,0])*0.25; mean2 = 0.75+f(Z1, X[:,:,0])*0.25
    
    """
    U = torch.rand( (D,1) )
    Z = torch.rand( (D,1) )
    #Z = torch.normal( torch.zeros((D,1)), torch.ones((D,1)) )    
    
    X1 = X[:,:,0]
    Z1 = Z.repeat((1,N))

    f = lambda x,z: torch.sigmoid( 50*(x-0.5) + 50*(z-0.5) )
    alpha = f(X1,Z1)*cnts + 1
    beta = (1-f(X1,Z1))*cnts + 1
    """
    
    # mode 1
    alpha1 = mean1*cnts1 + 1
    beta1 = 2 + cnts1 - alpha1
    
    # mode 2
    alpha2 = mean2*cnts2 + 1
    beta2 = 2 + cnts2 - alpha2

    alpha = mode1_ind * alpha1 + (1-mode1_ind) * alpha2
    beta = mode1_ind * beta1 + (1-mode1_ind) * beta2

    #import ipdb; ipdb.set_trace()

    import time
    start_time = time.time()
    success_p_np = Beta.ppf(U.repeat(1, alpha.shape[1]), alpha, beta)
    if verbose:
        print("--- %s seconds ---" % (time.time() - start_time))

    success_p = torch.Tensor(success_p_np)
    Y = torch.bernoulli(success_p)
    res = {}
    res['Z'] = Z
    res['X'] = X
    res['Y'] = Y
    res['alpha1'] = alpha1; res['beta1'] = beta1
    res['alpha2'] = alpha2; res['beta2'] = beta2
    res['mode'] = mode1_ind
    res['click_rate'] = success_p

    if ave_U:
        new_U = U*0+0.5
        success_p_np = Beta.ppf(new_U.repeat(1, alpha.shape[1]), alpha, beta)
        success_p = torch.Tensor(success_p_np)
        res['click_rate_ave_U'] = success_p

    if num_Us > 0:
        more_success_ps = []
        for i in range(num_Us):
            print(i)
            U = torch.rand( (D,1) ) 
            success_p_np = Beta.ppf(U.repeat(1, alpha.shape[1]), alpha, beta)
            success_p = torch.Tensor(success_p_np)
            more_success_ps.append(success_p)
        res['all_click_rate'] = torch.concatenate([x.unsqueeze(0) for x in more_success_ps])
    
    return res


def generate_data_beta_context_shareU_v2(D, N, dimX=1, cnts=5, binaryX=False, one_X_per_col=True, 
    num_Us=0, generator=None, ave_U=False, verbose=False, uniform=False, discU=None):
    '''
    Z: shape (D,1)
    X: shape (D,N,1)
    Y: shape (D,N)
    click_rate: shape (D,N)
    '''
    
    if one_X_per_col:
        #X = torch.rand( (1,N,dimX), generator=generator ).repeat( (D,1,1) )  
        X = torch.normal( mean=torch.zeros((1,N,dimX)), generator=generator ).repeat( (D,1,1) )  
    else:
        #X = torch.rand( (D,N,dimX), generator=generator )  
        X = torch.normal( mean=torch.zeros((D,N,dimX)), generator=generator )
    if binaryX:
        X = (X > 0.5).float()
    
    if uniform:
        if dimX > 1:
            raise ValueError("not implemented")
            
        U = torch.rand( (D,1), generator=generator ) 
        Z = torch.rand( (D,1), generator=generator )   
        if discU is not None:
            U = torch.round(U * discU) / discU    
    
        X1 = X[:,:,0]
        Z1 = Z.repeat((1,N))
            
        f = lambda x,z: (x + z)/2
        alpha = f(X1,Z1)*cnts + 1
        beta = (1-f(X1,Z1))*cnts + 1
    else:
        U = torch.rand( (D,1), generator=generator )
        Z = torch.rand( (D,1), generator=generator )
        #Z = torch.normal( mean=torch.zeros((D,1)), generator=generator )
        
        #Z = torch.normal( torch.zeros((D,1)), torch.ones((D,1)) )    
        if discU is not None:
            U = torch.round(U * discU) / discU

        if dimX == 0:
            X1 = X[:,:,0]
            Z1 = Z.repeat((1,N))

            f = lambda x,z: torch.sigmoid( 100*x + 100*z )
    
            #f = lambda x,z: torch.sigmoid( 50*(x-0.5) + 50*(z-0.5) )
            #f = lambda x,z: torch.sigmoid( 100*(x-0.5) + 100*(z-0.5) )
            #f = lambda x,z: torch.sigmoid( 10*(x-0.5) + 0.1*(z-0.5) )
            #f = lambda x,z: torch.sigmoid( (x-0.5) )
            #f = lambda x,z: torch.sigmoid( 50*(x-0.5) ) # + 0.01*(z-0.5)
            alpha = f(X1,Z1)*cnts + 1
            beta = (1-f(X1,Z1))*cnts + 1
            print(123)
        else:
            Z1 = Z.repeat((1,N))
            f = lambda x,z: torch.sigmoid( (100*x).sum(2) + (z-0.5) )

            #f = lambda x,z: torch.sigmoid( (100*(x-0.5)).sum(2) + 100*(z-0.5) )
            #f = lambda x,z: torch.sigmoid( (100*(x-0.5)).sum(2) + 100*(z-0.5) )
            #f = lambda x,z: torch.sigmoid( 10/np.sqrt(dimX)*(x-0.5).sum(2) ) #+ 100*(z-0.5)
            #multvec = torch.arange(dimX)/(dimX**2)+1; multvec = dimX * multvec / multvec.sum()
            #f = lambda x,z: torch.sigmoid( 200*( multvec*(X-0.5)).sum(2) + 200*(z-0.5) )   
            alpha = f(X,Z1)*cnts + 1
            beta = (1-f(X,Z1))*cnts + 1

    import time
    start_time = time.time()
    success_p_np = Beta.ppf(U.repeat(1, alpha.shape[1]), alpha, beta)
    if verbose:
        print("--- %s seconds ---" % (time.time() - start_time))

    success_p = torch.Tensor(success_p_np)
    Y = torch.bernoulli(success_p, generator=generator)
    res = {}
    res['Z'] = Z
    res['X'] = X
    res['Y'] = Y
    res['U'] = U
    res['click_rate'] = success_p

    #print(success_p)
    #import ipdb; ipdb.set_trace()

    if ave_U:
        new_U = U*0+0.5
        success_p_np = Beta.ppf(new_U.repeat(1, alpha.shape[1]), alpha, beta)
        success_p = torch.Tensor(success_p_np)
        res['click_rate_ave_U'] = success_p

    assert num_Us == 0
    
    return res

CONTEXT_DGPs = {
    'shareU_0702': lambda D,N,dimX,ave_U,one_X_per_col,g: generate_data_beta_context_shareU(D=D,N=N,binaryX=False,cnts=5,
        one_X_per_col=one_X_per_col,ave_U=ave_U,num_Us=0,uniform=False,generator=g),
    # like the above, but discretize U into 10
    'shareU_0703_discU10': lambda D,N,dimX,ave_U,one_X_per_col,g: generate_data_beta_context_shareU(D=D,N=N,binaryX=False,cnts=5,
        one_X_per_col=one_X_per_col,ave_U=ave_U,num_Us=0,uniform=False,generator=g,discU=10),
    # like the above, but discretize U into 20
    'shareU_0703_discU20': lambda D,N,dimX,ave_U,one_X_per_col,g: generate_data_beta_context_shareU(D=D,N=N,binaryX=False,cnts=5,
        one_X_per_col=one_X_per_col,ave_U=ave_U,num_Us=0,uniform=False,generator=g,discU=20),
    'shareU_0709_dimX': lambda D,N,dimX,ave_U,one_X_per_col,g: generate_data_beta_context_shareU(D=D,N=N,dimX=dimX,binaryX=False,cnts=5,
        one_X_per_col=one_X_per_col,ave_U=ave_U,num_Us=0,uniform=False,generator=g),
    'shareU_0717_dimX': lambda D,N,dimX,ave_U,one_X_per_col,g: generate_data_beta_context_shareU_v2(D=D,N=N,dimX=dimX,binaryX=False,cnts=25,
        one_X_per_col=one_X_per_col,ave_U=ave_U,num_Us=0,uniform=False,generator=g),
    '0722_logistic': lambda D,N,dimX,ave_U,one_X_per_col,g: generate_data_logistic_context_simple(D=D,N=N,dimX=dimX,ave_U=ave_U,binaryX=False,
        one_X_per_col=one_X_per_col,generator=g),
    '0723_logistic': lambda D,N,dimX,ave_U,one_X_per_col,g: generate_data_logistic_context_simple2(D=D,N=N,dimX=dimX,ave_U=ave_U,binaryX=False,
        one_X_per_col=one_X_per_col,generator=g),
    '0916_logistic': lambda D,N,dimX,ave_U,one_X_per_col,g: generate_data_logistic_context_simple3(D=D,N=N,dimX=dimX,ave_U=ave_U,binaryX=False,
        one_X_per_col=one_X_per_col,generator=g),
    '0916_logistic_10': lambda D,N,dimX,ave_U,one_X_per_col,g: generate_data_logistic_context_simple3(D=D,N=N,dimX=dimX,ave_U=ave_U,binaryX=False,
        one_X_per_col=one_X_per_col,generator=g,xfactor=10),
    '0916_logistic_3': lambda D,N,dimX,ave_U,one_X_per_col,g: generate_data_logistic_context_simple3(D=D,N=N,dimX=dimX,ave_U=ave_U,binaryX=False,
        one_X_per_col=one_X_per_col,generator=g,xfactor=3),
    '0916_logistic_2': lambda D,N,dimX,ave_U,one_X_per_col,g: generate_data_logistic_context_simple3(D=D,N=N,dimX=dimX,ave_U=ave_U,binaryX=False,
        one_X_per_col=one_X_per_col,generator=g,xfactor=2),
}

# old methods
#    if args.method=='sum_dumb':
#        generate_fn = lambda D,N,binary: generate_data_beta_context_sum_dumb(D=D,N=N,binaryX=binary,one_X_per_col=args.one_X_per_col, ave_U=True)                                                                  
#    elif args.method=='sum_dumb_z_ux':
#        generate_fn = lambda D,N,binary: generate_data_beta_context_sum_dumb_z_ux(D=D,N=N,binaryX=binary,one_X_per_col=args.one_X_per_col, ave_U=True)                                                             
#    elif args.method=='shareU':
#        generate_fn = lambda D,N,binary,ave_U: generate_data_beta_context_shareU(D=D,N=N,binaryX=binary,cnts=args.cnts,one_X_per_col=args.one_X_per_col,ave_U=ave_U,num_Us=1, uniform=True)                       
#    elif args.method=='shareU_new':
#        generate_fn = lambda D,N,binary,ave_U: generate_data_beta_context_shareU(D=D,N=N,binaryX=binary,cnts=args.cnts,one_X_per_col=args.one_X_per_col,ave_U=ave_U,num_Us=0, uniform=False)                       
#    elif args.method=='shareU_0702':
#        generate_fn = lambda D,N,binary,ave_U,g: generate_data_beta_context_shareU(D=D,N=N,binaryX=binary,cnts=args.cnts,one_X_per_col=args.one_X_per_col,ave_U=ave_U,num_Us=0, uniform=False, generator=g)        
#    elif args.method == 'shareU_bimodal':
#        generate_fn = lambda D,N,binary,ave_U: generate_data_beta_context_shareU_bimodal(D=D,N=N,binaryX=binary,cnts1=args.cnts,cnts2=args.cnts2,one_X_per_col=args.one_X_per_col,ave_U=ave_U,num_Us=1,dimX=args.dimX)


