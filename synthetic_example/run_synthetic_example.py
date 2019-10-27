# use autograd to calculate the gradient
import autograd.numpy as np
from autograd import grad

from matplotlib import pyplot as plt



### functions for solving QP problem  ###

# can use cvxopt or use the min_norm_solvers written by the author of MOO-MTL


#import cvxopt
#from cvxopt import matrix
#
#def cvxopt_solve_qp(P, q, G=None, h=None, A=None, b=None):
#    P = .5 * (P + P.T)  # make sure P is symmetric
#    
#    P = P.astype(np.double)
#    q = q.astype(np.double)
#    args = [matrix(P), matrix(q)]
#    if G is not None:
#        args.extend([matrix(G), matrix(h)])
#        if A is not None:
#            args.extend([matrix(A), matrix(b)])
#    sol = cvxopt.solvers.qp(*args)
#    if 'optimal' not in sol['status']:
#        return None
#    return np.array(sol['x']).reshape((P.shape[1],))


from min_norm_solvers_numpy import MinNormSolver


def get_d_moomtl(grads):
    """
    calculate the gradient direction for MOO-MTL 
    """
    
    nobj, dim = grads.shape
    
#    # use cvxopt to solve QP
#    P = np.dot(grads , grads.T)
#    
#    q = np.zeros(nobj)
#    
#    G =  - np.eye(nobj)
#    h = np.zeros(nobj)
#    
#    
#    A = np.ones(nobj).reshape(1,2)
#    b = np.ones(1)
#       
#    cvxopt.solvers.options['show_progress'] = False
#    sol = cvxopt_solve_qp(P, q, G, h, A, b)
    
    # use MinNormSolver to solve QP
    sol, nd = MinNormSolver.find_min_norm_element(grads)
    
    return sol



def get_d_paretomtl(grads,value,weights,i):
    # calculate the gradient direction for Pareto MTL
    nobj, dim = grads.shape
    
    # check active constraints
    normalized_current_weight = weights[i]/np.linalg.norm(weights[i])
    normalized_rest_weights = np.delete(weights, (i), axis=0) / np.linalg.norm(np.delete(weights, (i), axis=0), axis = 1,keepdims = True)
    w = normalized_rest_weights - normalized_current_weight
    
    
    # solve QP 
    gx =  np.dot(w,value/np.linalg.norm(value))
    idx = gx >  0
   
    
    vec =  np.concatenate((grads, np.dot(w[idx],grads)), axis = 0)
    
#    # use cvxopt to solve QP
#    
#    P = np.dot(vec , vec.T)
#    
#    q = np.zeros(nobj + np.sum(idx))
#    
#    G =  - np.eye(nobj + np.sum(idx) )
#    h = np.zeros(nobj + np.sum(idx))
#    
#
#    
#    A = np.ones(nobj + np.sum(idx)).reshape(1,nobj + np.sum(idx))
#    b = np.ones(1)
 
#    cvxopt.solvers.options['show_progress'] = False
#    sol = cvxopt_solve_qp(P, q, G, h, A, b)
  
    # use MinNormSolver to solve QP
    sol, nd = MinNormSolver.find_min_norm_element(vec)
   
    
    # reformulate ParetoMTL as linear scalarization method, return the weights
    weight0 =  sol[0] + np.sum(np.array([sol[j] * w[idx][j - 2,0] for j in np.arange(2,2 + np.sum(idx))]))
    weight1 = sol[1] + np.sum(np.array([sol[j] * w[idx][j - 2,1] for j in np.arange(2,2 + np.sum(idx))]))
    weight = np.stack([weight0,weight1])
   

    return weight



def get_d_paretomtl_init(grads,value,weights,i):
    # calculate the gradient direction for Pareto MTL initialization
    nobj, dim = grads.shape
    
    # check active constraints
    normalized_current_weight = weights[i]/np.linalg.norm(weights[i])
    normalized_rest_weights = np.delete(weights, (i), axis=0) / np.linalg.norm(np.delete(weights, (i), axis=0), axis = 1,keepdims = True)
    w = normalized_rest_weights - normalized_current_weight
    
    gx =  np.dot(w,value/np.linalg.norm(value))
    idx = gx >  0
    
    if np.sum(idx) <= 0:
        return np.zeros(nobj)
    if np.sum(idx) == 1:
        sol = np.ones(1)
    else:
        vec =  np.dot(w[idx],grads)
        sol, nd = MinNormSolver.find_min_norm_element(vec)

    # calculate the weights
    weight0 =  np.sum(np.array([sol[j] * w[idx][j ,0] for j in np.arange(0, np.sum(idx))]))
    weight1 =  np.sum(np.array([sol[j] * w[idx][j ,1] for j in np.arange(0, np.sum(idx))]))
    weight = np.stack([weight0,weight1])
   

    return weight



def circle_points(r, n):
    # generate evenly distributed preference vector
    circles = []
    for r, n in zip(r, n):
        t = np.linspace(0, 0.5 * np.pi, n)
        x = r * np.cos(t)
        y = r * np.sin(t)
        circles.append(np.c_[x, y])
    return circles


### the synthetic multi-objective problem ###
def f1(x):
    
    n = len(x)
    
    sum1 = np.sum([(x[i] - 1.0/np.sqrt(n)) ** 2 for i in range(n)])

    f1 = 1 - np.exp(- sum1)
    return f1

def f2(x):
    
    n = len(x)
    
    sum2 = np.sum([(x[i] + 1.0/np.sqrt(n)) ** 2 for i in range(n)])
   
    f2 = 1 - np.exp(- sum2)
    
    return f2

# calculate the gradients using autograd
f1_dx = grad(f1)
f2_dx = grad(f2)
    
def concave_fun_eval(x):
    """
    return the function values and gradient values
    """
    return np.stack([f1(x), f2(x)]), np.stack([f1_dx(x), f2_dx(x)])
    
    
### create the ground truth Pareto front ###
def create_pf():
    ps = np.linspace(-1/np.sqrt(2),1/np.sqrt(2))
    pf = []
    
    for x1 in ps:
        #generate solutions on the Pareto front:
        x = np.array([x1,x1])
        
        f, f_dx = concave_fun_eval(x)
        pf.append(f)
            
    pf = np.array(pf)
    
    return pf




### optimization method ###

def linear_scalarization_search(t_iter = 100, n_dim = 20, step_size = 1):
    """
    linear scalarization with randomly generated weights
    """
    r = np.random.rand(1)
    weights = np.stack([r, 1-r])
    
    x = np.random.uniform(-0.5,0.5,n_dim)
    
    for t in range(t_iter):
        f, f_dx = concave_fun_eval(x)
        x = x - step_size * np.dot(weights.T,f_dx).flatten()
    
    return x, f


def moo_mtl_search(t_iter = 100, n_dim = 20, step_size = 1):
    """
    MOO-MTL
    """
    x = np.random.uniform(-0.5,0.5,n_dim)
    
    for t in range(t_iter):
        f, f_dx = concave_fun_eval(x)
    
        weights = get_d_moomtl(f_dx)
     
        
        x = x - step_size * np.dot(weights.T,f_dx).flatten()


    return x, f


def pareto_mtl_search(ref_vecs,i,t_iter = 100, n_dim = 20, step_size = 1):
    """
    Pareto MTL
    """

    # randomly generate one solution
    x = np.random.uniform(-0.5,0.5,n_dim)
    
    # find the initial solution
    for t in range(int(t_iter * 0.2)):
        f, f_dx = concave_fun_eval(x)
    
        weights =  get_d_paretomtl_init(f_dx,f,ref_vecs,i)
     
        
        x = x - step_size * np.dot(weights.T,f_dx).flatten()
    
    # find the Pareto optimal solution
    for t in range(int(t_iter * 0.8)):
        f, f_dx = concave_fun_eval(x)
    
        weights =  get_d_paretomtl(f_dx,f,ref_vecs,i)
     
        
        x = x - step_size * np.dot(weights.T,f_dx).flatten()


    return x, f



def run(method = 'ParetoMTL', num = 10):
    """
    run method on the synthetic example
    method: optimization method {'ParetoMTL', 'MOOMTL', 'Linear'}
    num: number of solutions
    """
    
    pf = create_pf()
    f_value_list = []
    
    weights = circle_points([1], [num])[0]
    

    
    for i in range(num):
        
        print(i)
        
        if method == 'ParetoMTL':
            x, f = pareto_mtl_search(ref_vecs = weights,i = i)
        if method == 'MOOMTL':
            x, f = moo_mtl_search()
        if method == 'Linear':
            x, f = linear_scalarization_search()
        
        f_value_list.append(f)
        
       
    f_value = np.array(f_value_list)
    plt.plot(pf[:,0],pf[:,1])
    plt.scatter(f_value[:,0], f_value[:,1], c = 'r', s = 80)
    
    
run('ParetoMTL')

