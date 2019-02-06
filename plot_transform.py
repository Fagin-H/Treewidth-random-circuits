from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import expm

global sigx, sigy, sigz, eye, x, y, z

sigx = np.array([[0,1],[1,0]])
sigy = np.array([[0,-1.0j],[1.0j,0]])
sigz = np.array([[1,0],[0,-1]])

eye = np.array([[1,0],[0,1]])

sig = np.array([sigx,sigy,sigz])

U = np.matrix([[1,0,0,0],[0,0.5+0.5j,0.5-0.5j,0],[0,0.5-0.5j,0.5+0.5j,0],[0,0,0,-1]])

def partial_trace(rho, dims = [2,2], axis=0):
    dims_ = np.array(dims)
    reshaped_rho = rho.reshape(np.concatenate((dims_, dims_), axis=None))
    reshaped_rho = np.moveaxis(reshaped_rho, axis, -1)
    reshaped_rho = np.moveaxis(reshaped_rho, len(dims)+axis-1, -1)
    traced_out_rho = np.trace(reshaped_rho, axis1=-2, axis2=-1)
    dims_untraced = np.delete(dims_, axis)
    rho_dim = np.prod(dims_untraced)
    return traced_out_rho.reshape([rho_dim, rho_dim])
    

u = np.linspace(0, 2 * np.pi, 30)
v = np.linspace(0, np.pi, 30)

x = np.outer(np.cos(u), np.sin(v))
y = np.outer(np.sin(u), np.sin(v))
z = np.outer(np.ones(np.size(u)), np.cos(v))

def findrho(U,rhoin):
    rho = np.copy(rhoin)
    tol = 1
    while tol > 0.1:
        rhoall = np.kron(rhoin,rho)
        final = U*rhoall*U.getH()
        final = np.matrix(partial_trace(np.array(final), axis = 1))
        tol = np.linalg.norm(final-rho)
        rho = final
        
    return rho


def runplot(U,name):
    x_ = np.copy(x)
    y_ = np.copy(y)
    z_ = np.copy(z)
    
    
    for i in range(0,x.shape[0]):
        for j in range(0,x.shape[1]):
            
            rhoin = np.matrix((eye + x[i,j]*sigx+y[i,j]*sigy+z[i,j]*sigz)*0.5)
    
            rho = findrho(U,rhoin)
    
            
            rhoall = np.kron(rhoin,rho)
            final = U*rhoall*U.getH()
            rhoout = np.matrix(partial_trace(np.array(final), axis = 0))
    
            
            x_[i,j] = (rhoout[1,0]+rhoout[0,1])
            y_[i,j] = (1j*(rhoout[1,0]-rhoout[0,1])).real
            z_[i,j] = (rhoout[0,0]-rhoout[1,1])
            
    
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    ax.plot_wireframe(x, y, z, color = 'red', alpha=0.1)
    ax.plot_wireframe(x_, y_, z_)
    
    plt.savefig('gifimages\gif' + name + '.png')
    plt.close()
    
def makeimages(H,res = 30,ma = 1):
    ts = np.arange(0, ma, ma/res)
    
    for t in ts:
        U = np.matrix(expm(1j*t*H))
        runplot(U,str(t*res))
        
    
    
    
    
    
    
"""


        n = [x[i,j],y[i,j],z[i,j]]
        m = transform(U,n)
        
        rhoin = n[0]*sig[0]+n[1]*sig[1]+n[2]*sig[2]
        rho = m[0]*sig[0]+m[1]*sig[1]+m[2]*sig[2]
        rhoall = np.kron(rhoin,rho)
        final = U*rhoall*U.getH()
        final = np.matrix(partial_trace(np.array(final), axis = 1))
        
        x_[i,j] = (final[0,1]+final[1,0])*0.5
        y_[i,j] = -0.5j*(final[0,1]-final[0,1])
        z_[i,j] = final[0,0]

        
        
def solverho(m,U,n):
    rhoin = n[0]*sig[0]+n[1]*sig[1]+n[2]*sig[2]
    rho = m[0]*sig[0]+m[1]*sig[1]+m[2]*sig[2]
    rhoall = np.kron(rhoin,rho)
    final = U*rhoall*U.getH()
    final = np.matrix(partial_trace(np.array(final)))

    return np.linalg.norm((final-rho)**2)
    
def squaresum(x):
    return x[0]**2+x[1]**2+x[2]**2-1    

def transform(U,n):
    cons = {'type':'eq', 'fun': squaresum}
    res = minimize(solverho, [0.5,0.5,0.5], args=(U,n), constraints=cons)
    return res.x
"""
    