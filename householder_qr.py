import numpy as np
from scipy import linalg


def householder_reflection(A, b):

    m, n = A.shape

    for k in range(n):

        x = A[k:m,k].reshape(-1,1)
        s = np.sign(x[0])

        if s==0:
            s=1

        v  = x.copy()
        v[0] = v[0] + s*linalg.norm(x)
        v = v / linalg.norm(v)
        A[k:m, k:n] =  A[k:m, k:n] - 2*np.dot( v,  np.dot( v.T, A[k:m, k:n] ) )
        b[k:m] =  b[k:m] - 2*np.dot( v,  np.dot( v.T, b[k:m]) )

    return A , b

def generate_A(m, n):
    A = np.zeros([m,n])
    t = np.linspace(0, 1, m)
    for k in range(n):
        A[:,k] = t**k
    return A

def back_substitution(R, b):

    n = R.shape[1]
    x = np.zeros(n)

    for j in range(n-1,-1,-1):
        x[j] = ( b[j] - np.dot(R[j,j+1:], x[j+1:]) ) / R[j,j]

    return x


errors = []

for n in range(4,26,2):

    m = 2*n

    A = generate_A(m,n)
    x = np.ones([n,1])
    b = np.dot(A, x)
    A, b  = householder_reflection(A, b)
    x_approx = back_substitution(A, b)

    errors.append( linalg.norm( x - x_approx) )


import matplotlib.pyplot as plt


plt.plot(np.arange(4,26,2), errors)
plt.show()
print(A)
