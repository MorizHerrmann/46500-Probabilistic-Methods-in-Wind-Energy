def MysteriousLoadFunc(X):
    import numpy as np
    Alsq = np.array([-8.49821625e+03,  3.21111941e+03, -3.44642937e+03, -8.33496535e+02,
       -2.28585769e+02,  7.27203251e+03,  9.25677112e+02,  1.16755478e+02,
        1.18918605e+03,  5.07954840e+00, -3.79944018e+03, -3.99715920e+01,
        7.18288327e+01, -1.82188802e+02, -2.61134584e+00, -2.53566886e+02,
        1.11035167e+03,  8.03450614e+00, -1.49376926e+02,  6.50965086e+00])

    X1 = X[:,0]
    X2 = X[:,1]
    X3 = X[:,2]
    X0 = np.ones(X1.shape)
    Xmatrix = np.array((X0,X1,X2,X3,X1**2,X2**2,X1*X2,X1*X3,X2*X3,X1**3,X2**3,(X1**2)*X2, (X1**2)*X3, \
                       (X2**2)*X1,(X2**2)*X3,X1*X2*X3, X2**4, (X1**2)*(X2**2), X2**5, X2**6)).T

    Y = np.dot(Xmatrix,Alsq) + 50e3 + np.random.randn(X0.shape[0])*4.0e3
    return Y
