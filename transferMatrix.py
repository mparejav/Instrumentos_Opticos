import numpy as np
import sympy as sp

"""
In this module, we create and multiply matrices for an optical system.
Thus, we are going to use the conjugate planes theory
"""
# Set print options for better readability
np.set_printoptions(precision=3, suppress=True)

"""
Firstly we are doing the algebraic manipulation of the matrices
"""
# Define symbolic variables for algebraic manipulation
f_simbolic = sp.symbols('f')  # Symbolic focal length
d_simbolic = sp.symbols('d')  # Symbolic distance

#We are creating the principal matrices for the optical systems
propagationf = sp.Matrix([[1, f_simbolic], [0, 1]])  # Propagation matrix for distance f
thinLens = sp.Matrix([[1, 0], [-1/f_simbolic, 1]])  # Thin lens matrix
propagationd = sp.Matrix([[1, d_simbolic], [0, 1]])  # Propagation matrix for distance d
identity = sp.Matrix([[1, 0], [0, 1]])  # Identity matrix

#We are multipling the matrices for transfer s(xi,eta) to O(u,v) in terms of f
resultOalgebraic = propagationf * thinLens * propagationf * identity * propagationf * thinLens * propagationf

#We are multipling the matrices for transfer s(xi,eta) to U(x',y') in terms of f and d
resultUalgebraic = propagationf * thinLens * propagationd


"""
Secondly we are doing the numerical manipulation of the matrices
"""

f = 500  # Focal length of the lens in mm
d = 700  # Distance from the transmitance to lens in mm

#We are creating the matrix for O(u,v) in terms of f
matrixO = [[[1, f], [0, 1]],[[1, f], [0, 1]], [[1, 0], [-1/f, 1]], [[1, f], [0, 1]], 
           [[1, 0],[0, 1]],[[1, f], [0, 1]],[[1, 0], [-1/f, 1]],[[1, f], [0, 1]]]

#We are creating the matrix for U(u',v') in terms of f and d
matrixU = [[[1, f], [0, 1]], [[1, 0], [-1/f, 1]], [[1, d], [0, 1]]]

#We are multippling the matrices for transfer s(xi,eta) to O(u,v) or U(x',y')
def multiply_matrices(matrices):
    result = matrices[0]
    for j in range(1, len(matrices)):
        result = np.dot(result, matrices[j])
    return result

#We call the function to multiply the matrices
resultO = multiply_matrices(matrixO)
    
#We call the function to multiply the matrices
resultU = multiply_matrices(matrixU)

"""
Printing the results
"""

print ("The optic system matrix is \n", resultOalgebraic)
print ("The optic system matrix is \n", resultUalgebraic)


