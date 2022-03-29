########################################################################################################################################

# CS6375.001 Machine Learning - Assignment 1 - Linear Regression
# Name: Karneeshwar Sendilkumar Vijaya
# NetID: KXS200001

########################################################################################################################################

# Generating Sythetic Data

########################################################################################################################################

# The true function
def f_true(x):
  y = 6.0 * (np.sin(x + 2) + np.sin(2*x + 4))
  return y


import numpy as np                       # For all our math needs
n = 750                                  # Number of data points
X = np.random.uniform(-7.5, 7.5, n)      # Training examples, in one dimension
e = np.random.normal(0.0, 5.0, n)        # Random Gaussian noise
y = f_true(X) + e                        # True labels with noise

import matplotlib.pyplot as plt          # For all our plotting needs
plt.figure()

# Plot the data
plt.scatter(X, y, 12, marker='o')           

# Plot the true function, which is really "unknown"
x_true = np.arange(-7.5, 7.5, 0.05)
y_true = f_true(x_true)
plt.plot(x_true, y_true, marker='None', color='r')

# scikit-learn has many tools and utilities for model selection
from sklearn.model_selection import train_test_split
tst_frac = 0.3  # Fraction of examples to sample for the test set
val_frac = 0.1  # Fraction of examples to sample for the validation set

# First, we use train_test_split to partition (X, y) into training and test sets
X_trn, X_tst, y_trn, y_tst = train_test_split(X, y, test_size=tst_frac, random_state=42)

# Next, we use train_test_split to further partition (X_trn, y_trn) into training and validation sets
X_trn, X_val, y_trn, y_val = train_test_split(X_trn, y_trn, test_size=val_frac, random_state=42)

# Plot the three subsets
plt.figure()
plt.scatter(X_trn, y_trn, 12, marker='o', color='orange')
plt.scatter(X_val, y_val, 12, marker='o', color='green')
plt.scatter(X_tst, y_tst, 12, marker='o', color='blue')

########################################################################################################################################

# Problem 1: Regression with Polynomial Basis Functions

########################################################################################################################################

# 1.a: Vandermonde matrix of dimension d

########################################################################################################################################

# X float(n, ): univariate data
# d int: degree of polynomial  
def polynomial_transform(X, d):
    phi_2 = []
    for x in X:
        phi_1 = []
        for power in range(0,d+1):
            phi_1.append(x**power)
        phi_2.append(phi_1)
    return np.array(phi_2)

########################################################################################################################################

# 1.b: Ordinary least squares regression

########################################################################################################################################

# Phi float(n, d): transformed data
# y   float(n,  ): labels
def train_model(Phi, y):
    return (np.linalg.inv(np.transpose(Phi)@Phi))@(np.transpose(Phi)@y)

########################################################################################################################################

# 1.c: Mean squared error

########################################################################################################################################

# Phi float(n, d): transformed data
# y   float(n,  ): labels
# w   float(d,  ): linear regression model
def evaluate_model(Phi, y, w):
    summation = 0
    for i in range(len(y)):
        sub = y[i] - np.transpose(w)@Phi[i]
        sqr = sub ** 2
        summation += sqr
    return summation/len(y)

########################################################################################################################################

# 1.d: Discussion

########################################################################################################################################

w = {}               # Dictionary to store all the trained models
validationErr = {}   # Validation error of the models
testErr = {}         # Test error of all the models

for d in range(3, 25, 3):  # Iterate over polynomial degree
    Phi_trn = polynomial_transform(X_trn, d)                 # Transform training data into d dimensions
    w[d] = train_model(Phi_trn, y_trn)                       # Learn model on training data
    
    Phi_val = polynomial_transform(X_val, d)                 # Transform validation data into d dimensions
    validationErr[d] = evaluate_model(Phi_val, y_val, w[d])  # Evaluate model on validation data
    
    Phi_tst = polynomial_transform(X_tst, d)                 # Transform test data into d dimensions
    testErr[d] = evaluate_model(Phi_tst, y_tst, w[d])        # Evaluate model on test data

# Plot all the models
plt.figure()
plt.plot(validationErr.keys(), validationErr.values(), marker='o', linewidth=3, markersize=12)
plt.plot(testErr.keys(), testErr.values(), marker='s', linewidth=3, markersize=12)
plt.xlabel('Polynomial degree', fontsize=16)
plt.ylabel('Validation/Test error', fontsize=16)
plt.xticks(list(validationErr.keys()), fontsize=12)
plt.legend(['Validation Error', 'Test Error'], fontsize=16)
plt.axis([2, 25, 15, 60])

# Discussion:
# From the plot below, we can observe that degree 24 has very high error
# Others seem to have lesser error
# But degress 15, 18 and 21 have the least error when compared to other degrees
# Hence, these degress can be used to generalize the given model

plt.figure()
plt.plot(x_true, y_true, marker='None', linewidth=5, color='k')

for d in range(9, 25, 3):
  X_d = polynomial_transform(x_true, d)
  y_d = X_d @ w[d]
  plt.plot(x_true, y_d, marker='None', linewidth=2)

plt.legend(['true'] + list(range(9, 25, 3)))
plt.axis([-8, 8, -15, 15])

# Discussion:
# From the plot below, we can observe that the predicted curves with degree 15, 18 and 21 have closer fit to the actual curve

########################################################################################################################################

# Problem 2: Regression with Radial Basis Functions

########################################################################################################################################

# 2.a: Radial-basis kernel

########################################################################################################################################

# X float(n, ): univariate data
# B float(n, ): basis functions
# gamma float : standard deviation / scaling of radial basis kernel
import math
def radial_basis_transform(X, B, gamma=0.1):
    rphi_2 = []
    for x in X:
        rphi_1 = []
        for y in B:
            rphi_1.append(math.exp(-gamma*((x - y)**2)))
        rphi_2.append(rphi_1)
    return np.array(rphi_2)

#Testing, to be ignored
#print(radial_basis_transform(X_trn, X_trn))

########################################################################################################################################

# 2.b: Ridge regression

########################################################################################################################################

# Phi float(n, d): transformed data
# y   float(n,  ): labels
# lam float      : regularization parameter
def train_ridge_model(rPhi, y, lam):
    return np.linalg.inv(np.transpose(rPhi)@rPhi + lam*np.identity(len(rPhi[0])))@(np.transpose(rPhi)@y)
 
#Testing, to be ignored
#Phi_trn = radial_basis_transform(X_trn, X_trn)
#print(train_ridge_model(Phi_trn, y_trn, 10))

########################################################################################################################################

# 2.c: Fit and complexity

########################################################################################################################################

rw = {}               # Dictionary to store all the trained models
rvalidationErr = {}   # Validation error of the models
rtestErr = {}         # Test error of all the models
lmbdas = {}           # Dictionary to store Lambda values
llmbdas = {}
lmbda = 0.001         
i = 0
while lmbda <= 1000:  # Iterate
    rPhi_trn = radial_basis_transform(X_trn, X_trn)                # Transform training data into lmbda dimensions
    rw[i] = train_ridge_model(rPhi_trn, y_trn, lmbda)              # Learn model on training data
    
    rPhi_val = radial_basis_transform(X_val, X_trn)                # Transform validation data into lmbda dimensions
    rvalidationErr[i] = evaluate_model(rPhi_val, y_val, rw[i])     # Evaluate model on validation data
    
    rPhi_tst = radial_basis_transform(X_tst, X_trn)                # Transform test data into lmbda dimensions
    rtestErr[i] = evaluate_model(rPhi_tst, y_tst, rw[i])           # Evaluate model on test data
    lmbdas[i] = lmbda
    llmbdas[i] = math.log10(lmbda)
    i += 1
    lmbda *= 10
    
# Plot all the models
plt.figure()
plt.plot(lmbdas.values(), rvalidationErr.values(), marker='o', linewidth=3, markersize=12)
plt.plot(lmbdas.values(), rtestErr.values(), marker='s', linewidth=3, markersize=12)
plt.xlabel('Lambda', fontsize=16)
plt.ylabel('Validation/Test error', fontsize=16)
plt.xticks(list(lmbdas.values()), fontsize=12)
plt.legend(['Validation Error', 'Test Error'], fontsize=16)
plt.axis([-100, 1100, 25, 70])

# Plot all the models with respect to log(lambda) for a clear picture
plt.figure()
plt.plot(llmbdas.values(), rvalidationErr.values(), marker='o', linewidth=3, markersize=12)
plt.plot(llmbdas.values(), rtestErr.values(), marker='s', linewidth=3, markersize=12)
plt.xlabel('Log(Lambda)', fontsize=16)
plt.ylabel('Validation/Test error', fontsize=16)
plt.xticks(list(llmbdas.values()), fontsize=12)
plt.legend(['Validation Error', 'Test Error'], fontsize=16)
plt.axis([-4, 4, 25, 70])

# Discussion:
# From the plot below, 
# We can observe that the error is less when lambda is less 
# Fit is closer for lower values of Lambda such as 0.001 and 0.01
# The error is increasing as lambda increases

########################################################################################################################################

# 2.d: Discussion

########################################################################################################################################

plt.figure()
plt.plot(x_true, y_true, marker='None', linewidth=5, color='k')


for k in range(0,len(lmbdas)):
    rX_d = radial_basis_transform(x_true,X_trn)
    ry_d = rX_d @ rw[k]
    plt.plot(x_true, ry_d, marker='None', linewidth=2)

    
plt.legend(['true'] + list(lmbdas.values()))
plt.axis([-8, 8, -15, 15])

# Discussion:
# As Lambda increases, the sine nature of the function reduces
# and for higher value of Lambda such as 1000, the plot is almost flat

########################################################################################################################################