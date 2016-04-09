# ==========================================================
# class of useful linear regression function
# used in computing ordinaly linear regression
# all matrices will be stored as list of lists(columns)
# author: Rui Zhang
#
# vector: array
# matrix: array of arrays (i.e. matrix columns)
# ==========================================================

def vectorL2Norm(x):
    """ compute the norm of a vector
    Args:
        x: Array
    return:
        float
    """
    # calculate the mean and norms of each column
    return (sum([xElem ** 2 for xElem in x])) ** 0.5


def vectorAdd(x, y):
    """ compute addition of 2 vectors
    Args:
        x: array
        y: array
    return:
        array
    """
    # check vector size
    assert len(x) == len(y), 'vectorAdd: 2 array do not have same size.'
    # compute addition and return
    return [xElem + yElem for xElem, yElem in zip(x, y)]


def vectorInnerProd(x, y):
    """ compute inner product of 2 vectors
    Args:
        x: array
        y: array
    return:
        float
    """
    # check vector size
    assert len(x) == len(y), 'vectorInnerProd: 2 array do not have same size.'
    # compute inner product and return inner product
    return sum([xElem * yElem for xElem, yElem in zip(x, y)])


def vectorProject(x, y):
    """ compute the project of vector y onto vector x
    Args:
        x: array
        y: array
    return:
        factor: float
        projection: array
    """
    # check vector size
    assert len(x) == len(y), 'vectorInnerProd: 2 array do not have same size.'
    # compute inner product and return inner product
    factor = vectorInnerProd(x, y) / vectorInnerProd(x, x)
    return factor, vectorScaleProd(factor, x)


def matrixAdd(X, Y):
    """ compute addition of 2 matrix, X+Y
    Args:
        X: matrix
        Y: matrix
    return:
        matrix
    """
    # check matrix shape: column
    assert len(X) == len(Y), \
        'matrixAdd: 2 matrices do not have same column number.'
    # compute sum and return
    # check matrix shape: row
    return [vectorAdd(xCol, yCol) for xCol, yCol in zip(X, Y)]


def matrixProd(X, Y):
    """ compute product of 2 matrices: x.T * y
    Args:
        X: matrix
        Y: matrix
    return:
        matrix, X.T * Y
    """
    # compute product and return
    return [[vectorInnerProd(xCol, yCol) for xCol in X] for yCol in Y]


def vectorScaleProd(c, x):
    """ compute product of vector x and scale c: c * x
    Args:
        c: float
        x: vector
    return:
        vector
    """
    return [c * xElem for xElem in x]


def matrixTranspose(X):
    """ compute the transpose of a given matrix
    Args:
        X: matrix
    return:
        matrix
    """
    nbRow = len(X[0])
    return [[xCol[iRow] for xCol in X] for iRow in range(nbRow))]


def matrixNormalization(X):
    """ compute the normalized matrix of a given matrix, and record the standardized factors
    Args:
        x: matrix
    return:
        Array[means (of each matrix column), norms (of each matrix column), normalized matrix]
    """
    nbRow = float(len(X[0]))
    # calculate the norms of each column
    colNorms = [vectorL2Norm(xCol) / (nbRow ** 0.5) for xCol in X]
    # normalize the matrix by each column
    return colNorms, [vectorScaleProd(xCol, 1.0/colNorm)
                      for xCol, colNorm in zip(X, colNorms)]


def matrixOrthogonal(X):
    """ compute an orthogonalized version of a normalized matrix
    Args:
        X: matrix
    return:
        Array[operator: array, orthogonalized matrix: matrix]
    """
    X_orthogonal = []
    coefficients0 = []
    for ils in range(1, len(X)):
        X_temp = X[ils]
        coef0 = []
        for ilt in range(0, ils):
            slope, project_temp = vectorProject(X_orthogonal[ilt], X_temp)
            coef0.append(-1.0 * slope)
            X_temp = [X_temp_iter - project_temp_iter for X_temp_iter,project_temp_iter in zip(X_temp, project_temp)]
        coefficients.append(coef0)
        X_orthogonal.append(X_temp)
    return [coefficients, X_orthogonal]


def matrixOrthogonal(X):
    """ compute an orthogonalized version of a normalized matrix
    Args:
        X: matrix
    return:
        Array[operator: array, orthogonalized matrix: matrix]
    """
    XOrthogonal = []
    coefficients = []
    for xCol in X:
        xResidual = xCol
        coefficient = []
        for xOrthognoalCol in X_orthogonal:
            # compute prejection
            slope, projection = vectorProject(xOrthogonalCol, xResidual)
            coefficient.append(-slope)
            # remove prejection from x
            xResidual = vectorAdd(xCol, vectorScaleProd(-1, projection))
            coefficients.append(coefficient)
        X_orthogonal.append(xResidual)
    return coefficients, XOrthogonal


def inverseHat(covar):
    """ compute the (X^TX)^{-1} given the covariate matrix X by Gram Schmidt method
    Args:
        covar: matrix containing predictors, each column contains a single predictor observations
    return:
        (X^TX)^{-1}: matrix
    """
    assert len(covar[0]) == len(outcome), 'inverseHat: dependent and predictors do not have same length.'
    assert len(covar[0]) == len(outcome), 'GramSchmidt: dependent and predictors do not have same length.'
    n_obs = len(outcome)
    n_covar = len(covar)
    ######## perform standardization ########
    norms_covar, Z = matrixNormalization(covar)
    ######## orthogonalize covar columns one by one ########
    coefficients0, Z_orthogonal = matrixOrthogonal(Z)
    ######## check for collinearity #######
    Z_orthogonal_norms = [vectorNorm(Z) for Z in Z_orthogonal]
    assert min(Z_orthogonal_norms)/n_obs > 1e-5, 'There is high collinearity among the covariates, please use Lasso, Ridge or Elastic net to fit the linear model.'
    ######## find the QR decomposition corresponding operators ########
    transformR = []
    for ils in range(n_covar):
        list_temp = [0.0] * n_covar
        list_temp[ils] = 1.0
        if ils > 0:
            list_temp[:ils] = coefficients0[ils - 1]
        transformR.append(list_temp)
    transformR0 = []
    for ils in range(len(covar)):
        if ils < 2:
            transformR0.append(transformR[ils])
        else:
            list_temp = [0.0] * n_covar
            for ilt in range(ils):
                list_temp = [temp * coefficients0[ils - 1][ilt] + list0 for temp, list0 in zip(transformR0[ilt], list_temp)]
            list_temp[ils] = 1.0
            transformR0.append(list_temp)
    transformR0t = matrixTranspose(transformR0)
    ######## find the inverse hat matrix ############
    matrixP1 = []
    for index in range(len(covar)):
        temp = [0.0] * len(covar)
        temp[index] = 1.0/norms_covar[index]
        matrixP1.append(temp)
    matrixP2 = transformR0
    matrixP0 = matrixProd(matrixTranspose(matrixP1),matrixP2)
    all_orthogonal_norms_inv = [1/vectorNorm(X) for X in Z_orthogonal]
    matrixP0_normalized = [[temp1 * temp2 for temp1 in temp_list] for temp_list,temp2 in zip(matrixP0, all_orthogonal_norms_inv)]
    inverse_hat = matrixProd(matrixTranspose(matrixP0_normalized), matrixTranspose(matrixP0_normalized))
    return inverse_hat
