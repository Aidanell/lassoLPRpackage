#'Perform Lasso Local Polynomial Regression
#' 
#'
#'This function performs Lasso LPR on a given x and y. Lasso LPR uses
#'the LASSO (Least Absolute Shrinkage Selection Operator) via glmnet with local
#'polynomial regression in order to automatically shrink small derivative
#'estimates to 0.
#'
#'At each grid point, a local polynomial is fit with a penalty parameter lambda,
#'calculated with cv.glmnet's coordinate descent algorithm.
#'This returns a regularized estimate of the first function and its p derivatives at that point.
#'With the appropriate parameters, Lasso LPR will maintain
#'the function estimate while significantly improving derivative estimates.
#'Due to the randomness of obtaining optimal lambda via cv.glmnet, lassoLPR
#'will lowess smooth the lambdas and refit at each grid point. This significantly
#'reduces "spikes" in the estimates which come from cv.glmnet's nondeterministic optimization.
#'
#'
#'@param x vector of covariate observations.
#'@param y vector of response observations.
#'@param h bandwidth parameter. Distance from estimate point where data will be factored into calculation.
#'@param p degree of local polynomials.
#'@param nlambda number of lambdas tried for each polynomial estimate. More lambdas can significantly increase compute time
#'@return A list containing a numGridPoints by (p+1) matrix where element (i,j) is the (j-1)th derivative estimate at grid Point i. This list also contains a vector of lambdas which show how the penalty parameter changed across x.
#'@export
lassoLPR <- function(x, y, h, p=10, nlambda=25, numGridPoints=401){
  
  #All uniformly distributed points where polynomials are estimated
  gridPoints <- seq(min(x), max(x), length.out=numGridPoints)
  
  
  lassoOutput <- matrix(nrow=numGridPoints, ncol=p+1) #ith column shows estimates of (i-1)th derivative 
  lambdas <- numeric(numGridPoints) #Tracks lambda parameter over x
  
  
  #Estimate lasso polynomial at every gridpoint
  for(i in 1:numGridPoints){
    
    currentPoint <- gridPoints[i]
    X <- buildFeature(currentPoint, p, x)
    lassoWeights <- computeWeights(x, currentPoint, h)
    
    #Performing lasso regression, finding optimal lambda
    lassoFit <- glmnet::cv.glmnet(X, y, weights = lassoWeights, maxit=10**7, nlambda=nlambda)
    lambdas[i] <- lassoFit$lambda.min
    
    # Selects the desired coefficients with the best lambda
    lassoCoef <- coef(lassoFit, s="lambda.min")
    #The i'th row contains all estimated p+1 derivatives estimation at the i'th gridpoint
    lassoOutput[i, ] <- as.vector(lassoCoef) 
  }
  
  
  #cv.glmnet has randomness to it, causing jagged estimates. By smoothing the lambdas with lowess
  #across x, a much smoother estimate can be found. The regression at each gridpoint
  #is refit with the appropriate smoothed lambda.
  
  smoothLambdas <- stats::lowess(gridPoints, lambdas, f=1/10)$y
  smoothLassoOutput <- matrix(nrow=401, ncol=p+1)
  
  #Refit regression with smoothed lambdas
  for(i in 1:length(gridPoints)){
    currentPoint <- gridPoints[i]
    
    X <- buildFeature(currentPoint, p, x)
    lassoWeights <- computeWeights(x, currentPoint, h)
    lassoFit <- glmnet::glmnet(X, y, weights = lassoWeights, maxit=10**7)
    
    # Selects the desired coefficients with the best lambda
    smoothLassoCoef <- coef(lassoFit, s=smoothLambdas[i])
    
    #The i'th row contains all estimated p+1 derivatives estimation at the i'th gridpoint
    smoothLassoOutput[i, ] <- as.vector(smoothLassoCoef) 
  }
  
  return(list(lasso=smoothLassoOutput, lambdas=smoothLambdas))
}

#Helper Methods----
# This function builds our feature matrix.
buildFeature <- function(midpoint, p, xValues){
  # Every j'th column is the coefficient of the (j-1)'th derivative in the
  # Taylor polynomial formed at the i'th point.
  
  X <- matrix(nrow = length(xValues), ncol = p)
  
  for(i in 1:p){
    X[,i] <- ((xValues - midpoint)^(i))/factorial(i)
  }
  return(X)
}


#Function of the Epanechnikov kernel
epan <- function(x, bandwidth){
  ifelse(abs(x) <= 1, 3/4 * (1 - x^2), 0)
}


#Computes the scaled kernel weights for every input x
computeWeights <- function(x, midpoint, bandwidth, epan=TRUE){
  weights <- numeric(length(x))
  
  for(i in 1:length(x)){
    diff <- abs(x[i]-midpoint) / bandwidth
    if(epan){
      if(diff > 1){weights[i] <- 0}
      else{weights[i] <- epan(diff)/bandwidth}
    }else{
      weights[i] <- dnorm(diff)
    }
  }
  return(weights)
}








