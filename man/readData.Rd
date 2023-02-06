\name{readData}
\alias{readData}
\title{
  Read the original datasets
}
\description{
  Read the original datasets to be generated.
}
\usage{
readData(Y_input, X_input, RandomSeed = 99)
}
\arguments{
  \item{Y_input}{
  a data.frame consisting of continuous variables of the original data.
  It should consist only of numerical values.
}
  \item{X_input}{
  a data.frame consisting of categorical variables of the original data. 
  It should consist only of factor-type classes.
}
  \item{RandomSeed}{
  a random seed number
}
}
\value{
  \code{readData} returns an object of "\code{readData_passed}" class.
  
  An object of class "\code{readData_passed}" is a list containing the following components:
  \item{n_sample}{the size of the original datasets.}
  \item{p_Y}{the number of continuous variables.}
  \item{Y_mat_std}{a matrix that standardized \code{Y_input} with mean 0 and standard deviation 1.}
  \item{mean_Y_input}{the mean vectors of original \code{Y_input}.}
  \item{sd_Y_input}{the standard deviation vectors of original \code{Y_input}.}
  \item{NA_Y_mat}{an indicator matrix indicating a missing value (continuous).}
  \item{p_X}{the number of categorical variables}
  \item{D_l_vec}{the maximum number of levels each categorical variables.}
  \item{X_mat_std}{a numerical matrix recoded from categorical dataset.}
  \item{levels_X_input}{a list containing levels of categorical variables in \code{X_input}}
  \item{NA_X_mat}{an indicator matrix indicating a missing value (categorical).}
  \item{var_names}{a list containing variable names of \code{X_input} and \code{Y_input}.}
}
\seealso{
  \code{\link{multipleSyn}}, \code{\link{createModel}}
} 

