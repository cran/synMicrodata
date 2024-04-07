\name{readData}
\alias{readData}
\title{
  Read the original datasets
}
\description{
  Read the original input datasets to be learned for synthetic data generation. The package allows the input data to have missing values and impute them with the posterior predictive distribution, so no missing values exist in the synthetic data output.
}
\usage{
readData(Y_input, X_input, RandomSeed = 99)
}
\arguments{
  \item{Y_input}{
  data.frame consisting of continuous variables of the original data.
  It should consist only of \code{numeric}.
}
  \item{X_input}{
  data.frame consisting of categorical variables of the original data. 
  It should consist only of \code{factor}.
}
  \item{RandomSeed}{
  random seed number.
}
}
\value{
  \code{readData} returns an object of "\code{readData_passed}" class.
  
  An object of class "\code{readData_passed}" is a list containing the following components:
  \item{n_sample}{number of records in the input dataset.}
  \item{p_Y}{number of continuous variables.}
  \item{Y_mat_std}{matrix with standardized values of \code{Y_input}, with mean 0 and standard deviation 1.}
  \item{mean_Y_input}{mean vectors of original \code{Y_input}.}
  \item{sd_Y_input}{standard deviation vectors of original \code{Y_input}.}
  \item{NA_Y_mat}{matrix indicating missing values in \code{Y_input}.}
  \item{p_X}{number of categorical variables.}
  \item{D_l_vec}{numbers of levels of each categorical variable.}
  \item{X_mat_std}{matrix with the numeric-transformed values of \code{X_input}.}
  \item{levels_X_input}{list of levels of each categorical variable.}
  \item{NA_X_mat}{matrix indicating missing values in \code{X_input}.}
  \item{var_names}{list containing variable names of \code{X_input} and \code{Y_input}.}
  \item{orig_data}{original dataset.}
}
\seealso{
  \code{\link{multipleSyn}}, \code{\link{createModel}}
} 

