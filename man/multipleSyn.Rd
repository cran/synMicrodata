\name{multipleSyn}
\alias{multipleSyn}
\title{
Generate synthetic micro datasets
}
\description{
Generate synthetic micro datasets using hierarchically coupled mixture model with 
  local dependence (HCMM-LC).
}
\usage{
multipleSyn(data_obj, model_obj, n_burnin, m, interval_btw_Syn, show_iter = TRUE)
}
\arguments{
  \item{data_obj}{
  a data object passed \code{readData}
}
  \item{model_obj}{
  a model object passed \code{createModel}
}
  \item{n_burnin}{
  the size of burn-in
}
  \item{m}{
  number of synthetic micro data to be generated
}
  \item{interval_btw_Syn}{
  the size of interval between synthetic micro datasets
}
  \item{show_iter}{
  a logical value. If \code{TRUE}, \code{multipleSyn} will print 
  history of \code{(r,s,k)} components on console.
}
}

\value{
  \code{multipleSyn} returns a list containing the following conmponents:
  \item{Synt_Y_list }{a list of \code{m} continuous synthetic micro dataset(s).}
  \item{Synt_X_list }{a list of \code{m} categorical synthetic micro dataset(s).}
  \item{r_i_cube}{a matrix of the mixture component index for continuous variables.}
  \item{s_i_cube}{a matrix of the mixture component index for categorical variables.}
  \item{k_i_cube}{a matrix of the mixture component index for continuous & categorical structures.}
}
\references{
Murray, J. S. and Reiter, J. P. (2016). 
  Multiple imputation of missing categorical and continuous values via 
  Bayesian mixture models with local dependence. 
  \emph{Journal of the American Statistical Association}, \bold{111(516)}, pp.1466-1479.
}

\seealso{
  \code{\link{readData}}, \code{\link{createModel}}
}
\examples{
## preparing to generate synthetic datasets
dat_obj <- readData(iris[,1:4], iris[,5])
mod_obj <- createModel(dat_obj)

## generating synthetic datasets
syn_results <- multipleSyn(dat_obj, mod_obj, 
    n_burnin=100, m=5, interval_btw_Syn = 50, show_iter=FALSE)

head(syn_results$Synt_Y_list[[1]])
head(syn_results$Synt_X_list[[1]])

## table of mixture component index in each synthetic datasets
apply(syn_results$r_i_cube, MARGIN=1, table)
apply(syn_results$s_i_cube, MARGIN=1, table)
apply(syn_results$k_i_cube, MARGIN=1, table)

}

