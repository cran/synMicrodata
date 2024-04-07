\name{multipleSyn}
\alias{multipleSyn}
\alias{print.synMicro_object}
\title{
Generate synthetic micro datasets
}
\description{
Generate synthetic micro datasets using a hierarchically coupled mixture model with 
  local dependence (HCMM-LC).
}
\usage{
multipleSyn(data_obj, model_obj, n_burnin, m, interval_btw_Syn, show_iter = TRUE)

\method{print}{synMicro_object}(x, \dots)
}
\arguments{
  \item{data_obj}{
  data object produced by \code{readData}.
}
  \item{model_obj}{
  model object produced by \code{createModel}.
}
  \item{n_burnin}{
  size of burn-in.
}
  \item{m}{
  number of synthetic micro datasets to be generated.
}
  \item{interval_btw_Syn}{
  interval between MCMC iterations for generating synthetic micro datasets.
}
  \item{show_iter}{
  logical value. If \code{TRUE}, \code{multipleSyn} will print 
  history of \code{(r,s,k)} components on console.
}
  \item{x}{
  object of class \code{synMicro_object}; a result of a call to \code{multipleSyn()}.
  }
  \item{...}{
  further arguments passed to or from other methods.
  }
}

\value{
  \code{multipleSyn} returns a list of the following conmponents:
  \item{synt_data}{list of \code{m} synthetic micro datasets.}
  \item{comp_mat}{list of matrices of the mixture component indices.}
  \item{orig_data}{original dataset.}
  
}
\references{
Murray, J. S. and Reiter, J. P. (2016). 
  Multiple imputation of missing categorical and continuous values via 
  Bayesian mixture models with local dependence. 
  \emph{Journal of the American Statistical Association}, \bold{111(516)}, pp.1466-1479.
}

\seealso{
  \code{\link{readData}}, \code{\link{createModel}}, \code{\link{plot.synMicro_object}}
}
\examples{
## preparing to generate synthetic datsets
dat_obj <- readData(Y_input = iris[,1:4],
                    X_input = data.frame(Species = iris[,5]))
mod_obj <- createModel(dat_obj, max_R_S_K=c(30,50,20))

## generating synthetic datasets
res_obj <- multipleSyn(dat_obj, mod_obj, n_burnin = 100, m = 5, 
                       interval_btw_Syn = 50, show_iter = FALSE)

print(res_obj)
}

