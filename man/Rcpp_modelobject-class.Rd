\name{Rcpp_modelobject-class}
\Rdversion{1.1}
\docType{class}
\alias{Rcpp_modelobject-class}
\alias{Rcpp_modelobject}

\title{Class \code{"Rcpp_modelobject"}}
\description{
This class implements a joint modeling approach to generate synthetic microdata with continuous and categorical variables with possibly missing values. The method builds on the work of Murray and Reiter (2016)
}
\section{Extends}{
Class \code{"\linkS4class{C++Object}"}, directly.
}

\keyword{classes}

\section{Fields}{
  \itemize{
    \item \code{data_obj} input dataset generated from \code{\link{readData}}.
	}
}

\section{Methods}{
  \itemize{
    \item \code{multipleSyn} generates synthetic micro datasets.
  }
}

\details{
Rcpp_modelobject should be created with \code{\link{createModel}}. Please see the example below. 
}

\references{
Murray, J. S. and Reiter, J. P. (2016). 
  Multiple imputation of missing categorical and continuous values via 
  Bayesian mixture models with local dependence. 
  \emph{Journal of the American Statistical Association}, \bold{111(516)}, pp.1466-1479.
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