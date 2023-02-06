\name{Rcpp_modelobject-class}
\Rdversion{1.1}
\docType{class}
\alias{Rcpp_modelobject-class}
\alias{Rcpp_modelobject}

\title{Class \code{"Rcpp_modelobject"}}
\description{
This class implements the MCMC sampler for a joint modeling approach to multiple edit-imputation for continuous data. It provides methods for updating and monitoring the sampler. 
}
\section{Extends}{
Class \code{"\linkS4class{C++Object}"}, directly.
}

\keyword{classes}

\section{Fields}{
  \itemize{
    \item \code{data_obj}: input dataset generated from \code{\link{readData}}.
	}
}

\section{Methods}{
  \itemize{
    \item \code{multipleSyn}: Generate synthetic micro datasets.
  }
}

\details{
Rcpp_modelobject objects should be created with \code{\link{createModel}}. Please see the example below. 
}

\references{
Hang J. Kim, Lawrence H. Cox, Alan F. Karr, Jerome P. Reiter and Quanli Wang (2015). "Simultaneous Edit-Imputation for Continuous Microdata", Journal of the American Statistical Association, DOI: 10.1080/01621459.2015.1040881.
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