\name{summary.synMicro_object}
\alias{summary.synMicro_object}
\title{
  Summarizing synthesis results
}
\description{
  \code{summary} method for class "\code{summary.synMicro_object}".
}
\usage{
\method{summary}{synMicro_object}(object, max_print = 4, \dots)
}
\arguments{
  \item{object}{ \code{synMicro_object} object. }
  \item{max_print}{ maximum number of synthetic datset to print summaries}
  \item{\dots}{ other parameters to be passed through to other functions. }
}
\details{
  \code{summary} reports the synthesis results for each variable.
  \code{summary} reports the synthesis results for each variable. It compares the summary statistics of each variable for the original dataset(\code{Orig.}) and synthetic datasets(\code{synt.#}), their averaging(\code{Q_bar}), and between variance(\code{B_m}).
}

\seealso{
  \code{\link{multipleSyn}}
}
\examples{
## preparing to generate synthetic datsets
dat_obj <- readData(Y_input = iris[,1:4],
                    X_input = data.frame(Species = iris[,5]))
mod_obj <- createModel(dat_obj, max_R_S_K=c(30,50,20))

## generating synthetic datasets
res_obj <- multipleSyn(dat_obj, mod_obj, n_burnin = 100, m = 2, 
                       interval_btw_Syn = 50, show_iter = FALSE)

summary(res_obj)
}
