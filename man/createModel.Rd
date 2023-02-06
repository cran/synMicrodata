\name{createModel}
\alias{createModel}
\title{
  Create model object
}
\description{
  Create model object for \code{multipleSyn}.
}
\usage{
createModel(data_obj, max_R_S_K = c(30, 50, 20))
}
\arguments{
  \item{data_obj}{
  a data object passed \code{readData}
}
  \item{max_R_S_K}{
  a maximum value of the number of mixture component index (r, s, k).
}
}
\value{
  \code{createModel} returns a \code{\link{Rcpp_modelobject}}
}
\seealso{
  \code{\link{multipleSyn}}, \code{\link{readData}}
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

