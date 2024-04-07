\name{createModel}
\alias{createModel}
\title{
  Create a model object
}
\description{
  Create a model object for \code{multipleSyn}.
}
\usage{
createModel(data_obj, max_R_S_K = c(30, 50, 20))
}
\arguments{
  \item{data_obj}{
  data object produced by \code{readData}
}
  \item{max_R_S_K}{
  maximum value of the number of mixture component index (r, s, k).
}
}
\value{
  \code{createModel} returns a \code{\link{Rcpp_modelobject}}
}
\seealso{
  \code{\link{multipleSyn}}, \code{\link{readData}}
}

