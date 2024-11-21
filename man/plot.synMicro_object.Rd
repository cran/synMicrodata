\name{plot.synMicro_object}
\alias{plot.synMicro_object}
\title{
Plot Comparing Synthetic Data with Original Input Data
}
\description{
The \code{plot} method for \code{synMicro_object} object. 
This method compares synthetic datasets with original input data.
}
\usage{
\method{plot}{synMicro_object}(x, vars, plot_num = NULL, \dots)
}
\arguments{
  \item{x}{ \code{synMicro_object} object. }
  \item{vars}{ vector of names or indices of the variables to compare. }
  \item{plot_num}{ if \code{plot_num} is a number, returns a plot of the corresponding synthetic datset.}
  \item{...}{ other parameters to be passed through to plotting functions. }
}
\details{
The \code{plot} takes input variables and draws the graph.
The type of graph produced is contingent upon the number of categories in selected variables. 
\itemize{
  \item Putting a continuous variable produces a \emph{box plot} of the selected variable.
  \item Putting more than two continuous variables produces \emph{pairwise scatter plots} for each pair of selected variables.
  \item Putting categorical variables produce \emph{bar plot} of each selected variable.
  }
  
If \code{plot_num=NULL}, the function output plots for all generated synthetic datasets.
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

print(res_obj)

## plotting synthesis datasets
### box plot
par(mfrow=c(3,2))
plot(res_obj, vars = "Sepal.Length") ## variable names


### pairwise scatter plot
plot(res_obj, vars = c(1,2)) ## or variable index


### bar plot
plot(res_obj, vars = "Species")


### specify the synthetic dattaset
par(mfrow=c(1,1))
plot(res_obj, vars = "Petal.Length", plot_num=1)

}