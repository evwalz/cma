
compute_cma <- function(y, x){
  x_rank <- rank(x, ties.method = "average")
  y_rank <- rank(y, ties.method = "average")
  return(0.5*(cov(y_rank, x_rank) / cov(y_rank, y_rank) + 1))
}
