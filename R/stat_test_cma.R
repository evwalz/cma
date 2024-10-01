# Load necessary libraries
library(MASS)      # For mvrnorm
library(matrixStats)
library(pracma)    # For combn
library(stats)     # For rank and norm functions
library(rms)
# Function: Kernel P
kernel_p <- function(x_rank, y_rank, rho) {
  N <- length(y_rank)
  
  G_x <- 1/N * (x_rank - 0.5)
  G_y <- 1/N * (y_rank - 0.5)
  
  s_x <- sign(outer(x_rank, x_rank, "-"))
  s_y <- sign(outer(y_rank, y_rank, "-"))
  
  mean_product <- apply(s_x, 1, function(row_x) colMeans(row_x * t(s_y)))
  G_x_helper <- matrix(1, N , N)*G_x
  G_y_helper <- matrix(1, N , N)*G_y
  all_exp <- mean_product + 2 * G_x_helper + t(2 * G_y_helper) - 1
  
  g_1 <- colMeans(all_exp) / 4
  g_2 <- rowMeans(all_exp) / 4
  k_p <- 4 * (g_1 + g_2 + G_x * G_y - G_y - G_x) + 1 - rho
  
  return(k_p)
}

# Function: Comp Rho CMA
comp_rho_cma <- function(y_rank, x_rank) {
  N <- length(y_rank)
  mean_rank <- (N + 1) / 2
  var_y <- sum((y_rank - mean(y_rank))^2) / (N - 1)
  rho <- (12 / (N^2)) * (1 / (N - 1)) * sum((x_rank - mean_rank) * (y_rank - mean_rank))
  return(list(rho = rho, cmas = (cov(y_rank, x_rank) / var_y + 1) / 2))
}

# Function: One Dimensional Test
one_dim_test <- function(y_rank, x_rank) {
  N <- length(y_rank)
  
  zeta_3Y <- zeta_fun(y_rank)
  k_zeta <- prob_y(y_rank)^2 - zeta_3Y
  sigma_zeta <- 9 * mean(k_zeta^2)
  
  res <- comp_rho_cma(y_rank, x_rank)
  rho <- res$rho
  cmas <- res$cmas
  
  factor <- 1 / (1 - zeta_3Y)^2
  
  k_p <- kernel_p(x_rank, y_rank, rho)
  sigma_rho <- 9 * mean(k_p^2)
  sigma_pz <- 9 * mean(k_p * k_zeta)
  var <- factor * (sigma_rho + (2 * rho * sigma_pz) / (1 - zeta_3Y) + (rho^2 * sigma_zeta) / (1 - zeta_3Y)^2)
  sd_2 <- var / (4 * N)
  phalf <- 1 - pnorm((cmas - 0.5) / sqrt(var / (4 * N)))
  
  return(list(cmas = cmas, sd = sqrt(sd_2), p = phalf))
}

# Function: Probability of y
prob_y <- function(y) {
  unique_vals <- unique(y)
  counts <- table(y)
  probabilities <- counts / length(y)
  p <- probabilities[match(y, unique_vals)]
  return(p)
}

# Function: Zeta Function


zeta_fun2 <- function(y){
  N <- length(y)
  bin_N_3 = 6 / (N*(N-1)*(N-2))
  zeta_y = 0
  for (i in 1:(N-2)){
    for (j in (i+1):(N-1)){
      for (k in (j+1):N){
        if (y[i] == y[j] && y[i] == y[k]){
          zeta_y = zeta_y + 1
        }
      }
    }
  }
  zeta_y = bin_N_3* zeta_y
  return(zeta_y)
}

zeta_fun<- function(y) {
  N <- length(y)
  if (N < 3) return(0)
  
  # Calculate binomial coefficient
  bin_N_3 <- 6 / (N * (N - 1) * (N - 2))
  
  # Count the occurrences of each value in y
  counts <- table(y)
  
  # Only consider values that appear at least 3 times
  triplet_count <- sum(choose(counts[counts >= 3], 3))
  
  # Calculate the zeta value
  zeta_y <- bin_N_3 * triplet_count
  
  return(zeta_y)
}

# Function: Sigma
Sigma <- function(y_rank, xarray_ranks) {
  N <- length(y_rank)
  k <- nrow(xarray_ranks)
  
  zeta_3Y <- zeta_fun(y_rank)
  k_zeta <- prob_y(y_rank)^2 - zeta_3Y
  sigma_zeta <- 9 * mean(k_zeta^2)
  
  rhos <- numeric(k)
  cmas <- numeric(k)
  
  for (j in 1:k) {
    res <- comp_rho_cma(y_rank, xarray_ranks[j, ])
    rhos[j] <- res$rho
    cmas[j] <- res$cmas
  }
  
  factor <- 1 / (1 - zeta_3Y)^2
  phalf <- numeric(k)
  S <- matrix(0, nrow = k, ncol = k)
  
  for (j in 1:(k-1)) {
    k_p <- kernel_p(xarray_ranks[j, ], y_rank, rhos[j])
    sigma_rho <- 9 * mean(k_p^2)
    sigma_pz <- 9 * mean(k_p * k_zeta)
    var <- factor * (sigma_rho + (2 * rhos[j] * sigma_pz) / (1 - zeta_3Y) + (rhos[j]^2 * sigma_zeta) / (1 - zeta_3Y)^2)
    S[j, j] <- var / (4 * N)
    phalf[j] <- 1 - pnorm((cmas[j] - 0.5) / sqrt(var / (4 * N)))
    
    for (i in (j + 1):k) {
      k_p2 <- kernel_p(xarray_ranks[i, ], y_rank, rhos[i])
      sigma_rho2 <- 9 * mean(k_p * k_p2)
      sigma_pz2 <- 9 * mean(k_p2 * k_zeta)
      var <- factor * (sigma_rho2 + (rhos[j] * sigma_pz) / (1 - zeta_3Y) + (rhos[i] * sigma_pz2) / (1 - zeta_3Y) + (rhos[j] * rhos[i] * sigma_zeta) / (1 - zeta_3Y)^2)
      S[j, i] <- S[i, j] <- var / (4 * N)
    }
  }
  
  k_p <- kernel_p(xarray_ranks[k, ], y_rank, rhos[k])
  sigma_rho <- 9 * mean(k_p^2)
  sigma_pz <- 9 * mean(k_p * k_zeta)
  var <- factor * (sigma_rho + (2 * rhos[k] * sigma_pz) / (1 - zeta_3Y) + (rhos[k]^2 * sigma_zeta) / (1 - zeta_3Y)^2)
  S[k, k] <- var / (4 * N)
  phalf[k] <- 1 - pnorm((cmas[k] - 0.5) / sqrt(var / (4 * N)))
  
  cmas_pd <- data.frame(CMA = cmas, SD = sqrt(diag(S)), P = phalf)
  
  return(list(cmas = cmas, S = S, cmas_pd = cmas_pd))
}

# Class: Test Multiple
test_multiple <- function(cmas, differences, covariance, global_p, global_z) {
  list(
    cmas = cmas,
    differences = differences,
    covariance = covariance,
    global_z = global_z,
    global_p = global_p
  )
}

# Class: Test One
test_one <- function(cmas, sd, p) {
  list(
    cmas = cmas,
    sd = sd,
    p = p
  )
}

# Function: Calculate P-value using Chi-Square
calc_pvalue_chi_our <- function(aucs, S) {
  nauc <- length(aucs)
  
  L <- matrix(0, nrow = nauc * (nauc - 1) / 2, ncol = nauc)
  
  newa <- 0
  for (i in 1:(nauc - 1)) {
    newl <- nauc - i
    L[(newa + 1):(newa + newl), i] <- 1
    L[(newa + 1):(newa + newl), (i + 1):(i + newl)] <- -diag(newl)
    newa <- newa + newl
  }
  
  aucdiff <- L %*% aucs
  L_S_Lt <- L %*% S %*% t(L)
  
  # Solve the system
  L_S_Lt_inv <- rms::matinv(L_S_Lt)
  
  z <- t(aucdiff) %*% L_S_Lt_inv %*% aucdiff
  rank <- qr(L_S_Lt)$rank
  
  p <- pchisq(z, df = rank, lower.tail = FALSE)
  return(list(z = z, p = p, aucdiff = aucdiff))
}

# Function: Pairwise Testing
pairwise_testing_our <- function(nauc, S, aucdiff, conf_level) {
  cor_auc <- numeric(nauc * (nauc - 1) / 2)
  ci <- matrix(0, nrow = nauc * (nauc - 1) / 2, ncol = 2)
  pairp <- numeric(nauc * (nauc - 1) / 2)
  
  quantil <- qnorm(1 - (1 - conf_level) / 2)
  
  rows <- character(nauc * (nauc - 1) / 2)
  ctr <- 1
  
  for (i in 1:(nauc - 1)) {
    for (j in (i + 1):nauc) {
      cor_auc[ctr] <- S[i, j] / sqrt(S[i, i] * S[j, j])
      LSL <- c(1, -1) %*% S[c(j, i), c(j, i)] %*% c(1, -1)
      tmpz <- aucdiff[ctr] / sqrt(LSL)
      pairp[ctr] <- pchisq(aucdiff[ctr]^2 / LSL, df = 1, lower.tail = FALSE)
      ci[ctr, ] <- c(aucdiff[ctr] - quantil * sqrt(LSL), aucdiff[ctr] + quantil * sqrt(LSL))
      rows[ctr] <- paste(i, "vs.", j)
      ctr <- ctr + 1
    }
  }
  
  return(data.frame(Test = rows, `CMA diff` = aucdiff, `CI(lower)` = ci[, 1], `CI(upper)` = ci[, 2], p.value = pairp, correlation = cor_auc))
}

# Main Function: CMA Test
cma_test <- function(y, x, conf_level = 0.95) {
  if (length(dim(x)) == 0) {
    y_ranks <- rank(y)
    x_ranks <- rank(x)
    res <- one_dim_test(y_ranks, x_ranks)
    return(test_one(cmas = res$cmas, sd = res$sd, p = res$p))
  } else {
    y_ranks <- rank(y)
    xarray_ranks <- t(apply(x, 1, rank))
    
    res <- Sigma(y_ranks, xarray_ranks)
    
    global_test <- calc_pvalue_chi_our(res$cmas, res$S)
    pairwise_test <- pairwise_testing_our(length(res$cmas), res$S, global_test$aucdiff, conf_level)
    
    return(test_multiple(cmas = res$cmas_pd, differences = pairwise_test, covariance = res$S, global_z = global_test$z, global_p = global_test$p))
  }
}