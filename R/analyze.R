
#' perform wavelet transform on a grid of analysis frequencies
#'
#' @param x
#' @param m
#' @param f_start
#' @param f_end
#'
#' return tbd
wavelet_grid <- function(x, m, f_start, f_end) {
  # downsample analysis frequency range
  # as per Vistnes, eq. 14.17
  num_freqs <- 1 + log(f_end / f_start)/ log(1 + 1/(8 * K))
  freqs <- seq(f_start, f_end, length.out = floor(num_freqs))

  # transformed <- torch_zeros(
  #   num_freqs, dim(x)[1],
  #   dtype = torch_cfloat()
  # )
  # for(i in 1:num_freqs) {
  #   w <- wavelet_transform_fourier(x, freqs[i], K, fs)
  #   transformed[i, ] <- w
  # }
  list(transformed, freqs)
}

# TBD:
# - pass in wavelet, too
# - what to iterate over?
# - test
# - plotting
# - test for plotting
# - chaffinch example (README?)
