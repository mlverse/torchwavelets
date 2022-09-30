
#' The wavelet transform, computed in the Fourier domain
#'
#' @param x the signal, a vector (in the time domain)
#' @param m the wavelet to use (in Fourier representation)
#' @return the wavelet transform, in the time domain
wavelet_transform_fourier <- function(x, m) {

  x_fft <- torch::torch_fft_fft(x)
  prod <- x_fft * m
  w <- torch::torch_fft_ifft(prod)
  w
}
