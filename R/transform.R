
#' The wavelet transform, computed in the Fourier domain
#' Expects that the wavelet has already been constructed.
#'
#' @param x the signal, a vector (in the time domain)
#' @param m the wavelet to use (in Fourier representation)
#' @return the wavelet transform, in the time domain
#' @export
wavelet_transform_fourier <- function(x, m) {

  x_fft <- torch::torch_fft_fft(x)
  prod <- x_fft * m
  w <- torch::torch_fft_ifft(prod)
  w
}

#' The wavelet transform, computed in the Fourier domain
#' Constructs the wavelet for a desired scale and analysis frequency.
#'
#' @param type the type of wavelet. Currently has to be "morlet".
#' @param x the signal, a vector (in the time domain)
#' @param omega_a the analysis frequency to be used, in Hertz.
#' @param s the scale parameter to be used with the wavelets to be constructed.
#' @param fs the sampling frequency in Hertz.
#' @return the wavelet transform, in the time domain
#' @export
wavelet_transform_from_specs <- function(type, x, omega_a, s, fs) {
  N <- dim(x)[1]
  omega_bin <- omega_a / fs * N
  m <- morlet_fourier(s, omega_bin, 1:N)
  x_fft <- torch::torch_fft_fft(x)
  prod <- x_fft * m
  w <- torch::torch_fft_ifft(prod)
  w
}
