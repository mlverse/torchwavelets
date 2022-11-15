
#' The Morlet wavelet, in the time domain
#'
#' \deqn{\Psi_{\omega_{a},s,t_{k}}(t_n) = (e^{-i \omega_{a} (t_n - t_k)} - e^{-s^2}) \ e^{- \omega_a^2 (t_n - t_k )^2 /(2s )^2}}
#'
#' See Vistnes, Physics of Oscillations and Waves, eq. (14.8).
#'
#' @importFrom torch torch_exp
#' @importFrom torch torch_square
#' @param omega the analysis frequency to be used (a scalar, in Hertz).
#' @param s scale parameter for this wavelet (a scalar).
#' @param t_k location parameter for this wavelet (a scalar).
#' @param t a vector of measurement times.
#' @return a vector of wavelet values (complex).
#' @export
morlet <- function(omega, s, t_k, t) {
  (torch_exp(-1i * omega * (t - t_k)) -
     torch_exp(-torch_square(s))) *
    torch_exp(-torch_square(omega) * torch_square(t - t_k) /
                       torch_square(2 * s))
}



#' The Morlet wavelet, in Fourier domain
#'
#' \deqn{\hat{\Psi}_{\omega_{a},s}(\omega) = 2(e^{-(s (\omega - \omega_a)/\omega_a)^2}) - e^{-s^2} \ e^{- (s \omega/\omega_a)^2}}
#'
#' See Vistnes, Physics of Oscillations and Waves, eq. (14.12).
#'
#' @importFrom torch torch_exp
#' @importFrom torch torch_square
#' @param s scale parameter for this wavelet (a scalar).
#' @param omega_a the analysis frequency to be used (a scalar). Must be the DFT bin, not a frequency in Hertz.
#' @param omega a vector of frequencies to be considered.  Must be the DFT bin, not a frequency in Hertz.
#' @export
morlet_fourier <- function(s, omega_a, omega) {
  2 * (torch_exp(-torch_square(
    s * (omega - omega_a) / omega_a
  )) -
    torch_exp(-torch_square(s)) *
    torch_exp(-torch_square(s * omega / omega_a)))
}

