
#' Complex Morlet wavelet, centred at zero.
#'
#' @details
#' The standard version is:
#' \deqn{pi**-0.25 * exp(1j*w*x) * exp(-0.5*(x**2))}
#' This commonly used wavelet is often referred to simply as the Morlet wavelet.
#' Note that this simplified version can cause admissibility problems at low values
#' of `w`.
#' The complete version is:
#' \deqn{pi**-0.25 * (exp(1j*w*x) - exp(-0.5*(w**2))) * exp(-0.5*(x**2))}
#' The complete version of the Morlet wavelet, with a correction term to improve
#' admissibility. For `w` greater than 5, the correction term is negligible
#' Note that the energy of the return wavelet is not normalised according to `s`.
#' The fundamental frequency of this wavelet in Hz is given by `f = 2*s*w*r / M`
#' where r is the sampling rate.
#'
#' @importFrom torch torch_complex
#' @param w0 the nondimensional frequency constant. If this is set too low
#' then the wavelet does not sample very well: a value over 5 should be ok;
#' Terrence and Compo set it to 6.
#'
#' @seealso scipy.signal.gausspulse
#' @export
Morlet <- R6::R6Class(
  "Morlet",
  lock_objects = FALSE,
  public = list(
    initialize = function(s = 1, complete = TRUE, w0 = 6) {
      self$w0 <- w0
      if (w0 == 6) self$C_d <- 0.776   # value of C_d from TC98
    },
    #' @description Value of the wavelet at the given time
    #' @param t Time. If s is not specified, this can be used as the non-dimensional time t/s.
    #' @param s Scaling factor. Default is 1.
    #' @param complete Whether to use the complete or the standard version.
    time = function(t, s = 1, complete = TRUE) {
      w <- self$w0
      x <- t / s
      output <- torch_exp(torch_complex(0, 1) * w * x)
      if (isTRUE(complete)) {
        output <- output - torch_exp(-0.5 * (w^2))
      }
      output <- pi^(-0.25) * output * torch_exp(-0.5 * x^2)
      output
    },
    #' @description Equivalent Fourier period of Morlet
    #' @param s Scaling factor
    fourier_period = function(s) {
      4 * pi * s / (self$w0 + (2 + self$w0^2)^.5)
    },
    #' @description Compute the scale from the fourier period
    #' @param period Fourier period
    scale_from_period = function(period) {
      # Solve 4 * pi * scale / (w0 + (2 + w0 ** 2) ** .5)
      # for s to obtain this formula
      coeff <- sqrt(self$w0 * self$w0 + 2)
      (period * (coeff + self$w0)) / (4 * pi)
    },
    #' @description Frequency representation of Morlet
    #' @param w angular frequency. If s is not specified, i.e. set to 1,
    #' this can be used as the non-dimensional angular frequency w * s.
    #' @param s the scaling factor. Default is 1.
    frequency = function(w, s = 1) {
      x <- w * s
      # Heaviside mock
      Hw <- ifelse(w <= 0, 0, 1)
      f <- pi^-.25 * Hw * torch_exp((-(x - self$w0)^2) / 2)
      as.numeric(f)
    },
    #' @description The e folding time for the autocorrelation of wavelet
    #' power at each scale, i.e. the timescale over which an edge
    #' effect decays by a factor of 1/e^2.
    #' @param s Scaling factor
    coi = function(s) {
      # This can be worked out analytically by solving
      # |Y_0(T)|^2 / |Y_0(0)|^2 = 1 / e^2
      2^.5 * s
    }
  )
)

#' Derivative of Gaussian wavelet of order `m`
#'
#' @details
#' When m = 2, this is also known as the "Mexican hat", "Marr" or "Ricker" wavelet.
#' It models the function:
#' ``A d^m/dx^m exp(-x^2 / 2)``,
#' where ``A = (-1)^(m+1) / (gamma(m + 1/2))^.5`` and   ``x = t / s``.
#' Note that the energy of the return wavelet is not normalised according to `s`.
#'
#' @note The derivative of the Gaussian has a polynomial representation:
#' from http://en.wikipedia.org/wiki/Gaussian_function:
#' "Mathematically, the derivatives of the Gaussian function can be represented
#' using Hermite functions. The n-th derivative of the Gaussian is the Gaussian
#' function itself multiplied by the n-th Hermite polynomial, up to scale."
#' http://en.wikipedia.org/wiki/Hermite_polynomial
#' Here, we want the 'probabilists' Hermite polynomial (He_n),
#' which is computed by scipy.special.hermitenorm
#'
#' @param m the order
#'
#' @export
DerivativeOfGaussian <- R6::R6Class(
  "Derivative of Gaussian",
  lock_objects = FALSE,
  public = list(
    initialize = function(m) {
      if (m == 2) {
        # value of C_d from TC98
        self$C_d <- 3.541
      } else if (m == 6) {
        self$C_d <- 1.966
      }
      self$m <- m
    },
    #' @description Value of the wavelet at the given time
    #' @param t Time. If s is not specified, this can be used as the non-dimensional time t/s.
    #' @param s Scaling factor. Default is 1.
    time = function(t, s) {
      x <- t / s
      m <- self$m
      # compute the Hermite polynomial (used to evaluate the derivative of a Gaussian)
      h <- calculus::hermite(m)[[m]]$f
      eval(parse(text = paste('He_n <- function(x) {', h, '}', sep = '')))
      output <- He_n(x) * torch_exp(-x^2 / 2)
      const <- (-1)^(m + 1) / gamma(m + 0.5)^.5
      output <- output * const
      output
    },
    #' @description Equivalent Fourier period of wavelet
    #' @param s Scaling factor
    fourier_period = function(s) {
      2 * np.pi * s / (self$m + 0.5) ** .5
    },
    #' @description Compute the scale from the fourier period
    #' @param period Fourier period
    scale_from_period = function(period) {
      stop("not implemented")
    },
    #' @description Frequency representation of wavelet
    #' @param w angular frequency. If s is not specified, i.e. set to 1,
    #' this can be used as the non-dimensional angular frequency w * s.
    #' @param s the scaling factor. Default is 1.
    frequency = function(w, s = 1) {
      m <- self$m
      x <- w * s
      const = -torch_complex(0, 1)^m / gamma(m + 0.5)^.5
      output <- x^m * torch_exp(-x^2 / 2)
      output <- output * const
      as.numeric(output)
    },
    #' @description The e folding time for the autocorrelation of wavelet
    #' power at each scale, i.e. the timescale over which an edge
    #' effect decays by a factor of 1/e^2.
    #' @param s Scaling factor
    coi = function(s) {
      # This can be worked out analytically by solving
      # |Y_0(T)|^2 / |Y_0(0)|^2 = 1 / e^2
      2^.5 * s
    }
  )
)

#' Paul wavelet
#'
#' The Paul wavelet is defined as
#' \deqn{(2 ** m * i ** m * m!) / (pi * (2 * m)!) * (1 - i * t / s) ** -(m + 1)}
#'
#' @param m the order
#'
#' @export
Paul <- R6::R6Class(
  "Paul",
  lock_objects = FALSE,
  public = list(
    initialize = function(m = 4) {
      self$m <- m
    },
    #' @description Value of the wavelet at the given time
    #' @param t Time. If s is not specified, this can be used as the non-dimensional time t/s.
    #' @param s Scaling factor. Default is 1.
    #' @param complete Whether to use the complete or the standard version.
    time = function(t, s = 1) {
      m <- self$m
      x <- t / s
      const <- (2^m * torch_complex(0, 1)^m * factorial(m)) /
        (pi * factorial(2 * m))^.5
      functional_form <- (1 - torch_complex(0, 1) * x)^-(m + 1)
      output <- const * functional_form
      output
    },
    #' @description Equivalent Fourier period of wavelet
    #' @param s Scaling factor
    fourier_period = function(s) {
      4 * pi * s / (2 * self$m + 1)
    },
    #' @description Compute the scale from the fourier period
    #' @param period Fourier period
    scale_from_period = function(period) {
      stop("not implemented")
    },
    #' @description Frequency representation of wavelet
    #' @param w angular frequency. If s is not specified, i.e. set to 1,
    #' this can be used as the non-dimensional angular frequency w * s.
    #' @param s the scaling factor. Default is 1.
    frequency = function(w, s = 1) {
      m <- self$m
      x <- w * s
      # Heaviside mock
      Hw <- 0.5 * (sign(x) + 1)
      # prefactor
      const <- 2^m / (m * factorial(2 * m - 1))^ .5
      functional_form <- Hw * (x)^m * torch_exp(-x)
      output <- const * functional_form
      as.numeric(output)
    },
    #' @description The e folding time for the autocorrelation of wavelet
    #' power at each scale, i.e. the timescale over which an edge
    #' effect decays by a factor of 1/e^2.
    #' @param s Scaling factor
    coi = function(s) {
      # This can be worked out analytically by solving
      # |Y_0(T)|^2 / |Y_0(0)|^2 = 1 / e^2
      s / 2^.5
    }
  )
)

#' Mexican Hat
#'
#' @details
#' This is a derivative of a Gaussian of order 2.
#'
#' @export
MexicanHat <- R6::R6Class(
  "Mexican Hat",
  inherit = "DerivativeOfGaussian",
  public = list(
    initialize = function(m) {
      super()$initialize(m)
      self$C_d = 3.541
    }
  )
)

