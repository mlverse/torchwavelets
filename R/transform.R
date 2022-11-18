#' Wavelet transform
#'
#' @details
#' Continuous Wavelet Transform as described in
#' Torrence & Combo, A Practical Guide to Wavelet Analysis (BAMS, 1998)
#'
#' @param dt float, sample spacing
#' @param dj float, scale distribution parameter
#' @param wavelet wavelet object
#' @param unbias logical, whether to unbias the power spectrum
#' @cuda logical whether to run on cuda
#'
#' @export
WaveletTransform <- R6::R6Class(
  "WaveletTransform",
  lock_objects = FALSE,
  public = list(
    initialize = function(dt = 1, dj = 0.125, wavelet = Morlet$new(), unbias = FALSE, cuda = FALSE) {
      self$dt <- dt
      self$dj <- dj
      self$wavelet <- wavelet
      self$unbias <- unbias
      self$cuda <- cuda
      self$scale_minimum <- self$compute_minimum_scale()
      self$signal_length <- NULL  # initialize on first call
      self$scales  <- NULL        # initialize on first call
      self$filters <- NULL        # initialize on first call
      self$extractor <- filterBank(self$filters, cuda)
    }
  ),
  #' @description Implements the continuous wavelet transform on a batch of signals.
  #' All signals in the batch must have the same length, otherwise manual zero
  #' padding has to be applied. On the first call, the signal length is used to
  #' determines the optimal scale distribution and uses this for initialization
  #' of the wavelet filter bank. If there is only one example in the batch the
  #' batch dimension is squeezed.
  #' @param x `torch_tensor()`, batch of signals of shape `[n_batch,signal_length]`
  #' @return `torch_tensor()`, CWT for each signal in the batch `[n_batch,n_scales,signal_length]`
  cwt <- function(x) {
    if (x$ndim  == 1) {
      # Append batch_size and chn_in dimensions
      # [signal_length] => [n_batch,1,signal_length]
      x <- x$unsqueeze(1)$unsqueeze()
    } else if (x$ndim  == 2) {
      # Just append chn_in dimension
      # [n_batch,signal_length] => [n_batch,1,signal_length]
      x <- x$view(dim(x)[1], 1, dim(x)[2])
    }
    num_examples  <- x$shape[1]
    signal_length <- x$shape[-1]

    if (signal_length != self.signal_length || is_null(self.$filters)) {
      # First call initializtion, or change in signal length. Note that calling
      # this also determines the optimal scales and initialized the filter bank.
      self$signal_length <- signal_length
    }
    # Move to GPU and perform CWT computation
    x$requires_grad_(requires_grad = FALSE)
    if (is_true(self$cuda)) x <- x$cuda()
    cwt <- self$extractor(x)

    # Move back to CPU
    cwt <- cwt$detach()
    if (is_true(self$cuda)) cwt <- cwt$cpu()

    #  ### is this still necessary????????????????
    #if (is_true(self$complex_wavelet)) {
      # Combine real and imag parts, returns object of shape
      # [n_batch,n_scales,signal_length] of `type np.complex128`torch_complex()`
      # cwt <- cwt[:,:,0,:] + cwt[:,:,1,:] * torch_complex(0, 1)
    #} else {
      # Just squeeze the chn_out dimension (=1) to obtain an object of shape
      # [n_batch,n_scales,signal_length] of type np.float64
    #  cwt = np.squeeze(cwt, 2).astype(self.output_dtype)
    #}
    # Squeeze batch dimension if single example
    if (num_examples == 1) {
      cwt = cwt.squeeze(0)
    }
    cwt
  },
  #' @description Determines the optimal scale distribution (see. Torrence & Combo, Eq. 9-10),
  #' and then initializes the filter bank consisting of rescaled versions
  #' of the mother wavelet. Also includes normalization.
  build_filters <- function() {
    self$scale_minimum <- self$compute_minimum_scale()
    self$scales <- self$compute_optimal_scales()
    self$filters <- list()
    for (i in length(self$scales)) {
      # Number of points needed to capture wavelet
      M <- 10 * scale / self$dt
      # Times to use, centred at zero
      t <- torch_arange((-M + 1) / 2, (M + 1) / 2) * self$dt
      if (length(t) %% 2 == 0) {
        t = t[0:-1]  # requires odd filter size
      }
      # Sample wavelet and normalise
      norm <- (self$dt / scale)^.5
      self$filters[scale_idx] <- norm * self$wavelet(t, scale)
    }
  },
  #' @description Determines the optimal scale distribution (see. Torrence & Combo, Eq. 9-10).
  #' @return np.ndarray, collection of scales
  compute_optimal_scales <- function() {
    if (is_null(self$signal_length)) stop("Please specify signal_length before computing optimal scales.")
    J <- ceiling((1 / self$dj) * log2(self$signal_length * self$dt / self$scale_minimum))
    scales <- self$scale_minimum * 2^(self$dj * torch_arange(0, J + 1))
    scales
  },
  #' Choose s0 so that the equivalent Fourier period is 2 * dt.
  #' See Torrence & Combo Sections 3f and 3h.
  #' @return float, minimum scale level
  compute_minimal_scale <- function() {
    dt <- self$dt
    func_to_solve <- function(s) {
      self$fourier_period(s) - 2 * dt
    }
    ## tbd scipy.optimize.fsolve(func_to_solve, 1)[0]

  },
  #' @description Performs CWT and converts to a power spectrum (scalogram).
  #' See Torrence & Combo, Section 4d.
  #' @param x `torch_tensor()`, batch of input signals of shape [n_batch,signal_length]
  #' @return a `torch_tensor()`, scalogram for each signal [n_batch,n_scales,signal_length]
  power <- function() {
    ifelse(isTRUE(self$unbias), (torch_abs(self$cwt(x))$T^2 / self$scales)$T, torch_abs(self$cwt(x))^2)
  }

)
