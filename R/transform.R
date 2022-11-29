#######################################################################################
#
# Ported and refactored from:
# https://github.com/QUVA-Lab/PyTorchWavelets/blob/master/wavelets_pytorch/transform.py
# https://github.com/QUVA-Lab/PyTorchWavelets/blob/master/wavelets_pytorch/network.py
# Comments/documentation ported/adapted correspondingly.
#
# In-depth comments are taken from:
# https://github.com/aaren/wavelets/blob/master/wavelets/transform.py
# This is the "master repo" referred to by PyTorchWavelets as the reference
# for Torrence & Compo.
#
#######################################################################################


#' Continuous wavelet transform as described in Torrence and Compo,
#' "A Practical Guide to Wavelet Analysis".
#'
#' @details In contrast to _aaren/wavelets_, but in accordance with further
#' development in _QUVA-Lab/PyTorchWavelets_, this does not use the FFT,
#' but a filter bank consisting of `torch::nn_conv1d()` modules.
#' The code expects input sequences to be one-dimensional. It works with batches
#' of signals.
#'
#' @importFrom torch nn_module
#' @importFrom torch torch_stack
#' @importFrom torch torch_is_complex
#' @importFrom torch nn_module
#' @importFrom torch torch_tensor
#' @importFrom torch torch_cat
#' @importFrom torch torch_reciprocal
#' @importFrom torch nn_conv1d
#'
#' @param signal_length length of the signal to be processed
#' @param dt sample spacing, default is 1
#' @param dj scale distribution parameter, default is 0.125
#' @param wavelet wavelet object
#' @param unbias whether to unbias the power spectrum

wavelet_transform <- nn_module(

  initialize = function(signal_length, dt = 1, dj = 0.125, wavelet = Morlet$new(), unbias = FALSE) {
    self$signal_length <- signal_length
    self$dt <- dt
    self$dj <- dj
    self$wavelet <- wavelet
    self$unbias <- unbias
    self$scale_minimum <- self$compute_minimum_scale()
    self$scales <- self$compute_optimal_scales()
    self$filters <- self$build_filters()
    self$is_complex_wavelet <- torch_is_complex(self$filters[[1]])
    self$convs <- self$filter_bank(self$filters)
  },
  # Determines the optimal scale distribution (see Torrence & Combo, Eq. 9-10), and then initializes
  # the filter bank consisting of rescaled versions of the mother wavelet. Also includes normalization.
  build_filters = function() {
    filters <- list()
    for (i in 1:length(self$scales)) {
      # Number of points needed to capture wavelet
      M <- 10 * self$scales[i] / self$dt
      # Times to use, centered at zero
      t <- torch_arange((-M + 1) / 2, (M + 1) / 2) * self$dt
      if (length(t) %% 2 == 0) {
        t <- t[1:-2] # requires odd filter size
      }
      # Sample wavelet and normalize
      norm <- (self$dt / self$scales[i])^.5
      filters[[i]] <- norm * self$wavelet$time(t, s = self$scales[i])
    }
    filters
  },
  #' Given a list of temporal 1D filters of variable size, this method
  #' creates a list of `nn_conv1d()` objects that collectively form the filter bank.
 filter_bank = function(filters, padding_type = "same") {
    filter_bank <- list()
    chn_out <- if (self$is_complex_wavelet) 2 else 1
    for (i in 1:length(filters)) {
      filt_weights <- if (self$is_complex_wavelet) torch_cat(list(filters[[i]]$real$unsqueeze(1), filters[[i]]$imag$unsqueeze(1)), dim = 1) else filters[[i]]$unsqueeze(1)
      filt_weights <- filt_weights$unsqueeze(2) # append chn_in dimension
      filt_size <- dim(filt_weights)[3]            # filter length
      padding <- self$get_padding(padding_type, filt_size)
      conv <- nn_conv1d(1, chn_out, kernel_size = filt_size, padding = padding, bias = FALSE)
      conv$weight$requires_grad_(FALSE)
      conv$weight[ , , ] <- filt_weights
      filter_bank[[i]] <- conv
    }
    filter_bank
  },
 cwt = function(x) {
   # Takes a batch of signals and convolves each signal with all elements
   # in the filter bank. After convolving the entire filter bank, the method returns
   # a tensor of shape `[N,N_scales,1/2,T]` where the 1/2 number of channels depends
   # on whether the filter bank is composed of real or complex filters. If the filters
   # are complex the 2 channels represent `[real, imag]` parts.

   results <- list()
   for (i in 1:length(self$convs)) {
     results[[i]] <- self$convs[[i]](x)
   }
   results <- torch_stack(results)        # [n_scales,n_batch,2,t]
   results <- results$permute(c(2,1,3,4)) # [n_batch,n_scales,2,t]
   results
 },
  forward = function(x) {
    if (x$ndim == 1) {
      # Append batch_size and chn_in dimensions
      # [signal_length] => [n_batch,1,signal_length]
      x <- x$unsqueeze(1)$unsqueeze(1)
    } else if (x$ndim == 2) {
      # Just append chn_in dimension
      # [n_batch,signal_length] => [n_batch,1,signal_length]
      x <- x$view(c(dim(x)[1], 1, dim(x)[2]))
    }
    num_examples <- x$shape[1]
    signal_length <- x$shape[-1]

    cwt <- self$cwt(x)

    if (self$is_complex_wavelet) {
    # Combine real and imag parts, returns object of shape
    # [n_batch,n_scales,signal_length] of `type np.complex128`torch_complex()`
      cwt <- cwt[ , , 1, ] * torch_complex(1, 0) + cwt[ , , 2, ] * torch_complex(0, 1)
    } else {
    # Just squeeze the chn_out dimension (=1) to obtain an object of shape
    # [n_batch,n_scales,signal_length] of type np.float64
      cwt = cwt$squeeze(3)
    }
    # Squeeze batch dimension if single example
    if (num_examples == 1) {
      cwt <- cwt$squeeze(1)
    }
    cwt
  },
  # Determines the optimal scale distribution (see Torrence & Combo, Eq. 9-10).
  compute_optimal_scales = function() {
    J <- floor((1 / self$dj) * log2(self$signal_length * self$dt / self$scale_minimum))
    scales <- self$scale_minimum * 2^(self$dj * torch_arange(0, J))
    scales
  },
  # Performs CWT and converts to a power spectrum (scalogram). See Torrence & Combo, Section 4d.
  #' @param x `torch_tensor()`, batch of input signals of shape [n_batch,signal_length]
  #' @return a `torch_tensor()`, scalogram for each signal [n_batch,n_scales,signal_length]
  power = function() {
    if (isTRUE(self$unbias)) (torch_abs(self$cwt(x))$T^2 / self$scales)$T else torch_abs(self$cwt(x))^2
  },
  get_padding = function(padding_type, kernel_size) {
    if (padding_type == "same") floor((kernel_size - 1) / 2) else 0
  },
  # Choose s0 so that the equivalent Fourier period is 2 * dt. See Torrence & Combo Sections 3f and 3h.
  compute_minimum_scale = function() {
    dt <- self$dt
    f <- self$wavelet$fourier_period
    func_to_solve <- function(s) {
      f(s) - 2 * dt
    }
    uniroot(func_to_solve, c(0,10))$root
  },
  fourier_period = function(s) {
    self$wavelet$fourier_period(s)
  },
  scale_from_period = function(p) {
    self$wavelet$scale_from_period(p)
  },
  fourier_periods = function() {
    self$fourier_period(self$scales)
  },
  fourier_frequencies = function() {
    torch_reciprocal(self$fourier_periods())
  },
  complex_wavelet = function(value) {
    if (missing(value)) {
      torch_is_complex(private$.filters[0])
    } else {
      stop("Can't change wavelet properties.")
    }
  },
  output_dtype = function(value) {
    if (missing(value)) {
      if (self$complex_wavelet(private$.wavelet)) torch_cfloat else torch_get_default_dtype()
    } else {
      stop("Can't change wavelet properties.")
    }
  }
)

