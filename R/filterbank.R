
#' Temporal filter bank in PyTorch storing a collection of nn.Conv1d filters.
#'
#' @details When cuda=True, the convolutions are performed on the GPU. If initialized with filters=None,
#' the set_filters() method has to be called before actual running the convolutions.
#'
#' @importFrom torch nn_module
#' @importFrom torch torch_stack
#' @importFrom torch torch_is_complex
#'
#'
#' @param filters list, collection of variable sized 1D filters (default: `NULL`)
#' @param cuda logical, whether to run on GPU or not (default: FALSE)

filterBank <- torch::nn_module(

  initialize = function(filters = NULL, cuda = FALSE) {
    self$filters <- ifelse (is_null(filters), list(), self$set_filters(filters))
    self$cuda <- cuda
  },
  forward = function(x) {
    #' @description Takes a batch of signals and convolves each signal with all elements
    #' in the filter bank. After convolving the entire filter bank, the method returns
    #' a tensor of shape `[N,N_scales,1/2,T]` where the 1/2 number of channels depends
    #' on whether the filter bank is composed of real or complex filters. If the filters
    #' are complex the 2 channels represent `[real, imag]` parts.
    #' @param x torch.Variable, batch of input signals of shape `[N,1,T]`
    #' @return torch.Variable, batch of outputs of size `[N,N_scales,1/2,T]`
    if (is_null(self$filters)) stop('torch filters not initialized. Please call set_filters() first.')
    results <- list()
    for (i in length(self$filters)) {
      results[i] <- self$filters[i](x)
    }
    results <- torch_stack(results)        # [n_scales,n_batch,2,t]
    results <- results$permute(c(1,0,2,3)) # [n_batch,n_scales,2,t]
    results
  },
  #' @description Given a list of temporal 1D filters of variable size, this method
  #' creates a list of `nn_conv1d()` objects that collectively form the filter bank.
  #' @param filters list, collection of filters each a `torch_tensor`
  #' @param padding_type character, should be `"same"` or `"valid"`
  set_filters = function(filters) {
    for (i in length(filters)) {
      if (any(unlist(Map(torch_is_complex, ll)))) {
        chn_out <- 2
        filt_weights <- torch_tensor(list(self$filters[i]$real, self$filters[i]$imag))
      } else {
        chn_out <- 1
        filt_weights <- self$filters[i]$unsqueeze(1)# filt.astype(np.float32)[None,:]
      }
      filt_weights <- filt_weights$unsqueeze(3)# np.expand_dims(filt_weights, 1)  # append chn_in dimension
      filt_size <- filt_weights$ndim              # filter length
      padding <- self$get_padding(padding_type, filt_size)
      conv <- nn_conv1d(1, chn_out, kernel_size = filt_size, padding = padding, bias = FALSE)
      conv$weight$data <- filt_weights
      conv$weight$requires_grad_(FALSE)
      if (isTRUE(self$cuda)) conv$cuda()
      self$filters[i] <- conv
    }
  },
  get_padding = function(padding_type) {
    ifelse(padding_type == "same", floor((kernel_size - 1) / 2), 0)
  }
)

