#'
#' #' perform wavelet transform on a grid of analysis frequencies
#' #'
#' #' @importFrom torch torch_zeros
#' #' @importFrom torch torch_cfloat
#' #' @importFrom grDevices hcl.colors
#' #' @importFrom graphics image
#' #' @importFrom graphics mtext
#' #' @param x the signal
#' #' @param s the wavelet scale
#' #' @param f_start the lowest analysis frequency to use
#' #' @param f_end the highest analysis frequency to use
#' #' @param fs the sampling rate
#' #'
#' #' @return a list of transform results and analysis frequencies used
#' #' @export
#' wavelet_grid <- function(x, s, f_start, f_end, fs) {
#'   # downsample analysis frequency range
#'   # as per Vistnes, eq. 14.17
#'   num_freqs <- 1 + log(f_end / f_start)/ log(1 + 1/(8 * s))
#'   freqs <- seq(f_start, f_end, length.out = floor(num_freqs))
#'
#'   transformed <- torch_zeros(
#'     num_freqs, dim(x)[1],
#'     dtype = torch_cfloat()
#'   )
#'   for(i in 1:num_freqs) {
#'     w <- wavelet_transform_from_specs("morlet", x, freqs[i], s, fs)
#'     transformed[i, ] <- w
#'   }
#'   list(transformed, freqs)
#' }
#'
#' #' plot a wavelet diagram (scaleogram)
#' #'
#' #' @importFrom torch torch_square
#' #' @importFrom torch torch_sqrt
#' #' @importFrom torch torch_arange
#' #' @importFrom torch nnf_interpolate
#' #' @param x the signal
#' #' @param freqs the analysis frequencies used
#' #' @param grid the wavelet transform, a time-frequency grid
#' #' @param s the wavelet scale used
#' #' @param fs the sampling rate
#' #' @param f_end the highest analysis frequency to use
#' #' @param type what to plot; one of "Magnitude", "Magnitude squared", or "Magnitude (square root)"
#' #' @export
#'
#' plot_wavelet_diagram <- function(x,
#'                                  freqs,
#'                                  grid,
#'                                  s,
#'                                  fs,
#'                                  f_end,
#'                                  type = "magnitude") {
#'   grid <- switch(type,
#'                  magnitude = grid$abs(),
#'                  magnitude_squared = torch_square(grid$abs()),
#'                  magnitude_sqrt = torch_sqrt(grid$abs())
#'   )
#'
#'   # downsample time series
#'   # as per Vistnes, eq. 14.9
#'   new_x_take_every <- max(s / 24 * fs / f_end, 1)
#'   new_x_length <- floor(dim(grid)[2] / new_x_take_every)
#'   new_x <- torch_arange(
#'     x[1],
#'     x[dim(x)[1]],
#'     step = x[dim(x)[1]] / new_x_length
#'   )
#'
#'   # interpolate grid
#'   new_grid <- nnf_interpolate(
#'     grid$view(c(1, 1, dim(grid)[1], dim(grid)[2])),
#'     c(dim(grid)[1], new_x_length)
#'   )$squeeze()
#'   out <- as.matrix(new_grid)
#'
#'   # plot log frequencies
#'   freqs <- log10(freqs)
#'
#'   image(
#'     x = as.numeric(new_x),
#'     y = freqs,
#'     z = t(out),
#'     ylab = "log frequency [Hz]",
#'     xlab = "time [s]",
#'     col = hcl.colors(12, palette = "viridis")
#'   )
#'   main <- paste0("Wavelet Transform, s = ", s)
#'   sub <- switch(type,
#'                 magnitude = "Magnitude",
#'                 magnitude_squared = "Magnitude squared",
#'                 magnitude_sqrt = "Magnitude (square root)"
#'   )
#'
#'   mtext(side = 3, line = 2, at = 0, adj = 0, cex = 1.3, main)
#'   mtext(side = 3, line = 1, at = 0, adj = 0, cex = 1, sub)
#' }
#'
