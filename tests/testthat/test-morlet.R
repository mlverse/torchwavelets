test_that("morlet wavelet", {

  omega <- 6 * pi
  s <- 6
  t_k <- 5
  sample_time <- torch::torch_arange(3, 7, 0.01)
  morlet <- morlet(omega, s, t_k, sample_time)
  expect_equal(torch::torch_argmax(morlet$real), torch::torch_ceil(length(sample_time)/2))

})

test_that("morlet wavelet in Fourier representation", {

  fs <- 8000
  f1 <- 100
  s <- 5
  omega <- 2 * pi * f1
  bin <- f1/fs * 20
  m <- morlet_fourier(s, bin, 1:20)
  expect_equal(dim(m), 20)

})
