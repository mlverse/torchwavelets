test_that("wavelet transform in Fourier domain", {

  fs <- 8000
  f1 <- 100
  s <- 5
  omega <- 2 * pi * f1
  bin <- f1/fs * 20
  m <- morlet_fourier(s, bin, 1:20)

  x <- torch_arange(1, 20)

  expect_equal(dim(wavelet_transform_fourier(x, m)), 20)

})

test_that("wavelet transform from specs", {

  fs <- 8000
  f1 <- 100
  s <- 5
  omega <- 2 * pi * f1

  x <- torch_arange(1, 20)

  expect_equal(dim(wavelet_transform_from_specs("morlet", x, omega, s, fs)), 20)

})
