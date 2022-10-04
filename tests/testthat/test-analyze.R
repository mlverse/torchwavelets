test_that("wavelet grid", {

  fs <- 8000
  f1 <- 100
  s <- 5
  x <- torch::torch_arange(1, 20)

  f_start = 80
  f_end = 120
  grid <- wavelet_grid(x, s, f_start, f_end, fs)

  expect_equal(dim(grid[[1]])[1], length(grid[[2]]))

})
