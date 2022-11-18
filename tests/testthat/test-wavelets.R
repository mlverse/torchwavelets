library(torch)

test_that("morlet", {

  batch <- torch_arange(0, 7)

  m <- Morlet$new()
  expect_equal(m$w0, 6)

  x <- m$time(batch)
  y <- torch::torch_tensor(c(0.7511255330253157, 0.43743501749900315, 0.08578095013434033, 0.005509848274365431, 0.00010688231024455357, 4.317782084774805e-07, -1.4638570310887878e-09, -6.8792610280671006e-12))
  expect_true(torch::torch_allclose(x$real, y))
  y <- torch::torch_tensor(c(0.0, -0.12729630043984794, -0.054544669817373145, -0.0062664261398689275, -0.0002281826993787734, -2.765682701861861e-06, -1.1345579979228781e-08, -1.576305551635654e-11))
  expect_true(torch::torch_allclose(x$imag, y))

  x <- m$fourier_period(s = 1)
  y <- 1.0330436477492537
  expect_equal(x, y)

  x <- m$scale_from_period(m$fourier_period(s = 1))
  y <- 1
  expect_equal(x, y)

  x <- m$frequency(w = pi)
  y <- 0.012633178008637279
  expect_equal(x, y)

  x <- m$coi(s = 1)
  y <- 1.4142135623730951
  expect_equal(x, y)

})

test_that("Paul", {
  batch <- torch_arange(0, 7)

  p <- Paul$new()
  expect_equal(p$m, 4)

  x <- p$time(batch)
  y <- torch::torch_tensor(c(1.0789368501515768, -0.1348671062689471, 0.014155651473988687, 0.0034094404464789824, 0.000851838043563484, 0.00026116692048428863, 9.523789448268436e-05, 3.976011765230579e-05))
  expect_true(torch::torch_allclose(x$real, y))
  y <- torch::torch_tensor(c(0.0, -0.1348671062689471, -0.013119872097843174, -0.00012947242201818922, 0.00030699604781413695, 0.0001725372562309278, 8.78472720550949e-05, 4.630624330954543e-05))
  expect_true(torch::torch_allclose(x$imag, y, atol = 1e-6))

  x <- p$fourier_period(s = 1)
  y <- 1.3962634015954636
  expect_equal(x, y)

  expect_error(p$scale_from_period(m$fourier_period(s = 1)))

  x <- p$frequency(w = pi)
  y <- 0.47434885401835936
  expect_equal(x, y, tolerance = 1e-7)

  x <- p$coi(s = 1)
  y <- 0.7071067811865475
  expect_equal(x, y)

})

test_that("Derivative of Gaussian", {
  batch <- torch_arange(0, 7)

  d <- DerivativeOfGaussian$new()
  expect_equal(d$m, 2)

  x <- d$time(batch)
  y <- torch::torch_tensor(c(8.67325071e-01, -0.00000000e+00, -3.52139052e-01, -7.70808897e-02, -4.36432721e-03, -7.75732734e-05, -4.62327014e-07, -9.53253330e-10))
  expect_true(torch::torch_allclose(x, y))

  x <- d$fourier_period(s = 1)
  y <- 3.97383530631844
  expect_equal(x, y)

  expect_error(p$scale_from_period(m$fourier_period(s = 1)))

  x <- d$frequency(w = pi)
  y <- 0.06156363866852903
  expect_equal(x, y, tolerance = 1e-7)

  x <- d$coi(s = 1)
  y <- 1.4142135623730951
  expect_equal(x, y)

})

test_that("Mexican Hat", {
  batch <- torch_arange(0, 7)

  m <- MexicanHat$new()
  expect_equal(m$m, 2)

  d <- DerivativeOfGaussian$new()

  x1 <- m$time(batch)
  x2 <- d$time(batch)
  expect_true(torch::torch_allclose(x1, x2))

})


