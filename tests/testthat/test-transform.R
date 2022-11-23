test_that("simple transform", { # https://github.com/QUVA-Lab/PyTorchWavelets/blob/master/examples/simple_example.py

  dt <- 0.1               # sampling frequency
  dj <- 0.125             # scale distribution parameter
  batch_size <- 32

  t <- torch::torch_linspace(0, 10, floor(10/dt))
  frequencies <- torch::torch_tensor(runif(batch_size, -0.5, 2.0))
  batch <- torch::torch_zeros(batch_size, length(t))
  for (f in 1:length(frequencies)) {
    batch[f, ] <- torch::torch_sin(2 * pi * frequencies[f] * t)
  }

  wavelet <- Morlet$new()
  wtf <- WaveletTransform$new(dt, dj, wavelet)

  # test methods called in the constructor
  x <- wtf$compute_minimum_scale()
  y <- 1
  expect_equal(x, y)


  # test accessors
  x <- wtf$dt
  expect_equal(x, dt)
  expect_error({
    wtf$dj <- 1
  })
  expect_error({
    wtf$wavelet <- MexicanHat$new()
  })





  expect_equal(x, dt)

})
