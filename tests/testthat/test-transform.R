test_that("wavelet_transform fields and methods (using Morlet wavelet)", {
  # expected values computed from
  # https://github.com/QUVA-Lab/PyTorchWavelets/blob/master/examples/simple_example.py

  dt <- 0.1
  dj <- 0.125
  batch_size <- 32

  t <- torch::torch_linspace(0, 10, floor(10/dt))
  frequencies <- torch::torch_tensor(runif(batch_size, -0.5, 2.0))
  batch <- torch::torch_zeros(batch_size, length(t))
  for (f in 1:length(frequencies)) {
    batch[f, ] <- torch::torch_sin(2 * pi * frequencies[f] * t)
  }

  wavelet <- Morlet$new()
  wtf <- wavelet_transform(dim(batch)[2], dt, dj, wavelet)

  ### test access to wavelet methods ###
  x <- wtf$scale_from_period(wtf$fourier_period(s = 1))
  y <- wavelet$scale_from_period(wavelet$fourier_period(s = 1))
  expect_equal(x, y)


  ### test computed fields ###
  x <- wtf$scale_minimum
  y <- 0.1936027
  expect_equal(x, y, tolerance = 1e-7)

  x <- wtf$scales
  y <- torch::torch_tensor(c(0.19360266, 0.2111252 , 0.23023366, 0.25107159, 0.27379551,
        0.29857612, 0.32559957, 0.35506885, 0.38720532, 0.4222504 ,
        0.46046733, 0.50214318, 0.54759102, 0.59715224, 0.65119914,
        0.71013769, 0.77441065, 0.8445008 , 0.92093465, 1.00428636,
        1.09518204, 1.19430448, 1.30239827, 1.42027539, 1.54882129,
        1.6890016 , 1.8418693 , 2.00857272, 2.19036408, 2.38860897,
        2.60479655, 2.84055078, 3.09764259, 3.3780032 , 3.68373861,
        4.01714544, 4.38072816, 4.77721793, 5.2095931 , 5.68110156,
        6.19528518, 6.75600639, 7.36747721, 8.03429087, 8.76145632,
        9.55443587))
  expect_true(torch::torch_allclose(x, y))

  x <- wtf$fourier_periods()
  y <- torch::torch_tensor(c(0.2       , 0.21810155, 0.23784142, 0.25936791, 0.28284271,
                             0.30844217, 0.33635857, 0.36680162, 0.4       , 0.43620309,
                             0.47568285, 0.51873582, 0.56568542, 0.61688433, 0.67271713,
                             0.73360323, 0.8       , 0.87240619, 0.95136569, 1.03747164,
                             1.13137085, 1.23376866, 1.34543426, 1.46720647, 1.6       ,
                             1.74481237, 1.90273138, 2.07494329, 2.2627417 , 2.46753732,
                             2.69086853, 2.93441294, 3.2       , 3.48962474, 3.80546277,
                             4.14988657, 4.5254834 , 4.93507464, 5.38173706, 5.86882588,
                             6.4       , 6.97924949, 7.61092554, 8.29977315, 9.0509668 ,
                             9.87014928))
  expect_true(torch::torch_allclose(x, y))


  x <- wtf$fourier_frequencies()
  y <- torch::torch_tensor(c(5.        , 4.58502022, 4.20448208, 3.85552706, 3.53553391,
                             3.24209889, 2.97301779, 2.72626933, 2.5       , 2.29251011,
                             2.10224104, 1.92776353, 1.76776695, 1.62104944, 1.48650889,
                             1.36313467, 1.25      , 1.14625505, 1.05112052, 0.96388177,
                             0.88388348, 0.81052472, 0.74325445, 0.68156733, 0.625     ,
                             0.57312753, 0.52556026, 0.48194088, 0.44194174, 0.40526236,
                             0.37162722, 0.34078367, 0.3125    , 0.28656376, 0.26278013,
                             0.24097044, 0.22097087, 0.20263118, 0.18581361, 0.17039183,
                             0.15625   , 0.14328188, 0.13139006, 0.12048522, 0.11048543,
                             0.10131559))
  expect_true(torch::torch_allclose(x, y))

  ### test filters ###
  x <- wtf$build_filters()
  expect_equal(length(x), dim(wtf$scales)[1])
  x7r <- x[[7]]$real
  x7i <- x[[7]]$imag
  y7r = torch::torch_tensor(c(-2.29166061e-06, -7.08678266e-06,  5.18883299e-05, -2.17701978e-06,
                              -5.74560869e-04,  9.14609562e-04,  3.09507857e-03, -9.77803273e-03,
                              -4.74770664e-03,  4.74982141e-02, -2.90830556e-02, -1.15620252e-01,
                              1.64668398e-01,  1.15358557e-01, -3.55056884e-01,  5.38344647e-02,
                              3.81645308e-01, -2.43216348e-01, -1.92086012e-01,  2.39836912e-01,
                              1.39179477e-02, -1.12975440e-01,  3.01868772e-02,  2.63401590e-02,
                              -1.45634517e-02, -2.14933757e-03,  3.01939095e-03, -2.77525564e-04,
                              -3.14268673e-04,  7.85374253e-05,  1.43859297e-05, -7.29850356e-06,
                              7.54546487e-08))
  y7i = torch::torch_tensor(c(2.37992103e-06, -1.20308152e-05, -1.38241432e-05,  1.87910148e-04,
                              -1.67423620e-04, -1.47351440e-03,  3.36703963e-03,  4.98388955e-03,
                              -2.34909386e-02,  3.45140068e-03,  8.10560886e-02, -8.19246428e-02,
                              -1.33812101e-01,  2.65106548e-01,  4.94814646e-02, -4.00893268e-01,
                              1.63812558e-01,  3.02366422e-01, -2.68247628e-01, -8.74051242e-02,
                              1.79183098e-01, -2.22479297e-02, -5.99610318e-02,  2.39715911e-02,
                              9.14035740e-03, -7.24151938e-03, -5.00280209e-05,  1.06294194e-03,
                              -1.83017752e-04, -7.63826116e-05,  2.63625918e-05,  1.69022933e-06,
                              -1.69892494e-06))
  expect_true(torch::torch_allclose(x7r, y7r))
  expect_true(torch::torch_allclose(x7i, y7i, atol = 1e-7))

  x <- unlist(Map(function(t) dim(t)[1], x))
  y <- c(19, 21, 23, 25, 27, 29, 33, 35, 39, 43, 47, 51, 55, 59, 65, 71, 77, 85, 93, 101, 109, 119, 131, 143, 155, 169, 185, 201, 219, 239, 261, 285, 309, 337, 369, 401, 439, 477, 521, 569, 619, 675, 737, 803, 877, 955)
  expect_equal(x, y)

  ### test conv modules ###
  convs <- wtf$filter_bank(wtf$build_filters())

  x <- unlist(Map(function(c) c$kernel_size, convs))
  y <- c(19,  21,  23,  25,  27,  29,  33,  35,  39,  43,  47,  51,  55,
         59,  65,  71,  77,  85,  93, 101, 109, 119, 131, 143, 155, 169,
         185, 201, 219, 239, 261, 285, 309, 337, 369, 401, 439, 477, 521,
         569, 619, 675, 737, 803, 877, 955)
  expect_equal(x, y)

  x <- unlist(Map(function(c) c$padding, convs))
  y <- c(9,  10,  11,  12,  13,  14,  16,  17,  19,  21,  23,  25,  27,
         29,  32,  35,  38,  42,  46,  50,  54,  59,  65,  71,  77,  84,
         92, 100, 109, 119, 130, 142, 154, 168, 184, 200, 219, 238, 260,
         284, 309, 337, 368, 401, 438, 477)
  expect_equal(x, y)

  x <- as.numeric(convs[[46]]$weight$sum())
  y <- -5.8038e-06
  expect_equal(x, y, tolerance = 1e-6)

  x <- as.numeric(convs[[7]]$weight$std())
  y <- 0.1240
  expect_equal(x, y, tolerance = 1e-4)

  ### test cwt ###
  b <- batch$view(c(dim(batch)[1], 1, dim(batch)[2]))
  x <- wtf$cwt(b)$shape
  y <- c(32, 46, 2, 100)
  expect_equal(x, y)

  ### test forward ###
  x <- c(as.numeric(wtf(batch)$real$mean()), as.numeric(wtf(batch)$real$min()), as.numeric(wtf(batch)$real$max()))
  y <- c(-0.0026392932315294822, -4.567781448364258, 4.5740203857421875)
  expect_equal(x, y, tolerance = 1e-1)
  x <- c(as.numeric(wtf(batch)$imag$mean()), as.numeric(wtf(batch)$imag$min()), as.numeric(wtf(batch)$imag$max()))
  y <- c(-0.008708901385664894, -4.636211395263672, 4.621674537658691)
  expect_equal(x, y, tolerance = 1e-1)
  x <- c(as.numeric(wtf(batch)$abs()$mean()), as.numeric(wtf(batch)$abs()$min()), as.numeric(wtf(batch)$abs()$max()))
  y <- c(0.3943239924987704, 7.787456582134456e-09, 4.649060297586245)
  expect_equal(x, y, tolerance = 1e-1)

  }
)

test_that("wavelet_transform, Mexican Hat", {
  # expected values computed from
  # https://github.com/QUVA-Lab/PyTorchWavelets/blob/master/examples/simple_example.py

  dt <- 0.1
  dj <- 0.125
  batch_size <- 32

  t <- torch::torch_linspace(0, 10, floor(10/dt))
  frequencies <- torch::torch_tensor(runif(batch_size, -0.5, 2.0))
  batch <- torch::torch_zeros(batch_size, length(t))
  for (f in 1:length(frequencies)) {
    batch[f, ] <- torch::torch_sin(2 * pi * frequencies[f] * t)
  }

  wavelet <- MexicanHat$new()
  wtf <- wavelet_transform(dim(batch)[2], dt, dj, wavelet)
  transform <- wtf(batch)

  x <- transform$shape
  y <- c(32, 62, 100)
  expect_equal(x, y)

  x <- c(as.numeric(transform$mean()), as.numeric(transform$min()), as.numeric(transform$max()))
  y <- c(-0.01580384674525963, -6.3885040283203125, 6.596841335296631)
  expect_equal(x, y, tolerance = 1e-1)

}
)

test_that("wavelet_transform, Paul wavelet", {
  # expected values computed from
  # https://github.com/QUVA-Lab/PyTorchWavelets/blob/master/examples/simple_example.py

  dt <- 0.1
  dj <- 0.125
  batch_size <- 32

  t <- torch::torch_linspace(0, 10, floor(10/dt))
  frequencies <- torch::torch_tensor(runif(batch_size, -0.5, 2.0))
  batch <- torch::torch_zeros(batch_size, length(t))
  for (f in 1:length(frequencies)) {
    batch[f, ] <- torch::torch_sin(2 * pi * frequencies[f] * t)
  }

  wavelet <- Paul$new()
  wtf <- wavelet_transform(dim(batch)[2], dt, dj, wavelet)
  transform <- wtf(batch)

  x <- transform$shape
  y <- c(32, 50, 100)
  expect_equal(x, y)

  x <- c(as.numeric(wtf(batch)$real$mean()), as.numeric(wtf(batch)$real$min()), as.numeric(wtf(batch)$real$max()))
  y <- c(-0.0029783361621269462, -4.427924633026123, 4.5088677406311035)
  expect_equal(x, y, tolerance = 1e-0) # ??
  x <- c(as.numeric(wtf(batch)$imag$mean()), as.numeric(wtf(batch)$imag$min()), as.numeric(wtf(batch)$imag$max()))
  y <- c(-0.002633194155512286, -5.00217866897583, 5.001322269439697)
  expect_equal(x, y, tolerance = 1e-0) # ??
  x <- c(as.numeric(wtf(batch)$abs()$mean()), as.numeric(wtf(batch)$abs()$min()), as.numeric(wtf(batch)$abs()$max()))
  y <- c(0.676334777904302, 1.9461160282119255e-06, 5.00279809480148)
  expect_equal(x, y, tolerance = 1e-0) # ??

}
)


