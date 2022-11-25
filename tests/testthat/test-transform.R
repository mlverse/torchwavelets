test_that("basic functionality", { # https://github.com/QUVA-Lab/PyTorchWavelets/blob/master/examples/simple_example.py

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
}
)
