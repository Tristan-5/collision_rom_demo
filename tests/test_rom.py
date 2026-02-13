import numpy as np
from rom import rom_prediction

def test_rom_prediction
  N = 500
  mean_v = 0.0
  var_v = 2.0

  t, predicted = rom_prediction(N, mean_v, var_v)

  assert len(t) == N
  assert len(predicted == N)
  assert predicted[-1] == var_v

