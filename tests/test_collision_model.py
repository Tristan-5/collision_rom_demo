import numpy as np
from collision_model import generate_collision_velocity

def test_reproducibility_with_seed() :
  v1 = generate_collision_velocity(N=1000, seed=123)
  v2 = generate_collision_velocity(N=1000, seed=123)

  assert np.array_equal(v1, v2), "Velocity should be reproducible with fixed seed"
