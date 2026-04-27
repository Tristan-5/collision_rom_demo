def collision_step_variance(step_size: float, p_forward: float) -> float:
  """
  Calculates the variance of a single collision step.
  Based on Bernoulli trial variance. 4*p*(1-p)*s^2
  """
  return 4.0 * p_forward * (1.0 - p_forward) * step_size**2
