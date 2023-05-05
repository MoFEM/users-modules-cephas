import math

def surface(x, y, z, eta):
	return -math.tanh((-0.01 - x) / (eta * math.sqrt(2.)))
  