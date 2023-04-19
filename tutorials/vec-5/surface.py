import math

def surface(x, y, z, eta):
	water_height = 0.;
	return math.tanh((water_height - y) / (eta * math.sqrt(2.)))
  