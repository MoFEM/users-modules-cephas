import math

# SDF Indenter

# Negative level set represents interior of the indenter. 
# This normal points outside of the indenter.

r = 1


def sdf(t, x, y, z, tx, ty, tz):
	return CylinderZ.sDF(r, 0, -0.5-r, x, y)


def grad_sdf(t, x, y, z, tx, ty, tz):
	return CylinderZ.gradSdf(0, -0.5-r, x, y)


def hess_sdf(t, x, y, z, tx, ty, tz):
	return CylinderZ.hessSdf(0, -0.5-r, x, y)

# Indenters

class yPlane:
	def sDF(shift, y):
		return y - shift

	def gradSdf():
		# nx, ny, nz
		return [0, 1, 0]
	
	def hessSdf():
		# xx, yx, zx, yy, zy, zz
		return [0, 0, 0, 0, 0, 0]


class CylinderZ:
	
	def sDF(r, xc, yc, x, y):
		a = pow(x-xc, 2)+pow(y-yc, 2)
		c_val = math.sqrt(a)-r
		return c_val
	
	def gradSdf(xc, yc, x, y):
		a = pow(x-xc, 2)+pow(y-yc, 2)
		c_val = math.sqrt(a)
		c_val_A = 1/c_val
		c_val_dx = c_val_A * (x-xc)
		c_val_dy = c_val_A * (y-yc)
		# x, y, z
		return [c_val_dx, c_val_dy, 0]
	
	def hessSdf(xc, yc, x, y):
		a = pow(x-xc, 2)+pow(y-yc, 2)
		c_val = math.sqrt(a)
		c_val_A = 1./c_val
		c_val_B = 1./pow(a, 3./2.)
		c_val_dx_dx = c_val_A - c_val_B * pow(x-xc, 2)
		c_val_dx_dy = -c_val_B * (x-xc)*(y-yc)
		c_val_dy_dy = c_val_A - c_val_B * pow(y-yc, 2)
		# xx, yx, zx, yy, zy, zz
		return [c_val_dx_dx, c_val_dx_dy, 0, c_val_dy_dy, 0, 0]

class Sphere:
	
	def sDF(r, xc, yc, zc, x, y, z):
		a = pow(x-xc, 2)+pow(y-yc, 2)+pow(z-zc, 2)
		c_val = math.sqrt(a)-r
		return c_val
	
	def gradSdf(xc, yc, zc, x, y, z):
		a = pow(x-xc, 2)+pow(y-yc, 2)+pow(z-zc, 2)
		c_val = math.sqrt(a)
		c_val_A = 1/c_val
		c_val_dx = c_val_A * (x-xc)
		c_val_dy = c_val_A * (y-yc)
		c_val_dz = c_val_A * (z-zc)
		# x, y, z
		return [c_val_dx, c_val_dy, c_val_dz]
	
	def hessSdf(xc, yc, zc, x, y, z):
		a = pow(x-xc, 2)+pow(y-yc, 2)+pow(z-zc, 2)
		c_val = math.sqrt(a)
		c_val_A = 1./c_val
		c_val_B = 1./pow(a, 3./2.)
		c_val_dx_dx = c_val_A - c_val_B * pow(x-xc, 2)
		c_val_dx_dy = -c_val_B * (x-xc)*(y-yc)
		c_val_dy_dy = c_val_A - c_val_B * pow(y-yc, 2)
		# xx, yx, zx, yy, zy, zz
		return [c_val_dx_dx, c_val_dx_dy, 0, c_val_dy_dy, 0, 0]


