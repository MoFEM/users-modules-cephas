import math

# Indenters

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

# SDF Indenter

r = 1

def sdf(t, x, y, z):
  return CylinderZ.sDF(1, 0, -0.5-r, x, y)
  
def grad_sdf(t, x, y, z):
  return CylinderZ.gradSdf(0, -0.5-r, x, y)

def hess_sdf(t, x, y, z):
  return CylinderZ.hessSdf(0, -0.5-r, x, y)






