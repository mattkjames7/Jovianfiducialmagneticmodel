import numpy as np
from numba import njit

@njit
def _BphiScalar(rho,abs_z,z,i_rho,d):
	'''
	numba-compatible fucntion for calculating the azimuthal field.
	

	New to CAN2020 (not included in CAN1981): radial current 
	produces an azimuthal field, so Bphi is nonzero

	Inputs
	======
	rho : float
		distance in the x-z plane of the current sheet in Rj.
	abs_z : float
		absolute value of the z-coordinate
	z : float
		signed version of the z-coordinate
		
	Returns
	=======
	Bphi : float
		Azimuthal component of the magnetic field.

	'''
	Bphi = 2.7975*i_rho/rho

	if abs_z < d:
		Bphi *= (abs_z/d)
	if z > 0:
		Bphi = -Bphi

		
	return Bphi
		
	
@njit
def _BphiVector(rho,abs_z,z,i_rho,d):
	'''
	numba-compatible fucntion for calculating the azimuthal field.
	

	New to CAN2020 (not included in CAN1981): radial current 
	produces an azimuthal field, so Bphi is nonzero

	Inputs
	======
	rho : float
		distance in the x-z plane of the current sheet in Rj.
	abs_z : float
		absolute value of the z-coordinate
	z : float
		signed version of the z-coordinate
		
	Returns
	=======
	Bphi : float
		Azimuthal component of the magnetic field.

	'''
	Bphi = 2.7975*i_rho/rho


	ind = np.where(abs_z < d)[0]
	if ind.size > 0:
		Bphi[ind] *= (abs_z[ind]/d)
	ind = np.where(z > 0)[0]
	if ind.size > 0:
		Bphi[ind] = -Bphi[ind]
		
	return Bphi
		
	
