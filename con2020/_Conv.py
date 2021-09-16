import numpy as np
from numba import jit,njit

@njit
def _ConvInputCart(x0,y0,z0,cosxp,sinxp,cosxt,sinxt):
	'''
	Converts input coordinates from Cartesian right-handed System 
	III to current sheet coordinates.
	
	Inputs
	======
	x0 : float
		System III x-coordinate (Rj).
	y0 : float
		System III y-coordinate (Rj).
	z0 : float
		System III z-coordinate (Rj).
		
	Returns
	=======
	x1 : float
		x current sheet coordinate
	y1 : float
		y current sheet coordinate
	z1 : float
		z current sheet coordinate
	rho1 : float
		distance from z-axis (Rj).
	abs_z1 : float
		abs(z1) (Rj).
	cost : float
		cos(theta) - where theta is the colatitude
	sint : float
		sin(theta)
	cosp : float
		cos(phi) - where phi is east longitude
	sinp : float	
		sin(phi)
	'''

	rho0_sq = x0*x0 + y0*y0
	rho0 = np.sqrt(rho0_sq)
	r = np.sqrt(rho0_sq + z0**2)
	cost = z0/r
	sint = rho0/r
	sinp = y0/rho0
	cosp = x0/rho0

	#rotate x and y to align with the current sheet longitude
	x = rho0*(cosp*cosxp + sinp*sinxp)
	y1 = rho0*(sinp*cosxp - cosp*sinxp)

	#rotate about y axis to align with current sheet
	x1 = x*cosxt + z0*sinxt
	z1 = z0*cosxt - x*sinxt	

	#some other bits we need for the model
	rho1 = np.sqrt(x1*x1 + y1*y1)
	abs_z1 = np.abs(z1)
			
	return x1,y1,z1,rho1,abs_z1,cost,sint,cosp,sinp
		
@njit
def _ConvOutputCart(cost,sint,cosp,sinp,x1,y1,rho1,Brho1,Bphi1,Bz1,
					cosxp,sinxp,cosxt,sinxt):
	'''
	Convert the output magnetic field from cylindrical current 
	sheet coordinates to Cartesian right-handed System III
	coordinates.
	
	Inputs
	======
	cost : float (dummy)
		cos(theta) - where theta is the colatitude
	sint : float (dummy)
		sin(theta)
	cosp : float (dummy)
		cos(phi) - where phi is east longitude
	sinp : float (dummy)	
		sin(phi)
	x1 : float
		x-position in current sheet coords (Rj).
	y1 : float
		y-position in current sheet coords (Rj).
	rho1 : float
		distance from z-axis (Rj).
	Brho1 : float	
		Rho component of magnetic field (nT).
	Bphi1 : float
		Phi (azimuthal) component of the magnetic field (nT).
	Bz1 : float
		z component of the magnetic field (nT).
		
	Returns
	=======
	Bx0 : float
		x-component of magnetic field in right-handed System III
		coordinates (nT).
	By0 : float
		y-component of magnetic field in right-handed System III
		coordinates (nT).
	Bz0 : float
		z-component of magnetic field in right-handed System III
		coordinates (nT).
		
		
	'''
	cosphi1 = x1/rho1
	sinphi1 = y1/rho1
	
	Bx1 = Brho1*cosphi1 - Bphi1*sinphi1
	By1 = Brho1*sinphi1 + Bphi1*cosphi1 		

	Bx = Bx1*cosxt - Bz1*sinxt
	Bz0 = Bx1*sinxt + Bz1*cosxt		

	Bx0 = Bx*cosxp - By1*sinxp
	By0 = By1*cosxp + Bx*sinxp	
	
	return Bx0,By0,Bz0


@njit
def _ConvInputPol(r,theta,phi,cosxp,sinxp,cosxt,sinxt):
	'''
	Converts input coordinates from spherical polar right-handed 
	System III to Cartesian current sheet coordinates.
	
	Inputs
	======
	r : float
		System III radial distance (Rj).
	theta : float
		System III colatitude (rad).
	phi : float
		System III east longitude (rad).
			
	Returns
	=======
	x1 : float
		x current sheet coordinate
	y1 : float
		y current sheet coordinate
	z1 : float
		z current sheet coordinate
	rho1 : float
		distance from z-axis (Rj).
	abs_z1 : float
		abs(z1) (Rj).
	cost : float
		cos(theta) - where theta is the colatitude
	sint : float
		sin(theta)
	cosp : float
		cos(phi) - where phi is east longitude
	sinp : float	
		sin(phi)
	'''		
	
	sint = np.sin(theta)
	cost = np.cos(theta)
	sinp = np.sin(phi)
	cosp = np.cos(phi)

	#surprisingly this is slightly (~20%) quicker than 
	#x = r*sint*np.cos(phi - self._dipole_shift) etc.
	x = r*sint*(cosp*cosxp + sinp*sinxp)
	y1 = r*sint*(sinp*cosxp - cosp*sinxp)
	z = r*cost
		
	x1 = x*cosxt + z*sinxt
	z1 = z*cosxt - x*sinxt	

	#some other bits we need for the model
	rho1 = np.sqrt(x1*x1 + y1*y1)
	abs_z1 = np.abs(z1)
			
	return x1,y1,z1,rho1,abs_z1,cost,sint,cosp,sinp	
		

@njit
def _ConvOutputPol(cost,sint,cosp,sinp,x1,y1,rho1,Brho1,Bphi1,Bz1,
					cosxp,sinxp,cosxt,sinxt):
	'''
	Convert the output magnetic field from cylindrical current 
	sheet coordinates to spherical polar right-handed System III
	coordinates.
	
	Inputs
	======
	cost : float
		cos(theta) - where theta is the colatitude
	sint : float
		sin(theta)
	cosp : float
		cos(phi) - where phi is east longitude
	sinp : float	
		sin(phi)
	x1 : float
		x-position in current sheet coords (Rj).
	y1 : float
		y-position in current sheet coords (Rj).
	rho1 : float
		distance from z-axis (Rj).
	Brho1 : float	
		Rho component of magnetic field (nT).
	Bphi1 : float
		Phi (azimuthal) component of the magnetic field (nT).
	Bz1 : float
		z component of the magnetic field (nT).
		
	Returns
	=======
	Br : float
		Radial component of magnetic field in right-handed System 
		III coordinates (nT).
	Bt : float
		Meridional component of magnetic field in right-handed 
		System III coordinates (nT).
	Bp : float
		Azimuthal component of magnetic field in right-handed System 
		III coordinates (nT).
		
	
	'''		
		
	#this now runs in about 60% of the time it used to
	cosphi1 = x1/rho1
	sinphi1 = y1/rho1
	
	Bx1 = Brho1*cosphi1 - Bphi1*sinphi1
	By1 = Brho1*sinphi1 + Bphi1*cosphi1 		
	Bx = Bx1*cosxt - Bz1*sinxt
	Bz = Bx1*sinxt + Bz1*cosxt		
	Bx2 = Bx*cosxp - By1*sinxp
	By2 = By1*cosxp + Bx*sinxp	

	Br =  Bx2*sint*cosp+By2*sint*sinp+Bz*cost
	Bt =  Bx2*cost*cosp+By2*cost*sinp-Bz*sint
	Bp = -Bx2*     sinp+By2*     cosp
	
	return Br,Bt,Bp
