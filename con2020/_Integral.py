import numpy as np
import numba
from scipy.special import j0,j1
from ._Integrate import _Integrate

@numba.njit
def _callj0(z):
	y = np.empty_like(z) 
	n = len(z) 
	for i in range(0,n): 
		y[i] = j0(z[i])  
	return y 

@numba.njit
def _callj1(z):
	y = np.empty_like(z) 
	n = len(z) 
	for i in range(0,n): 
		y[i] = j1(z[i])  
	return y 



@numba.njit
def _IntegralScalar(rho,z,abs_z,d,mu_i,
					lambda_int_brho,lambda_int_bz,
					beselj_rho_r0_0,beselj_z_r0_0,
					dlambda_brho,dlambda_bz):

	#do the integration
	lr = lambda_int_brho*rho
	beselj_rho_rho1_1 = _callj1(lr)
	beselj_z_rho1_0   = _callj0(lambda_int_bz*rho)
	if (abs_z > d): #% Connerney et al. 1981 eqs. 14 and 15
		brho_int_funct = beselj_rho_rho1_1*beselj_rho_r0_0 \
							*np.sinh(d*lambda_int_brho) \
							*np.exp(-abs_z*lambda_int_brho) \
							/lambda_int_brho
		bz_int_funct   = beselj_z_rho1_0 *beselj_z_r0_0 \
							*np.sinh(d*lambda_int_bz) \
							*np.exp(-abs_z*lambda_int_bz) \
							/lambda_int_bz  
		Brho = mu_i*2.0*_Integrate(brho_int_funct,dlambda_brho)
		if z < 0:
			Brho = -Brho
	else:
		brho_int_funct = beselj_rho_rho1_1*beselj_rho_r0_0 \
							*(np.sinh(z*lambda_int_brho) \
							*np.exp(-d*lambda_int_brho)) \
							/lambda_int_brho
		bz_int_funct   = beselj_z_rho1_0  *beselj_z_r0_0 \
							*(1.0 -np.cosh(z*lambda_int_bz) \
							*np.exp(-d*lambda_int_bz)) \
							/lambda_int_bz
		Brho = mu_i*2.0*_Integrate(brho_int_funct,dlambda_brho)#
	Bz = mu_i*2.0*_Integrate(bz_int_funct,dlambda_bz)

	return Brho,Bz
