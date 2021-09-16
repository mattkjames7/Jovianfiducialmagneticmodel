import numpy as np
from numba import njit,jit
from scipy.special import j0,j1
from ._Integrate import _Integrate

def _IntegralScalar(rho,z,abs_z,d,mu_i,
					lambda_int_brho,lambda_int_bz,
					beselj_rho_r0_0,beselj_z_r0_0,
					dlambda_brho,dlambda_bz):

	#do the integration
	beselj_rho_rho1_1 = j1(lambda_int_brho*rho)
	beselj_z_rho1_0   = j0(lambda_int_bz*rho)
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
