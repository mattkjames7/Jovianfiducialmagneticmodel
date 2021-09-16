import numpy as np
from scipy.special import jv,j0,j1
from ._Switcher import _Switcher
from ._Analytic import _AnalyticEdwards,_AnalyticEdwardsScalar,_AnalyticEdwardsVector
from ._Conv import _ConvInputCart,_ConvInputPol,_ConvOutputCart,_ConvOutputPol
from ._Integrate import _Integrate
import time
from numba import njit,jit
from ._Integral import _IntegralScalar,_IntegralVector

class Model(object):
	def __init__(self,**kwargs):
		'''
		Code to calculate the perturbation magnetic field produced by the 
		Connerney (CAN) current sheet, which is represented by a finite disk 
		of current.	This disk has variable parameters including the current 
		density mu0i0, inner edge R0, outer edge R1, thickness D. The disk 
		is centered on the magnetic equator (shifted in longitude and tilted 
		according to the dipole field parameters of an internal field model 
		like VIP4 or JRM09). This 2020 version includes a radial current per 
		Connerney et al. (2020), 
		https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2020JA028138

		Keyword Arguments (shorthand keywords in brackets)
		=================
		mu_i_div2__current_density_nT (mu_i): float
			mu0i0/2 term (current sheet current density), in nT
		i_rho__radial_current_density_nT (i_rho) : float
			radial current term from Connerney et al., 2020
			NOTE: The default value (16.7 nT) is the average value from
			Connerney et al 2020. This value was shown to vary from one 
			pass to the next, where Table 2 provides radial current 
			density values for 23 of the first 24
			perijoves.
		r0__inner_rj (r0) : float
			Inner edge of current disk in Rj
		r1__outer_rj (r1) : float
			Outer edge of current disk in Rj
		d__cs_half_thickness_rj (d) : float
			Current sheet half thickness in Rj
		xt__cs_tilt_degs (xt) : float
			Current sheet tilt in degrees
		xp__cs_rhs_azimuthal_angle_of_tilt_degs (xp) : float
			Current sheet tilt longitude (right handed) in degrees
		equation_type: str
			Define method for calculating the current sheet field, may be 
			one of the following: 'hybrid'|'analytic'|'integral'
			See notes below for more information.
		error_check : bool
			If True (default) then inputs will be checked for potential errors.		
		CartesianIn : bool
			If True (default) the inputs to the model will be expected to be 
			in Cartesian right-handed System III coordinates. If False, then
			the inputs should be in spherical polar coordinates.
		CartesianOut : bool
			If True (default) the output magnetic field will be in Cartesian
			right-handed System III coordinates. Otherwise, the magnetic 
			field components produced will be radial, meridional and 
			azimuthal.

		Returns
		========
		model : object
			This is an instance of the con2020.Model object. To obtain the
			magnetic field, call the Field() member function, e.g.:
			
			model = con2020.Model()
			B = model.Field(x,y,z)

		This code takes a hybrid approach to calculating the current sheet 
		field, using the integral equations in some regions and the analytic 
		equations in others. Following Connerney et al. 1981, figure A1, and 
		Edwards et al. (2001), figure 2, the choice of integral vs. analytic 
		equations is most important near rho = r0 and z = 0.
		
		By default, this code uses the analytic equations everywhere except 
		|Z| < D*1.5 and |Rho-R0| < 2.

		Analytic Equations
		==================
		For the analytic equations, we use the equations  
		provided by Edwards et al. 2001: 
		https://www.sciencedirect.com/science/article/abs/pii/S0032063300001641
		
		
		Integral Equations
		==================
		For the integral equations we use the Bessel functions from 
		Connerney et al. 1981, eqs. 14, 15, 17, 18.
		
		We do not integrate lambda from zero to infinity, but vary the 
		integration limit depending on the value of the Bessel functions.
		
		Other Notes
		===========
		
		Keyword equation_type can be set to 'integral' or 'analytic' if the 
		user wants to force using the integral or analytic equations ,by 
		Marissa Vogt, March 2021.
		
		RJ Wilson did some speedups and re-formatting of lines, also March 2021
		'''		
		
		#list the default arguments here
		defargs = {	'mu_i'			: 139.6,
					'i_rho' 		: 16.7,
					'r0'			: 7.8,
					'r1'			: 51.4,
					'd'				: 3.6,
					'xt'			: 9.3,
					'xp'			: -24.2,
					'equation_type'	: 'hybrid',
					'error_check'	: True,
					'CartesianIn'	: True,
					'CartesianOut'	: True}
					
		#list the long names
		longnames = {	'mu_i'	: 'mu_i_div2__current_density_nT',
						'r0'	: 'r0__inner_rj',
						'r1'	: 'r1__outer_rj',
						'd'		: 'd__cs_half_thickness_rj',
						'xt'	: 'xt__cs_tilt_degs',
						'xp'	: 'xp__cs_rhs_azimuthal_angle_of_tilt_degs',
						'i_rho'	: 'i_rho__radial_current_density_nT'		  }
						
		#check input kwargs
		#for those which exist (either in long or short name form) add
		#them to this object using the short name as the object tag
		#Otherwise use the default value
		
		#the input keys
		ikeys = list(kwargs.keys())
		
		#default keys
		dkeys = list(defargs.keys())
		
		#short and long name keys
		skeys = list(longnames.keys())
		lkeys = [longnames[k] for k in skeys]

		#some constants
		self._Deg2Rad = np.pi/180.0
		
			
		#loop through each one		
		for k in dkeys:
			if k in ikeys:
				#short name found in kwargs - add to this object
				kw = kwargs[k]
			elif longnames.get(k,'') in ikeys:
				#long name found - add to object
				kw = kwargs[longnames[k]]
			else:
				#key not found, use default
				kw = defargs[k]
			setattr(self,k,kw)
		
		#check for additional keys and issue a warning
		for k in ikeys:
			if not ((k in skeys) or (k in lkeys) or (k in dkeys)):
				print("Keyword argument {:s} unrecognized, ignoring.".format(k))
		
		#now do the checks
	
		ckeys = ['mu_i','r0','r1','d','xt']
		for k in ckeys:
			x = getattr(self,k)
			if (x <= 0) or (np.isfinite(x) == False):
				raise SystemExit("'{:s}' should be greater than 0 and finite".format(k))	

		if (np.isfinite(self.xp) == False):
			raise SystemExit("'xp' should be finite")	
			
				


	#the following variables are set as properties, so that if someone 
	#changes xt or xp after the object has been created, it will 
	#automatically update the cos/sin values of each
	@property
	def xp(self):
		return self._xp
	
	@xp.setter
	def xp(self,value):
		self._xp = value
		self._dipole_shift = self._xp*self._Deg2Rad # xp is longitude of the current sheet
		self._cosxp = np.cos(self._dipole_shift)
		self._sinxp = np.sin(self._dipole_shift)	
	
	@property
	def xt(self):
		return self._xt
	
	@xt.setter
	def xt(self,value):
		self._xt = value
		self._theta_cs = self._xt*self._Deg2Rad # current sheet tilt
		self._cosxt = np.cos(self._theta_cs)
		self._sinxt = np.sin(self._theta_cs)
	
	#do a similar thing for equation type
	@property
	def equation_type(self):
		return self._eq_type
	
	@equation_type.setter
	def equation_type(self,value):
		_eq_type = value.lower()
		if not _eq_type in ['analytic','hybrid','integral']:
			raise SystemExit("ERROR: 'equation_type' has unrecognized string - it should be 'analytic'|'hybrid'|'integral'")			
		self._eq_type = _eq_type

		#set the integral functions (scalar and vector)
		if self._eq_type == 'analytic':
			self._ModelFunc = self._Analytic
		elif self._eq_type == 'integral':
			self._ModelFunc = self._Integral
		else:
			self._ModelFunc = self._Hybrid
		
		if self._eq_type != 'analytic' and not hasattr(self,'_dlambda_brho'):
			self._UpdateBessel()
	
	

	#also for CartesianIn/CartesianOut
	@property
	def CartesianIn(self):
		return self._CartIn
	
	@CartesianIn.setter
	def CartesianIn(self,value):
		self._CartIn = value
		#set the coordinate conversion functions for input
		self._SetInputConv()
	
	@property
	def CartesianOut(self):
		return self._CartOut
	
	@CartesianOut.setter
	def CartesianOut(self,value):
		self._CartOut = value		
		#set the output functions
		if self._CartOut:
			self._OutputConv = _ConvOutputCart
		else:
			self._OutputConv = _ConvOutputPol		
						

	@property
	def r0(self):
		return self._r0
	
	@r0.setter
	def r0(self,value):
		self._r0 = value
		#update the bessel functions
		if hasattr(self,'_eq_type'):
			if self._eq_type != 'analytic':
				self._UpdateBessel()
	
	@property
	def error_check(self):
		return self._err_chk
	
	@error_check.setter
	def error_check(self,value):
		self._err_chk = value
		if hasattr(self,'_CartIn'):
			self._SetInputConv()
		
	def _SetInputConv(self):
		'''
		This function sets the appropriate function pointers for the 
		input coordinate conversion.
		
		'''
		if self._CartIn:
			if self._err_chk:
				self._InputConv = self._ConvInputCartSafe
			else:
				self._InputConv = _ConvInputCart
		else:
			if self._err_chk:
				self._InputConv = self._ConvInputPolSafe
			else:
				self._InputConv = _ConvInputPol		
		
	
	def _UpdateBessel(self):
		'''
		This function updates a bunch of internal parameters and arrays 
		which are used for integration.
		
		'''
		#this stuff is for integration
		self._dlambda_brho    = 1e-4  #% default step size for Brho function
		self._dlambda_bz      = 5e-5  #% default step size for Bz function
			
		#each of the following variables will be indexed by zcase (starting at 0)
		self._lambda_max_brho = [4,4,40,40,100,100]
		self._lambda_max_bz = [100,20,100,20,100,20]
			
		self._lambda_int_brho = []
		self._lambda_int_bz = []
			
		self._beselj_rho_r0_0 = []
		self._beselj_z_r0_0 = []

		for i in range(0,6):
			#save the lambda arrays
			self._lambda_int_brho.append(np.arange(self._dlambda_brho,self._dlambda_brho*(self._lambda_max_brho[i]/self._dlambda_brho),self._dlambda_brho))
			self._lambda_int_bz.append(np.arange(self._dlambda_bz,self._dlambda_bz*(self._lambda_max_bz[i]/self._dlambda_bz),self._dlambda_bz))
				
			#save the Bessel functions
			self._beselj_rho_r0_0.append(j0(self._lambda_int_brho[i]*self.r0))
			self._beselj_z_r0_0.append(j0(self._lambda_int_bz[i]*self.r0))		
					
		
	def _ConvInputCartSafe(self,x0,y0,z0,cosxp,sinxp,cosxt,sinxt):
		'''
		Converts input coordinates from Cartesian right-handed System 
		III to current sheet coordinates - with error checking.
		
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
		#check input
		self._CheckInputCart(x0,y0,z0)
		
		#now convert it
		return _ConvInputCart(x0,y0,z0,cosxp,sinxp,cosxt,sinxt)

	
	def _ConvInputPolSafe(self,r,theta,phi,cosxp,sinxp,cosxt,sinxt):
		'''
		Converts input coordinates from spherical polar right-handed 
		System III to Cartesian current sheet coordinates - with error
		checks.
		
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
		#check the input coordinates
		self._CheckInputPol(r,theta,phi)
		
		#now convert coordinates
		return _ConvInputPol(r,theta,phi,cosxp,sinxp,cosxt,sinxt)


		
	def _CheckInputCart(self,x,y,z):
		'''
		Check the Cartesian inputs - if the checks fail then the
		function raises an error.

		
		Inputs
		======
		x0 : float
			System III x-coordinate (Rj).
		y0 : float
			System III y-coordinate (Rj).
		z0 : float
			System III z-coordinate (Rj).
			
		'''
		if (np.size(x) != np.size(y)) or (np.size(x) != np.size(z)):
			raise SystemExit ('ERROR: Input coordinate arrays must all be of the same length. Returning...')
		
		#calculate r
		r = np.sqrt(x*x + y*y + z*z)

		if np.min(r) <= 0 or np.max(r) >= 200:
			raise SystemExit ('ERROR: Radial distance r must be in units of Rj and >0 but <200 only, and not outside that range (did you use km instead?). Returning...')


	def _CheckInputPol(self,r,theta,phi):
		'''
		Check the spherical polar inputs - if the checks fail then the
		function raises an error.

		
		Inputs
		======
		r : float
			System III radial distance (Rj).
		theta : float
			System III colatitude (rad).
		phi : float
			System III east longitude (rad).
			
		'''
		if np.min(r) <= 0 or np.max(r) >= 200:
			raise SystemExit ('ERROR: Radial distance r must be in units of Rj and >0 but <200 only, and not outside that range (did you use km instead?). Returning...')

		if np.min(theta) < 0 or np.max(theta) > np.pi:
			raise SystemExit ('ERROR: CoLat must be in radians of 0 to pi only, and not outside that range (did you use degrees instead?). Returning...')

		if np.min(phi)  < -2*np.pi or np.max(phi) > 2*np.pi:
			raise SystemExit ('ERROR: Long must be in radians of -2pi to 2pi only, and not outside that range (did you use degrees instead?). Returning...')	
			
		if (np.size(r) != np.size(phi)) or (np.size(r) != np.size(theta)):
			raise SystemExit ('ERROR: Input coordinate arrays must all be of the same length. Returning...')

	def _Bphi(self,rho,abs_z,z):
		'''
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
		Bphi = 2.7975*self.i_rho/rho
		
		if np.size(rho) == 1:
			if abs_z < self.d:
				Bphi *= (abs_z/self.d)
			if z > 0:
				Bphi = -Bphi
		else:
			ind = np.where(abs_z < self.d)[0]
			if ind.size > 0:
				Bphi[ind] *= (abs_z[ind]/self.d)
			ind = np.where(z > 0)[0]
			if ind.size > 0:
				Bphi[ind] = -Bphi[ind]
		
		return Bphi
		
	def _Analytic(self,rho,abs_z,z):
		'''		
		Calculate the magnetic field associated with the current sheet
		using analytical equations either from Connerney et al 1981 or
		the divergence-free equations from Edwards et al 2001 (defualt).
		
		The equations used the "Edwards" ones.
		Using equations 9a and 9b for the small rho approximation
		and 13a and 13b for the large rho approximation of Edwards
		et al.
		
			
		Inputs
		======
		rho : float
			rho coordinate (Rj).
		abs_z : float
			absolute value of the z coordinate (Rj)
		z : float
			z coordinate (Rj)
		
		Returns
		=======
		Brho : float
			rho-component of the magnetic field.
		Bphi : float
			phi-component of the magnetic field.
		Bz : float
			z-component of the magnetic field.
		
		'''

		#calculate the analytic solution first for Brho and Bz
		Brho,Bz = _AnalyticEdwards(rho,z,self.d,self.r0,self.mu_i)

		#calculate Bphi
		Bphi = self._Bphi(rho,abs_z,z)

		#subtract outer edge contribution
		Brho_fin,Bz_fin = _AnalyticEdwards(rho,z,self.d,self.r1,self.mu_i)

		#Bphi_fin = -self.i_rho*Brho_fin/self.mu_i
		Brho -= Brho_fin
		#Bphi -= Bphi_fin
		Bz -= Bz_fin

		return Brho,Bphi,Bz
		
		
	def _IntegralScalar(self,rho,abs_z,z):
		'''
		Integrates the model equations for an single set of input 
		coordinates.
		
		Inputs
		======
		rho : float
			rho coordinate (Rj).
		abs_z : float
			absolute value of the z coordinate (Rj)
		z : float
			z coordinate (Rj)
		
		Returns
		=======
		Brho : float
			rho-component of the magnetic field.
		Bz : float
			z-component of the magnetic field.
		
		'''				

		#check which "zcase" we need for this vector
		check1 = np.abs(abs_z - self.d)		
		check2 = abs_z <= self.d*1.1
		
		if check1 >= 0.7:
			#case 1 or 2
			zc = 1
		elif (check1 < 0.7) and (check1 >= 0.1):
			#case 3 or 4
			zc = 3
		else:
			#case 5 or 6
			zc = 5
		#this bit does two things - it both takes into account the
		#check2 thing and it makes zc an index in range 0 to 5 as 
		#opposed to the zcase 1 to 6, so zi = zcase -1
		zc -= np.int(check2)
		
		return _IntegralScalar(rho,z,abs_z,self.d,self.mu_i,
					self._lambda_int_brho[zc],self._lambda_int_bz[zc],
					self._beselj_rho_r0_0[zc],self._beselj_z_r0_0[zc],
					self._dlambda_brho,self._dlambda_bz)
		
		#do the integration
		beselj_rho_rho1_1 = j1(self._lambda_int_brho[zc]*rho)
		beselj_z_rho1_0   = j0(self._lambda_int_bz[zc]*rho)
		if (abs_z > self.d): #% Connerney et al. 1981 eqs. 14 and 15
			brho_int_funct = beselj_rho_rho1_1*self._beselj_rho_r0_0[zc] \
							*np.sinh(self.d*self._lambda_int_brho[zc]) \
							*np.exp(-abs_z*self._lambda_int_brho[zc]) \
							/self._lambda_int_brho[zc]
			bz_int_funct   = beselj_z_rho1_0 *self._beselj_z_r0_0[zc] \
							*np.sinh(self.d*self._lambda_int_bz[zc]) \
							*np.exp(-abs_z*self._lambda_int_bz[zc]) \
							/self._lambda_int_bz[zc]  
			Brho = self.mu_i*2.0*_Integrate(brho_int_funct,self._dlambda_brho)
			if z < 0:
				Brho = -Brho
		else:
			brho_int_funct = beselj_rho_rho1_1*self._beselj_rho_r0_0[zc] \
							*(np.sinh(z*self._lambda_int_brho[zc]) \
							*np.exp(-self.d*self._lambda_int_brho[zc])) \
							/self._lambda_int_brho[zc]
			bz_int_funct   = beselj_z_rho1_0  *self._beselj_z_r0_0[zc] \
							*(1.0 -np.cosh(z*self._lambda_int_bz[zc]) \
							*np.exp(-self.d*self._lambda_int_bz[zc])) \
							/self._lambda_int_bz[zc]
			Brho = self.mu_i*2.0*_Integrate(brho_int_funct,self._dlambda_brho)#
		Bz = self.mu_i*2.0*_Integrate(bz_int_funct,self._dlambda_bz)

		return Brho,Bz


	def _IntegralVector(self,rho,abs_z,z):
		'''
		Integrates the model equations for an array of input coordinates.
		
		Inputs
		======
		rho : float
			rho coordinate (Rj).
		abs_z : float
			absolute value of the z coordinate (Rj)
		z : float
			z coordinate (Rj)
			
		Returns
		=======
		Brho : float
			rho-component of the magnetic field.
		Bz : float
			z-component of the magnetic field.
		
		'''		
		
		#check which "zcase" we need for this vector
		check1 = np.abs(abs_z - self.d)		
		check2 = abs_z <= self.d*1.1
		s = _Switcher(check1,check2)

		#create the output arrays for this function
		Brho = np.zeros(np.size(rho),dtype='float64')
		Bz = np.zeros(np.size(rho),dtype='float64')

		for zcase in range(1,7):
			
			ind_case,lambda_max_brho,lambda_max_bz = s.FetchCase(zcase)
			n_ind_case=len(ind_case)
			zc = zcase - 1
			
			if n_ind_case > 0:
				Brho[ind_case],Bz[ind_case] = \
					_IntegralVector(rho[ind_case],z[ind_case],
					abs_z[ind_case],self.d,self.mu_i,
					self._lambda_int_brho[zc],self._lambda_int_bz[zc],
					self._beselj_rho_r0_0[zc],self._beselj_z_r0_0[zc],
					self._dlambda_brho,self._dlambda_bz)
				# for zi in range(0,n_ind_case):
					# ind_for_integral = ind_case[zi] #;% sub-indices of sub-indices!

					# beselj_rho_rho1_1 = j1(self._lambda_int_brho[zc]*rho[ind_for_integral])
					# beselj_z_rho1_0   = j0(self._lambda_int_bz[zc]*rho[ind_for_integral] )
					# if (abs_z[ind_for_integral] > self.d): #% Connerney et al. 1981 eqs. 14 and 15
						# brho_int_funct = beselj_rho_rho1_1*self._beselj_rho_r0_0[zc] \
										# *np.sinh(self.d*self._lambda_int_brho[zc]) \
										# *np.exp(-abs_z[ind_for_integral]*self._lambda_int_brho[zc]) \
										# /self._lambda_int_brho[zc]
						# bz_int_funct   = beselj_z_rho1_0*self._beselj_z_r0_0[zc] \
										# *np.sinh(self.d*self._lambda_int_bz[zc]) \
										# *np.exp(-abs_z[ind_for_integral]*self._lambda_int_bz[zc]) \
										# /self._lambda_int_bz[zc]
						# Brho[ind_for_integral] = self.mu_i*2.0*_Integrate(brho_int_funct,self._dlambda_brho)
						# if z[ind_for_integral] < 0:
							# Brho[ind_for_integral] = -Brho[ind_for_integral]
					# else:
						# brho_int_funct = beselj_rho_rho1_1*self._beselj_rho_r0_0[zc] \
										# *(np.sinh(z[ind_for_integral]*self._lambda_int_brho[zc]) \
										# *np.exp(-self.d*self._lambda_int_brho[zc])) \
										# /self._lambda_int_brho[zc]
						# bz_int_funct   = beselj_z_rho1_0*self._beselj_z_r0_0[zc] \
										# *(1.0 -np.cosh(z[ind_for_integral]*self._lambda_int_bz[zc]) \
										# *np.exp(-self.d*self._lambda_int_bz[zc])) \
										# /self._lambda_int_bz[zc]  
						# Brho[ind_for_integral] = self.mu_i*2.0*_Integrate(brho_int_funct,self._dlambda_brho)
					# Bz[ind_for_integral]   = self.mu_i*2.0*_Integrate(bz_int_funct,self._dlambda_bz)
		
		return Brho,Bz			
	
			
		
	def _Integral(self,rho,abs_z,z):
		'''		
		Calculate the magnetic field associated with the current sheet
		by integrating equations 14, 15, 17 and 18 of Connerney et al
		1981.
		
		Inputs
		======
		rho : float
			rho coordinate (Rj).
		abs_z : float
			absolute value of the z coordinate (Rj)
		z : float
			z coordinate (Rj)
		
		Returns
		=======
		Brho : float
			rho-component of the magnetic field.
		Bphi : float
			phi-component of the magnetic field.
		Bz : float
			z-component of the magnetic field.
		
		'''		

		if np.size(rho) == 1:
			#scalar version of the code
			Brho,Bz = self._IntegralScalar(rho,abs_z,z)
		else:
			#vectorized version
			Brho,Bz = self._IntegralVector(rho,abs_z,z)
		
		#calculate Bphi
		Bphi = self._Bphi(rho,abs_z,z)
		
		#subtract outer edge contribution
		Brho_fin,Bz_fin = _AnalyticEdwards(rho,z,self.d,self.r1,self.mu_i)
		#Bphi_fin = -self.i_rho*Brho_fin/self.mu_i
		Brho -= Brho_fin
		#Bphi -= Bphi_fin
		Bz -= Bz_fin
		
		return Brho,Bphi,Bz		
		
		
	def _Hybrid(self,rho,abs_z,z):
		'''		
		Calculate the magnetic field associated with the current sheet
		by using a combination of analytical equations and numerical
		integration.
		
		Inputs
		======
		rho : float
			rho coordinate (Rj).
		abs_z : float
			absolute value of the z coordinate (Rj)
		z : float
			z coordinate (Rj)
		
		Returns
		=======
		Brho : float
			rho-component of the magnetic field.
		Bphi : float
			phi-component of the magnetic field.
		Bz : float
			z-component of the magnetic field.
		
		'''		


		if np.size(rho) == 1:
			#do the scalar version
			
			#check if we need to integrate numerically, or use analytical equations
			if (abs_z <= self.d*1.5) and (np.abs(rho - self.r0) <= 2.0):
				#use integration
				Brho,Bz = self._IntegralScalar(rho,abs_z,z)
			else:
				#analytical
				Brho,Bz = _AnalyticEdwardsScalar(rho,z,self.d,self.r0,self.mu_i)

		else:
			#this would be the vectorized version
			n = np.size(rho)
			Brho = np.zeros(n,dtype='float64')
			Bz = np.zeros(n,dtype='float64')

			doint = (abs_z <= self.d*1.5) & (np.abs(rho-self.r0) <= 2)
			Iint = np.where(doint)[0]
			Iana = np.where(doint == False)[0]
			
			if Iint.size > 0:
				Brho[Iint],Bz[Iint] = self._IntegralVector(rho[Iint],abs_z[Iint],z[Iint])
			
			if Iana.size > 0:
				Brho[Iana],Bz[Iana] = _AnalyticEdwardsVector(rho[Iana],z[Iana],self.d,self.r0,self.mu_i)


		#calculate Bphi
		Bphi = self._Bphi(rho,abs_z,z)
		
		#subtract outer edge contribution
		Brho_fin,Bz_fin = _AnalyticEdwards(rho,z,self.d,self.r1,self.mu_i)
		#Bphi_fin = -self.i_rho*Brho_fin/self.mu_i
		Brho -= Brho_fin
		#Bphi -= Bphi_fin
		Bz -= Bz_fin
		
		return Brho,Bphi,Bz		
				
		
				
	def Field(self,in0,in1,in2):
		'''
		Return the magnetic field vector(s) for a given input position
		in right-handed System III coordinates.
		
		Inputs
		======
		in0 : float
			First input coordinate(s) - x or r (in Rj).
		in1 : float
			Second input coordinate(s) - y (in Rj) or theta (in rad).
		in2 : float
			Third input coordinate(s) - z (in Rj) or phi (in rad).
			
		Whether or not the input coordinates are treated as Cartesian or
		spherical polar depends upon how the model was initialized with
		the "CartesianIn" keyword.
		
		e.g.:
		# for Cartesian input coordinates:
		B = Model.Field(x,y,z)
		
		#or spherical polar coordinates:
		B = Model.Field(r,theta,phi)
		
		Returns
		=======
		B : float
			(n,3) shaped array containing the magnetic field vectors in
			either Cartesian SIII coordinates or spherical polar ones,
			depending upon how the model was initialized, where "n" is
			the number of elements contained in the input arguments.
		'''
		t0 = time.time()
		#rotate and check input SIII coordinates to current sheet coords
		x,y,z,rho,abs_z,cost,sint,cosp,sinp = self._InputConv(in0,in1,
											in2,self._cosxp,self._sinxp,
											self._cosxt,self._sinxt)
		t1 = time.time()
		
		#create the output arrays
		n = np.size(rho)
		Bout = np.zeros((n,3),dtype='float64')
		
		
		#call the model function
		Brho,Bphi,Bz = self._ModelFunc(rho,abs_z,z)
		t2 = time.time()
		   
		#return to SIII coordinates
		Bout[:,0],Bout[:,1],Bout[:,2] = self._OutputConv(cost,sint,cosp,
											sinp,x,y,rho,Brho,Bphi,Bz,
											self._cosxp,self._sinxp,
											self._cosxt,self._sinxt)
		t3 = time.time()
		#turn into a nx3 array

		
		print('-------------------------------------------------------')
		print('Field Timing:')
		print('-------------------------------------------------------')
		print('Coordinate conversion: {:f}μs ({:6.2f}%)'.format((t1-t0)*1e6,100*(t1-t0)/(t3-t0)))
		print('Call Model: {:f}μs ({:6.2f}%)'.format((t2-t1)*1e6,100*(t2-t1)/(t3-t0)))
		print('Output Conversion: {:f}μs ({:6.2f}%)'.format((t3-t2)*1e6,100*(t3-t2)/(t3-t0)))
		print('Total: {:f}μs'.format((t3-t0)*1e6))

		return Bout
