To perform an analysis with thermo-plasticity use the following command line

-omega_0 2e-3 -omega_h 2e-3 -omega_inf 2e-3 are the parameters, which regulate the sensibility of yield stress, linear hardening and saturation stress to temperature. If they are 0.0 no effect from temperature is considered and the model is coupled only in a single direction.

-number_of_cycles_in_total_time 0 this parameter allows the user to create a monotonic load case, when equal to 0. If it is larger than zero, it indicates the number of full-cycle over the -ts_max_time imposed (period or inverse of frequency). The unit used needs to reflect the units adopted for all other physical quantities.

The sinusoid function is implemented as follows:

y = amplitude_cycle*sin(2*pi*time-phase_shift)*number_of_cycles+amplitude_shift

-amplitude_cycle 0.5 regulates the amplitude of the sinusoid function
-amplitude_shift 0.5 regulates the shift in amplitude respect the horizontal axis
-phase_shift 0.8 regulates the phase of the sinusoid function

-fraction_of_dissipation 0.3 this parameter regulates how much plastic dissipation energy is converted into heat, default value is 0.9 from plasticity theory.

./thermo_plastic_2d -file_name adv2_thermo_plast_2D.cub -ts_max_time 0.085 -ts_adapt_type basic -ts_dt 0.001 -ts_adapt_dt_max 0.002 -ts_adapt_dt_min 0.0005 -snes_atol 1e-6 -snes_rtol 1e-6 -snes_max_it 40 -order 2 -ts_max_snes_failures -1 -ts_type theta -ts_theta_initial_guess_extrapolate 1 -ts_theta_theta 1 -ts_exact_final_time matchstep 1 -calculate_reactions 1 -reaction_id 5 -fix_skin_flux 0 -snes_linesearch_type l2 -number_of_cycles_in_total_time 0 -Qinf 265 -b_iso 17 -hardening_viscous 100 -omega_0 2e-3 -omega_h 2e-3 -omega_inf 2e-3 -capacity 3.6 2>&1 | tee log_temp_sy_h_Q

To test thermo-elasticity the following command line can be used:

./thermo_plastic_2d -file_name adv2_thermo_elast_2D.cub -ts_max_time 1 -ts_adapt_type basic -ts_dt 0.25 -ts_adapt_dt_max 0.25 -ts_adapt_dt_min 0.005 -snes_atol 1e-7 -snes_rtol 1e-7 -snes_max_it 40 -order 2 -ts_max_snes_failures -1 -ts_theta_initial_guess_extrapolate 1 -ts_exact_final_time matchstep 1 -fix_skin_flux 0 -snes_linesearch_type l2 -young_modulus 206913 -yield_stress 450 -hardening 130 -fraction_of_dissipation 0.9 -Qinf 265 -b_iso 17 -omega_0 0 -omega_h 0 -omega_inf 0 -capacity 3.6 -conductivity 18.2 -coeff_expansion 1e-6 2>&1 | tee log

From analytical formula du = (alpha _ dT _ Length^2) 2\*h
The Cantilever has L = 100 and h = 10 the dT imposed is 200 and alpha = 1e-6. The expected du = 0.1
