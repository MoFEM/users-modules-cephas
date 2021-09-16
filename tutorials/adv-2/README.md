To perform an analysis with thermo-plasticity use the following command line

-omega_0 2e-3 -omega_h 2e-3 -omega_inf 2e-3 are the parameters, which regulate the sensibility of yield stress, linear hardening and saturation stress to temperature. If they are 0.0 no effect from temperature is considered and the model is coupled only in a single direction.

-number_of_cycles_in_one_hour 0 this parameter allows user to create a monotonic load case, when equal to 0, or a cyclic one (sinusoid load amplitude) if is larger than 0.

-fraction_of_dissipation 0.3 this parameter regulates how much plastic dissipation energy is converted into heat, default value is 0.9 from plasticity theory.

./thermo_plastic_2d -file_name adv2_thermo_plast_2D.cub -ts_max_time 1.0 -ts_adapt_type basic -ts_dt 0.001 -ts_adapt_dt_max 0.002 -ts_adapt_dt_min 0.0005 -snes_atol 1e-6 -snes_rtol 1e-6 -snes_max_it 40 -order 2 -ts_max_snes_failures -1 -ts_type theta -ts_theta_initial_guess_extrapolate 1 -ts_theta_theta 1 -ts_exact_final_time matchstep 1 -calculate_reactions 1 -reaction_id 5 -fix_skin_flux 0 -snes_linesearch_type l2 -number_of_cycles_in_one_hour 0 -Qinf 265 -b_iso 17 -hardening_viscous 100 -omega_0 2e-3 -omega_h 2e-3 -omega_inf 2e-3 -capacity 3.6 2>&1 | tee log_temp_sy_h_Q
