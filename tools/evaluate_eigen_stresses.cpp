/** \file reading_med.cpp

  \brief Reading med files

*/

/* This file is part of MoFEM.
 * MoFEM is free software: you can redistribute it and/or modify it under
 * the terms of the GNU Lesser General Public License as published by the
 * Free Software Foundation, either version 3 of the License, or (at your
 * option) any later version.
 *
 * MoFEM is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public
 * License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with MoFEM. If not, see <http://www.gnu.org/licenses/>. */

#include <MoFEM.hpp>
using namespace MoFEM;

static char help[] = "...\n\n";

int main(int argc, char *argv[]) {
  MoFEM::Core::Initialize(&argc, &argv, (char *)0, help);

  try {

    moab::Core mb_instance;
    moab::Interface &moab = mb_instance;
    ParallelComm *pcomm = ParallelComm::get_pcomm(&moab, MYPCOMM_INDEX);
    if (pcomm == NULL)
      pcomm = new ParallelComm(&moab, PETSC_COMM_WORLD);

    // Create MoFEM database
    MoFEM::Core core(moab);
    MoFEM::Interface &m_field = core;

    PetscBool flg_file, flg_field;
    char mesh_out_file[255] = "out.h5m";
    char mesh_file_name[255];
    char actual_strain_field[255];
    char actual_stress_field[255];
    char youngs_modulus[255];

    int time_step = 0;
    CHKERR PetscOptionsBegin(m_field.get_comm(), "", "Read MED tool", "none");
    CHKERR PetscOptionsInt("-med_time_step", "time step", "", time_step,
                           &time_step, PETSC_NULL);
    CHKERR PetscOptionsString("-output_file", "output mesh file name", "",
                              "out.h5m", mesh_out_file, 255, PETSC_NULL);

    CHKERR PetscOptionsString("-my_file", "mesh file name", "", "mesh.h5m",
                              mesh_file_name, 255, &flg_file);

    CHKERR PetscOptionsString("-my_actual_strains", "mesh file name", "", "",
                              actual_strain_field, 255, &flg_field);

    CHKERR PetscOptionsString("-my_actual_stress", "mesh file name", "", "",
                              actual_stress_field, 255, &flg_field);

    CHKERR PetscOptionsString("-my_youngs_modulus", "mesh file name", "", "",
                              youngs_modulus, 255, &flg_field);


    ierr = PetscOptionsEnd();
    CHKERRQ(ierr);

    if (flg_file != PETSC_TRUE) {
      SETERRQ(PETSC_COMM_SELF, 1, "*** ERROR -my_file (MESH FILE NEEDED)");
    }

    if (flg_field != PETSC_TRUE) {
      SETERRQ(PETSC_COMM_SELF, 1,
              "*** ERROR -my_field_name (FIELD NAME NEEDED)");
    }

    const char *option;
    option = "";
    CHKERR moab.load_file(mesh_file_name, 0, option);

    Tag th_actual_strain;
    CHKERR m_field.get_moab().tag_get_handle(actual_strain_field, th_actual_strain);

    Tag th_actual_stress;
    CHKERR m_field.get_moab().tag_get_handle(actual_stress_field, th_actual_stress);

    Tag th_youngs_modulus;
    CHKERR m_field.get_moab().tag_get_handle(youngs_modulus, th_youngs_modulus);

    Range tets;
    CHKERR m_field.get_moab().get_entities_by_type(0, MBTET, tets);

    auto get_tag = [&](const std::string name) {
      std::array<double, 9> def;
      std::fill(def.begin(), def.end(), 0);
      Tag th;
      CHKERR m_field.get_moab().tag_get_handle(name.c_str(), 9, MB_TYPE_DOUBLE,
                                               th, MB_TAG_CREAT | MB_TAG_SPARSE,
                                               def.data());
      return th;
    };


    auto get_scalar = [&](const std::string name) {
      double def_val[] = {0};
      Tag th;
      CHKERR m_field.get_moab().tag_get_handle(name.c_str(), 1, MB_TYPE_DOUBLE,
                                               th, MB_TAG_CREAT | MB_TAG_SPARSE,
                                               def_val);
      return th;
    };


    auto th_new_data = get_tag("MED_EIGEN_STRESSES_EVALUATED");
    auto th_difference = get_tag("EIGEN_STRESSES_DIFFERENCE");
    auto th_eigen_from_med = get_tag("MED_ST_HET_G");
    auto th_rel_error = get_tag("EIGEN_STRESSES_REL_ERROR");
    auto th_energy_error = get_scalar("EIGEN_ENERGY_ERROR");
    auto th_check_eigen_stress = get_tag("CHECK_EIGEN_STRESS");


    cerr << actual_strain_field<<"\n" ;
    cerr << actual_stress_field<<"\n" ;
    cerr << youngs_modulus<<"\n" ;

      MatrixDouble stiffness_matrix;
      stiffness_matrix.resize(9, 9, false);


    for (Range::iterator pit = tets.begin(); pit != tets.end(); ++pit) {
        stiffness_matrix.clear();
      // auto get_vec_from_array = [](double  *array_help) {
      //   return FTensor::Tensor1<VectorDouble9 , 6>(&array_help[0], &array_help[1],
      //                                        &array_help[2], &array_help[3], &array_help[4],
      //                                        &array_help[5]);
      // };


      VectorDouble9 vector_values_eigen(9);
      CHKERR m_field.get_moab().tag_get_data(th_new_data, &*pit, 1,
                                             &*vector_values_eigen.begin());

      VectorDouble9 vector_actual_strain(9);
          CHKERR m_field.get_moab().tag_get_data(th_actual_strain, &*pit, 1,
                                                &*vector_actual_strain.begin());

      VectorDouble9 vector_actual_stress(9);
      CHKERR m_field.get_moab().tag_get_data(th_actual_stress, &*pit, 1,
                                             &*vector_actual_stress.begin());


      VectorDouble vector_youngs_modulus;
      vector_youngs_modulus.resize(1);
      vector_youngs_modulus.clear();
      CHKERR m_field.get_moab().tag_get_data(th_youngs_modulus, &*pit, 1,
                                             &*vector_youngs_modulus.begin());


      // for (int ii = 0; ii != 9; ++ii) {
      //   vector_values[ii] = tet_map.at(*pit)[ii];
      // }

      const double poisson_ratio = 0.2;
      double one_min_two_poisson = (1 - 2. * poisson_ratio);
      
      double mult_ratio = vector_youngs_modulus[0] / ( (1 + poisson_ratio) * one_min_two_poisson);
      for (int ii = 0; ii != 3; ++ii){
        stiffness_matrix(ii, ii) = mult_ratio * (1 - poisson_ratio);
        switch(ii){
        case(0):
        stiffness_matrix(ii, 1) = stiffness_matrix(ii, 2) = mult_ratio * poisson_ratio;
          break;
        case(1):
        stiffness_matrix(ii, 0) = stiffness_matrix(ii, 2) = mult_ratio * poisson_ratio;
          break;
        case(2):
        stiffness_matrix(ii, 0) = stiffness_matrix(ii, 1) = mult_ratio * poisson_ratio;
          break;
        }
      }

      for (int ii = 3; ii != 6; ++ii) {
        stiffness_matrix(ii, ii) = 0.5 * mult_ratio * one_min_two_poisson;
      }

      VectorDouble9 thermal_values(9);
      noalias(thermal_values) = prod(stiffness_matrix, vector_actual_strain);
      vector_values_eigen = vector_actual_stress - thermal_values;
      double *eigen_stresses_vaules;
      CHKERR m_field.get_moab().tag_get_by_ptr(
          th_new_data, &*pit, 1, (const void **)&eigen_stresses_vaules);
      
      for (int ii = 0; ii != 6; ++ii) {
        eigen_stresses_vaules[ii] = vector_values_eigen(ii);
      }

      VectorDouble9 vector_eigen_from_med(9);
      CHKERR m_field.get_moab().tag_get_data(th_eigen_from_med, &*pit, 1,
                                             &*vector_eigen_from_med.begin());

      double *eigen_stresses_difference;
      CHKERR m_field.get_moab().tag_get_by_ptr(
          th_difference, &*pit, 1, (const void **)&eigen_stresses_difference);

      double *eigen_rel_error;
      CHKERR m_field.get_moab().tag_get_by_ptr(
          th_rel_error, &*pit, 1, (const void **)&eigen_rel_error);

      for (int ii = 0; ii != 6; ++ii) {
        eigen_stresses_difference[ii] =
            vector_values_eigen(ii) - vector_eigen_from_med(ii);
            eigen_rel_error[ii] = abs(eigen_stresses_difference[ii]) / abs(vector_eigen_from_med(ii));
      }



  auto get_symm_tensor = [](auto &m) {
    FTensor::Tensor2_symmetric<double *, 3> t(

        &m(0), &m(3), &m(4),

        &m(1), &m(5),

        &m(2)

    );
    return t;
  };

  auto t_stress_med = get_symm_tensor(vector_eigen_from_med);
  auto t_stress_calc = get_symm_tensor(vector_values_eigen);
  auto t_stress_actual = get_symm_tensor(vector_actual_stress);

    FTensor::Index<'i', 3> i;
    FTensor::Index<'j', 3> j;
    FTensor::Index<'k', 3> k;
    FTensor::Index<'l', 3> l;

    double *stress_energy_error;
    CHKERR m_field.get_moab().tag_get_by_ptr(
        th_energy_error, &*pit, 1, (const void **)&stress_energy_error);
    stress_energy_error[0] = sqrt((t_stress_calc(i, j) - t_stress_med(i, j)) *
                                  (t_stress_calc(i, j) - t_stress_med(i, j)) /
                                  (t_stress_calc(i, j) * t_stress_calc(i, j)));

    
  //   auto t_strain_sym = get_symm_tensor(vector_actual_strain);

  //    FTensor::Ddg<double, 3, 3> t_D;

  //   const double young = vector_youngs_modulus[0];
  //   const double poisson = 0.2;

  //   const double coefficient = young / ((1 + poisson) * (1 - 2 * poisson));

  //   t_D(i, j, k, l) = 0.;
  //   t_D(0, 0, 0, 0) = t_D(1, 1, 1, 1) = t_D(2, 2, 2, 2) = 1 - poisson;
  //   t_D(0, 1, 0, 1) = t_D(0, 2, 0, 2) = t_D(1, 2, 1, 2) =
  //       0.5 * (1 - 2 * poisson);
  //   t_D(0, 0, 1, 1) = t_D(1, 1, 0, 0) = t_D(0, 0, 2, 2) = t_D(2, 2, 0, 0) =
  //       t_D(1, 1, 2, 2) = t_D(2, 2, 1, 1) = poisson;
  //   t_D(i, j, k, l) *= coefficient;
    
  //   FTensor::Tensor2_symmetric<double, 3> t_eigen_stress;
  //   t_eigen_stress(i, j) = t_stress_actual(i, j) - t_D(i, j, k, l) * t_strain_sym(k, l);


  // auto tensor_to_tensor = [](const auto &t1, auto &t2) {
  //   t2(0, 0) = t1(0, 0);
  //   t2(1, 1) = t1(1, 1);
  //   t2(2, 2) = t1(2, 2);
  //   t2(0, 1) = t2(1, 0) = t1(1, 0);
  //   t2(0, 2) = t2(2, 0) = t1(2, 0);
  //   t2(1, 2) = t2(2, 1) = t1(2, 1);
  // };

  // FTensor::Tensor2<double, 3, 3> t_stress;

  // tensor_to_tensor(t_eigen_stress, t_stress);

  // CHKERR m_field.get_moab().tag_set_data(th_check_eigen_stress, &*pit, 1,
  //                                  &t_stress(0, 0));

  // cerr << "stress_energy_error  " << stress_energy_error[0] << "\n";
  // for (int ii = 0; ii != 9; ++ii) {
  //   cerr << "vector_values_eigen   " << vector_values_eigen[ii] << "\n";
  // }
    }

    CHKERR moab.write_file(mesh_out_file);
  }
  CATCH_ERRORS;

  MoFEM::Core::Finalize();

  return 0;
}
