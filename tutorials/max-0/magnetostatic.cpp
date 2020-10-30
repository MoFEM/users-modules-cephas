/** \file magnetostatic.cpp
 * \example magnetostatic.cpp
 * \ingroup maxwell_element
 *
 * \brief Example implementation of magnetostatics problem
 *
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

#include <BasicFiniteElements.hpp>
#include <MagneticElement.hpp>
using namespace MoFEM;

static char help[] = "-my_file mesh file name\n"
                     "-my_order default approximation order\n"
                     "-my_is_partitioned set if mesh is partitioned\n"
                     "-regression_test check solution vector against expected value\n"
                     "\n\n";

int main(int argc, char *argv[]) {

  const string default_options = "-ksp_type fgmres \n"
                                 "-pc_type lu \n"
                                 "-pc_factor_mat_solver_package mumps \n"
                                 "-mat_mumps_icntl_13 1 \n"
                                 "-ksp_monitor \n"
                                 "-mat_mumps_icntl_24 1 \n"
                                 "-mat_mumps_icntl_13 1 \n";

  string param_file = "param_file.petsc";
  if (!static_cast<bool>(ifstream(param_file))) {
    std::ofstream file(param_file.c_str(), std::ios::ate);
    if (file.is_open()) {
      file << default_options;
      file.close();
    }
  }

  MoFEM::Core::Initialize(&argc, &argv, param_file.c_str(), help);

  try {

    moab::Core mb_instance;
    moab::Interface &moab = mb_instance;

    ParallelComm *pcomm = ParallelComm::get_pcomm(&moab, MYPCOMM_INDEX);
    if (pcomm == NULL)
      pcomm = new ParallelComm(&moab, PETSC_COMM_WORLD);

    // Read parameters from line command
    PetscBool flg_file;
    char mesh_file_name[255];
    PetscInt order = 2;
    PetscBool is_partitioned = PETSC_FALSE;
    PetscBool regression_test = PETSC_FALSE;

    CHKERR PetscOptionsBegin(PETSC_COMM_WORLD, "", "Magnetostatics options",
                             "none");
    CHKERR PetscOptionsString("-my_file", "mesh file name", "", "mesh.h5m",
                              mesh_file_name, 255, &flg_file);
    CHKERR PetscOptionsInt("-my_order", "default approximation order", "",
                           order, &order, PETSC_NULL);
    CHKERR PetscOptionsBool(
        "-regression_test",
        "if set norm of solution vector is check agains expected value ",
        "",
        PETSC_FALSE, &regression_test, PETSC_NULL);

    ierr = PetscOptionsEnd();
    CHKERRG(ierr);

    // Read mesh to MOAB
    const char *option;
    option = "PARALLEL=READ_PART;"
             "PARALLEL_RESOLVE_SHARED_ENTS;"
             "PARTITION=PARALLEL_PARTITION;";
    CHKERR moab.load_file(mesh_file_name, 0, option);

    // Create mofem interface
    MoFEM::Core core(moab);
    MoFEM::Interface &m_field = core;

    MagneticElement magnetic(m_field);
    magnetic.blockData.oRder = order;
    CHKERR magnetic.getNaturalBc();
    CHKERR magnetic.getEssentialBc();
    CHKERR magnetic.createFields();
    CHKERR magnetic.createElements();
    CHKERR magnetic.createProblem();
    CHKERR magnetic.solveProblem(regression_test == PETSC_TRUE);
    CHKERR magnetic.postProcessResults();
    CHKERR magnetic.destroyProblem();

    // write solution to file (can be used by lorentz_force example)
    CHKERR moab.write_file("solution.h5m", "MOAB", "PARALLEL=WRITE_PART");
  }
  CATCH_ERRORS;

  MoFEM::Core::Finalize();
  return 0;
}
