/** \file magnetostatic.cpp
 * \example magnetostatic.cpp
 * \ingroup maxwell_element
 *
 * \brief Example implementation of magnetostatics problem
 *
 */

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
                                 "-pc_factor_mat_solver_type mumps \n"
                                 "-mat_mumps_icntl_20 0 \n"
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
    auto moab_comm_wrap =
        boost::make_shared<WrapMPIComm>(PETSC_COMM_WORLD, false);
    if (pcomm == NULL)
      pcomm =
          new ParallelComm(&moab, moab_comm_wrap->get_comm());

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
