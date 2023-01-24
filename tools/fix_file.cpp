/** \file fix_file.cpp
  \brief Load and save file
*/

#include <MoFEM.hpp>
using namespace MoFEM;

static char help[] = "...\n\n";

int main(int argc, char *argv[]) {

  MoFEM::Core::Initialize(&argc, &argv, (char *)0, help);

  try {
    char mesh_file_name[255] = "in.h5m";
    char out_file_name[255] = "out.h5m";
    PetscBool in_flg_file = PETSC_FALSE;
    PetscBool out_flg_file = PETSC_FALSE;
    CHKERR PetscOptionsBegin(PETSC_COMM_WORLD, "", "Fix file options", "none");
    CHKERR PetscOptionsString("-file_name", "mesh file name", "", "mesh.h5m",
                              mesh_file_name, 255, &out_flg_file);
    CHKERR PetscOptionsString("-out_file_name", "mesh file name", "",
                              "mesh.h5m", out_file_name, 255, &out_flg_file);
    ierr = PetscOptionsEnd();
    CHKERRG(ierr);

    moab::Core mb_instance;
    moab::Interface &moab = mb_instance;
    ParallelComm *pcomm = ParallelComm::get_pcomm(&moab, MYPCOMM_INDEX);
    const char *option;
    option = "";

    MOFEM_LOG("WORLD", Sev::inform)
        << "In " << mesh_file_name << " out " << out_file_name;

    CHKERR moab.load_file(mesh_file_name, 0, option);
		CHKERR BitRefManager::fixTagSize(moab);
    CHKERR moab.write_file(out_file_name);
  }

  CATCH_ERRORS;

  CHKERR MoFEM::Core::Finalize();
}