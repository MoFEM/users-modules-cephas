#include <MoFEM.hpp>
#include <BasicFiniteElements.hpp>

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

    // global variables
    char mesh_file_name[255] = "mesh.h5m";
    char mesh_out_file[255] = "out.h5m";
    int dim = 3;
    PetscBool flg_file;

    ierr = PetscOptionsBegin(PETSC_COMM_WORLD, "", "none", "none");
    CHKERRQ(ierr);

    CHKERR PetscOptionsString("-file_name", "mesh file name", "", "mesh.h5m",
                              mesh_file_name, 255, &flg_file);
    CHKERR PetscOptionsString("-output_file", "output mesh file name", "",
                              mesh_out_file, mesh_out_file, 255, PETSC_NULL);
    CHKERR PetscOptionsInt("-dim", "mesh dimension", "", dim, &dim, PETSC_NULL);


    ierr = PetscOptionsEnd();
    CHKERRQ(ierr);

    if (flg_file != PETSC_TRUE)
      SETERRQ(PETSC_COMM_SELF, 1,
              "*** ERROR -my_file (-file_name) (MESH FILE NEEDED)");

    MOFEM_LOG("WORLD", Sev::inform) << "In file " << mesh_file_name;
    MOFEM_LOG("WORLD", Sev::inform) << "Out file " << mesh_out_file;
    MOFEM_LOG("WORLD", Sev::inform) << "Mesh dimension " << dim;

    const char *option;
    option = "";
    CHKERR moab.load_file(mesh_file_name, 0, option);

    Range ents, skin;
    CHKERR moab.get_entities_by_dimension(0, dim, ents);
    MOFEM_LOG("WORLD", Sev::verbose) << "Ents:\n" << ents;

    Skinner skinner(&moab);
    CHKERR skinner.find_skin(0, ents, false, skin);
    MOFEM_LOG("WORLD", Sev::verbose) << "Skin:\n" << skin;

    EntityHandle set;
    CHKERR moab.create_meshset(MESHSET_SET, set);
    CHKERR moab.add_entities(set, skin);

    CHKERR moab.write_file(mesh_out_file, "MOAB", "", &set, 1);
  }
  CATCH_ERRORS;

  CHKERR MoFEM::Core::Finalize();

  return 0;
}