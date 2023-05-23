/** \file extrude_prisms.cpp
 * \brief Extrude prisms from surface elements block
 * \example mesh_cut.cpp
 *
 * \ingroup mesh_cut
 */



#include <MoFEM.hpp>

using namespace MoFEM;

static char help[] = "mesh cutting\n\n";

int main(int argc, char *argv[]) {

  MoFEM::Core::Initialize(&argc, &argv, (char *)0, help);

  try {

    PetscBool flg_myfile = PETSC_TRUE;
    char mesh_file_name[255];
    char mesh_out_file[255] = "out.h5m";

    CHKERR PetscOptionsBegin(PETSC_COMM_WORLD, "", "Prism surface", "none");
    CHKERR PetscOptionsString("-my_file", "mesh file name", "", "mesh.h5m",
                              mesh_file_name, 255, &flg_myfile);
    CHKERR PetscOptionsString("-output_file", "output mesh file name", "",
                              "out.h5m", mesh_out_file, 255, PETSC_NULL);
    ierr = PetscOptionsEnd();
    CHKERRG(ierr);

    if (flg_myfile != PETSC_TRUE)
      SETERRQ(PETSC_COMM_SELF, MOFEM_INVALID_DATA,
              "error -my_file (mesh file needed)");

    moab::Core mb_instance;
    moab::Interface &moab = mb_instance;
    ParallelComm *pcomm = ParallelComm::get_pcomm(&moab, MYPCOMM_INDEX);
    if (pcomm == NULL)
      pcomm = new ParallelComm(&moab, PETSC_COMM_WORLD);

    const char *option;
    option = "";
    CHKERR moab.load_file(mesh_file_name, 0, option);

    MoFEM::Core core(moab);
    MoFEM::CoreInterface &m_field =
        *(core.getInterface<MoFEM::CoreInterface>());

    PrismsFromSurfaceInterface *prisms_from_surface_interface;
    CHKERR m_field.getInterface(prisms_from_surface_interface);

    const std::string extrude_block_name = "EXTRUDE_PRISMS";
    for (_IT_CUBITMESHSETS_BY_SET_TYPE_FOR_LOOP_(m_field, BLOCKSET, it)) {
      if (it->getName().compare(0, extrude_block_name.length(),
                                extrude_block_name) == 0) {
        std::vector<double> thickness;
        CHKERR it->getAttributes(thickness);
        if (thickness.size() != 2)
          SETERRQ(PETSC_COMM_SELF, MOFEM_DATA_INCONSISTENCY,
                  "Data inconsistency");
        Range tris;
        CHKERR m_field.get_moab().get_entities_by_type(it->meshset, MBTRI, tris,
                                                       true);
        Range block_prisms;
        CHKERR prisms_from_surface_interface->createPrisms(
            tris, PrismsFromSurfaceInterface::NO_SWAP, block_prisms);
        CHKERR prisms_from_surface_interface->setNormalThickness(
            block_prisms, thickness[0], thickness[1]);
        CHKERR prisms_from_surface_interface->updateMeshestByEdgeBlock(
            block_prisms);
        CHKERR prisms_from_surface_interface->updateMeshestByTriBlock(
            block_prisms);

        std::cout << "Extrude block " << it->getMeshsetId() << " set prisms "
                  << block_prisms.size() << endl;
      }
    }

    CHKERR moab.write_file(mesh_out_file);
  }
  CATCH_ERRORS;

  MoFEM::Core::Finalize();

  return 0;
}