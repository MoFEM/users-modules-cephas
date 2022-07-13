/** \file reading_med.cpp

  \brief Reading med files

*/



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

    char mesh_out_file[255] = "out.h5m";

    int time_step = 0;
    CHKERR PetscOptionsBegin(m_field.get_comm(), "", "Read MED tool", "none");
    CHKERR PetscOptionsInt("-med_time_step", "time step", "", time_step,
                           &time_step, PETSC_NULL);
    CHKERR PetscOptionsString("-output_file", "output mesh file name", "",
                              "out.h5m", mesh_out_file, 255, PETSC_NULL);
    ierr = PetscOptionsEnd();
    CHKERRQ(ierr);

    MedInterface *med_interface_ptr;
    CHKERR m_field.getInterface(med_interface_ptr);
    CHKERR med_interface_ptr->readMed();
    CHKERR med_interface_ptr->medGetFieldNames();

    for (std::map<std::string, MedInterface::FieldData>::iterator fit =
             med_interface_ptr->fieldNames.begin();
         fit != med_interface_ptr->fieldNames.end(); fit++) {
      CHKERR med_interface_ptr->readFields(med_interface_ptr->medFileName,
                                           fit->first, false, time_step);
    }

    // Add meshsets if config file provided
    MeshsetsManager *meshsets_interface_ptr;
    CHKERR m_field.getInterface(meshsets_interface_ptr);
    CHKERR meshsets_interface_ptr->setMeshsetFromFile();

    MOFEM_LOG_CHANNEL("WORLD");
    MOFEM_LOG_TAG("WORLD", "read_med")
    MOFEM_LOG("WORLD", Sev::inform)
        << "Print all meshsets (old and added from meshsets "
           "configurational file";
    for (auto cit = meshsets_interface_ptr->getBegin();
         cit != meshsets_interface_ptr->getEnd(); cit++)
      MOFEM_LOG("WORLD", Sev::inform) << *cit;

    CHKERR moab.write_file(mesh_out_file);
  }
  CATCH_ERRORS;

  MoFEM::Core::Finalize();

  return 0;
}
