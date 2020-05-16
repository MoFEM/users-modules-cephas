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
    MOFEM_LOG("WORLD", sev::inform)
        << "Print all meshsets (old and added from meshsets "
           "configurational file";
    for (auto cit = meshsets_interface_ptr->getBegin();
         cit != meshsets_interface_ptr->getEnd(); cit++)
      MOFEM_LOG("WORLD", sev::inform) << *cit;

    CHKERR moab.write_file(mesh_out_file);
  }
  CATCH_ERRORS;

  MoFEM::Core::Finalize();

  return 0;
}
