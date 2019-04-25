/** \file meshset_to_vtk.cpp
 * \example meshset_to_vtk.cpp
 * \brief Print all meshset to VTK file
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

#include <MoFEM.hpp>
using namespace MoFEM;

static char help[] = "...\n\n";

int main(int argc, char *argv[]) {

  MoFEM::Core::Initialize(&argc, &argv, (char *)0, help);

  try {

    // global variables
    char mesh_file_name[255];
    PetscBool flg_file = PETSC_FALSE;

    ierr = PetscOptionsBegin(PETSC_COMM_WORLD, "", "none", "none");
    CHKERRQ(ierr);

    CHKERR PetscOptionsString("-my_file", "mesh file name", "", "mesh.h5m",
                              mesh_file_name, 255, &flg_file);

    ierr = PetscOptionsEnd();
    CHKERRQ(ierr);

    moab::Core mb_instance;
    moab::Interface &moab = mb_instance;
    ParallelComm *pcomm = ParallelComm::get_pcomm(&moab, MYPCOMM_INDEX);
    if (pcomm == NULL)
      pcomm = new ParallelComm(&moab, PETSC_COMM_WORLD);

    const char *option;
    option = "";
    CHKERR moab.load_file(mesh_file_name, 0, option);

    // Create MoFEM  database
    MoFEM::Core core(moab);
    MoFEM::Interface &m_field = core;

    if (flg_file != PETSC_TRUE) {
      SETERRQ(PETSC_COMM_SELF, 1, "*** ERROR -my_file (MESH FILE NEEDED)");
    }

    MeshsetsManager *meshsets_manager;
    CHKERR m_field.getInterface(meshsets_manager);
    CHKERR meshsets_manager->setMeshsetFromFile();

    const CubitBCType mask = CubitBCType(BLOCKSET | SIDESET | NODESET);

    for (auto &cit : meshsets_manager->getMeshsetsMultindex()) {

      std::string type = "_";
      if((cit.getBcType() & CubitBCType(BLOCKSET)).any())
        type += "BLOCKSET_";
      if((cit.getBcType() & CubitBCType(SIDESET)).any())
        type += "SIDESET_";
      if((cit.getBcType() & CubitBCType(NODESET)).any())
        type += "NODESET_";

      std::string name = "meshset" + type +
                         boost::lexical_cast<std::string>(cit.getMeshsetId()) +
                         ".vtk";
      std::cout << "Writting file: " << name << endl;

      CHKERR meshsets_manager->saveMeshsetToFile(
          cit.getMeshsetId(), (cit.getBcType() & mask).to_ulong(), name);
    }
  }
  CATCH_ERRORS;

  MoFEM::Core::Finalize();
  CHKERRQ(ierr);

  return 0;
}
