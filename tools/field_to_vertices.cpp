/** \file field_to_vertices.cpp
  \brief Field to vertices
  \example field_to_vertices.cpp

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
#include <BasicFiniteElements.hpp>

using namespace MoFEM;

static char help[] = "...\n\n";

int main(int argc, char *argv[]) {
  MoFEM::Core::Initialize(&argc, &argv, (char *)0, help);

  try {

    // global variables
    char mesh_file_name[255];
    PetscBool flg_file = PETSC_FALSE;
    char field_name_param[255] = "RHO";
    CHKERR PetscOptionsBegin(PETSC_COMM_WORLD, "", "Field to vertices options", "none");
    CHKERR PetscOptionsString("-my_file", "mesh file name", "", "mesh.h5m",
                              mesh_file_name, 255, &flg_file);
    CHKERR PetscOptionsString("-my_field", "field name", "", "FIELD",
                              field_name_param, 255, PETSC_NULL);
    ierr = PetscOptionsEnd(); CHKERRG(ierr);

    std::string field_name(field_name_param);

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
      SETERRQ(PETSC_COMM_SELF, MOFEM_INVALID_DATA,
              "*** ERROR -my_file (MESH FILE NEEDED)");
    }
    BitRefLevel bit_level0;
    bit_level0.set(0);
    CHKERR m_field.getInterface<BitRefManager>()->setBitRefLevelByDim(
        0, 3, bit_level0);
    CHKERR m_field.build_fields();

    bool field_flg = false;
    auto fields_ptr = m_field.get_fields();
    for(auto field : (*fields_ptr)) {
      bool check_space = field->getSpace() == H1;
      if(field->getName() == field_name && check_space) field_flg = true;
    }
     if (!field_flg) {
      SETERRQ(PETSC_COMM_SELF, MOFEM_INVALID_DATA,
              "*** ERROR -my_field (FIELD (in H1 space) is NOT FOUND)");
    }


    SaveVertexDofOnTag ent_method(m_field, field_name.c_str());

    CHKERR m_field.loop_dofs(field_name.c_str(),ent_method);
    PetscPrintf(PETSC_COMM_WORLD, "\nDone. Saving files... \n");

    //TODO: Higher order field mapping
    CHKERR m_field.getInterface<BitRefManager>()->writeBitLevelByType(
        bit_level0, BitRefLevel().set(), MBTET, "out_mesh.vtk", "VTK", "");
    CHKERR moab.write_file("out.h5m");
  }
  CATCH_ERRORS;

  CHKERR MoFEM::Core::Finalize();

  return 0;
}
