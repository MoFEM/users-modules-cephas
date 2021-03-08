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
    PetscBool flg_file_2 = PETSC_FALSE;
    char field_name_param[255] = "SPATIAL_POSITION";
    char mesh_out_file[255] = "deformed_mesh.h5m";
    PetscBool flg_use_displ = PETSC_FALSE;
    CHKERR PetscOptionsBegin(PETSC_COMM_WORLD, "", "Field to vertices options",
                             "none");
    CHKERR PetscOptionsString("-my_file", "mesh file name", "", "mesh.h5m",
                              mesh_file_name, 255, &flg_file);
    CHKERR PetscOptionsString("-file_name", "mesh file name", "", "mesh.h5m",
                              mesh_file_name, 255, &flg_file_2);
    CHKERR PetscOptionsString("-my_field", "field name", "", "FIELD",
                              field_name_param, 255, PETSC_NULL);
    CHKERR PetscOptionsString("-output_file", "output mesh file name", "",
                              "deformed_mesh.h5m", mesh_out_file, 255,
                              PETSC_NULL);
    CHKERR PetscOptionsBool("-flg_use_displ", "true to squash bits at the end",
                            "", flg_use_displ, &flg_use_displ, PETSC_NULL);
    ierr = PetscOptionsEnd();
    CHKERRG(ierr);

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

    if (!flg_file && !flg_file_2) {
      SETERRQ(PETSC_COMM_SELF, MOFEM_INVALID_DATA,
              "*** ERROR -my_file (MESH FILE NEEDED)");
    }
    BitRefLevel bit_level0;
    bit_level0.set(0);
    CHKERR m_field.getInterface<BitRefManager>()->setBitRefLevelByDim(
        0, 3, bit_level0);

    Range ents_vertices;
    rval = moab.get_entities_by_dimension(0, 0, ents_vertices, false);

    Tag th_spatial_positions;
    CHKERR m_field.get_moab().tag_get_handle(field_name.c_str(),
                                             th_spatial_positions);

    double tag_val[3];
    double coords[3];

    for (Range::iterator nit = ents_vertices.begin();
         nit != ents_vertices.end(); nit++) {
      CHKERR m_field.get_moab().tag_get_data(th_spatial_positions, &*nit, 1,
                                             tag_val);
      if (flg_use_displ) {
        CHKERR moab.get_coords(&*nit, 1, coords);
        for (int ii = 0; ii != 3; ++ii)
          coords[ii] += tag_val[ii];

        CHKERR moab.set_coords(&*nit, 1, coords);
      } else {
        CHKERR moab.set_coords(&*nit, 1, tag_val);
      }
    }

    CHKERR moab.write_file(mesh_out_file);
  }
  CATCH_ERRORS;

  CHKERR MoFEM::Core::Finalize();

  return 0;
}
