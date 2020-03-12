/** \file pavy_surface.cpp
  \example pavy_surface.cpp
  \brief Creating wavy surface

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

namespace bio = boost::iostreams;
using bio::stream;
using bio::tee_device;

using namespace MoFEM;

static char help[] = "...\n\n";
static int debug = 1;

int main(int argc, char *argv[]) {

  MoFEM::Core::Initialize(&argc, &argv, (char *)0, help);

  try {

    moab::Core mb_instance;
    moab::Interface &moab = mb_instance;
    int rank;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);

    // Read parameters from line command
    PetscBool flg_mesh_file = PETSC_TRUE;
    PetscBool flg_out_file = PETSC_TRUE;
    char mesh_file_name[255];
    char out_file_name[255] = "out.h5m";

    double lambda = 1.0;
    double delta = 0.01;
    double height = 1.0;
    int dim = 2;

    CHKERR PetscOptionsBegin(PETSC_COMM_WORLD, "", "ADD_PRISMS_LAYER", "none");

    CHKERR PetscOptionsString("-my_file", "mesh file name", "", "mesh.h5m",
                              mesh_file_name, 255, &flg_mesh_file);
    CHKERR PetscOptionsString("-my_out_file", "out file name", "", "out.h5m",
                              out_file_name, 255, &flg_out_file);

    CHKERR PetscOptionsInt("-my_dim", "dimension (2 or 3)", "", dim, &dim,
                           PETSC_NULL);

    CHKERR PetscOptionsReal("-my_lambda", "roughness wavelength", "", lambda,
                            &lambda, PETSC_NULL);
    CHKERR PetscOptionsReal("-my_delta", "roughness amplitude", "", delta,
                            &delta, PETSC_NULL);
    CHKERR PetscOptionsReal("-my_height", "vertical dimension of the mesh", "",
                            height, &height, PETSC_NULL);

    int ierr = PetscOptionsEnd();
    CHKERRG(ierr);

    if (flg_mesh_file != PETSC_TRUE) {
      SETERRQ(PETSC_COMM_SELF, 1, "*** ERROR -my_file (MESH FILE NEEDED)");
    }

    // Read mesh to MOAB
    const char *option;
    option = ""; //"PARALLEL=BCAST;";//;DEBUG_IO";
    CHKERR moab.load_file(mesh_file_name, 0, option);
    ParallelComm *pcomm = ParallelComm::get_pcomm(&moab, MYPCOMM_INDEX);
    if (pcomm == NULL)
      pcomm = new ParallelComm(&moab, PETSC_COMM_WORLD);

    // Create MoFEM (Joseph) databas
    MoFEM::Core core(moab);
    MoFEM::Interface &m_field = core;

    Range tets, tris, nodes, all_nodes;
    for (_IT_CUBITMESHSETS_BY_SET_TYPE_FOR_LOOP_(m_field, BLOCKSET, bit)) {
      if (bit->getName().compare(0, 11, "MAT_ELASTIC") == 0) {
        const int id = bit->getMeshsetId();
        if (id == 1) {
          CHKERR m_field.get_moab().get_entities_by_dimension(bit->getMeshset(),
                                                              3, tets, true);
          CHKERR m_field.get_moab().get_entities_by_dimension(bit->getMeshset(),
                                                              2, tris, true);
        }
      }
    }

    CHKERR m_field.get_moab().get_connectivity(tets, nodes);
    all_nodes.merge(nodes);
    nodes.clear();
    CHKERR m_field.get_moab().get_connectivity(tris, nodes);
    all_nodes.merge(nodes);

    double coords[3];
    for (Range::iterator nit = all_nodes.begin(); nit != all_nodes.end();
         nit++) {
      CHKERR moab.get_coords(&*nit, 1, coords);
      double x = coords[0];
      double y = coords[1];
      double z = coords[2];
      double coef = (height + z) / height;
      switch (dim) {
      case 2:
        coords[2] -= coef * delta * (1. - cos(2. * M_PI * x / lambda));
        break;
      case 3:
        coords[2] -=
            coef * delta *
            (1. - cos(2. * M_PI * x / lambda) * cos(2. * M_PI * y / lambda));
        break;
      default:
        SETERRQ1(PETSC_COMM_SELF, MOFEM_DATA_INCONSISTENCY,
                 "Wrong dimension = %d", dim);
      }

      CHKERR moab.set_coords(&*nit, 1, coords);
    }

    EntityHandle rootset = moab.get_root_set();
    CHKERR moab.write_file(out_file_name, "MOAB", "", &rootset, 1);
    CHKERR PetscPrintf(PETSC_COMM_WORLD, "Write file %s\n", out_file_name);
  }
  CATCH_ERRORS;

  MoFEM::Core::Finalize();
  return 0;
}
