/** \file reading_med.cpp

  \brief Partition mesh and configuring blocksets

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
    char mesh_out_file[255] = "out.h5m";
    PetscBool flg_file = PETSC_FALSE;
    PetscBool flg_n_part = PETSC_FALSE;
    PetscInt n_partas = 1;
    PetscBool create_lower_dim_ents = PETSC_TRUE;
    PetscInt dim = 3;
    PetscInt adj_dim = 2;

    ierr = PetscOptionsBegin(PETSC_COMM_WORLD, "", "none", "none");
    CHKERRQ(ierr);

    CHKERR PetscOptionsString("-my_file", "mesh file name", "", "mesh.h5m",
                              mesh_file_name, 255, &flg_file);
    CHKERR PetscOptionsString("-file_name", "mesh file name", "", "mesh.h5m",
                              mesh_file_name, 255, &flg_file);
    CHKERR PetscOptionsString("-output_file", "output mesh file name", "",
                              "out.h5m", mesh_out_file, 255, PETSC_NULL);
    CHKERR PetscOptionsInt("-my_nparts", "number of parts", "", n_partas,
                           &n_partas, &flg_n_part);
    CHKERR PetscOptionsInt("-dim", "adjacency dim", "", dim, &dim, PETSC_NULL);
    adj_dim = dim - 1;
    CHKERR PetscOptionsInt("-adj_dim", "adjacency dim", "", adj_dim, &adj_dim,
                           PETSC_NULL);
    CHKERR PetscOptionsBool(
        "-my_create_lower_dim_ents", "if tru create lower dimension entireties",
        "", create_lower_dim_ents, &create_lower_dim_ents, NULL);

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

    if (flg_n_part != PETSC_TRUE) {
      SETERRQ(PETSC_COMM_SELF, 1, "*** ERROR partitioning number not given");
    }

    MeshsetsManager *meshsets_interface_ptr;
    CHKERR m_field.getInterface(meshsets_interface_ptr);
    CHKERR meshsets_interface_ptr->setMeshsetFromFile();

    for (CubitMeshSet_multiIndex::iterator cit =
             meshsets_interface_ptr->getBegin();
         cit != meshsets_interface_ptr->getEnd(); cit++) {
      std::cout << *cit << endl;
    }

    {
      Range ents_dim;
      CHKERR moab.get_entities_by_dimension(0, dim, ents_dim, false);
      if (create_lower_dim_ents) {
        if (dim == 3) {
          Range faces;
          CHKERR moab.get_adjacencies(ents_dim, 2, true, faces,
                                      moab::Interface::UNION);
        }
        if (dim > 2) {
          Range edges;
          CHKERR moab.get_adjacencies(ents_dim, 1, true, edges,
                                      moab::Interface::UNION);
        }
      }
      ProblemsManager *prb_mng_ptr;
      CHKERR m_field.getInterface(prb_mng_ptr);
      CHKERR prb_mng_ptr->partitionMesh(ents_dim, dim, adj_dim, n_partas);
    }

    CHKERR moab.write_file(mesh_out_file);
  }
  CATCH_ERRORS;

  MoFEM::Core::Finalize();
  CHKERRQ(ierr);

  return 0;
}
