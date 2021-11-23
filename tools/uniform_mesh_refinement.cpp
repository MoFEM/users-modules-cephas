/** \file uniform_mesh_refinement.cpp

  \brief Uniformly refine mesh

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

static char help[] =
    "Uniform mesh refinement\n\n"

    "Usage example:\n"

    "$ ./uniform_mesh_refinement -my_file mesh.h5m -output_file refined_mesh.h5m"
    
    "\n\n";

int main(int argc, char *argv[]) {

  MoFEM::Core::Initialize(&argc, &argv, (char *)0, help);

  try {

    // global variables
    char mesh_file_name[255];
    char mesh_out_file[255] = "out.h5m";
    PetscBool flg_file = PETSC_FALSE;

    ierr = PetscOptionsBegin(PETSC_COMM_WORLD, "", "none", "none");
    CHKERRQ(ierr);

    CHKERR PetscOptionsString("-my_file", "mesh file name", "", "mesh.h5m",
                              mesh_file_name, 255, &flg_file);
    CHKERR PetscOptionsString("-output_file", "output mesh file name", "",
                              "mesh.h5m", mesh_out_file, 255, PETSC_NULL);

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

    BitRefManager *bit_ref_manager;
    CHKERR m_field.getInterface(bit_ref_manager);

    CHKERR bit_ref_manager->setBitRefLevelByDim(0, 3, BitRefLevel().set(0));

    Range ents3d;
    rval = moab.get_entities_by_dimension(0, 3, ents3d, false);
    Range edges;
    CHKERR moab.get_adjacencies(ents3d, 1, true, edges, moab::Interface::UNION);

    EntityHandle meshset_ref_edges;
    CHKERR moab.create_meshset(MESHSET_SET, meshset_ref_edges);
    CHKERR moab.add_entities(meshset_ref_edges, edges);

    MeshRefinement *refine;
    CHKERR m_field.getInterface(refine);

    CHKERR refine->addVerticesInTheMiddleOfEdges(meshset_ref_edges,
                                                      BitRefLevel().set(1));
    CHKERR refine->refineTets(0, BitRefLevel().set(1));

    // update cubit meshsets
    for (_IT_CUBITMESHSETS_FOR_LOOP_(m_field, ciit)) {
      EntityHandle cubit_meshset = ciit->meshset;
      CHKERR bit_ref_manager->updateMeshsetByEntitiesChildren(
          cubit_meshset, BitRefLevel().set(1), cubit_meshset, MBVERTEX, true);
      CHKERR bit_ref_manager->updateMeshsetByEntitiesChildren(
          cubit_meshset, BitRefLevel().set(1), cubit_meshset, MBEDGE, true);
      CHKERR bit_ref_manager->updateMeshsetByEntitiesChildren(
          cubit_meshset, BitRefLevel().set(1), cubit_meshset, MBTRI, true);
      CHKERR bit_ref_manager->updateMeshsetByEntitiesChildren(
          cubit_meshset, BitRefLevel().set(1), cubit_meshset, MBTET, true);
    }

    CHKERR core.getInterface<BitRefManager>()->shiftRightBitRef(
        1, BitRefLevel().set(), VERBOSE);

    CHKERR moab.delete_entities(&meshset_ref_edges, 1);
    CHKERR moab.write_file(mesh_out_file);
  }
  CATCH_ERRORS;

  MoFEM::Core::Finalize();
  CHKERRQ(ierr);

  return 0;
}
