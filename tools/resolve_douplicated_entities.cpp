/** \file delete_ho_nodes.cpp
  \brief Delete higher order nodes
  \example delete_ho_nodes.cpp

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
    int cutting_side_set = 200;
    char mesh_out_file[255] = "out.h5m";
    PetscBool flg_cutting_side_set;

    CHKERR PetscOptionsBegin(PETSC_COMM_WORLD, "", "Field to vertices options",
                             "none");
    CHKERR PetscOptionsString("-my_file", "mesh file name", "", "mesh.h5m",
                              mesh_file_name, 255, &flg_file);
    CHKERR PetscOptionsInt("-cutting_side_set", "crete side set", "",
                           cutting_side_set, &cutting_side_set,
                           &flg_cutting_side_set);

    CHKERR PetscOptionsString("-output_file", "output mesh file name", "",
                              mesh_out_file, mesh_out_file, 255, PETSC_NULL);
    ierr = PetscOptionsEnd();
    CHKERRG(ierr);

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

    MoFEM::CoreInterface &m_field =
        *(core.getInterface<MoFEM::CoreInterface>());

    MeshsetsManager *meshset_manager;
    CHKERR m_field.getInterface(meshset_manager);

    if (flg_file != PETSC_TRUE) {
      SETERRQ(PETSC_COMM_SELF, MOFEM_INVALID_DATA,
              "*** ERROR -my_file (MESH FILE NEEDED)");
    }

    MOFEM_LOG("WORLD", Sev::inform) << "In file " << mesh_file_name;
    MOFEM_LOG("WORLD", Sev::inform) << "Out file " << mesh_out_file;
    MOFEM_LOG("WORLD", Sev::inform)
        << "Cutting surface sideset " << cutting_side_set;

    Range ents, skin;
    CHKERR moab.get_entities_by_dimension(0, 3, ents);
    MOFEM_LOG("WORLD", Sev::verbose) << "Ents:\n" << ents;

    Skinner skinner(&moab);
    CHKERR skinner.find_skin(0, ents, false, skin);

    MOFEM_LOG("WORLD", Sev::verbose) << "Skin:\n" << skin;

    Range check_vol;
    CHKERR moab.get_entities_by_type(0, MBTET, check_vol);

    Range edges_on_root_vols;
    ErrorCode tmp_result;
    tmp_result = moab.get_adjacencies(check_vol, 1, true, edges_on_root_vols,
                                      moab::Interface::UNION);
    if (MB_SUCCESS != tmp_result) {
      PetscPrintf(PETSC_COMM_WORLD, "Multiplicity in edges.\n");
    }

    Range root_edges;
    CHKERR moab.get_entities_by_type(0, MBEDGE, root_edges);

    Range edges = subtract(root_edges, edges_on_root_vols);
    Range cut_surface_edges;
    if (meshset_manager->checkMeshset(cutting_side_set, SIDESET))
      CHKERR meshset_manager->getEntitiesByDimension(
          cutting_side_set, SIDESET, 1, cut_surface_edges, true);

    // preserve edges on cutting surfaces
    edges = subtract(edges, cut_surface_edges);

    CHKERR moab.delete_entities(edges);
    
    CHKERR moab.write_file(mesh_out_file);
  }
  CATCH_ERRORS;

  MoFEM::Core::Finalize();

  return 0;
}