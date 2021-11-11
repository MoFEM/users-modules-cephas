/** \file split_sideset.cpp
  \brief Split sidesets
  \example split_sideset.cpp

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
constexpr bool debug = false;

int main(int argc, char *argv[]) {
  MoFEM::Core::Initialize(&argc, &argv, (char *)0, help);

  auto core_log = logging::core::get();
  core_log->add_sink(
      LogManager::createSink(LogManager::getStrmWorld(), "SPLIT"));
  LogManager::setLog("SPLIT");
  MOFEM_LOG_TAG("SPLIT", "split");

  try {

    // global variables
    char mesh_file_name[255];
    PetscBool flg_file = PETSC_FALSE;
    PetscBool squash_bit_levels = PETSC_TRUE;
    PetscBool flg_list_of_sidesets = PETSC_FALSE;
    PetscBool output_vtk = PETSC_TRUE;
    PetscBool add_prisms = PETSC_FALSE;
    PetscBool split_corner_edges = PETSC_FALSE;
    int nb_sidesets = 10;
    int sidesets[nb_sidesets];

    CHKERR PetscOptionsBegin(PETSC_COMM_WORLD, "", "Split sides options",
                             "none");
    CHKERR PetscOptionsString("-my_file", "mesh file name", "", "mesh.h5m",
                              mesh_file_name, 255, &flg_file);
    CHKERR PetscOptionsBool("-squash_bit_levels", "squash bit levels", "",
                            squash_bit_levels, &squash_bit_levels, NULL);
    CHKERR PetscOptionsIntArray("-side_sets", "get list of sidesets", "",
                                sidesets, &nb_sidesets, &flg_list_of_sidesets);
    CHKERR PetscOptionsBool("-output_vtk", "if true outout vtk file", "",
                            output_vtk, &output_vtk, PETSC_NULL);
    CHKERR PetscOptionsBool("-add_prisms", "if true outout vtk file", "",
                            add_prisms, &add_prisms, PETSC_NULL);
    CHKERR PetscOptionsBool("-split_corner_edges", "if true outout vtk file",
                            "", split_corner_edges, &split_corner_edges,
                            PETSC_NULL);
    CHKERR PetscOptionsBool("-output_vtk", "if true outout vtk file", "",
                            output_vtk, &output_vtk, PETSC_NULL);
    ierr = PetscOptionsEnd();
    CHKERRG(ierr);

    moab::Core mb_instance;
    moab::Interface &moab = mb_instance;

    const char *option;
    option = "";
    CHKERR moab.load_file(mesh_file_name, 0, option);

    // Create MoFEM  database
    MoFEM::Core core(moab);
    MoFEM::Interface &m_field = core;

    if (flg_file != PETSC_TRUE) {
      SETERRQ(PETSC_COMM_SELF, MOFEM_INVALID_DATA,
              "*** ERROR -my_file (mesh file needed)");
    }
    if (flg_list_of_sidesets != PETSC_TRUE) {
      SETERRQ(PETSC_COMM_SELF, MOFEM_INVALID_DATA,
              "List of sidesets not given -my_side_sets ...");
    }

    // Get interface to meshsets manager
    auto m_mng = m_field.getInterface<MeshsetsManager>();
    // Get interface for splitting manager
    auto interface_ptr = m_field.getInterface<PrismInterface>();
    // Managing bits
    auto bit_mng = m_field.getInterface<BitRefManager>();
    // Refine mesh
    auto refine_mng = m_field.getInterface<MeshRefinement>();

    auto &meshsets_index = m_mng->getMeshsetsMultindex();
    auto &m_by_type = meshsets_index.get<CubitMeshSets_mask_meshset_mi_tag>();
    auto &m_by_id_and_type =
        meshsets_index.get<Composite_Cubit_msId_And_MeshSetType_mi_tag>();

    for (auto mit = m_by_type.lower_bound(SIDESET);
         mit != m_by_type.upper_bound(SIDESET); mit++) {
      MOFEM_LOG("SPLIT", Sev::verbose)
          << "Sideset on the mesh id = " << mit->getMeshsetId();
    }

    CHKERR bit_mng->setBitRefLevelByDim(0, 3, BitRefLevel().set(0));
    std::vector<BitRefLevel> bit_levels;
    bit_levels.push_back(BitRefLevel().set(0));

    auto update_meshsets = [&]() {
      MoFEMFunctionBegin;
      for (_IT_CUBITMESHSETS_FOR_LOOP_(m_field, ciit)) {
        EntityHandle cubit_meshset = ciit->meshset;
        CHKERR bit_mng->updateMeshsetByEntitiesChildren(
            cubit_meshset, bit_levels.back(), cubit_meshset, MBVERTEX, true);
        CHKERR bit_mng->updateMeshsetByEntitiesChildren(
            cubit_meshset, bit_levels.back(), cubit_meshset, MBEDGE, true);
        CHKERR bit_mng->updateMeshsetByEntitiesChildren(
            cubit_meshset, bit_levels.back(), cubit_meshset, MBTRI, true);
        CHKERR bit_mng->updateMeshsetByEntitiesChildren(
            cubit_meshset, bit_levels.back(), cubit_meshset, MBTET, true);
      }
      MoFEMFunctionReturn(0);
    };

    // refine corner mesh
    if (split_corner_edges) {
      Skinner skin(&m_field.get_moab());
      auto meshset_of_edges_to_refine_ptr = get_temp_meshset_ptr(moab);

      for (int mm = 0; mm != nb_sidesets; mm++) {

        // find side set
        auto mit =
            m_by_id_and_type.find(boost::make_tuple(sidesets[mm], SIDESET));
        if (mit == m_by_id_and_type.end())
          SETERRQ1(PETSC_COMM_SELF, MOFEM_INVALID_DATA,
                   "No sideset in database id = %d", sidesets[mm]);

        Range faces;
        CHKERR moab.get_entities_by_type(mit->getMeshset(), MBTRI, faces, true);
        Range faces_edges;
        CHKERR moab.get_adjacencies(faces, 1, true, faces_edges,
                                    moab::Interface::UNION);

        Range skin_edges;
        CHKERR skin.find_skin(0, faces, false, skin_edges);
        Range skin_verts;
        CHKERR moab.get_connectivity(skin_edges, skin_verts, true);
        Range vertex_edges;
        CHKERR moab.get_adjacencies(skin_verts, 1, true, vertex_edges,
                                    moab::Interface::UNION);
        Range vertex_edges_verts;
        CHKERR moab.get_connectivity(vertex_edges, vertex_edges_verts, true);
        vertex_edges_verts = subtract(vertex_edges_verts, skin_verts);
        Range vertex_edges_verts_edges;
        CHKERR moab.get_adjacencies(vertex_edges_verts, 1, true,
                                    vertex_edges_verts_edges,
                                    moab::Interface::UNION);
        vertex_edges = subtract(vertex_edges, vertex_edges_verts_edges);
        vertex_edges = subtract(vertex_edges, skin_edges);
        vertex_edges = intersect(vertex_edges, faces_edges);
        CHKERR moab.add_entities(*meshset_of_edges_to_refine_ptr, vertex_edges);
      }

      int nb_tris;
      CHKERR moab.get_number_entities_by_type(*meshset_of_edges_to_refine_ptr,
                                              MBEDGE, nb_tris, true);
      MOFEM_LOG("SPLIT", Sev::inform) << "Refine corner edges " << nb_tris;

      if (debug)
        CHKERR moab.write_file("out_edges_to_refine.vtk", "VTK", "",
                               meshset_of_edges_to_refine_ptr->get_ptr(), 1);

      bit_levels.push_back(BitRefLevel().set(1));
      CHKERR refine_mng->addVerticesInTheMiddleOfEdges(
          *meshset_of_edges_to_refine_ptr, bit_levels.back());
      CHKERR refine_mng->refineTets(0, bit_levels.back());
      CHKERR update_meshsets();
    }

    // iterate sideset and split
    for (int mm = 0; mm != nb_sidesets; mm++) {

      // find side set
      auto mit =
          m_by_id_and_type.find(boost::make_tuple(sidesets[mm], SIDESET));
      if (mit == m_by_id_and_type.end())
        SETERRQ1(PETSC_COMM_SELF, MOFEM_INVALID_DATA,
                 "No sideset in database id = %d", sidesets[mm]);

      MOFEM_LOG("SPLIT", Sev::inform)
          << "Split sideset " << mit->getMeshsetId();

      const auto cubit_meshset = mit->getMeshset();
      {
        // get tet entities form back bit_level
        auto ref_level_meshset_ptr = get_temp_meshset_ptr(moab);
        CHKERR bit_mng->getEntitiesByTypeAndRefLevel(bit_levels.back(),
                                                     BitRefLevel().set(), MBTET,
                                                     *ref_level_meshset_ptr);
        CHKERR bit_mng->getEntitiesByTypeAndRefLevel(
            bit_levels.back(), BitRefLevel().set(), MBPRISM,
            *ref_level_meshset_ptr);
        Range ref_level_tets;
        CHKERR moab.get_entities_by_handle(*ref_level_meshset_ptr,
                                           ref_level_tets, true);

        // get faces and test to split
        CHKERR interface_ptr->getSides(cubit_meshset, bit_levels.back(), true,
                                       0);
        // set new bit level
        MOFEM_LOG("SPLIT", Sev::verbose)
            << "Add bit level " << bit_levels.size();
        bit_levels.push_back(BitRefLevel().set(bit_levels.size()));
        // split faces and
        CHKERR interface_ptr->splitSides(*ref_level_meshset_ptr,
                                         bit_levels.back(), cubit_meshset,
                                         add_prisms, true, 0);
      }
      // Update cubit meshsets
      CHKERR update_meshsets();
    }

    if (squash_bit_levels == PETSC_TRUE) {
      for (unsigned int ll = 0; ll != bit_levels.size() - 1; ll++) {
        CHKERR m_field.delete_ents_by_bit_ref(bit_levels[ll], bit_levels[ll],
                                              true);
      }
      CHKERR bit_mng->shiftRightBitRef(bit_levels.size() - 1);
    }

    if (output_vtk) {
      auto meshset_ptr = get_temp_meshset_ptr(moab);
      BitRefLevel bit;
      if (squash_bit_levels)
        bit = bit_levels[0];
      else
        bit = bit_levels.back();
      CHKERR bit_mng->getEntitiesByTypeAndRefLevel(bit, BitRefLevel().set(),
                                                   MBTET, *meshset_ptr);
      CHKERR moab.write_file("out.vtk", "VTK", "", meshset_ptr->get_ptr(), 1);
    }

    ParallelComm *pcomm = ParallelComm::get_pcomm(&moab, MYPCOMM_INDEX);
    if (pcomm == NULL)
      SETERRQ(PETSC_COMM_SELF, MOFEM_DATA_INCONSISTENCY,
              "Communicator should be allocated");

    CHKERR pcomm->assign_global_ids(0, 3, 1, false);
    CHKERR moab.write_file("out.h5m");
  }
  CATCH_ERRORS;

  MoFEM::Core::Finalize();

  return 0;
}
