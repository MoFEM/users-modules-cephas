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

static char help[] = "Uniform mesh refinement\n\n"

                     "Usage example:\n"

                     "$ ./uniform_mesh_refinement -my_file mesh.h5m "
                     "-output_file refined_mesh.h5m"

                     "\n\n";

int main(int argc, char *argv[]) {

  MoFEM::Core::Initialize(&argc, &argv, (char *)0, help);

  try {

    // global variables
    char mesh_file_name[255];
    char mesh_out_file[255] = "out.h5m";
    int dim = 3;
    int nb_levels = 1;
    PetscBool shift = PETSC_TRUE;
    PetscBool flg_file = PETSC_FALSE;

    ierr = PetscOptionsBegin(PETSC_COMM_WORLD, "", "none", "none");
    CHKERRQ(ierr);

    CHKERR PetscOptionsString("-my_file", "mesh file name", "", "mesh.h5m",
                              mesh_file_name, 255, &flg_file);
    CHKERR PetscOptionsString("-output_file", "output mesh file name", "",
                              "mesh.h5m", mesh_out_file, 255, PETSC_NULL);
    CHKERR PetscOptionsInt("-dim", "entities dim", "", dim, &dim, PETSC_NULL);
    CHKERR PetscOptionsInt("-nb_levels", "number of refinement levels", "",
                           nb_levels, &nb_levels, PETSC_NULL);

    CHKERR PetscOptionsBool("-shift",
                            "shift bits, squash entities of refined elements",
                            "", shift, &shift, PETSC_NULL);

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
      SETERRQ(PETSC_COMM_SELF, MOFEM_DATA_INCONSISTENCY,
              "*** ERROR -my_file (-file_name) (mesh file needed)");
    }

    BitRefManager *bit_ref_manager;
    CHKERR m_field.getInterface(bit_ref_manager);

    auto bit = [](auto l) { return BitRefLevel().set(l); };

    CHKERR bit_ref_manager->setBitRefLevelByDim(0, dim, bit(0));

    for (auto l = 0; l != nb_levels; ++l) {
      Range ents;
      CHKERR bit_ref_manager->getEntitiesByDimAndRefLevel(
          bit(l), BitRefLevel().set(), dim, ents);

      Range edges;
      CHKERR moab.get_adjacencies(ents, 1, true, edges, moab::Interface::UNION);

      EntityHandle meshset_ref_edges;
      CHKERR moab.create_meshset(MESHSET_SET, meshset_ref_edges);
      CHKERR moab.add_entities(meshset_ref_edges, edges);

      MeshRefinement *refine = m_field.getInterface<MeshRefinement>();

      CHKERR refine->addVerticesInTheMiddleOfEdges(meshset_ref_edges,
                                                   bit(l + 1));
      if (dim == 3) {
        CHKERR refine->refineTets(ents, bit(l + 1));
      } else if (dim == 2) {
        CHKERR refine->refineTris(ents, bit(l + 1));
      } else {
        SETERRQ(PETSC_COMM_SELF, MOFEM_NOT_IMPLEMENTED,
                "Refinment implemented only for three and two dimensions");
      }

      auto update_meshsets = [&]() {
        MoFEMFunctionBegin;
        // update cubit meshsets
        for (_IT_CUBITMESHSETS_FOR_LOOP_(m_field, ciit)) {
          EntityHandle cubit_meshset = ciit->meshset;
          CHKERR bit_ref_manager->updateMeshsetByEntitiesChildren(
              cubit_meshset, bit(l + 1), cubit_meshset, MBMAXTYPE, true);
        }
        MoFEMFunctionReturn(0);
      };

      auto update_partition_sets = [&]() {
        MoFEMFunctionBegin;

        ParallelComm *pcomm = ParallelComm::get_pcomm(
            &m_field.get_moab(), m_field.get_basic_entity_data_ptr()->pcommID);
        Tag part_tag = pcomm->part_tag();

        Range tagged_sets;
        CHKERR m_field.get_moab().get_entities_by_type_and_tag(
            0, MBENTITYSET, &part_tag, NULL, 1, tagged_sets,
            moab::Interface::UNION);

        for (auto m : tagged_sets) {

          int part;
          CHKERR moab.tag_get_data(part_tag, &m, 1, &part);

          for (auto t = CN::TypeDimensionMap[dim].first;
               t <= CN::TypeDimensionMap[dim].second; t++) {

            // Refinement is only implemented for simplexes in 2d and 3d
            if (t == MBTRI || t == MBTET) {
              Range ents;
              CHKERR moab.get_entities_by_type(m, t, ents, true);
              CHKERR bit_ref_manager->filterEntitiesByRefLevel(
                  bit(l), BitRefLevel().set(), ents);

              Range children;
              CHKERR bit_ref_manager->updateRangeByChildren(ents, children);
              CHKERR bit_ref_manager->filterEntitiesByRefLevel(
                  bit(l + 1), BitRefLevel().set(), children);
              children = subtract(children, children.subset_by_type(MBVERTEX));

              Range adj;
              for (auto d = 1; d != dim; ++d) {
                CHKERR moab.get_adjacencies(children.subset_by_dimension(dim),
                                            d, false, adj,
                                            moab::Interface::UNION);
              }
              children.merge(adj);

              CHKERR moab.add_entities(m, children);
              CHKERR moab.tag_clear_data(part_tag, children, &part);
            }
          }
        }
        MoFEMFunctionReturn(0);
      };

      CHKERR update_partition_sets();
      CHKERR update_meshsets();

      CHKERR moab.delete_entities(&meshset_ref_edges, 1);
    }

    if (shift == PETSC_TRUE)
      CHKERR core.getInterface<BitRefManager>()->shiftRightBitRef(
          nb_levels, BitRefLevel().set(), VERBOSE);

    CHKERR moab.write_file(mesh_out_file);
  }
  CATCH_ERRORS;

  MoFEM::Core::Finalize();
  CHKERRQ(ierr);

  return 0;
}
