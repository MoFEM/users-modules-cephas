/** \file uniform_mesh_refinement.cpp

  \brief Uniformly refine mesh

*/

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
    PetscBool debug = PETSC_FALSE;

    ierr = PetscOptionsBegin(PETSC_COMM_WORLD, "", "none", "none");
    CHKERRQ(ierr);

    CHKERR PetscOptionsString("-my_file", "mesh file name", "", "mesh.h5m",
                              mesh_file_name, 255, &flg_file);
    if (flg_file != PETSC_TRUE)
      CHKERR PetscOptionsString("-file_name", "mesh file name", "", "mesh.h5m",
                                mesh_file_name, 255, &flg_file);
    CHKERR PetscOptionsString("-output_file", "output mesh file name", "",
                              "mesh.h5m", mesh_out_file, 255, PETSC_NULL);
    CHKERR PetscOptionsInt("-dim", "entities dim", "", dim, &dim, PETSC_NULL);
    CHKERR PetscOptionsInt("-nb_levels", "number of refinement levels", "",
                           nb_levels, &nb_levels, PETSC_NULL);
    CHKERR PetscOptionsBool("-shift",
                            "shift bits, squash entities of refined elements",
                            "", shift, &shift, PETSC_NULL);
    CHKERR PetscOptionsBool("-debug",
                            "write additional files with bit ref levels", "",
                            debug, &debug, PETSC_NULL);

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
                "Refinement implemented only for three and two dimensions");
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

        Range r_tagged_sets;
        CHKERR m_field.get_moab().get_entities_by_type_and_tag(
            0, MBENTITYSET, &part_tag, NULL, 1, r_tagged_sets,
            moab::Interface::UNION);

        std::vector<EntityHandle> tagged_sets(r_tagged_sets.size());
        std::copy(r_tagged_sets.begin(), r_tagged_sets.end(),
                  tagged_sets.begin());

        auto order_tagged_sets = [&]() {
          MoFEMFunctionBegin;
          std::vector<int> parts(tagged_sets.size());
          CHKERR m_field.get_moab().tag_get_data(
              part_tag, &*tagged_sets.begin(), tagged_sets.size(),
              &*parts.begin());
          map<int, EntityHandle> m_tagged_sets;
          for (int p = 0; p != tagged_sets.size(); ++p) {
            m_tagged_sets[parts[p]] = tagged_sets[p];
          }
          for (int p = 0; p != tagged_sets.size(); ++p) {
            tagged_sets[p] = m_tagged_sets.at(p);
          }
          MoFEMFunctionReturn(0);
        };

        auto add_children = [&]() {
          MoFEMFunctionBegin;
          std::vector<Range> part_ents(tagged_sets.size());

          for (int p = 0; p != tagged_sets.size(); ++p) {
            Range ents;
            CHKERR moab.get_entities_by_dimension(tagged_sets[p], dim, ents,
                                                  true);
            CHKERR bit_ref_manager->filterEntitiesByRefLevel(
                bit(l), BitRefLevel().set(), ents);

            Range children;
            CHKERR bit_ref_manager->updateRangeByChildren(ents, children);
            children = children.subset_by_dimension(dim);
            CHKERR bit_ref_manager->filterEntitiesByRefLevel(
                bit(l + 1), BitRefLevel().set(), children);

            Range adj;
            for (auto d = 1; d != dim; ++d) {
              CHKERR moab.get_adjacencies(children.subset_by_dimension(dim), d,
                                          false, adj, moab::Interface::UNION);
            }

            part_ents[p].merge(children);
            part_ents[p].merge(adj);
          }

          for (int p = 1; p != tagged_sets.size(); ++p) {
            for (int pp = 0; pp != p; pp++) {
              part_ents[p] = subtract(part_ents[p], part_ents[pp]);
            }
          }

          for (int p = 0; p != tagged_sets.size(); ++p) {
            CHKERR moab.add_entities(tagged_sets[p], part_ents[p]);
            CHKERR moab.tag_clear_data(part_tag, part_ents[p], &p);
          }

          if (debug) {

            auto save_range = [&](const std::string name, const Range &r) {
              MoFEMFunctionBegin;
              auto meshset_ptr = get_temp_meshset_ptr(m_field.get_moab());
              CHKERR m_field.get_moab().add_entities(*meshset_ptr, r);
              CHKERR m_field.get_moab().write_file(name.c_str(), "VTK", "",
                                                   meshset_ptr->get_ptr(), 1);
              MoFEMFunctionReturn(0);
            };

            for (int p = 0; p != tagged_sets.size(); ++p) {
              MOFEM_LOG("WORLD", Sev::inform)
                  << "Write part " << p << " level " << l;
              Range ents;
              CHKERR m_field.get_moab().get_entities_by_handle(tagged_sets[p],
                                                               ents, true);
              CHKERR bit_ref_manager->filterEntitiesByRefLevel(
                  bit(l + 1), BitRefLevel().set(), ents);
              CHKERR save_range("part" + boost::lexical_cast<std::string>(p) +
                                    "_" + boost::lexical_cast<std::string>(l) +
                                    ".vtk",
                                ents);
            }
          }

          MoFEMFunctionReturn(0);
        };

        CHKERR order_tagged_sets();
        CHKERR add_children();

        MoFEMFunctionReturn(0);
      };

      CHKERR update_partition_sets();
      CHKERR update_meshsets();

      CHKERR moab.delete_entities(&meshset_ref_edges, 1);
    }

    if (debug) {
      for (int l = 0; l <= nb_levels; ++l) {
        MOFEM_LOG("WORLD", Sev::inform) << "Write level " << l;
        CHKERR bit_ref_manager->writeBitLevel(
            bit(l), BitRefLevel().set(),
            ("level" + boost::lexical_cast<std::string>(l) + ".vtk").c_str(),
            "VTK", "");
      }
    }

    if (shift == PETSC_TRUE) {
      MOFEM_LOG("WORLD", Sev::inform) << "Shift bits";
      CHKERR core.getInterface<BitRefManager>()->shiftRightBitRef(
          nb_levels, BitRefLevel().set(), VERBOSE);
    }

    CHKERR moab.write_file(mesh_out_file);
  }
  CATCH_ERRORS;

  MoFEM::Core::Finalize();
  CHKERRQ(ierr);

  return 0;
}
