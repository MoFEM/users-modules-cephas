/** \file reading_med.cpp

  \brief Partition mesh and configuring blocksets

*/



#include <MoFEM.hpp>
using namespace MoFEM;

static char help[] = "...\n\n";

int main(int argc, char *argv[]) {

  MoFEM::Core::Initialize(&argc, &argv, (char *)0, help);

  try {

    moab::Core mb_instance;
    moab::Interface &moab = mb_instance;
    ParallelComm *pcomm = ParallelComm::get_pcomm(&moab, MYPCOMM_INDEX);
    if (pcomm == NULL)
      pcomm = new ParallelComm(&moab, PETSC_COMM_WORLD);

    // global variables
    char mesh_file_name[255];
    char mesh_out_file[255] = "out.h5m";
    PetscBool flg_file = PETSC_FALSE;
    PetscBool flg_n_part = PETSC_FALSE;
    PetscBool flg_part = PETSC_FALSE;
    PetscBool only_tags = PETSC_FALSE;
    PetscInt n_partas = 1;
    PetscBool create_lower_dim_ents = PETSC_TRUE;
    PetscInt dim = 3;
    PetscInt adj_dim = 2;

    ierr = PetscOptionsBegin(PETSC_COMM_WORLD, "", "none", "none");
    CHKERRQ(ierr);

    CHKERR PetscOptionsString("-my_file", "mesh file name", "", "mesh.h5m",
                              mesh_file_name, 255, &flg_file);
    if (flg_file != PETSC_TRUE)
      CHKERR PetscOptionsString("-file_name", "mesh file name", "", "mesh.h5m",
                                mesh_file_name, 255, &flg_file);
    if (flg_file != PETSC_TRUE)
      SETERRQ(PETSC_COMM_SELF, 1,
              "*** ERROR -my_file (-file_name) (mesh file needed)");

    const char *option;
    option = "";
    CHKERR moab.load_file(mesh_file_name, 0, option);

    CHKERR PetscOptionsString("-output_file", "output mesh file name", "",
                              "out.h5m", mesh_out_file, 255, PETSC_NULL);
    CHKERR PetscOptionsInt("-my_nparts", "number of parts", "", n_partas,
                           &n_partas, &flg_n_part);
    CHKERR PetscOptionsInt("-nparts", "number of parts", "", n_partas,
                           &n_partas, &flg_part);

    if (!flg_n_part && !flg_part)
      SETERRQ(PETSC_COMM_SELF, 1,
              "*** ERROR partitioning number not given (-nparts)");

    auto get_nb_ents_by_dim = [&](const int dim) {
      int nb;
      CHKERR moab.get_number_entities_by_dimension(0, dim, nb);
      return nb;
    };
    for (; dim >= 0; dim--) {
      if (get_nb_ents_by_dim(dim))
        break;
    }

    if (!dim)
      SETERRQ(PETSC_COMM_WORLD, MOFEM_DATA_INCONSISTENCY,
              "Dimension of entities to partition not found");

    CHKERR PetscOptionsInt("-dim", "entities dim", "", dim, &dim, PETSC_NULL);
    adj_dim = dim - 1;
    CHKERR PetscOptionsInt("-adj_dim", "adjacency dim", "", adj_dim, &adj_dim,
                           PETSC_NULL);
    CHKERR PetscOptionsBool(
        "-my_create_lower_dim_ents", "if tru create lower dimension entireties",
        "", create_lower_dim_ents, &create_lower_dim_ents, PETSC_NULL);
    CHKERR PetscOptionsBool("-block_tags", "only block and meshsests tags", "",
                            only_tags, &only_tags, PETSC_NULL);

    ierr = PetscOptionsEnd();
    CHKERRQ(ierr);

    // Create MoFEM  database
    MoFEM::Core core(moab);
    MoFEM::Interface &m_field = core;

    auto meshsets_interface_ptr = m_field.getInterface<MeshsetsManager>();
    CHKERR meshsets_interface_ptr->setMeshsetFromFile();

    MOFEM_LOG_CHANNEL("WORLD");
    MOFEM_LOG_TAG("WORLD", "mofem_part")
    for (CubitMeshSet_multiIndex::iterator cit =
             meshsets_interface_ptr->getBegin();
         cit != meshsets_interface_ptr->getEnd(); cit++)
      MOFEM_LOG("WORLD", Sev::inform) << *cit;
    MOFEM_LOG_CHANNEL("WORLD");

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
      CommInterface *comm_interafce_ptr = m_field.getInterface<CommInterface>();
      CHKERR comm_interafce_ptr->partitionMesh(
          ents_dim, dim, adj_dim, n_partas, nullptr, nullptr, nullptr, VERBOSE);
    }

    auto get_tag_list = [&]() {
      std::vector<Tag> tags_list;
      auto meshsets_mng = m_field.getInterface<MeshsetsManager>();
      auto &list = meshsets_mng->getMeshsetsMultindex();
      for (auto &m : list) {
        auto meshset = m.getMeshset();
        std::vector<Tag> tmp_tags_list;
        CHKERR m_field.get_moab().tag_get_tags_on_entity(meshset,
                                                         tmp_tags_list);
        for (auto t : tmp_tags_list) {
          tags_list.push_back(t);
        }
      }

      return tags_list;
    };

    EntityHandle root_mesh = 0;
    if (m_field.get_comm_rank() == 0) {
      if (only_tags) {
        auto tags_list = get_tag_list();
        std::sort(tags_list.begin(), tags_list.end());
        auto new_end = std::unique(tags_list.begin(), tags_list.end());
        tags_list.resize(std::distance(tags_list.begin(), new_end));
        tags_list.push_back(pcomm->part_tag());
        // tags_list.push_back(m_field.get_moab().globalId_tag());
        CHKERR moab.write_file(mesh_out_file, "MOAB", "", &root_mesh, 1,
                               &*tags_list.begin(), tags_list.size());
      } else {
        CHKERR moab.write_file(mesh_out_file, "MOAB", "");
      }
    }
  }
  CATCH_ERRORS;

  MoFEM::Core::Finalize();
  CHKERRQ(ierr);

  return 0;
}
