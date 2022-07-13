/** \file delete_ho_nodes.cpp
  \brief Delete higher order nodes
  \example delete_ho_nodes.cpp

*/



#include <MoFEM.hpp>

using namespace MoFEM;

// struct MyAddRemove: public moab::Interface::HONodeAddedRemoved {

//  void node_added (EntityHandle 	node, EntityHandle 	element ) {
//      cerr << "Node has beed added..." << endl;
//  }
//  void node_removed (EntityHandle node) {
//     cerr << "Node has beed deleted..." << endl;
//  }

// };

static char help[] = "...\n\n";

int main(int argc, char *argv[]) {
  MoFEM::Core::Initialize(&argc, &argv, (char *)0, help);

  try {

    // global variables
    char mesh_file_name[255];
    PetscBool flg_file = PETSC_FALSE;
    CHKERR PetscOptionsBegin(PETSC_COMM_WORLD, "", "Field to vertices options",
                             "none");
    CHKERR PetscOptionsString("-my_file", "mesh file name", "", "mesh.h5m",
                              mesh_file_name, 255, &flg_file);
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

    if (flg_file != PETSC_TRUE) {
      SETERRQ(PETSC_COMM_SELF, MOFEM_INVALID_DATA,
              "*** ERROR -my_file (MESH FILE NEEDED)");
    }

    Range all_tets, topo_nodes, mid_nodes;
    CHKERR moab.get_entities_by_type(0, MBTET, all_tets);
    CHKERR moab.get_connectivity(all_tets, topo_nodes, true);
    CHKERR moab.get_connectivity(all_tets, mid_nodes, false);
    std::cout << "\n Mesh contains of " << mid_nodes.size()
              << " nodes in total. ";
    mid_nodes = subtract(mid_nodes, topo_nodes);
    std::cout << mid_nodes.size() << " higher order nodes. \n";

    for (int dim : {1, 2, 3}) {
      EntityHandle sub_meshset;
      CHKERR moab.create_meshset(MESHSET_SET, sub_meshset);
      Range edges;
      CHKERR moab.get_entities_by_dimension(0, dim, edges, false);
      CHKERR moab.add_entities(sub_meshset, edges);
      CHKERR moab.convert_entities(sub_meshset, false, false,
                                   false /*,&my_add_remove*/);
    }

    Range new_nodes;
    CHKERR moab.get_entities_by_type(0, MBVERTEX, new_nodes);
    PetscPrintf(PETSC_COMM_WORLD, "New number of nodes: %d. \n",
                new_nodes.size());
    PetscPrintf(PETSC_COMM_WORLD, "Saving file out.h5m... \n");

    CHKERR moab.write_file("out.h5m");
  }
  CATCH_ERRORS;

  MoFEM::Core::Finalize();

  return 0;
}
