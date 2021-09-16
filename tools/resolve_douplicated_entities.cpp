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
    int cutting_side_set = 200;
    PetscBool flg_cutting_side_set;
    
    CHKERR PetscOptionsBegin(PETSC_COMM_WORLD, "", "Field to vertices options",
                             "none");
    CHKERR PetscOptionsString("-my_file", "mesh file name", "", "mesh.h5m",
                              mesh_file_name, 255, &flg_file);
    CHKERR PetscOptionsInt("-cutting_side_set", "crete side set", "",
                           cutting_side_set, &cutting_side_set,
                           &flg_cutting_side_set);
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

    // Range all_tets, topo_nodes, mid_nodes;
    // CHKERR moab.get_entities_by_type(0, MBTET, all_tets);
    // CHKERR moab.get_connectivity(all_tets, topo_nodes, true);
    // CHKERR moab.get_connectivity(all_tets, mid_nodes, false);
    // std::cout << "\n Mesh contains of " << mid_nodes.size()
    //           << " nodes in total. ";
    // mid_nodes = subtract(mid_nodes, topo_nodes);
    // std::cout << mid_nodes.size() << " higher order nodes. \n";

    Range new_faces_4;
    CHKERR moab.get_entities_by_type(0, MBTRI, new_faces_4);
    cerr << "new_faces_4 " << new_faces_4.size() << "\n";

    Range new_vol;
    CHKERR moab.get_entities_by_type(0, MBTET, new_vol);
    cerr << "new_vol " << new_vol.size() << "\n";


    for (int dim : {2, 1}) {
      // EntityHandle sub_meshset = 0;
      // CHKERR moab.create_meshset(MESHSET_SET, sub_meshset);
      Range edges_1, edges_2;
      ErrorCode tmp_result;
      tmp_result = moab.get_adjacencies(new_faces_4, dim, true, edges_1,
                                  moab::Interface::UNION);

    if (MB_SUCCESS != tmp_result){
          cerr << "check multiplicity first " << dim << "\n";
          cerr << "size " << edges_1.size() <<"\n";
    }
      }


    for (int dim : {3, 2, 1}) {
      // EntityHandle sub_meshset = 0;
      // CHKERR moab.create_meshset(MESHSET_SET, sub_meshset);
      Range edges_1, edges_2;
      ErrorCode tmp_result;
      tmp_result = moab.get_adjacencies(new_vol, dim, true, edges_1,
                                  moab::Interface::UNION);

    if (MB_SUCCESS != tmp_result){
          cerr << "check multiplicity first " << dim << "\n";
          cerr << "size " << edges_1.size() <<"\n";
    }
      }


    Range ents, skin;
    CHKERR moab.get_entities_by_dimension(0, 3, ents);
    MOFEM_LOG("WORLD", Sev::verbose) << "Ents:\n" << ents;

    Skinner skinner(&moab);
    CHKERR skinner.find_skin(0, ents, false, skin);
    
    MOFEM_LOG("WORLD", Sev::verbose) << "Skin:\n" << skin;

    Range delete_faces;
    Range delete_volume;
    Range delete_edges;
    Range delete_verts;
    for (int dim : {3, 2, 1}) {
      // EntityHandle sub_meshset = 0;
      // CHKERR moab.create_meshset(MESHSET_SET, sub_meshset);
      Range edges_1, edges_2;
      ErrorCode tmp_result;
      tmp_result = moab.get_adjacencies(ents, dim, false, edges_1,
                                  moab::Interface::UNION);

    if (MB_MULTIPLE_ENTITIES_FOUND == tmp_result)
          cerr << "check multiplicity " << dim << "\n";

      if (dim == 2 || dim == 3 ) {
        
        for (Range::iterator edge_1_it = edges_1.begin();
             edge_1_it != edges_1.end(); edge_1_it++) {

          ErrorCode tmp_result_1;
          Range tri_edges;
          // tmp_result_1 = moab.get_adjacencies(&*edge_1_it, 1, false, tri_edges,
          //                                     moab::Interface::UNION);
          tmp_result_1 = moab.get_adjacencies(&*edge_1_it, 1, 1,
                                                   true, tri_edges);
          

          ErrorCode check_check;
          
          EntityHandle meshset = 0;
          CHKERR moab.create_meshset(MESHSET_SET, meshset);
          CHKERR moab.add_entities(meshset, tri_edges);
          // check_check = moab.check_adjacencies(&*edge_1_it, 1);

          // if (MB_MULTIPLE_ENTITIES_FOUND == check_check)
          //   cerr << "check_check " << dim << "\n";
          if (tri_edges.size() != 6 && dim == 3) {
            cerr << "tri_edges " << tri_edges.size() << "\n";
            delete_volume.insert(*edge_1_it);
          }

          if (tri_edges.size() != 3 && dim == 2) {
            cerr << "tri_edges " << tri_edges.size() << "\n";
            delete_faces.insert(*edge_1_it);
            //delete_edges.insert(tri_edges.begin(), tri_edges.end());
            Range tri_vertex;
          tmp_result_1 =
              moab.get_adjacencies(&*edge_1_it, 1, 0, false, tri_vertex);
          cerr << "tri_vertex " << tri_vertex.size() << "\n";
          // delete_faces.insert(tri_vertex.begin(), tri_vertex.end());

            // Range vol_ele;
            // tmp_result_1 = moab.get_adjacencies(&*edge_1_it, 1, 3,
            //                                          false, vol_ele);
            //                                          cerr << "vol_ele " <<
            //                                          vol_ele.size() <<"\n";

            // Range vertex_ele;
            // tmp_result_1 = moab.get_adjacencies(&*edge_1_it, 1, 0,
            //                                          false, vertex_ele);
            //                                          cerr << "vertex_ele " <<
            //                                          vertex_ele.size()
            //                                          <<"\n";
        }
        
        

        }

      // EntityHandle meshset = 0;
      // CHKERR moab.create_meshset(MESHSET_SET, meshset);
      cerr << "delete_faces " << delete_faces.size() << "\n";
      cerr << "delete_volume " << delete_volume.size() << "\n";
      // CHKERR moab.add_entities(meshset, delete_faces);

        // CHKERR moab.delete_entities(&meshset, 1);

        // Range meshsets;
        // CHKERR moab.get_entities_by_type(0, MBENTITYSET, meshsets, false);
        cerr << "1\n";
        Range meshsets;
        CHKERR moab.get_entities_by_type(0, MBENTITYSET, meshsets, true);
        // for (auto m : meshsets) {
        //   CHKERR moab.remove_entities(m, delete_faces);
        // }
          
          CHKERR moab.get_connectivity(delete_faces, delete_verts, false);
          
          
          

          ents.clear();
          CHKERR moab.get_entities_by_dimension(0, 3, ents);
          edges_1.clear();
          // sking staff

          skin.clear();
          CHKERR skinner.find_skin(0, ents, false, skin);
          ErrorCode tmp_result_3;
          tmp_result_3 = moab.get_adjacencies(skin, dim, false, edges_1,
                                            moab::Interface::UNION);

          if (MB_MULTIPLE_ENTITIES_FOUND == tmp_result_3)
            cerr << "check multiplicity again!! " << dim << "\n";

          for (Range::iterator edge_1_it = edges_1.begin();
               edge_1_it != edges_1.end(); edge_1_it++) {

            ErrorCode tmp_result_2;
            Range tri_edges_e;
            // tmp_result_2 = moab.get_adjacencies(&*edge_1_it, 1, false,
            // tri_edges_e,
            //                                     moab::Interface::UNION);
            tmp_result_2 =
                moab.get_adjacencies(&*edge_1_it, 1, 1, false, tri_edges_e);
                if(tri_edges_e.size() == 0)
            cerr << "again! " << tri_edges_e.size() << " dim " << dim << "\n";
          }
        }

        
        // tmp_result = moab.get_adjacencies(skin, dim, true, edges_2,
        //                                   moab::Interface::UNION);

        // if (MB_MULTIPLE_ENTITIES_FOUND == tmp_result)
        //   cerr << "check multiplicity 2 " << dim << "\n";

        cerr << "dim " << dim << " edges_1: " << edges_1.size()
             /*<< " edges_2: " << edges_2.size()*/ << "\n";
        // edges_2 = intersect(edges_1, edges_2);
        // cerr << "edges_2 intersect " << edges_2.size() << "\n";
        // CHKERR moab.get_entities_by_dimension(0, dim, edges, false);
        // CHKERR moab.add_entities(sub_meshset, edges);
        // CHKERR moab.convert_entities(sub_meshset, false, false,
        //                              false /*,&my_add_remove*/);
      }

    Range new_faces;
    CHKERR moab.get_entities_by_type(0, MBTRI, new_faces);
    cerr << "new_faces " << new_faces.size()<< "\n";
    // PetscPrintf(PETSC_COMM_WORLD, "New number of nodes: %d. \n",
    //             new_nodes.size());
    PetscPrintf(PETSC_COMM_WORLD, "Saving file out.h5m... \n");

    Range meshsets;
    CHKERR moab.get_entities_by_type(0, MBENTITYSET, meshsets, true);
    for (auto m : meshsets) {
      CHKERR moab.remove_entities(m, delete_faces);
    }

      // CHKERR moab.delete_entities(delete_faces);
      // CHKERR moab.delete_entities(delete_volume);

    {
      Range new_faces_2;
      CHKERR moab.get_entities_by_type(0, MBTRI, new_faces_2);
      cerr << "new_faces_2 " << new_faces_2.size() << "\n";

      Range new_faces_3;
      CHKERR moab.get_entities_by_type(0, MBTRI, new_faces_3);
      cerr << "new_faces_3 " << new_faces_3.size() << "\n";

      Range new_edge;
      CHKERR moab.get_entities_by_type(0, MBEDGE, new_edge);
      cerr << "new_edge " << new_edge.size() << "\n";

      
    }
      Range check_vol;
      CHKERR moab.get_entities_by_type(0, MBTET, check_vol);
      cerr << "check_vol " << check_vol.size() << "\n";
      
    for (int dim : {3, 2, 1}) {
      // EntityHandle sub_meshset = 0;
      // CHKERR moab.create_meshset(MESHSET_SET, sub_meshset);
      Range edges_1, edges_2;
      ErrorCode tmp_result;
      tmp_result = moab.get_adjacencies(check_vol, dim, true, edges_1,
                                  moab::Interface::UNION);
cerr << "size " << edges_1.size() << " dim " << dim <<"\n";
    if (MB_SUCCESS != tmp_result){
          cerr << "check multiplicity first " << dim << "\n";
          cerr << "size " << edges_1.size() <<"\n";
    }
    if (dim == 2) {
      Range new_faces_3;
      CHKERR moab.get_entities_by_type(0, MBTRI, new_faces_3);

      Range faces = subtract(new_faces_3, edges_1);
       Range surface;
    if (meshset_manager->checkMeshset(cutting_side_set, SIDESET))
      CHKERR meshset_manager->getEntitiesByDimension(cutting_side_set, SIDESET,
                                                     2, surface, true);

Range search_edges;
    ErrorCode tmp_result;
    tmp_result =
        moab.get_adjacencies(surface, 1, false, search_edges, moab::Interface::UNION);

    if (MB_MULTIPLE_ENTITIES_FOUND == tmp_result)
      cerr << "check multiplicity cutting face " << "\n";

    // for (Range::iterator search_edges_it = search_edges.begin();
    //      search_edges_it != search_edges.end(); search_edges_it++) {

    //   Range search_tri;
    //   CHKERR moab.get_adjacencies(&*search_edges_it, 1, 2, true, search_tri);

    //   // CHKERR mField.get_moab().get_adjacencies(&*vertex_it_slave, 1, 1, true,
    //   //                                                  slave_edges);

    //   if (search_tri.size() > 2) {
    //     cerr << "More than 2 i.e. " << search_tri.size() << "\n";
    //   } else {
    //     cerr << "HO LOOK! " << search_tri.size() << "\n";
    //   }
    // }

      faces = subtract(faces, surface);
      cerr << "surface " << surface.size() << "\n";                                         
      // CHKERR moab.delete_entities(faces);
    } else if (dim == 1) {
            Range new_edges_3;
      CHKERR moab.get_entities_by_type(0, MBEDGE, new_edges_3);

      Range edges = subtract(new_edges_3, edges_1);
        Range cut_surface_edges;
    if (meshset_manager->checkMeshset(cutting_side_set, SIDESET))
      CHKERR meshset_manager->getEntitiesByDimension(cutting_side_set, SIDESET,
                                                     dim, cut_surface_edges, true);
      edges = subtract(edges, cut_surface_edges);
      CHKERR moab.delete_entities(edges);
    }
      }

    {
      Range new_faces_2;
      CHKERR moab.get_entities_by_type(0, MBTRI, new_faces_2);
      cerr << "new_faces_2 " << new_faces_2.size() << "\n";

      Range new_faces_3;
      CHKERR moab.get_entities_by_type(0, MBTRI, new_faces_3);
      cerr << "new_faces_3 " << new_faces_3.size() << "\n";

      Range new_edge;
      CHKERR moab.get_entities_by_type(0, MBEDGE, new_edge);
      cerr << "new_edge " << new_edge.size() << "\n";

      check_vol.clear();
      CHKERR moab.get_entities_by_type(0, MBTET, check_vol);
      cerr << "check_vol " << check_vol.size() << "\n";
    }

    ents.clear();
    CHKERR moab.get_entities_by_dimension(0, 3, ents);
    Range skin_2;
    CHKERR skinner.find_skin(0, ents, false, skin_2);

    EntityHandle skin_faces_ent_handle;
    CHKERR moab.create_meshset(MESHSET_SET, skin_faces_ent_handle);

    CHKERR
    moab.add_entities(skin_faces_ent_handle, skin_2);

    // CHKERR m_field.getInterface<BitRefManager>()->setBitRefLevelByDim(
    //     skin_faces_ent_handle, 3, bit_levels.back());

    CHKERR moab.write_mesh("edges_check.vtk", &skin_faces_ent_handle, 1);

    // CHKERR moab.delete_entities(delete_verts);
    CHKERR moab.write_file("out.h5m");
  }
  CATCH_ERRORS;

  MoFEM::Core::Finalize();

  return 0;
}
