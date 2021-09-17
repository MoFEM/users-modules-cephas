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
cerr << "\n\nWTH 1\n\n";
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
    cerr << "\n\nWTH -1\n\n";
    if (pcomm == NULL)
      pcomm = new ParallelComm(&moab, PETSC_COMM_WORLD);

    const char *option;
    option = "";
    cerr << "\n\nWTH - 2\n\n";
    CHKERR moab.load_file(mesh_file_name, 0, option);
    cerr << "\n\nWTH\n\n";
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

    // Range new_vol;
    // CHKERR moab.get_entities_by_type(0, MBTET, new_vol);
    // cerr << "new_vol " << new_vol.size() << "\n";

    // for (int dim : {3, 2, 1}) {
    //   Range edges_1, edges_2;
    //   ErrorCode tmp_result;
    //   tmp_result = moab.get_adjacencies(new_vol, dim, true, edges_1,
    //                                     moab::Interface::UNION);

    //   if (MB_SUCCESS != tmp_result) {
    //     cerr << "check multiplicity first " << dim << "\n";
    //     cerr << "size " << edges_1.size() << "\n";
    //   }
    // }

    Range ents, skin;
    CHKERR moab.get_entities_by_dimension(0, 3, ents);
    MOFEM_LOG("WORLD", Sev::verbose) << "Ents:\n" << ents;

    Skinner skinner(&moab);
    CHKERR skinner.find_skin(0, ents, false, skin);
    
    MOFEM_LOG("WORLD", Sev::verbose) << "Skin:\n" << skin;

    PetscPrintf(PETSC_COMM_WORLD, "Saving file out.h5m... \n");

      // CHKERR moab.delete_entities(delete_faces);
      // CHKERR moab.delete_entities(delete_volume);

    // {
    //   Range new_faces_2;
    //   CHKERR moab.get_entities_by_type(0, MBTRI, new_faces_2);
    //   cerr << "new_faces_2 " << new_faces_2.size() << "\n";

    //   Range new_faces_3;
    //   CHKERR moab.get_entities_by_type(0, MBTRI, new_faces_3);
    //   cerr << "new_faces_3 " << new_faces_3.size() << "\n";

    //   Range new_edge;
    //   CHKERR moab.get_entities_by_type(0, MBEDGE, new_edge);
    //   cerr << "new_edge " << new_edge.size() << "\n";
 
    // }

      Range check_vol;
      CHKERR moab.get_entities_by_type(0, MBTET, check_vol);
      cerr << "check_vol " << check_vol.size() << "\n";
      
    // for (int dim : {1, 2, 3}) {
    //   // EntityHandle sub_meshset = 0;
    //   // CHKERR moab.create_meshset(MESHSET_SET, sub_meshset);
    //   Range ents_interogated;
    //   ErrorCode tmp_result;
    //   tmp_result = moab.get_adjacencies(check_vol, dim, true, ents_interogated,
    //                               moab::Interface::UNION);
    // cerr << "size " << ents_interogated.size() << " dim " << dim <<"\n";
    // if (MB_SUCCESS != tmp_result){
    //       cerr << "check multiplicity first " << dim << "\n";
    //       cerr << "size " << ents_interogated.size() <<"\n";
    // }
    // if (dim == 2) {
    //   Range new_faces_3;
    //   CHKERR moab.get_entities_by_type(0, MBTRI, new_faces_3);

    //   Range faces = subtract(new_faces_3, ents_interogated);
    //    Range surface;
    // if (meshset_manager->checkMeshset(cutting_side_set, SIDESET))
    //   CHKERR meshset_manager->getEntitiesByDimension(cutting_side_set, SIDESET,
    //                                                  2, surface, true);

    // Range search_edges;
    // ErrorCode tmp_result;
    // tmp_result =
    //     moab.get_adjacencies(surface, 1, false, search_edges, moab::Interface::UNION);

    // if (MB_MULTIPLE_ENTITIES_FOUND == tmp_result)
    //   cerr << "check multiplicity cutting face " << "\n";

    //   faces = subtract(faces, surface);
    //   cerr << "surface " << surface.size() << "\n";                                         
    //   // CHKERR moab.delete_entities(faces);
    // } else if (dim == 1) {
    //   Range new_edges_3;
    //   CHKERR moab.get_entities_by_type(0, MBEDGE, new_edges_3);

    //   Range edges = subtract(new_edges_3, ents_interogated);
    //     Range cut_surface_edges;
    // if (meshset_manager->checkMeshset(cutting_side_set, SIDESET))
    //   CHKERR meshset_manager->getEntitiesByDimension(cutting_side_set, SIDESET,
    //                                                  dim, cut_surface_edges, true);
    //   cerr << "edges before: " << edges.size()<< "\n";
    //   cerr << "cut_surface_edges " << cut_surface_edges.size() <<"\n";
    //   edges = subtract(edges, cut_surface_edges);
    //   cerr << "edges after: " << edges.size()<< "\n";
    //   CHKERR moab.delete_entities(edges);
    // }
    //   }



      Range ents_interogated;
      ErrorCode tmp_result;
      tmp_result = moab.get_adjacencies(check_vol, 1, true, ents_interogated,
                                  moab::Interface::UNION);
    cerr << "size " << ents_interogated.size() << " dim " << 1 <<"\n";
    if (MB_SUCCESS != tmp_result){
          cerr << "check multiplicity first " << 1 << "\n";
          cerr << "size " << ents_interogated.size() <<"\n";
    }
    
       Range new_edges_3;
      CHKERR moab.get_entities_by_type(0, MBEDGE, new_edges_3);

      Range edges = subtract(new_edges_3, ents_interogated);
        Range cut_surface_edges;
    if (meshset_manager->checkMeshset(cutting_side_set, SIDESET))
      CHKERR meshset_manager->getEntitiesByDimension(cutting_side_set, SIDESET,
                                                     1, cut_surface_edges, true);

      cerr << "edges before: " << edges.size()<< "\n";
      cerr << "cut_surface_edges " << cut_surface_edges.size() <<"\n";
      edges = subtract(edges, cut_surface_edges);
      cerr << "edges after: " << edges.size()<< "\n";
      CHKERR moab.delete_entities(edges);

    // {
    //   Range new_faces_2;
    //   CHKERR moab.get_entities_by_type(0, MBTRI, new_faces_2);
    //   cerr << "2 new_faces_2 " << new_faces_2.size() << "\n";

    //   Range new_faces_3;
    //   CHKERR moab.get_entities_by_type(0, MBTRI, new_faces_3);
    //   cerr << "2 new_faces_3 " << new_faces_3.size() << "\n";

    //   Range new_edge;
    //   CHKERR moab.get_entities_by_type(0, MBEDGE, new_edge);
    //   cerr << "2 new_edge " << new_edge.size() << "\n";

    //   check_vol.clear();
    //   CHKERR moab.get_entities_by_type(0, MBTET, check_vol);
    //   cerr << "2 check_vol " << check_vol.size() << "\n";
    // }

    // ents.clear();
    // CHKERR moab.get_entities_by_dimension(0, 3, ents);
    // Range skin_2;
    // CHKERR skinner.find_skin(0, ents, false, skin_2);

    // EntityHandle skin_faces_ent_handle;
    // CHKERR moab.create_meshset(MESHSET_SET, skin_faces_ent_handle);

    // CHKERR
    // moab.add_entities(skin_faces_ent_handle, skin_2);

    // CHKERR moab.write_mesh("edges_check.vtk", &skin_faces_ent_handle, 1);

    // CHKERR moab.delete_entities(delete_verts);
    CHKERR moab.write_file("out.h5m");
  }
  CATCH_ERRORS;

  MoFEM::Core::Finalize();

  return 0;
}
