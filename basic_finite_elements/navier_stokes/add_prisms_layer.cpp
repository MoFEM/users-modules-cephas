/** \file prisms_elements_from_surface.cpp
  \example prisms_elements_from_surface.cpp
  \brief Adding prims on the surface

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

namespace bio = boost::iostreams;
using bio::stream;
using bio::tee_device;

using namespace MoFEM;

static char help[] = "...\n\n";
static int debug = 1;

int main(int argc, char *argv[]) {

  MoFEM::Core::Initialize(&argc, &argv, (char *)0, help);

  try {

    moab::Core mb_instance;
    moab::Interface &moab = mb_instance;
    int rank;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);

    // Read parameters from line command
    PetscBool flg_mesh_file = PETSC_TRUE;
    char mesh_file_name[255];

    PetscBool fluid_surface = PETSC_FALSE;

    int nb_layers = 1; // default number of prism layers

    CHKERR PetscOptionsBegin(PETSC_COMM_WORLD, "", "ADD_PRISMS_LAYER", "none");

    CHKERR PetscOptionsString("-my_file", "mesh file name", "", "mesh.h5m",
                              mesh_file_name, 255, &flg_mesh_file);

    CHKERR PetscOptionsInt("-my_nb_layers", "number oflayers", "", nb_layers,
                           &nb_layers, PETSC_NULL);

    CHKERR PetscOptionsBool("-my_fluid_surface", "", "", PETSC_FALSE,
                            &fluid_surface, PETSC_NULL);

    int ierr = PetscOptionsEnd();
    CHKERRG(ierr);

    if (flg_mesh_file != PETSC_TRUE) {
      SETERRQ(PETSC_COMM_SELF, 1, "*** ERROR -my_file (MESH FILE NEEDED)");
    }

    // Read mesh to MOAB
    const char *option;
    option = ""; //"PARALLEL=BCAST;";//;DEBUG_IO";
    CHKERR moab.load_file(mesh_file_name, 0, option);
    ParallelComm *pcomm = ParallelComm::get_pcomm(&moab, MYPCOMM_INDEX);
    if (pcomm == NULL)
      pcomm = new ParallelComm(&moab, PETSC_COMM_WORLD);

    // Create MoFEM (Joseph) databas
    MoFEM::Core core(moab);
    MoFEM::Interface &m_field = core;

    Skinner skinner(&moab);

    const int ID_SOLID_SURFACE = 1000;
    const int ID_INLET_EDGE = 1001;
    const int ID_OUTLET_EDGE = 1002;
    const int ID_SYMMETRY_EDGE = 1003;

    const int ID_FLUID_BLOCK = 2000;
    const int ID_INLET_PRESS = 2001;
    const int ID_OUTLET_PRESS = 2002;
    const int ID_SYMMETRY_VEL = 2003;
    const int ID_INLET_VEL = 2004;
    const int ID_OUTLET_VEL = 2005;
    const int ID_FIXED_VEL = 2006;

    const int ID_FLUID_SURFACE = 2010;

    // const int ID_FAR_FIELD_VEL = 2007;

    PrismsFromSurfaceInterface *prisms_from_surface_interface;
    CHKERR m_field.getInterface(prisms_from_surface_interface);
 
    MeshsetsManager *mmanager_ptr;
    Range bot_tris;
    CHKERR m_field.query_interface(mmanager_ptr);

    CHKERR mmanager_ptr->getEntitiesByDimension(ID_SOLID_SURFACE, BLOCKSET, 2,
                                                bot_tris, true);

    cout << "Solid surface: " << bot_tris.size() << " face(s)" << endl;
    // bot_tris.print();

    Range nodes;
    CHKERR moab.get_connectivity(bot_tris, nodes);
    double coords[3], director[3];
    double B = 0.1;
    double L = 1.0;
    for (Range::iterator nit = nodes.begin(); nit != nodes.end(); nit++) {
      CHKERR moab.get_coords(&*nit, 1, coords);
      director[0] = 0.0;
      director[1] = 0.0;
      // director[2] = -B * (1. - cos(2 * M_PI * coords[0] / L) *
      //                              cos(2 * M_PI * coords[1] / L));
     //  director[2] = -B - B * (1. - cos(2 * M_PI * coords[0] / L));                             
      director[2] = -B * coords[0];
      cblas_daxpy(3, 1, director, 1, coords, 1);
      CHKERR moab.set_coords(&*nit, 1, coords);
    }

    auto set_thickness = [&](const Range &prisms, int nb_layers) {
      MoFEMFunctionBegin;
      Range nodes_f4;
      int ff = 4;
      for (Range::iterator pit = prisms.begin(); pit != prisms.end(); pit++) {
        EntityHandle face;
        CHKERR moab.side_element(*pit, 2, ff, face);
        const EntityHandle *conn;
        int number_nodes = 0;
        CHKERR moab.get_connectivity(face, conn, number_nodes, false);
        nodes_f4.insert(&conn[0], &conn[number_nodes]);
      }
      double coords[3], director[3];
      for (Range::iterator nit = nodes_f4.begin(); nit != nodes_f4.end();
           nit++) {
        CHKERR moab.get_coords(&*nit, 1, coords);
        if (coords[2] > 0.0) {
          SETERRQ(PETSC_COMM_SELF, MOFEM_DATA_INCONSISTENCY,
                  "Solid surface is on the wrong side of the plane z = 0");
        }
        director[0] = 0.0;
        director[1] = 0.0;
        director[2] = -coords[2] / (double)nb_layers;
        cblas_daxpy(3, 1, director, 1, coords, 1);
        CHKERR moab.set_coords(&*nit, 1, coords);
      }
      MoFEMFunctionReturn(0);
    };

    Range prisms, new_layer, prev_layer;
    CHKERR prisms_from_surface_interface->createPrisms(bot_tris, new_layer);
    prisms.merge(new_layer);
    // prisms_from_surface_interface->createdVertices.clear();
    CHKERR set_thickness(new_layer, nb_layers);

    for (int i = 1; i < nb_layers; i++) {
      prev_layer = new_layer;
      new_layer.clear();
      CHKERR prisms_from_surface_interface->createPrismsFromPrisms(
          prev_layer, false, new_layer);
      prisms.merge(new_layer);
      CHKERR set_thickness(new_layer, nb_layers - i);
    }

    CHKERR mmanager_ptr->addMeshset(BLOCKSET, ID_FLUID_BLOCK, "FLUID");
    CHKERR mmanager_ptr->addEntitiesToMeshset(BLOCKSET, ID_FLUID_BLOCK, prisms);

    EntityHandle meshset;
    CHKERR mmanager_ptr->getMeshset(ID_FLUID_BLOCK, BLOCKSET, meshset);

    BitRefLevel bit_level0;
    bit_level0.set(0);
    CHKERR m_field.getInterface<BitRefManager>()->setBitRefLevelByDim(
        meshset, 3, bit_level0);
    CHKERR prisms_from_surface_interface->seedPrismsEntities(prisms,
                                                             bit_level0);

    // cout << "FLUID: " << prisms.size() << " prism(s)" << endl;
    // prisms.print();

    auto add_quads_set = [&](const int old_set_id, const int new_set_id,
                             const std::string new_set_name) {
      MoFEMFunctionBegin;

      cout << new_set_name << endl;

      Range edges, quads, quads_layer;
      CHKERR mmanager_ptr->getEntitiesByDimension(old_set_id, BLOCKSET, 1,
                                                  edges, true);
      CHKERR moab.get_adjacencies(edges, 2, true, quads,
                                  moab::Interface::UNION);

      quads = quads.subset_by_type(MBQUAD);
      // cout << "first layer faces: " << quads.size() << endl;
      // quads.print();

      quads_layer = quads;
      for (int i = 1; i < nb_layers; i++) {
        Range prisms_layer, prisms_next_layer, tris_layer;
        Range adj_edges, adj_quads;

        CHKERR moab.get_adjacencies(quads_layer, 1, true, adj_edges,
                                    moab::Interface::UNION);
        CHKERR moab.get_adjacencies(adj_edges, 2, true, adj_quads,
                                    moab::Interface::UNION);
        adj_quads = adj_quads.subset_by_type(MBQUAD);
        adj_quads = subtract(adj_quads, quads);
        // cout << "adj quads: " << adj_quads.size() << endl;
        // adj_quads.print();

        CHKERR moab.get_adjacencies(quads_layer, 3, true, prisms_layer,
                                    moab::Interface::UNION);
        CHKERR moab.get_adjacencies(prisms_layer, 2, true, tris_layer,
                                    moab::Interface::UNION);
        tris_layer = tris_layer.subset_by_type(MBTRI);
        // cout << "prism layer bot_tris: " << tris_layer.size() << endl;
        // tris_layer.print();

        CHKERR moab.get_adjacencies(tris_layer, 3, true, prisms_next_layer,
                                    moab::Interface::UNION);
        prisms_next_layer = subtract(prisms_next_layer, prisms_layer);

        quads_layer.clear();

        CHKERR moab.get_adjacencies(prisms_next_layer, 2, true, quads_layer,
                                    moab::Interface::UNION);
        quads_layer = quads_layer.subset_by_type(MBQUAD);

        quads_layer = intersect(quads_layer, adj_quads);

        // cout << "quads layer: " << quads_layer.size() << endl;
        // quads_layer.print();

        Range nodes;
        CHKERR moab.get_connectivity(quads_layer, nodes);
        MatrixDouble coords(nodes.size(), 3);
        CHKERR moab.get_coords(nodes, &coords(0, 0));

        // for (int j = 0; j < nodes.size(); j++) {
        //   cout << coords(j, 0) << " " << coords(j, 1) << " "
        //        << coords(j, 2) << endl;
        // }

        quads.merge(quads_layer);
      }

      // cout << new_set_name << ": " << quads.size() << " face(s)" << endl;
      // quads.print();

      CHKERR mmanager_ptr->addMeshset(BLOCKSET, new_set_id, new_set_name);
      CHKERR mmanager_ptr->addEntitiesToMeshset(BLOCKSET, new_set_id, quads);

      MoFEMFunctionReturn(0);
    };

    auto add_skin_tris_set = [&](const int new_set_id,
                                 const std::string new_set_name) {
      MoFEMFunctionBegin;
      Range faces;
      CHKERR skinner.find_skin(0, prisms, 2, faces);
      faces = faces.subset_by_type(MBTRI);

      cout << new_set_name << ": " << faces.size() << " faces(s)" << endl;
      faces.print();

      CHKERR mmanager_ptr->addMeshset(BLOCKSET, new_set_id, new_set_name);
      CHKERR mmanager_ptr->addEntitiesToMeshset(BLOCKSET, new_set_id, faces);

      MoFEMFunctionReturn(0);
    };

    auto add_top_tris_set = [&](const int new_set_id,
                                const std::string new_set_name) {
      MoFEMFunctionBegin;
      Range faces;
      CHKERR skinner.find_skin(0, prisms, 2, faces);
      faces = faces.subset_by_type(MBTRI);

      faces = subtract(faces, bot_tris);

      cout << new_set_name << ": " << faces.size() << " faces(s)" << endl;
      // faces.print();

      CHKERR mmanager_ptr->addMeshset(BLOCKSET, new_set_id, new_set_name);
      CHKERR mmanager_ptr->addEntitiesToMeshset(BLOCKSET, new_set_id, faces);

      MoFEMFunctionReturn(0);
    };

    auto add_edges_set = [&](const int old_set_id, const int new_set_id,
                             const std::string new_set_name) {
      MoFEMFunctionBegin;

      Range edges, quads, new_edges;
      CHKERR mmanager_ptr->getEntitiesByDimension(old_set_id, BLOCKSET, 1,
                                                  edges, true);
      CHKERR moab.get_adjacencies(edges, 2, true, quads,
                                  moab::Interface::UNION);

      quads = quads.subset_by_type(MBQUAD);
      Range quads_edges;
      CHKERR moab.get_adjacencies(quads, 1, true, quads_edges,
                                  moab::Interface::UNION);

      Range top_tris;
      CHKERR skinner.find_skin(0, prisms, 2, top_tris);
      top_tris = top_tris.subset_by_type(MBTRI);
      top_tris = subtract(top_tris, bot_tris);
      Range top_tris_edges;
      CHKERR moab.get_adjacencies(top_tris, 1, true, top_tris_edges,
                                  moab::Interface::UNION);

      new_edges = intersect(quads_edges, top_tris_edges);

      CHKERR mmanager_ptr->addMeshset(BLOCKSET, new_set_id, new_set_name);
      CHKERR mmanager_ptr->addEntitiesToMeshset(BLOCKSET, new_set_id,
                                                new_edges);

      MoFEMFunctionReturn(0);
    };

    if (fluid_surface) {
      CHKERR add_top_tris_set(ID_FLUID_SURFACE, "FLUID_SURFACE");
      CHKERR add_edges_set(ID_INLET_EDGE, ID_INLET_PRESS, "INLET_PRESS");
      CHKERR add_edges_set(ID_OUTLET_EDGE, ID_OUTLET_PRESS, "OUTLET_PRESS");
    } 
    else {
      CHKERR add_quads_set(ID_INLET_EDGE, ID_INLET_PRESS, "INLET_PRESS");
      CHKERR add_quads_set(ID_OUTLET_EDGE, ID_OUTLET_PRESS, "OUTLET_PRESS");

      CHKERR add_quads_set(ID_SYMMETRY_EDGE, ID_SYMMETRY_VEL, "SYMMETRY_VEL");
      CHKERR add_quads_set(ID_INLET_EDGE, ID_INLET_VEL, "INLET_VEL");
      CHKERR add_quads_set(ID_OUTLET_EDGE, ID_OUTLET_VEL, "OUTLET_VEL");

      CHKERR add_skin_tris_set(ID_FIXED_VEL, "FIXED_VEL");
    }

    CHKERR mmanager_ptr->setMeshsetFromFile("bc.cfg");

    CHKERR mmanager_ptr->printDisplacementSet();
    CHKERR mmanager_ptr->printMaterialsSet();
    CHKERR mmanager_ptr->printPressureSet();

    EntityHandle rootset = moab.get_root_set();

    Range faces;
    CHKERR skinner.find_skin(0, prisms, 2, faces);
    EntityHandle out_meshset_skin;
    CHKERR moab.create_meshset(MESHSET_SET, out_meshset_skin);
    CHKERR moab.add_entities(out_meshset_skin, faces);

    CHKERR moab.write_file("prisms_layer.h5m", "MOAB", "", &rootset, 1);
    CHKERR moab.write_file("prisms_layer.vtk", "VTK", "", &rootset, 1);
    CHKERR moab.write_file("prisms_layer_skin.vtk", "VTK", "",
                           &out_meshset_skin, 1);

    CHKERR moab.delete_entities(&out_meshset_skin, 1);
  }
  CATCH_ERRORS;

  MoFEM::Core::Finalize();
  return 0;
}
