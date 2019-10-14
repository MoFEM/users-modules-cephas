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
    PetscBool flg = PETSC_TRUE;
    char mesh_file_name[255];
#if PETSC_VERSION_GE(3, 6, 4)
    CHKERR PetscOptionsGetString(PETSC_NULL, "", "-my_file", mesh_file_name,
                                 255, &flg);
#else
    CHKERR PetscOptionsGetString(PETSC_NULL, PETSC_NULL, "-my_file",
                                 mesh_file_name, 255, &flg);
#endif
    if (flg != PETSC_TRUE) {
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

    PrismsFromSurfaceInterface *prisms_from_surface_interface;
    CHKERR m_field.getInterface(prisms_from_surface_interface);

    const int ID_SOLID_SURFACE = 1000;
    const int ID_INLET_EDGE    = 1001;
    const int ID_OUTLET_EDGE   = 1002;
    const int ID_SYMMETRY_EDGE = 1003;

    const int ID_FLUID_BLOCK   = 2000;
    const int ID_INLET_PRESS   = 2001;
    const int ID_OUTLET_PRESS  = 2002;
    const int ID_SYMMETRY_DISP = 2003;
    const int ID_INLET_DISP    = 2004;
    const int ID_OUTLET_DISP   = 2005;
    const int ID_FIXED_DISP    = 2006;

    const int ID_FAR_FIELD_DISP = 2007;
   
    MeshsetsManager *mmanager_ptr;
    Range tris;
    CHKERR m_field.query_interface(mmanager_ptr);
    //Range ents_2nd_layer;
    CHKERR mmanager_ptr->getEntitiesByDimension(ID_SOLID_SURFACE, SIDESET, 2,
                                                tris, true);

    cout << "!!! TRIS: " << tris.size() << endl;
    tris.print();

    Range prisms;
    CHKERR prisms_from_surface_interface->createPrisms(tris, prisms);
    prisms_from_surface_interface->createdVertices.clear();
    
    //Range add_prims_layer;
    // CHKERR prisms_from_surface_interface->createPrismsFromPrisms(
    //     prisms, true, add_prims_layer);
    //prisms.merge(add_prims_layer);

    //double d0[3] = {0, 0, 0};
    //double d1[3] = {0, 0, 1};
    //CHKERR prisms_from_surface_interface->setThickness(prisms, d0, d1);

    auto set_thickness = [&](const Range &prisms) {
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
        director[2] = -coords[2];
        cblas_daxpy(3, 1, director, 1, coords, 1);
        CHKERR moab.set_coords(&*nit, 1, coords);
      }
      MoFEMFunctionReturn(0);
    };

    CHKERR set_thickness(prisms);

    CHKERR mmanager_ptr->addMeshset(BLOCKSET, ID_FLUID_BLOCK, "FLUID");
    CHKERR mmanager_ptr->addEntitiesToMeshset(BLOCKSET, ID_FLUID_BLOCK, prisms);

    // std::vector<double> aTtr;
    // aTtr.resize(2);
    // aTtr[0] = 1.0;
    // aTtr[1] = 1.0;
    // CHKERR mmanager_ptr->setAtributes(BLOCKSET, ID_FLUID_BLOCK, aTtr);

    EntityHandle meshset;
    CHKERR mmanager_ptr->getMeshset(ID_FLUID_BLOCK, BLOCKSET, meshset);

    BitRefLevel bit_level0;
    bit_level0.set(0);
    CHKERR m_field.getInterface<BitRefManager>()->setBitRefLevelByDim(
        meshset, 3, bit_level0);
    CHKERR prisms_from_surface_interface->seedPrismsEntities(prisms,
                                                             bit_level0);

    cout << "FLUID: " << prisms.size() << " prism(s)" << endl;
    prisms.print();

    auto add_quad_set = [&](const int old_set_id, const int new_set_id,
                            const std::string new_set_name) {
      MoFEMFunctionBegin;

      Range edges, faces;
      CHKERR mmanager_ptr->getEntitiesByDimension(old_set_id, SIDESET, 1, edges,
                                                  true);
      CHKERR moab.get_adjacencies(edges, 2, true, faces,
                                  moab::Interface::UNION);
      faces = faces.subset_by_type(MBQUAD);

      cout << new_set_name << ": " << faces.size() << " face(s)" << endl;

      faces.print();

      CHKERR mmanager_ptr->addMeshset(BLOCKSET, new_set_id, new_set_name);
      CHKERR mmanager_ptr->addEntitiesToMeshset(BLOCKSET, new_set_id, faces);

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

      faces = subtract(faces, tris);

      cout << new_set_name << ": " << faces.size() << " faces(s)" << endl;
      faces.print();

      CHKERR mmanager_ptr->addMeshset(BLOCKSET, new_set_id, new_set_name);
      CHKERR mmanager_ptr->addEntitiesToMeshset(BLOCKSET, new_set_id, faces);

      MoFEMFunctionReturn(0);
    };

    // add_quad_set(ID_INLET_EDGE, ID_INLET_PRESS, "MESHSET_INLET_PRESS");
    // add_quad_set(ID_OUTLET_EDGE, ID_OUTLET_PRESS, "MESHSET_OUTLET_PRESS");

    add_quad_set(ID_SYMMETRY_EDGE, ID_SYMMETRY_DISP, "SYMMETRY_DISP");
    add_quad_set(ID_INLET_EDGE, ID_INLET_DISP, "INLET_DISP");
    add_quad_set(ID_OUTLET_EDGE, ID_OUTLET_DISP, "OUTLET_DISP");

    add_top_tris_set(ID_FAR_FIELD_DISP, "FAR_FIELD_DISP");

    CHKERR mmanager_ptr->addMeshset(BLOCKSET, ID_FIXED_DISP, "FIXED_DISP");
    CHKERR mmanager_ptr->addEntitiesToMeshset(BLOCKSET, ID_FIXED_DISP,
    tris); cout << "FIXED_DISP"
         << ": " << tris.size() << " faces(s)" << endl;
    tris.print();

    CHKERR mmanager_ptr->setMeshsetFromFile("bc.cfg");

    // CHKERR moab.create_meshset(MESHSET_SET | MESHSET_TRACK_OWNER,
    // meshset); CHKERR moab.add_entities(meshset, prisms);

    CHKERR mmanager_ptr->printDisplacementSet();
    // CHKERR mmanager_ptr->printMaterialsSet();
    // CHKERR mmanager_ptr->printPressureSet();

    EntityHandle rootset = moab.get_root_set();

    CHKERR moab.write_file("prisms_layer.h5m", "MOAB", "", &rootset, 1);
    CHKERR moab.write_file("prisms_layer.vtk", "VTK", "", &rootset, 1);

    } CATCH_ERRORS;

    MoFEM::Core::Finalize();
    return 0;
}
