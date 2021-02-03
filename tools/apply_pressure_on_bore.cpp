/** \file reading_med.cpp

  \brief Reading med files

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

    moab::Core mb_instance;
    moab::Interface &moab = mb_instance;
    ParallelComm *pcomm = ParallelComm::get_pcomm(&moab, MYPCOMM_INDEX);
    if (pcomm == NULL)
      pcomm = new ParallelComm(&moab, PETSC_COMM_WORLD);

    // Create MoFEM database
    MoFEM::Core core(moab);
    MoFEM::Interface &m_field = core;

    PetscBool flg_file, flg_field;
    char mesh_out_file[255] = "out.h5m";
    char mesh_file_name[255];
    char actual_strain_field[255];
    char actual_stress_field[255];
    char youngs_modulus[255];

    int time_step = 0;
    CHKERR PetscOptionsBegin(m_field.get_comm(), "", "Read MED tool", "none");
    CHKERR PetscOptionsInt("-med_time_step", "time step", "", time_step,
                           &time_step, PETSC_NULL);
    CHKERR PetscOptionsString("-output_file", "output mesh file name", "",
                              "out.h5m", mesh_out_file, 255, PETSC_NULL);

    CHKERR PetscOptionsString("-my_file", "mesh file name", "", "mesh.h5m",
                              mesh_file_name, 255, &flg_file);

    ierr = PetscOptionsEnd();
    CHKERRQ(ierr);

    if (flg_file != PETSC_TRUE) {
      SETERRQ(PETSC_COMM_SELF, 1, "*** ERROR -my_file (MESH FILE NEEDED)");
    }


    const char *option;
    option = "";
    CHKERR moab.load_file(mesh_file_name, 0, option);

    Range tets;
    CHKERR m_field.get_moab().get_entities_by_type(0, MBTET, tets);
    Skinner skin(&m_field.get_moab());
    Range skin_faces; // skin faces from 3d ents
    CHKERR skin.find_skin(0, tets, false, skin_faces);

    Range pressure_tris;
    for (Range::iterator pit = skin_faces.begin(); pit != skin_faces.end(); ++pit) {
      Range surface_verts;
      CHKERR m_field.get_moab().get_connectivity(&*pit, 1, surface_verts, true);

    double coords[3];
    double check = true;
    for (Range::iterator nit = surface_verts.begin(); nit != surface_verts.end();
         ++nit) {
           CHKERR m_field.get_moab().get_coords(&*nit, 1, coords);
           double x = coords[0];
           double y = coords[1];
           double r = sqrt(x * x + y * y);
           if( r > 131.9 || r < 130.1)
           check = false;
    }

    if(check)
    pressure_tris.insert(*pit);

    }
      EntityHandle meshset;
      CHKERR moab.create_meshset(MESHSET_SET, meshset);
      CHKERR moab.add_entities(meshset, pressure_tris);
      CHKERR moab.write_file("out_check_bore_tris.vtk", "VTK", "", &meshset, 1);
      CHKERR moab.delete_entities(&meshset, 1);

      MeshsetsManager *mmanager_ptr;
      CHKERR m_field.getInterface(mmanager_ptr);
      int pressure_bc_block_id = 33;
      CHKERR mmanager_ptr->addMeshset(SIDESET, pressure_bc_block_id);
      CHKERR mmanager_ptr->addEntitiesToMeshset(SIDESET, pressure_bc_block_id,
                                                pressure_tris);

      PressureCubitBcData pressure_bc;
      std::memcpy(pressure_bc.data.name, "Pressure", 8);
      pressure_bc.data.flag1 = 0;
      pressure_bc.data.flag2 = 0;
      pressure_bc.data.value1 = 1.;

      CHKERR mmanager_ptr->setBcData(SIDESET, pressure_bc_block_id,
                                     pressure_bc);

      cerr << "pressure_tris   " << pressure_tris.size() << "\n";
      CHKERR moab.write_file(mesh_out_file);
  }
  CATCH_ERRORS;

  MoFEM::Core::Finalize();

  return 0;
}
