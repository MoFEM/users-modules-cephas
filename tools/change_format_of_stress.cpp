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
    char input_stresses[255];
    char output_stress[255];
    
    int time_step = 0;
    CHKERR PetscOptionsBegin(m_field.get_comm(), "", "Read MED tool", "none");
    CHKERR PetscOptionsInt("-med_time_step", "time step", "", time_step,
                           &time_step, PETSC_NULL);
    CHKERR PetscOptionsString("-output_file", "output mesh file name", "",
                              "out.h5m", mesh_out_file, 255, PETSC_NULL);

    CHKERR PetscOptionsString("-my_file", "mesh file name", "", "mesh.h5m",
                              mesh_file_name, 255, &flg_file);

    CHKERR PetscOptionsString("-my_input_stress", "mesh file name", "", "",
                              input_stresses, 255, &flg_field);

    CHKERR PetscOptionsString("-my_output_stress", "mesh file name", "", "",
                              output_stress, 255, &flg_field);

    ierr = PetscOptionsEnd();
    CHKERRQ(ierr);

    if (flg_file != PETSC_TRUE) {
      SETERRQ(PETSC_COMM_SELF, 1, "*** ERROR -my_file (MESH FILE NEEDED)");
    }

    if (flg_field != PETSC_TRUE) {
      SETERRQ(PETSC_COMM_SELF, 1,
              "*** ERROR -my_field_name (FIELD NAME NEEDED)");
    }

    const char *option;
    option = "";
    CHKERR moab.load_file(mesh_file_name, 0, option);

    Tag th_input_stress;
    CHKERR m_field.get_moab().tag_get_handle(input_stresses, th_input_stress);

    Range tets;
    CHKERR m_field.get_moab().get_entities_by_type(0, MBTET, tets);

    auto get_tag = [&](const std::string name) {
      std::array<double, 9> def;
      std::fill(def.begin(), def.end(), 0);
      Tag th;
      CHKERR m_field.get_moab().tag_get_handle(name.c_str(), 9, MB_TYPE_DOUBLE,
                                               th, MB_TAG_CREAT | MB_TAG_SPARSE,
                                               def.data());
      return th;
    };


    auto th_output_stress = get_tag(output_stress);

    cerr << input_stresses << " " << tets.size() <<"\n" ;
    cerr << output_stress<<"\n" ;
    

    for (Range::iterator pit = tets.begin(); pit != tets.end(); ++pit) {
        
      VectorDouble9 vector_input_stress(9);
      // CHKERR m_field.get_moab().tag_get_data(th_input_stress, &*pit, 1,
      //                                        &*vector_input_stress.begin());
    Range surface_verts;
    CHKERR m_field.get_moab().get_connectivity(&*pit, 1, surface_verts, true);

    double *output_stress;
      CHKERR m_field.get_moab().tag_get_by_ptr(
          th_output_stress, &*pit, 1, (const void **)&output_stress);



for (Range::iterator vit = surface_verts.begin(); vit != surface_verts.end(); ++vit) {

      CHKERR m_field.get_moab().tag_get_data(th_input_stress, &*vit, 1,
                                             &*vector_input_stress.begin());

output_stress[0] += 0.25 * vector_input_stress(0);
output_stress[1] += 0.25 * vector_input_stress(4);
output_stress[2] += 0.25 * vector_input_stress(8);
output_stress[3] += 0.25 * vector_input_stress(1);
output_stress[4] += 0.25 * vector_input_stress(2);
output_stress[5] += 0.25 * vector_input_stress(5);
}



cerr <<  "output_stress[0]   " << output_stress[0] << "\n";

    }

    CHKERR moab.write_file(mesh_out_file);
  }
  CATCH_ERRORS;

  MoFEM::Core::Finalize();

  return 0;
}
