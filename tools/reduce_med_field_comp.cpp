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
    char copy_field_name[255];
    PetscInt numb_comp = 9;

    int time_step = 0;
    CHKERR PetscOptionsBegin(m_field.get_comm(), "", "Read MED tool", "none");
    CHKERR PetscOptionsInt("-med_time_step", "time step", "", time_step,
                           &time_step, PETSC_NULL);
    CHKERR PetscOptionsString("-output_file", "output mesh file name", "",
                              "out.h5m", mesh_out_file, 255, PETSC_NULL);

    CHKERR PetscOptionsString("-my_file", "mesh file name", "", "mesh.h5m",
                              mesh_file_name, 255, &flg_file);

    CHKERR PetscOptionsString("-my_field_name", "mesh file name", "", "",
                              copy_field_name, 255, &flg_field);

    CHKERR PetscOptionsInt("-my_number_of_components", "target number of components", "", numb_comp,
                           &numb_comp, &flg_field);

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

    Tag th_field;
    CHKERR m_field.get_moab().tag_get_handle(copy_field_name, th_field);

    Range tets;
    CHKERR m_field.get_moab().get_entities_by_type(0, MBTET, tets);
    for (Range::iterator eit = tets.begin(); eit != tets.end(); ++eit) {
      double *vector_values;
      CHKERR m_field.get_moab().tag_get_by_ptr(th_field, &*eit, 1,
                                               (const void **)&vector_values);
    }

if(numb_comp == 9){
    map<EntityHandle, std::array<double, 9>> tet_map;
    for (Range::iterator pit = tets.begin(); pit != tets.end(); ++pit) {

      double *vector_values;
      CHKERR m_field.get_moab().tag_get_by_ptr(th_field, &*pit, 1,
                                               (const void **)&vector_values);

      std::array<double, 9> values;
      for (int ii = 0; ii != 9; ++ii) {
        values[ii] = vector_values[ii];
      }
      tet_map.insert(
          std::pair<EntityHandle, std::array<double, 9>>(*pit, values));
    }
    CHKERR m_field.get_moab().tag_delete(th_field);

    auto get_tag = [&](const std::string name) {
      std::array<double, 9> def;
      std::fill(def.begin(), def.end(), 0);
      Tag th;
      CHKERR m_field.get_moab().tag_get_handle(name.c_str(), 9, MB_TYPE_DOUBLE,
                                               th, MB_TAG_CREAT | MB_TAG_SPARSE,
                                               def.data());
      return th;
    };

    auto th_new_data = get_tag(copy_field_name);

    for (Range::iterator pit = tets.begin(); pit != tets.end(); ++pit) {
      std::array<double, 9> def;
      double *vector_values;
      CHKERR m_field.get_moab().tag_get_by_ptr(th_new_data, &*pit, 1,
                                               (const void **)&vector_values);

      for (int ii = 0; ii != 9; ++ii) {
        if (ii < 3)
        vector_values[ii] = tet_map.at(*pit)[ii];
        else
        vector_values[ii] = 0.5 * tet_map.at(*pit)[ii];
      }
    }
} else if(numb_comp == 1){


    map<EntityHandle, std::array<double, 1>> tet_map;
    for (Range::iterator pit = tets.begin(); pit != tets.end(); ++pit) {

      double *vector_values;
      CHKERR m_field.get_moab().tag_get_by_ptr(th_field, &*pit, 1,
                                               (const void **)&vector_values);

      std::array<double, 1> values;
      for (int ii = 0; ii != 1; ++ii) {
        values[ii] = vector_values[ii];
      }
      tet_map.insert(
          std::pair<EntityHandle, std::array<double, 1>>(*pit, values));
    }
    CHKERR m_field.get_moab().tag_delete(th_field);

    auto get_tag = [&](const std::string name) {
      std::array<double, 1> def;
      std::fill(def.begin(), def.end(), 0);
      Tag th;
      CHKERR m_field.get_moab().tag_get_handle(name.c_str(), 1, MB_TYPE_DOUBLE,
                                               th, MB_TAG_CREAT | MB_TAG_SPARSE,
                                               def.data());
      return th;
    };

    auto th_new_data = get_tag(copy_field_name);

    for (Range::iterator pit = tets.begin(); pit != tets.end(); ++pit) {
      std::array<double, 1> def;
      double *vector_values;
      CHKERR m_field.get_moab().tag_get_by_ptr(th_new_data, &*pit, 1,
                                               (const void **)&vector_values);

      for (int ii = 0; ii != 1; ++ii) {
        vector_values[ii] = tet_map.at(*pit)[ii];
      }
    }

}

    CHKERR moab.write_file(mesh_out_file);
  }
  CATCH_ERRORS;

  MoFEM::Core::Finalize();

  return 0;
}
