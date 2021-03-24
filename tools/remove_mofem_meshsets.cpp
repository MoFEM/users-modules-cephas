/** \file field_to_vertices.cpp
  \brief Field to vertices
  \example field_to_vertices.cpp

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
#include <BasicFiniteElements.hpp>

using namespace MoFEM;

static char help[] = "...\n\n";

int main(int argc, char *argv[]) {
  MoFEM::Core::Initialize(&argc, &argv, (char *)0, help);

  try {

    // global variables
    char mesh_file_name[255];
    PetscBool flg_file = PETSC_FALSE;
    char mesh_out_file[255] = "out.h5m";

    CHKERR PetscOptionsBegin(PETSC_COMM_WORLD, "", "Field to vertices options",
                             "none");
    CHKERR PetscOptionsString("-file_name", "mesh file name", "", "mesh.h5m",
                              mesh_file_name, 255, &flg_file);
    CHKERR PetscOptionsString("-output_file", "output mesh file name", "",
                              "out.h5m", mesh_out_file, 255, PETSC_NULL);

    ierr = PetscOptionsEnd(); CHKERRG(ierr);

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

    // Add logging channel for example
    auto core_log = logging::core::get();
    core_log->add_sink(
        LogManager::createSink(LogManager::getStrmWorld(), "REMOVER"));
    LogManager::setLog("REMOVER");
    MOFEM_LOG_TAG("REMOVER", "remover");

    auto prb_ptr = m_field.get_problems();
    std::vector<std::string> prb_list;
    for(auto &it : *prb_ptr) 
      prb_list.push_back(it.getName());

    for (auto &it : prb_list) {
      MOFEM_LOG("REMOVER", Sev::inform) << "Delete problem " << it;
      CHKERR m_field.delete_problem(it);
    }

    auto fe_ptr = m_field.get_finite_elements();
    std::vector<std::string> fe_list;
    for (auto &it : *fe_ptr) 
      fe_list.push_back(it->getName());

    for (auto &it : fe_list) {
      MOFEM_LOG("REMOVER", Sev::inform)
          << "Delete finite element " << it;
      CHKERR m_field.delete_finite_element(it);
    }

    auto field_ptr = m_field.get_fields();
     std::vector<std::string> field_list;
     for (auto &it : *field_ptr)
       field_list.push_back(it->getName());

     for (auto &it : field_list) {
       MOFEM_LOG("REMOVER", Sev::inform) << "Delete field " << it;
       CHKERR m_field.delete_field(it);
    }

    CHKERR moab.write_file(mesh_out_file);

  }
  CATCH_ERRORS;

  CHKERR MoFEM::Core::Finalize();

  return 0;
}
