/** \file thermal_steady.cpp
  \ingroup mofem_thermal_elem
  \brief Example of steady thermal analysis

  TODO:
  \todo Make it work in distributed meshes with multigird solver. At the moment
  it is not working efficient as can.
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

#include <BasicFiniteElements.hpp>
using namespace MoFEM;

namespace bio = boost::iostreams;
using bio::stream;
using bio::tee_device;

static char help[] = "...\n\n";

int main(int argc, char *argv[]) {

  const string default_options = "-ksp_type fgmres \n"
                                 "-pc_type lu \n"
                                 "-pc_factor_mat_solver_package mumps \n"
                                 "-ksp_monitor \n";

  string param_file = "param_file.petsc";
  if (!static_cast<bool>(ifstream(param_file))) {
    std::ofstream file(param_file.c_str(), std::ios::ate);
    if (file.is_open()) {
      file << default_options;
      file.close();
    }
  }

  MoFEM::Core::Initialize(&argc, &argv, param_file.c_str(), help);

  try {

    moab::Core mb_instance;
    moab::Interface &moab = mb_instance;
    int rank;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);

    PetscBool flg = PETSC_TRUE;
    char mesh_file_name[255];
    CHKERR PetscOptionsGetString(PETSC_NULL, PETSC_NULL, "-my_file",
                                 mesh_file_name, 255, &flg);
    if (flg != PETSC_TRUE) {
      SETERRQ(PETSC_COMM_SELF, MOFEM_NOT_FOUND,
              "*** ERROR -my_file (MESH FILE NEEDED)");
    }

    const char *option;
    option = ""; //"PARALLEL=BCAST;";//;DEBUG_IO";
    CHKERR moab.load_file(mesh_file_name, 0, option);

    // Create MoFEM (Joseph) database
    MoFEM::Core core(moab);
    MoFEM::Interface &m_field = core;

    // set entities bit level
    BitRefLevel bit_level0;
    bit_level0.set(0);
    EntityHandle meshset_level0;
    CHKERR moab.create_meshset(MESHSET_SET, meshset_level0);
    CHKERR m_field.getInterface<BitRefManager>()->setBitRefLevelByDim(
        0, 3, bit_level0);

    // Fields
    CHKERR m_field.add_field("TEMP", H1, AINSWORTH_LEGENDRE_BASE, 1);

    // Problem
    CHKERR m_field.add_problem("THERMAL_PROBLEM");

    // set refinement level for problem
    CHKERR m_field.modify_problem_ref_level_add_bit("THERMAL_PROBLEM",
                                                    bit_level0);

    // meshset consisting all entities in mesh
    EntityHandle root_set = moab.get_root_set();
    // add entities to field
    CHKERR m_field.add_ents_to_field_by_type(root_set, MBTET, "TEMP");

    // set app. order
    // see Hierarchic Finite Element Bases on Unstructured Tetrahedral Meshes
    // (Mark Ainsworth & Joe Coyle)
    PetscInt order;
    CHKERR PetscOptionsGetInt(PETSC_NULL, PETSC_NULL, "-my_order", &order,
                              &flg);
    if (flg != PETSC_TRUE) {
      order = 2;
    }

    CHKERR m_field.set_field_order(root_set, MBTET, "TEMP", order);
    CHKERR m_field.set_field_order(root_set, MBTRI, "TEMP", order);
    CHKERR m_field.set_field_order(root_set, MBEDGE, "TEMP", order);
    CHKERR m_field.set_field_order(root_set, MBVERTEX, "TEMP", 1);

    CHKERR m_field.add_field("MESH_NODE_POSITIONS", H1, AINSWORTH_LEGENDRE_BASE,
                             3);
    CHKERR m_field.add_ents_to_field_by_type(root_set, MBTET,
                                             "MESH_NODE_POSITIONS");
    CHKERR m_field.set_field_order(0, MBTET, "MESH_NODE_POSITIONS", 2);
    CHKERR m_field.set_field_order(0, MBTRI, "MESH_NODE_POSITIONS", 2);
    CHKERR m_field.set_field_order(0, MBEDGE, "MESH_NODE_POSITIONS", 2);
    CHKERR m_field.set_field_order(0, MBVERTEX, "MESH_NODE_POSITIONS", 1);

    ThermalElement thermal_elements(m_field);
    CHKERR thermal_elements.addThermalElements("TEMP");
    CHKERR thermal_elements.addThermalFluxElement("TEMP");
    CHKERR thermal_elements.addThermalConvectionElement("TEMP");

    CHKERR m_field.modify_problem_add_finite_element("THERMAL_PROBLEM",
                                                     "THERMAL_FE");
    CHKERR m_field.modify_problem_add_finite_element("THERMAL_PROBLEM",
                                                     "THERMAL_FLUX_FE");
    CHKERR m_field.modify_problem_add_finite_element("THERMAL_PROBLEM",
                                                     "THERMAL_CONVECTION_FE");

    /****/
    // build database
    // build field
    CHKERR m_field.build_fields();
    // build finite elemnts
    CHKERR m_field.build_finite_elements();
    // build adjacencies
    CHKERR m_field.build_adjacencies(bit_level0);

    ProblemsManager *prb_mng_ptr;
    CHKERR m_field.getInterface(prb_mng_ptr);
    // build problem
    CHKERR prb_mng_ptr->buildProblem("THERMAL_PROBLEM", true);

    Projection10NodeCoordsOnField ent_method_material(m_field,
                                                      "MESH_NODE_POSITIONS");
    CHKERR m_field.loop_dofs("MESH_NODE_POSITIONS", ent_method_material);

    /****/
    // mesh partitioning
    // partition
    CHKERR prb_mng_ptr->partitionProblem("THERMAL_PROBLEM");
    CHKERR prb_mng_ptr->partitionFiniteElements("THERMAL_PROBLEM");
    // what are ghost nodes, see Petsc Manual
    CHKERR prb_mng_ptr->partitionGhostDofs("THERMAL_PROBLEM");

    Vec F;
    CHKERR m_field.getInterface<VecManager>()->vecCreateGhost("THERMAL_PROBLEM",
                                                              ROW, &F);
    Vec T;
    CHKERR VecDuplicate(F, &T);
    Mat A;
    CHKERR m_field.getInterface<MatrixManager>()
        ->createMPIAIJWithArrays<PetscGlobalIdx_mi_tag>("THERMAL_PROBLEM", &A);

    DirichletTemperatureBc my_dirichlet_bc(m_field, "TEMP", A, T, F);
    CHKERR thermal_elements.setThermalFiniteElementRhsOperators("TEMP", F);
    CHKERR thermal_elements.setThermalFiniteElementLhsOperators("TEMP", A);
    CHKERR thermal_elements.setThermalFluxFiniteElementRhsOperators("TEMP", F);
    CHKERR thermal_elements.setThermalConvectionFiniteElementLhsOperators(
        "TEMP", A);
    CHKERR thermal_elements.setThermalConvectionFiniteElementRhsOperators(
        "TEMP", F);

    CHKERR VecZeroEntries(T);
    CHKERR VecGhostUpdateBegin(T, INSERT_VALUES, SCATTER_FORWARD);
    CHKERR VecGhostUpdateEnd(T, INSERT_VALUES, SCATTER_FORWARD);
    CHKERR VecZeroEntries(F);
    CHKERR VecGhostUpdateBegin(F, INSERT_VALUES, SCATTER_FORWARD);
    CHKERR VecGhostUpdateEnd(F, INSERT_VALUES, SCATTER_FORWARD);
    CHKERR MatZeroEntries(A);

    // preproc
    CHKERR m_field.problem_basic_method_preProcess("THERMAL_PROBLEM",
                                                   my_dirichlet_bc);
    CHKERR m_field.getInterface<VecManager>()->setGlobalGhostVector(
        "THERMAL_PROBLEM", ROW, T, INSERT_VALUES, SCATTER_REVERSE);

    CHKERR m_field.loop_finite_elements("THERMAL_PROBLEM", "THERMAL_FE",
                                        thermal_elements.getLoopFeRhs());
    CHKERR m_field.loop_finite_elements("THERMAL_PROBLEM", "THERMAL_FE",
                                        thermal_elements.getLoopFeLhs());
    CHKERR m_field.loop_finite_elements("THERMAL_PROBLEM", "THERMAL_FLUX_FE",
                                        thermal_elements.getLoopFeFlux());
    CHKERR m_field.loop_finite_elements(
        "THERMAL_PROBLEM", "THERMAL_CONVECTION_FE",
        thermal_elements.getLoopFeConvectionRhs());
    CHKERR m_field.loop_finite_elements(
        "THERMAL_PROBLEM", "THERMAL_CONVECTION_FE",
        thermal_elements.getLoopFeConvectionLhs());

    // postproc
    CHKERR m_field.problem_basic_method_postProcess("THERMAL_PROBLEM",
                                                    my_dirichlet_bc);

    CHKERR VecGhostUpdateBegin(F, ADD_VALUES, SCATTER_REVERSE);
    CHKERR VecGhostUpdateEnd(F, ADD_VALUES, SCATTER_REVERSE);
    CHKERR VecAssemblyBegin(F);
    CHKERR VecAssemblyEnd(F);
    CHKERR MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);
    CHKERR MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);

    CHKERR VecScale(F, -1);

    // Solver
    KSP solver;
    CHKERR KSPCreate(PETSC_COMM_WORLD, &solver);
    CHKERR KSPSetOperators(solver, A, A);
    CHKERR KSPSetFromOptions(solver);
    CHKERR KSPSetUp(solver);

    CHKERR KSPSolve(solver, F, T);
    CHKERR VecGhostUpdateBegin(T, INSERT_VALUES, SCATTER_FORWARD);
    CHKERR VecGhostUpdateEnd(T, INSERT_VALUES, SCATTER_FORWARD);

    CHKERR m_field.problem_basic_method_preProcess("THERMAL_PROBLEM",
                                                   my_dirichlet_bc);

    // Save data on mesh
    CHKERR m_field.getInterface<VecManager>()->setGlobalGhostVector(
        "THERMAL_PROBLEM", ROW, T, INSERT_VALUES, SCATTER_REVERSE);

    if (m_field.get_comm_rank() == 0) {
      CHKERR moab.write_file("solution.h5m");
    }

    ProjectionFieldOn10NodeTet ent_method_on_10nodeTet(m_field, "TEMP", true,
                                                       false, "TEMP");
    CHKERR m_field.loop_dofs("TEMP", ent_method_on_10nodeTet);
    ent_method_on_10nodeTet.setNodes = false;
    CHKERR m_field.loop_dofs("TEMP", ent_method_on_10nodeTet);

    if (m_field.get_comm_rank() == 0) {
      EntityHandle out_meshset;
      CHKERR moab.create_meshset(MESHSET_SET, out_meshset);
      CHKERR m_field.get_problem_finite_elements_entities(
          "THERMAL_PROBLEM", "THERMAL_FE", out_meshset);
      CHKERR moab.write_file("out.vtk", "VTK", "", &out_meshset, 1);
      CHKERR moab.delete_entities(&out_meshset, 1);
    }

    CHKERR MatDestroy(&A);
    CHKERR VecDestroy(&F);
    CHKERR VecDestroy(&T);
    CHKERR KSPDestroy(&solver);
  }
  CATCH_ERRORS;

  return MoFEM::Core::Finalize();
}
