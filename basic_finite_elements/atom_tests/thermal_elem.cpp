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

  MoFEM::Core::Initialize(&argc, &argv, (char *)0, help);

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
      SETERRQ(PETSC_COMM_SELF, 1, "*** ERROR -my_file (MESH FILE NEEDED)");
    }

    const char *option;
    option = "";
    CHKERR moab.load_file(mesh_file_name, 0, option);

    // Create MoFEM (Joseph) database
    MoFEM::Core core(moab);
    MoFEM::Interface &m_field = core;

    // set entitities bit level
    BitRefLevel bit_level0;
    bit_level0.set(0);
    EntityHandle meshset_level0;
    CHKERR moab.create_meshset(MESHSET_SET, meshset_level0);
    CHKERR m_field.getInterface<BitRefManager>()->setBitRefLevelByDim(
        0, 3, bit_level0);

    // Fields
    CHKERR m_field.add_field("TEMP", H1, AINSWORTH_LEGENDRE_BASE, 1);

    // Problem
    CHKERR m_field.add_problem("TEST_PROBLEM");

    // set refinement level for problem
    CHKERR m_field.modify_problem_ref_level_add_bit("TEST_PROBLEM", bit_level0);

    // meshset consisting all entities in mesh
    EntityHandle root_set = moab.get_root_set();
    // add entities to field
    CHKERR m_field.add_ents_to_field_by_type(root_set, MBTET, "TEMP");

    // set app. order
    // see Hierarchic Finite Element Bases on Unstructured Tetrahedral Meshes
    // (Mark Ainsworth & Joe Coyle)
    int order = 4;
    CHKERR m_field.set_field_order(root_set, MBTET, "TEMP", order);
    CHKERR m_field.set_field_order(root_set, MBTRI, "TEMP", order);
    CHKERR m_field.set_field_order(root_set, MBEDGE, "TEMP", order);
    CHKERR m_field.set_field_order(root_set, MBVERTEX, "TEMP", 1);

    ThermalElement thermal_elements(m_field);
    CHKERR thermal_elements.addThermalElements("TEMP");
    CHKERR thermal_elements.addThermalFluxElement("TEMP");

    CHKERR m_field.modify_problem_add_finite_element("TEST_PROBLEM",
                                                     "THERMAL_FE");
    CHKERR m_field.modify_problem_add_finite_element("TEST_PROBLEM",
                                                     "THERMAL_FLUX_FE");

    /****/
    // build database
    // build field
    CHKERR m_field.build_fields();
    // build finite elemnts
    CHKERR m_field.build_finite_elements();
    // build adjacencies
    CHKERR m_field.build_adjacencies(bit_level0);

    /****/
    // mesh partitioning
    // build problem
    ProblemsManager *prb_mng_ptr;
    CHKERR m_field.getInterface(prb_mng_ptr);
    CHKERR prb_mng_ptr->buildProblem("TEST_PROBLEM", true);
    // partition
    CHKERR prb_mng_ptr->partitionSimpleProblem("TEST_PROBLEM");
    CHKERR prb_mng_ptr->partitionFiniteElements("TEST_PROBLEM");
    // what are ghost nodes, see Petsc Manual
    CHKERR prb_mng_ptr->partitionGhostDofs("TEST_PROBLEM");

    Vec F;
    CHKERR m_field.getInterface<VecManager>()->vecCreateGhost("TEST_PROBLEM",
                                                              ROW, &F);
    Vec T;
    CHKERR VecDuplicate(F, &T);
    Mat A;
    CHKERR m_field.getInterface<MatrixManager>()
        ->createMPIAIJWithArrays<PetscGlobalIdx_mi_tag>("TEST_PROBLEM", &A);

    DirichletTemperatureBc my_dirichlet_bc(m_field, "TEMP", A, T, F);
    CHKERR thermal_elements.setThermalFiniteElementRhsOperators("TEMP", F);
    CHKERR thermal_elements.setThermalFiniteElementLhsOperators("TEMP", A);
    CHKERR thermal_elements.setThermalFluxFiniteElementRhsOperators("TEMP", F);

    CHKERR VecZeroEntries(T);
    CHKERR VecZeroEntries(F);
    CHKERR MatZeroEntries(A);

    // preproc
    CHKERR m_field.problem_basic_method_preProcess("TEST_PROBLEM",
                                                   my_dirichlet_bc);
    CHKERR m_field.getInterface<VecManager>()->setGlobalGhostVector(
        "TEST_PROBLEM", ROW, T, INSERT_VALUES, SCATTER_REVERSE);

    CHKERR m_field.loop_finite_elements("TEST_PROBLEM", "THERMAL_FE",
                                        thermal_elements.getLoopFeRhs());
    CHKERR m_field.loop_finite_elements("TEST_PROBLEM", "THERMAL_FE",
                                        thermal_elements.getLoopFeLhs());
    CHKERR m_field.loop_finite_elements("TEST_PROBLEM", "THERMAL_FLUX_FE",
                                        thermal_elements.getLoopFeFlux());

    // postproc
    CHKERR m_field.problem_basic_method_postProcess("TEST_PROBLEM",
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

    CHKERR m_field.problem_basic_method_preProcess("TEST_PROBLEM",
                                                   my_dirichlet_bc);

    // Save data on mesh
    CHKERR m_field.getInterface<VecManager>()->setGlobalGhostVector(
        "TEST_PROBLEM", ROW, T, INSERT_VALUES, SCATTER_REVERSE);

    double sum = 0;
    CHKERR VecSum(F, &sum);
    CHKERR PetscPrintf(PETSC_COMM_WORLD, "sum  = %9.8f\n", sum);
    double fnorm;
    CHKERR VecNorm(F, NORM_2, &fnorm);
    CHKERR PetscPrintf(PETSC_COMM_WORLD, "fnorm  = %9.8e\n", fnorm);
    if (fabs(sum + 0.59583333) > 1e-7) {
      SETERRQ(PETSC_COMM_WORLD, MOFEM_ATOM_TEST_INVALID, "Failed to pass test");
    }
    if (fabs(fnorm - 2.32872499e-01) > 1e-6) {
      SETERRQ(PETSC_COMM_WORLD, MOFEM_ATOM_TEST_INVALID, "Failed to pass test");
    }

    CHKERR MatDestroy(&A);
    CHKERR VecDestroy(&F);
    CHKERR VecDestroy(&T);
    CHKERR KSPDestroy(&solver);

  }
  CATCH_ERRORS;

  MoFEM::Core::Finalize();

  return 0;
}
