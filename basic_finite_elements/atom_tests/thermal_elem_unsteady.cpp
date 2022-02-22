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
    rval = moab.load_file(mesh_file_name, 0, option);

    // Create MoFEM (Joseph) database
    MoFEM::Core core(moab);
    MoFEM::Interface &m_field = core;

    // set entitities bit level
    BitRefLevel bit_level0;
    bit_level0.set(0);
    EntityHandle meshset_level0;
    rval = moab.create_meshset(MESHSET_SET, meshset_level0);
    CHKERR m_field.getInterface<BitRefManager>()->setBitRefLevelByDim(
        0, 3, bit_level0);

    // Fields
    CHKERR m_field.add_field("TEMP", H1, AINSWORTH_LEGENDRE_BASE, 1);
    CHKERR m_field.add_field("TEMP_RATE", H1, AINSWORTH_LEGENDRE_BASE, 1);

    // Problem
    CHKERR m_field.add_problem("TEST_PROBLEM");

    // set refinement level for problem
    CHKERR m_field.modify_problem_ref_level_add_bit("TEST_PROBLEM", bit_level0);

    // meshset consisting all entities in mesh
    EntityHandle root_set = moab.get_root_set();
    // add entities to field
    CHKERR m_field.add_ents_to_field_by_type(root_set, MBTET, "TEMP");
    CHKERR m_field.add_ents_to_field_by_type(root_set, MBTET, "TEMP_RATE");

    // set app. order
    // see Hierarchic Finite Element Bases on Unstructured Tetrahedral Meshes
    // (Mark Ainsworth & Joe Coyle)
    int order = 2;
    CHKERR m_field.set_field_order(root_set, MBTET, "TEMP", order);
    CHKERR m_field.set_field_order(root_set, MBTRI, "TEMP", order);
    CHKERR m_field.set_field_order(root_set, MBEDGE, "TEMP", order);
    CHKERR m_field.set_field_order(root_set, MBVERTEX, "TEMP", 1);

    CHKERR m_field.set_field_order(root_set, MBTET, "TEMP_RATE", order);
    CHKERR m_field.set_field_order(root_set, MBTRI, "TEMP_RATE", order);
    CHKERR m_field.set_field_order(root_set, MBEDGE, "TEMP_RATE", order);
    CHKERR m_field.set_field_order(root_set, MBVERTEX, "TEMP_RATE", 1);

    ThermalElement thermal_elements(m_field);
    CHKERR thermal_elements.addThermalElements("TEMP");
    CHKERR thermal_elements.addThermalFluxElement("TEMP");
    // add rate of temerature to data field of finite element
    CHKERR m_field.modify_finite_element_add_field_data("THERMAL_FE",
                                                        "TEMP_RATE");

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
    // build problem
    ProblemsManager *prb_mng_ptr;
    CHKERR m_field.getInterface(prb_mng_ptr);
    CHKERR prb_mng_ptr->buildProblem("TEST_PROBLEM", true);

    /****/
    // mesh partitioning
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

    // TS
    TsCtx ts_ctx(m_field, "TEST_PROBLEM");
    TS ts;
    CHKERR TSCreate(PETSC_COMM_WORLD, &ts);
    CHKERR TSSetType(ts, TSBEULER);

    DirichletTemperatureBc my_dirichlet_bc(m_field, "TEMP", A, T, F);
    ThermalElement::UpdateAndControl update_velocities(m_field, "TEMP",
                                                       "TEMP_RATE");
    ThermalElement::TimeSeriesMonitor monitor(m_field, "THEMP_SERIES", "TEMP");

    // preprocess
    ts_ctx.getPreProcessIFunction().push_back(&update_velocities);
    ts_ctx.getPreProcessIFunction().push_back(&my_dirichlet_bc);
    ts_ctx.getPreProcessIJacobian().push_back(&my_dirichlet_bc);

    // and temperature element functions
    CHKERR thermal_elements.setTimeSteppingProblem(ts_ctx, "TEMP", "TEMP_RATE");

    // postprocess
    ts_ctx.getPostProcessIFunction().push_back(&my_dirichlet_bc);
    ts_ctx.getPostProcessIJacobian().push_back(&my_dirichlet_bc);
    ts_ctx.getPostProcessMonitor().push_back(&monitor);

    CHKERR TSSetIFunction(ts, F, TsSetIFunction, &ts_ctx);
    CHKERR TSSetIJacobian(ts, A, A, TsSetIJacobian, &ts_ctx);
    CHKERR TSMonitorSet(ts, TsMonitorSet, &ts_ctx, PETSC_NULL);

    double ftime = 1;
    CHKERR TSSetDuration(ts, PETSC_DEFAULT, ftime);
    CHKERR TSSetSolution(ts, T);
    CHKERR TSSetFromOptions(ts);

    SeriesRecorder *recorder_ptr;
    CHKERR m_field.getInterface(recorder_ptr);
    CHKERR recorder_ptr->add_series_recorder("THEMP_SERIES");
    CHKERR recorder_ptr->initialize_series_recorder("THEMP_SERIES");

#if PETSC_VERSION_GE(3, 7, 0)
    CHKERR TSSetExactFinalTime(ts, TS_EXACTFINALTIME_STEPOVER);
#endif
    CHKERR TSSolve(ts, T);
    CHKERR TSGetTime(ts, &ftime);

    CHKERR recorder_ptr->finalize_series_recorder("THEMP_SERIES");

    PetscInt steps, snesfails, rejects, nonlinits, linits;
    CHKERR TSGetTimeStepNumber(ts, &steps);
    CHKERR TSGetSNESFailures(ts, &snesfails);
    CHKERR TSGetStepRejections(ts, &rejects);
    CHKERR TSGetSNESIterations(ts, &nonlinits);
    CHKERR TSGetKSPIterations(ts, &linits);
    PetscPrintf(PETSC_COMM_WORLD,
                "steps %D (%D rejected, %D SNES fails), ftime %g, nonlinits "
                "%D, linits %D\n",
                steps, rejects, snesfails, ftime, nonlinits, linits);

    // PetscViewer viewer;
    // PetscViewerASCIIOpen(PETSC_COMM_WORLD,"thermal_elem_unsteady.txt",&viewer);

    double sum = 0;
    double fnorm = 0;

    for (_IT_SERIES_STEPS_BY_NAME_FOR_LOOP_(recorder_ptr, "THEMP_SERIES",
                                            sit)) {

      CHKERR recorder_ptr->load_series_data("THEMP_SERIES",
                                            sit->get_step_number());
      CHKERR m_field.getInterface<VecManager>()->setLocalGhostVector(
          "TEST_PROBLEM", ROW, T, INSERT_VALUES, SCATTER_FORWARD);

      double sum0;
      CHKERR VecSum(T, &sum0);
      CHKERR PetscPrintf(PETSC_COMM_WORLD, "sum0  = %9.8e\n", sum0);
      double fnorm0;
      CHKERR VecNorm(T, NORM_2, &fnorm0);
      CHKERR PetscPrintf(PETSC_COMM_WORLD, "fnorm0  = %9.8e\n", fnorm0);

      sum += sum0;
      fnorm += fnorm0;

      // CHKERR VecChop(T,1e-4);
      // CHKERR VecView(T,viewer);
    }
    CHKERR PetscPrintf(PETSC_COMM_WORLD, "sum  = %9.8e\n", sum);
    CHKERR PetscPrintf(PETSC_COMM_WORLD, "fnorm  = %9.8e\n", fnorm);
    if (fabs(sum + 1.32314077e+01) > 1e-7) {
      SETERRQ(PETSC_COMM_WORLD, MOFEM_ATOM_TEST_INVALID, "Failed to pass test");
    }
    if (fabs(fnorm - 4.59664623e+00) > 1e-6) {
      SETERRQ(PETSC_COMM_WORLD, MOFEM_ATOM_TEST_INVALID, "Failed to pass test");
    }

    // CHKERR PetscViewerDestroy(&viewer);

    /*PostProcVertexMethod ent_method(moab,"TEMP");
    CHKERR m_field.loop_dofs("TEST_PROBLEM","TEMP",ROW,ent_method);
    if(pcomm->rank()==0) {
    EntityHandle out_meshset;
    rval = moab.create_meshset(MESHSET_SET,out_meshset);
    CHKERR
  m_field.get_problem_finite_elements_entities("TEST_PROBLEM","THERMAL_FE",out_meshset);
    rval = moab.write_file("out.vtk","VTK","",&out_meshset,1);
    rval = moab.delete_entities(&out_meshset,1);
  }*/

    CHKERR TSDestroy(&ts);
    CHKERR MatDestroy(&A);
    CHKERR VecDestroy(&F);
    CHKERR VecDestroy(&T);
  }
  CATCH_ERRORS;

  MoFEM::Core::Finalize();

  return 0;
}
