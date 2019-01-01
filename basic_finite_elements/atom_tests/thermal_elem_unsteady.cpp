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
using bio::tee_device;
using bio::stream;

static char help[] = "...\n\n";

int main(int argc, char *argv[]) {

  MoFEM::Core::Initialize(&argc,&argv,(char *)0,help);

  try {

    moab::Core mb_instance;
    moab::Interface& moab = mb_instance;
    int rank;
    MPI_Comm_rank(PETSC_COMM_WORLD,&rank);

    PetscBool flg = PETSC_TRUE;
    char mesh_file_name[255];
    ierr = PetscOptionsGetString(PETSC_NULL,PETSC_NULL,"-my_file",mesh_file_name,255,&flg); CHKERRG(ierr);
    if(flg != PETSC_TRUE) {
      SETERRQ(PETSC_COMM_SELF,1,"*** ERROR -my_file (MESH FILE NEEDED)");
    }

    ParallelComm* pcomm = ParallelComm::get_pcomm(&moab,MYPCOMM_INDEX);
    if(pcomm == NULL) pcomm =  new ParallelComm(&moab,PETSC_COMM_WORLD);

    const char *option;
    option = "";//"PARALLEL=BCAST;";//;DEBUG_IO";
    BARRIER_PCOMM_RANK_START(pcomm)
    rval = moab.load_file(mesh_file_name, 0, option); CHKERRG(rval);
    BARRIER_PCOMM_RANK_END(pcomm)

    //Create MoFEM (Joseph) database
    MoFEM::Core core(moab);
    MoFEM::Interface& m_field = core;

    //set entitities bit level
    BitRefLevel bit_level0;
    bit_level0.set(0);
    EntityHandle meshset_level0;
    rval = moab.create_meshset(MESHSET_SET,meshset_level0); CHKERRG(rval);
    ierr = m_field.getInterface<BitRefManager>()->setBitRefLevelByDim(0,3,bit_level0); CHKERRG(ierr);

    //Fields
    ierr = m_field.add_field("TEMP",H1,AINSWORTH_LEGENDRE_BASE,1); CHKERRG(ierr);
    ierr = m_field.add_field("TEMP_RATE",H1,AINSWORTH_LEGENDRE_BASE,1); CHKERRG(ierr);

    //Problem
    ierr = m_field.add_problem("TEST_PROBLEM"); CHKERRG(ierr);

    //set refinement level for problem
    ierr = m_field.modify_problem_ref_level_add_bit("TEST_PROBLEM",bit_level0); CHKERRG(ierr);

    //meshset consisting all entities in mesh
    EntityHandle root_set = moab.get_root_set();
    //add entities to field
    ierr = m_field.add_ents_to_field_by_type(root_set,MBTET,"TEMP"); CHKERRG(ierr);
    ierr = m_field.add_ents_to_field_by_type(root_set,MBTET,"TEMP_RATE"); CHKERRG(ierr);

    //set app. order
    //see Hierarchic Finite Element Bases on Unstructured Tetrahedral Meshes (Mark Ainsworth & Joe Coyle)
    int order = 2;
    ierr = m_field.set_field_order(root_set,MBTET,"TEMP",order); CHKERRG(ierr);
    ierr = m_field.set_field_order(root_set,MBTRI,"TEMP",order); CHKERRG(ierr);
    ierr = m_field.set_field_order(root_set,MBEDGE,"TEMP",order); CHKERRG(ierr);
    ierr = m_field.set_field_order(root_set,MBVERTEX,"TEMP",1); CHKERRG(ierr);

    ierr = m_field.set_field_order(root_set,MBTET,"TEMP_RATE",order); CHKERRG(ierr);
    ierr = m_field.set_field_order(root_set,MBTRI,"TEMP_RATE",order); CHKERRG(ierr);
    ierr = m_field.set_field_order(root_set,MBEDGE,"TEMP_RATE",order); CHKERRG(ierr);
    ierr = m_field.set_field_order(root_set,MBVERTEX,"TEMP_RATE",1); CHKERRG(ierr);

    ThermalElement thermal_elements(m_field);
    ierr = thermal_elements.addThermalElements("TEMP"); CHKERRG(ierr);
    ierr = thermal_elements.addThermalFluxElement("TEMP"); CHKERRG(ierr);
    //add rate of temerature to data field of finite element
    ierr = m_field.modify_finite_element_add_field_data("THERMAL_FE","TEMP_RATE"); CHKERRG(ierr);

    ierr = m_field.modify_problem_add_finite_element("TEST_PROBLEM","THERMAL_FE"); CHKERRG(ierr);
    ierr = m_field.modify_problem_add_finite_element("TEST_PROBLEM","THERMAL_FLUX_FE"); CHKERRG(ierr);

    /****/
    //build database
    //build field
    ierr = m_field.build_fields(); CHKERRG(ierr);
    //build finite elemnts
    ierr = m_field.build_finite_elements(); CHKERRG(ierr);
    //build adjacencies
    ierr = m_field.build_adjacencies(bit_level0); CHKERRG(ierr);
    //build problem
    ProblemsManager *prb_mng_ptr;
    ierr = m_field.getInterface(prb_mng_ptr); CHKERRG(ierr);
    ierr = prb_mng_ptr->buildProblem("TEST_PROBLEM",true); CHKERRG(ierr);

    /****/
    //mesh partitioning
    //partition
    ierr = prb_mng_ptr->partitionSimpleProblem("TEST_PROBLEM"); CHKERRG(ierr);
    ierr = prb_mng_ptr->partitionFiniteElements("TEST_PROBLEM"); CHKERRG(ierr);
    //what are ghost nodes, see Petsc Manual
    ierr = prb_mng_ptr->partitionGhostDofs("TEST_PROBLEM"); CHKERRG(ierr);

    Vec F;
    ierr = m_field.getInterface<VecManager>()->vecCreateGhost("TEST_PROBLEM",ROW,&F); CHKERRG(ierr);
    Vec T;
    ierr = VecDuplicate(F,&T); CHKERRG(ierr);
    Mat A;
    ierr = m_field.MatCreateMPIAIJWithArrays("TEST_PROBLEM",&A); CHKERRG(ierr);

    //TS
    TsCtx ts_ctx(m_field,"TEST_PROBLEM");
    TS ts;
    ierr = TSCreate(PETSC_COMM_WORLD,&ts); CHKERRG(ierr);
    ierr = TSSetType(ts,TSBEULER); CHKERRG(ierr);

    DirichletTemperatureBc my_dirichlet_bc(m_field,"TEMP",A,T,F);
    ThermalElement::UpdateAndControl update_velocities(m_field,"TEMP","TEMP_RATE");
    ThermalElement::TimeSeriesMonitor monitor(m_field,"THEMP_SERIES","TEMP");

    //preprocess
    ts_ctx.get_preProcess_to_do_IFunction().push_back(&update_velocities);
    ts_ctx.get_preProcess_to_do_IFunction().push_back(&my_dirichlet_bc);
    ts_ctx.get_preProcess_to_do_IJacobian().push_back(&my_dirichlet_bc);

    //and temperature element functions
    ierr = thermal_elements.setTimeSteppingProblem(ts_ctx,"TEMP","TEMP_RATE"); CHKERRG(ierr);

    //postprocess
    ts_ctx.get_postProcess_to_do_IFunction().push_back(&my_dirichlet_bc);
    ts_ctx.get_postProcess_to_do_IJacobian().push_back(&my_dirichlet_bc);
    ts_ctx.get_postProcess_to_do_Monitor().push_back(&monitor);

    ierr = TSSetIFunction(ts,F,TsSetIFunction,&ts_ctx); CHKERRG(ierr);
    ierr = TSSetIJacobian(ts,A,A,TsSetIJacobian,&ts_ctx); CHKERRG(ierr);
    ierr = TSMonitorSet(ts,TsMonitorSet,&ts_ctx,PETSC_NULL); CHKERRG(ierr);

    double ftime = 1;
    ierr = TSSetDuration(ts,PETSC_DEFAULT,ftime); CHKERRG(ierr);
    ierr = TSSetSolution(ts,T); CHKERRG(ierr);
    ierr = TSSetFromOptions(ts); CHKERRG(ierr);

    SeriesRecorder *recorder_ptr;
    ierr = m_field.getInterface(recorder_ptr); CHKERRG(ierr);
    ierr = recorder_ptr->add_series_recorder("THEMP_SERIES"); CHKERRG(ierr);
    ierr = recorder_ptr->initialize_series_recorder("THEMP_SERIES"); CHKERRG(ierr);

    #if PETSC_VERSION_GE(3,7,0)
    ierr = TSSetExactFinalTime(ts,TS_EXACTFINALTIME_STEPOVER); CHKERRG(ierr);
    #endif
    ierr = TSSolve(ts,T); CHKERRG(ierr);
    ierr = TSGetTime(ts,&ftime); CHKERRG(ierr);

    ierr = recorder_ptr->finalize_series_recorder("THEMP_SERIES"); CHKERRG(ierr);

    PetscInt steps,snesfails,rejects,nonlinits,linits;
    ierr = TSGetTimeStepNumber(ts,&steps); CHKERRG(ierr);
    ierr = TSGetSNESFailures(ts,&snesfails); CHKERRG(ierr);
    ierr = TSGetStepRejections(ts,&rejects); CHKERRG(ierr);
    ierr = TSGetSNESIterations(ts,&nonlinits); CHKERRG(ierr);
    ierr = TSGetKSPIterations(ts,&linits); CHKERRG(ierr);
    PetscPrintf(PETSC_COMM_WORLD,
      "steps %D (%D rejected, %D SNES fails), ftime %g, nonlinits %D, linits %D\n",
      steps,rejects,snesfails,ftime,nonlinits,linits);

      // PetscViewer viewer;
      // PetscViewerASCIIOpen(PETSC_COMM_WORLD,"thermal_elem_unsteady.txt",&viewer);

      double sum = 0;
      double fnorm = 0;

      for(_IT_SERIES_STEPS_BY_NAME_FOR_LOOP_(recorder_ptr,"THEMP_SERIES",sit)) {

        ierr = recorder_ptr->load_series_data("THEMP_SERIES",sit->get_step_number()); CHKERRG(ierr);
        ierr = m_field.getInterface<VecManager>()->setLocalGhostVector("TEST_PROBLEM",ROW,T,INSERT_VALUES,SCATTER_FORWARD); CHKERRG(ierr);

        double sum0;
        ierr = VecSum(T,&sum0); CHKERRG(ierr);
        ierr = PetscPrintf(PETSC_COMM_WORLD,"sum0  = %9.8e\n",sum0); CHKERRG(ierr);
        double fnorm0;
        ierr = VecNorm(T,NORM_2,&fnorm0); CHKERRG(ierr);
        ierr = PetscPrintf(PETSC_COMM_WORLD,"fnorm0  = %9.8e\n",fnorm0); CHKERRG(ierr);

        sum += sum0;
        fnorm += fnorm0;

        // ierr = VecChop(T,1e-4); CHKERRG(ierr);
        // ierr = VecView(T,viewer); CHKERRG(ierr);

      }
      ierr = PetscPrintf(PETSC_COMM_WORLD,"sum  = %9.8e\n",sum); CHKERRG(ierr);
      ierr = PetscPrintf(PETSC_COMM_WORLD,"fnorm  = %9.8e\n",fnorm); CHKERRG(ierr);
      if(fabs(sum+1.32314077e+01)>1e-7) {
        SETERRQ(PETSC_COMM_WORLD,MOFEM_ATOM_TEST_INVALID,"Failed to pass test");
      }
      if(fabs(fnorm-4.59664623e+00)>1e-6) {
        SETERRQ(PETSC_COMM_WORLD,MOFEM_ATOM_TEST_INVALID,"Failed to pass test");
      }


      // ierr = PetscViewerDestroy(&viewer); CHKERRG(ierr);


      /*PostProcVertexMethod ent_method(moab,"TEMP");
      ierr = m_field.loop_dofs("TEST_PROBLEM","TEMP",ROW,ent_method); CHKERRG(ierr);
      if(pcomm->rank()==0) {
      EntityHandle out_meshset;
      rval = moab.create_meshset(MESHSET_SET,out_meshset); CHKERRG(rval);
      ierr = m_field.get_problem_finite_elements_entities("TEST_PROBLEM","THERMAL_FE",out_meshset); CHKERRG(ierr);
      rval = moab.write_file("out.vtk","VTK","",&out_meshset,1); CHKERRG(rval);
      rval = moab.delete_entities(&out_meshset,1); CHKERRG(rval);
    }*/

    ierr = TSDestroy(&ts);CHKERRG(ierr);
    ierr = MatDestroy(&A); CHKERRG(ierr);
    ierr = VecDestroy(&F); CHKERRG(ierr);
    ierr = VecDestroy(&T); CHKERRG(ierr);


  } 
  CATCH_ERRORS;

  MoFEM::Core::Finalize(); 

  return 0;

}
