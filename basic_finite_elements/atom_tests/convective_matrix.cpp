/** \file convective_matrix.cpp

 * \ingroup convective_mass_elem
 * \ingroup nonlinear_elastic_elem
 *
 * Atom test for convective mass element
 *
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

    const char *option;
    option = "";//"PARALLEL=BCAST;";//;DEBUG_IO";
    rval = moab.load_file(mesh_file_name, 0, option); CHKERRG(rval);
    ParallelComm* pcomm = ParallelComm::get_pcomm(&moab,MYPCOMM_INDEX);
    if(pcomm == NULL) pcomm =  new ParallelComm(&moab,PETSC_COMM_WORLD);

    MoFEM::Core core(moab);
    MoFEM::Interface& m_field = core;

    //ref meshset ref level 0
    BitRefLevel bit_level0;
    bit_level0.set(0);
    EntityHandle meshset_level0;
    CHKERR moab.create_meshset(MESHSET_SET,meshset_level0); 
    CHKERR m_field.getInterface<BitRefManager>()->setBitRefLevelByDim(
        0, 3, bit_level0);
    CHKERR m_field.getInterface<BitRefManager>()->getEntitiesByRefLevel(bit_level0,BitRefLevel().set(),meshset_level0); 

    //Fields
    ierr = m_field.add_field("SPATIAL_POSITION",H1,AINSWORTH_LEGENDRE_BASE,3); CHKERRG(ierr);

    //FE
    ierr = m_field.add_finite_element("ELASTIC"); CHKERRG(ierr);

    //Define rows/cols and element data
    ierr = m_field.modify_finite_element_add_field_row("ELASTIC","SPATIAL_POSITION"); CHKERRG(ierr);
    ierr = m_field.modify_finite_element_add_field_col("ELASTIC","SPATIAL_POSITION"); CHKERRG(ierr);
    ierr = m_field.modify_finite_element_add_field_data("ELASTIC","SPATIAL_POSITION"); CHKERRG(ierr);

    //define problems
    ierr = m_field.add_problem("ELASTIC_MECHANICS"); CHKERRG(ierr);

    //set finite elements for problems
    ierr = m_field.modify_problem_add_finite_element("ELASTIC_MECHANICS","ELASTIC"); CHKERRG(ierr);

    //set refinement level for problem
    ierr = m_field.modify_problem_ref_level_add_bit("ELASTIC_MECHANICS",bit_level0); CHKERRG(ierr);

    //add entitities (by tets) to the field
    ierr = m_field.add_ents_to_field_by_type(0,MBTET,"SPATIAL_POSITION"); CHKERRG(ierr);

    //add finite elements entities
    ierr = m_field.add_ents_to_finite_element_by_bit_ref(bit_level0,BitRefLevel().set(),"ELASTIC",MBTET); CHKERRG(ierr);

    //set app. order
    PetscInt order;
    ierr = PetscOptionsGetInt(PETSC_NULL,PETSC_NULL,"-my_order",&order,&flg); CHKERRG(ierr);
    if(flg != PETSC_TRUE) {
      order = 1;
    }
    ierr = m_field.set_field_order(0,MBTET,"SPATIAL_POSITION",order); CHKERRG(ierr);
    ierr = m_field.set_field_order(0,MBTRI,"SPATIAL_POSITION",order); CHKERRG(ierr);
    ierr = m_field.set_field_order(0,MBEDGE,"SPATIAL_POSITION",order); CHKERRG(ierr);
    ierr = m_field.set_field_order(0,MBVERTEX,"SPATIAL_POSITION",1); CHKERRG(ierr);

    ierr = m_field.add_finite_element("NEUAMNN_FE"); CHKERRG(ierr);
    ierr = m_field.modify_finite_element_add_field_row("NEUAMNN_FE","SPATIAL_POSITION"); CHKERRG(ierr);
    ierr = m_field.modify_finite_element_add_field_col("NEUAMNN_FE","SPATIAL_POSITION"); CHKERRG(ierr);
    ierr = m_field.modify_finite_element_add_field_data("NEUAMNN_FE","SPATIAL_POSITION"); CHKERRG(ierr);
    ierr = m_field.modify_problem_add_finite_element("ELASTIC_MECHANICS","NEUAMNN_FE"); CHKERRG(ierr);
    for(_IT_CUBITMESHSETS_BY_BCDATA_TYPE_FOR_LOOP_(m_field,NODESET|FORCESET,it)) {
      Range tris;
      rval = moab.get_entities_by_type(it->meshset,MBTRI,tris,true); CHKERRG(rval);
      ierr = m_field.add_ents_to_finite_element_by_type(tris,MBTRI,"NEUAMNN_FE"); CHKERRG(ierr);
    }
    for(_IT_CUBITMESHSETS_BY_BCDATA_TYPE_FOR_LOOP_(m_field,SIDESET|PRESSURESET,it)) {
      Range tris;
      rval = moab.get_entities_by_type(it->meshset,MBTRI,tris,true); CHKERRG(rval);
      ierr = m_field.add_ents_to_finite_element_by_type(tris,MBTRI,"NEUAMNN_FE"); CHKERRG(ierr);
    }

    //Velocity
    ierr = m_field.add_field("SPATIAL_VELOCITY",H1,AINSWORTH_LEGENDRE_BASE,3); CHKERRG(ierr);
    ierr = m_field.add_ents_to_field_by_type(0,MBTET,"SPATIAL_VELOCITY"); CHKERRG(ierr);
    int order_velocity = 1;
    ierr = m_field.set_field_order(0,MBTET,"SPATIAL_VELOCITY",order_velocity); CHKERRG(ierr);
    ierr = m_field.set_field_order(0,MBTRI,"SPATIAL_VELOCITY",order_velocity); CHKERRG(ierr);
    ierr = m_field.set_field_order(0,MBEDGE,"SPATIAL_VELOCITY",order_velocity); CHKERRG(ierr);
    ierr = m_field.set_field_order(0,MBVERTEX,"SPATIAL_VELOCITY",1); CHKERRG(ierr);

    ierr = m_field.add_field("DOT_SPATIAL_POSITION",H1,AINSWORTH_LEGENDRE_BASE,3); CHKERRG(ierr);
    ierr = m_field.add_ents_to_field_by_type(0,MBTET,"DOT_SPATIAL_POSITION"); CHKERRG(ierr);
    ierr = m_field.set_field_order(0,MBTET,"DOT_SPATIAL_POSITION",order); CHKERRG(ierr);
    ierr = m_field.set_field_order(0,MBTRI,"DOT_SPATIAL_POSITION",order); CHKERRG(ierr);
    ierr = m_field.set_field_order(0,MBEDGE,"DOT_SPATIAL_POSITION",order); CHKERRG(ierr);
    ierr = m_field.set_field_order(0,MBVERTEX,"DOT_SPATIAL_POSITION",1); CHKERRG(ierr);
    ierr = m_field.add_field("DOT_SPATIAL_VELOCITY",H1,AINSWORTH_LEGENDRE_BASE,3); CHKERRG(ierr);
    ierr = m_field.add_ents_to_field_by_type(0,MBTET,"DOT_SPATIAL_VELOCITY"); CHKERRG(ierr);
    ierr = m_field.set_field_order(0,MBTET,"DOT_SPATIAL_VELOCITY",order_velocity); CHKERRG(ierr);
    ierr = m_field.set_field_order(0,MBTRI,"DOT_SPATIAL_VELOCITY",order_velocity); CHKERRG(ierr);
    ierr = m_field.set_field_order(0,MBEDGE,"DOT_SPATIAL_VELOCITY",order_velocity); CHKERRG(ierr);
    ierr = m_field.set_field_order(0,MBVERTEX,"DOT_SPATIAL_VELOCITY",1); CHKERRG(ierr);

    ConvectiveMassElement inertia(m_field,1);
    ierr = inertia.setBlocks(); CHKERRG(ierr);
    ierr = inertia.addConvectiveMassElement("MASS_ELEMENT","SPATIAL_VELOCITY","SPATIAL_POSITION"); CHKERRG(ierr);
    ierr = inertia.addVelocityElement("VELOCITY_ELEMENT","SPATIAL_VELOCITY","SPATIAL_POSITION"); CHKERRG(ierr);
    ierr = m_field.modify_problem_add_finite_element("ELASTIC_MECHANICS","MASS_ELEMENT"); CHKERRG(ierr);
    ierr = m_field.modify_problem_add_finite_element("ELASTIC_MECHANICS","VELOCITY_ELEMENT"); CHKERRG(ierr);

    //build field
    ierr = m_field.build_fields(); CHKERRG(ierr);

    double scale_positions = 2;
    {
      EntityHandle node = 0;
      double coords[3];
      for(_IT_GET_DOFS_FIELD_BY_NAME_FOR_LOOP_(m_field,"SPATIAL_POSITION",dof_ptr)) {
        if(dof_ptr->get()->getEntType()!=MBVERTEX) continue;
        EntityHandle ent = dof_ptr->get()->getEnt();
        int dof_rank = dof_ptr->get()->getDofCoeffIdx();
        double &fval = dof_ptr->get()->getFieldData();
        if(node!=ent) {
          rval = moab.get_coords(&ent,1,coords); CHKERRG(rval);
          node = ent;
        }
        fval = scale_positions*coords[dof_rank];
      }
    }

    double scale_velocities = 4;
    {
      EntityHandle node = 0;
      double coords[3];
      for(_IT_GET_DOFS_FIELD_BY_NAME_FOR_LOOP_(m_field,"DOT_SPATIAL_POSITION",dof_ptr)) {
        if(dof_ptr->get()->getEntType()!=MBVERTEX) continue;
        EntityHandle ent = dof_ptr->get()->getEnt();
        int dof_rank = dof_ptr->get()->getDofCoeffIdx();
        double &fval = dof_ptr->get()->getFieldData();
        if(node!=ent) {
          rval = moab.get_coords(&ent,1,coords); CHKERRG(rval);
          node = ent;
        }
        fval = scale_velocities*coords[dof_rank];
      }
    }

    //build finite elemnts
    ierr = m_field.build_finite_elements(); CHKERRG(ierr);

    //build adjacencies
    ierr = m_field.build_adjacencies(bit_level0); CHKERRG(ierr);

    //build problem

    ProblemsManager *prb_mng_ptr;
    ierr = m_field.getInterface(prb_mng_ptr); CHKERRG(ierr);

    ierr = prb_mng_ptr->buildProblem("ELASTIC_MECHANICS",true); CHKERRG(ierr);

    //partition
    ierr = prb_mng_ptr->partitionProblem("ELASTIC_MECHANICS"); CHKERRG(ierr);
    ierr = prb_mng_ptr->partitionFiniteElements("ELASTIC_MECHANICS"); CHKERRG(ierr);
    ierr = prb_mng_ptr->partitionGhostDofs("ELASTIC_MECHANICS"); CHKERRG(ierr);

    //create matrices
    Vec F;
    ierr = m_field.getInterface<VecManager>()->vecCreateGhost("ELASTIC_MECHANICS",COL,&F); CHKERRG(ierr);
    Vec D;
    ierr = VecDuplicate(F,&D); CHKERRG(ierr);
    Mat Aij;
    ierr = m_field.MatCreateMPIAIJWithArrays("ELASTIC_MECHANICS",&Aij); CHKERRG(ierr);

    ierr = inertia.setConvectiveMassOperators("SPATIAL_VELOCITY","SPATIAL_POSITION"); CHKERRG(ierr);
    ierr = inertia.setVelocityOperators("SPATIAL_VELOCITY","SPATIAL_POSITION"); CHKERRG(ierr);

    inertia.getLoopFeMassRhs().ts_F = F;
    inertia.getLoopFeMassRhs().ts_a = 1;
    inertia.getLoopFeMassLhs().ts_B = Aij;
    inertia.getLoopFeMassLhs().ts_a = 1;

    inertia.getLoopFeVelRhs().ts_F = F;
    inertia.getLoopFeVelRhs().ts_a = 1;
    inertia.getLoopFeVelLhs().ts_B = Aij;
    inertia.getLoopFeVelLhs().ts_a = 1;

    ierr = m_field.loop_finite_elements("ELASTIC_MECHANICS","MASS_ELEMENT",inertia.getLoopFeMassRhs()); CHKERRG(ierr);
    ierr = m_field.loop_finite_elements("ELASTIC_MECHANICS","VELOCITY_ELEMENT",inertia.getLoopFeVelRhs()); CHKERRG(ierr);
    ierr = VecGhostUpdateBegin(F,ADD_VALUES,SCATTER_REVERSE); CHKERRG(ierr);
    ierr = VecGhostUpdateEnd(F,ADD_VALUES,SCATTER_REVERSE); CHKERRG(ierr);
    ierr = VecAssemblyBegin(F); CHKERRG(ierr);
    ierr = VecAssemblyEnd(F); CHKERRG(ierr);

    ierr = m_field.loop_finite_elements("ELASTIC_MECHANICS","MASS_ELEMENT",inertia.getLoopFeMassLhs()); CHKERRG(ierr);
    ierr = m_field.loop_finite_elements("ELASTIC_MECHANICS","VELOCITY_ELEMENT",inertia.getLoopFeVelLhs()); CHKERRG(ierr);
    ierr = MatAssemblyBegin(Aij,MAT_FINAL_ASSEMBLY); CHKERRG(ierr);
    ierr = MatAssemblyEnd(Aij,MAT_FINAL_ASSEMBLY); CHKERRG(ierr);

    // PetscViewer viewer;
    // ierr = PetscViewerASCIIOpen(PETSC_COMM_WORLD,"convective_matrix.txt",&viewer); CHKERRG(ierr);
    //ierr = PetscViewerSetFormat(viewer,PETSC_VIEWER_ASCII_SYMMODU); CHKERRG(ierr);

    //ierr = VecChop(F,1e-4); CHKERRG(ierr);
    // ierr = VecView(F,PETSC_VIEWER_STDOUT_WORLD); CHKERRG(ierr);
    // ierr = VecView(F,viewer); CHKERRG(ierr);

    //MatView(Aij,PETSC_VIEWER_DRAW_WORLD);
    // MatChop(Aij,1e-4);
    // MatView(Aij,PETSC_VIEWER_STDOUT_WORLD);
    // MatView(Aij,viewer);
    //std::string wait;
    //std::cin >> wait;
    //
    //  ierr = PetscViewerDestroy(&viewer); CHKERRG(ierr);

    double sum = 0;
    ierr = VecSum(F,&sum); CHKERRG(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"sum  = %9.8e\n",sum); CHKERRG(ierr);
    double fnorm;
    ierr = VecNorm(F,NORM_2,&fnorm); CHKERRG(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"fnorm  = %9.8e\n",fnorm); CHKERRG(ierr);

    double mnorm;
    ierr = MatNorm(Aij,NORM_1,&mnorm); CHKERRG(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"mnorm  = %9.8e\n",mnorm); CHKERRG(ierr);


    if(fabs(sum-6.27285463e+00)>1e-8) {
      SETERRQ(PETSC_COMM_WORLD,MOFEM_ATOM_TEST_INVALID,"Failed to pass test");
    }
    if(fabs(fnorm-1.28223353e+00)>1e-6) {
      SETERRQ(PETSC_COMM_WORLD,MOFEM_ATOM_TEST_INVALID,"Failed to pass test");
    }
    if(fabs(mnorm-1.31250000e+00)>1e-6) {
      SETERRQ(PETSC_COMM_WORLD,MOFEM_ATOM_TEST_INVALID,"Failed to pass test");
    }




    ierr = VecDestroy(&F); CHKERRG(ierr);
    ierr = VecDestroy(&D); CHKERRG(ierr);
    ierr = MatDestroy(&Aij); CHKERRG(ierr);



  } CATCH_ERRORS;

  MoFEM::Core::Finalize();

  return 0;


}
