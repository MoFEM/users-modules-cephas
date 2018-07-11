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
    ierr = m_field.add_field("DISP",H1,AINSWORTH_LEGENDRE_BASE,3); CHKERRG(ierr);
    ierr = m_field.add_field("TEMP",H1,AINSWORTH_LEGENDRE_BASE,1); CHKERRG(ierr);

    //Problem
    ierr = m_field.add_problem("PROB"); CHKERRG(ierr);

    //set refinement level for problem
    ierr = m_field.modify_problem_ref_level_add_bit("PROB",bit_level0); CHKERRG(ierr);

    //meshset consisting all entities in mesh
    EntityHandle root_set = moab.get_root_set();
    //add entities to field
    ierr = m_field.add_ents_to_field_by_type(root_set,MBTET,"TEMP"); CHKERRG(ierr);
    ierr = m_field.add_ents_to_field_by_type(root_set,MBTET,"DISP"); CHKERRG(ierr);

    //set app. order
    //see Hierarchic Finite Element Bases on Unstructured Tetrahedral Meshes (Mark Ainsworth & Joe Coyle)
    int order_temp = 2;
    ierr = m_field.set_field_order(root_set,MBTET,"TEMP",order_temp); CHKERRG(ierr);
    ierr = m_field.set_field_order(root_set,MBTRI,"TEMP",order_temp); CHKERRG(ierr);
    ierr = m_field.set_field_order(root_set,MBEDGE,"TEMP",order_temp); CHKERRG(ierr);
    ierr = m_field.set_field_order(root_set,MBVERTEX,"TEMP",1); CHKERRG(ierr);

    int order_disp = 3;
    ierr = m_field.set_field_order(root_set,MBTET,"DISP",order_disp); CHKERRG(ierr);
    ierr = m_field.set_field_order(root_set,MBTRI,"DISP",order_disp); CHKERRG(ierr);
    ierr = m_field.set_field_order(root_set,MBEDGE,"DISP",order_disp); CHKERRG(ierr);
    ierr = m_field.set_field_order(root_set,MBVERTEX,"DISP",1); CHKERRG(ierr);

    ThermalStressElement thermal_stress_elem(m_field);
    ierr = thermal_stress_elem.addThermalStressElement("ELAS","DISP","TEMP"); CHKERRG(ierr);
    ierr = m_field.modify_problem_add_finite_element("PROB","ELAS"); CHKERRG(ierr);

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
    ierr = prb_mng_ptr->buildProblem("PROB",true); CHKERRG(ierr);
    //mesh partitioning
    //partition
    ierr = prb_mng_ptr->partitionSimpleProblem("PROB"); CHKERRG(ierr);
    ierr = prb_mng_ptr->partitionFiniteElements("PROB"); CHKERRG(ierr);
    //what are ghost nodes, see Petsc Manual
    ierr = prb_mng_ptr->partitionGhostDofs("PROB"); CHKERRG(ierr);

    //set temerature at nodes
    for(_IT_GET_DOFS_FIELD_BY_NAME_AND_TYPE_FOR_LOOP_(m_field,"TEMP",MBVERTEX,dof)) {
      EntityHandle ent = dof->get()->getEnt();
      VectorDouble coords(3);
      rval = moab.get_coords(&ent,1,&coords[0]); CHKERRG(rval);
      dof->get()->getFieldData() = 1;
    }

    Vec F;
    ierr = m_field.getInterface<VecManager>()->vecCreateGhost("PROB",ROW,&F); CHKERRG(ierr);
    ierr = thermal_stress_elem.setThermalStressRhsOperators("DISP","TEMP",F,1); CHKERRG(ierr);

    ierr = m_field.loop_finite_elements("PROB","ELAS",thermal_stress_elem.getLoopThermalStressRhs()); CHKERRG(ierr);
    ierr = VecAssemblyBegin(F); CHKERRG(ierr);
    ierr = VecAssemblyEnd(F); CHKERRG(ierr);

    // PetscViewer viewer;
    // ierr = PetscViewerASCIIOpen(PETSC_COMM_WORLD,"forces_and_sources_thermal_stress_elem.txt",&viewer); CHKERRG(ierr);
    // ierr = VecChop(F,1e-4); CHKERRG(ierr);
    // ierr = VecView(F,viewer); CHKERRG(ierr);
    // ierr = PetscViewerDestroy(&viewer); CHKERRG(ierr);

    double sum = 0;
    ierr = VecSum(F,&sum); CHKERRG(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"sum  = %9.8f\n",sum); CHKERRG(ierr);
    double fnorm;
    ierr = VecNorm(F,NORM_2,&fnorm); CHKERRG(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"fnorm  = %9.8e\n",fnorm); CHKERRG(ierr);
    if(fabs(sum)>1e-7) {
      SETERRQ(PETSC_COMM_WORLD,MOFEM_ATOM_TEST_INVALID,"Failed to pass test");
    }
    if(fabs(fnorm-2.64638118e+00)>1e-7) {
      SETERRQ(PETSC_COMM_WORLD,MOFEM_ATOM_TEST_INVALID,"Failed to pass test");
    }


    ierr = VecZeroEntries(F); CHKERRG(ierr);

    ierr = VecDestroy(&F); CHKERRG(ierr);


  } 
  CATCH_ERRORS;

  MoFEM::Core::Finalize();

  return 0;

}
