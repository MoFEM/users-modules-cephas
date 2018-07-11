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

namespace bio = boost::iostreams;
using bio::tee_device;
using bio::stream;

#define HOON

static char help[] = "...\n\n";

/// Example approx. function
struct MyFunApprox {

  std::vector<VectorDouble > result;

  std::vector<VectorDouble >& operator()(double x, double y, double z) {
    result.resize(1);
    result[0].resize(3);
    (result[0])[0] = x;
    (result[0])[1] = y;
    (result[0])[2] = z*z;
    return result;
  }

};

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
    option = "";
    rval = moab.load_file(mesh_file_name, 0, option); CHKERRG(rval);
    ParallelComm* pcomm = ParallelComm::get_pcomm(&moab,MYPCOMM_INDEX);
    if(pcomm == NULL) pcomm =  new ParallelComm(&moab,PETSC_COMM_WORLD);



    //Create MoFEM (Joseph) database
    MoFEM::Core core(moab);
    MoFEM::Interface& m_field = core;


    //set entities bit level
    BitRefLevel bit_level0;
    bit_level0.set(0);
    EntityHandle meshset_level0;
    rval = moab.create_meshset(MESHSET_SET,meshset_level0); CHKERRG(rval);
    ierr = m_field.getInterface<BitRefManager>()->setBitRefLevelByDim(0,3,bit_level0); CHKERRG(ierr);

    //Fields
    ierr = m_field.add_field("FIELD1",H1,AINSWORTH_LEGENDRE_BASE,3); CHKERRG(ierr);
    #ifdef HOON
    ierr = m_field.add_field("MESH_NODE_POSITIONS",H1,AINSWORTH_LEGENDRE_BASE,3,MB_TAG_SPARSE,MF_ZERO); CHKERRG(ierr);
    #endif

    //FE
    ierr = m_field.add_finite_element("TEST_FE"); CHKERRG(ierr);

    //Define rows/cols and element data
    ierr = m_field.modify_finite_element_add_field_row("TEST_FE","FIELD1"); CHKERRG(ierr);
    ierr = m_field.modify_finite_element_add_field_col("TEST_FE","FIELD1"); CHKERRG(ierr);
    ierr = m_field.modify_finite_element_add_field_data("TEST_FE","FIELD1"); CHKERRG(ierr);
    #ifdef HOON
    ierr = m_field.modify_finite_element_add_field_data("TEST_FE","MESH_NODE_POSITIONS"); CHKERRG(ierr);
    #endif

    //Problem
    ierr = m_field.add_problem("TEST_PROBLEM"); CHKERRG(ierr);

    //set finite elements for problem
    ierr = m_field.modify_problem_add_finite_element("TEST_PROBLEM","TEST_FE"); CHKERRG(ierr);
    //set refinement level for problem
    ierr = m_field.modify_problem_ref_level_add_bit("TEST_PROBLEM",bit_level0); CHKERRG(ierr);

    //meshset consisting all entities in mesh
    EntityHandle root_set = moab.get_root_set();
    //add entities to field
    ierr = m_field.add_ents_to_field_by_type(root_set,MBTET,"FIELD1"); CHKERRG(ierr);
    #ifdef HOON
    ierr = m_field.add_ents_to_field_by_type(root_set,MBTET,"MESH_NODE_POSITIONS"); CHKERRG(ierr);
    #endif
    //add entities to finite element
    ierr = m_field.add_ents_to_finite_element_by_type(root_set,MBTET,"TEST_FE"); CHKERRG(ierr);


    //set app. order
    //see Hierarchic Finite Element Bases on Unstructured Tetrahedral Meshes (Mark Ainsworth & Joe Coyle)
    int order = 3;
    ierr = PetscOptionsGetInt(PETSC_NULL,PETSC_NULL,"-my_order",&order,&flg); CHKERRG(ierr);
    if(flg != PETSC_TRUE) {
      order = 3;
    }
    ierr = m_field.set_field_order(root_set,MBTET,"FIELD1",order); CHKERRG(ierr);
    ierr = m_field.set_field_order(root_set,MBTRI,"FIELD1",order); CHKERRG(ierr);
    ierr = m_field.set_field_order(root_set,MBEDGE,"FIELD1",order); CHKERRG(ierr);
    ierr = m_field.set_field_order(root_set,MBVERTEX,"FIELD1",1); CHKERRG(ierr);
    #ifdef HOON
    ierr = m_field.set_field_order(root_set,MBTET,"MESH_NODE_POSITIONS",2); CHKERRG(ierr);
    ierr = m_field.set_field_order(root_set,MBTRI,"MESH_NODE_POSITIONS",2); CHKERRG(ierr);
    ierr = m_field.set_field_order(root_set,MBEDGE,"MESH_NODE_POSITIONS",2); CHKERRG(ierr);
    ierr = m_field.set_field_order(root_set,MBVERTEX,"MESH_NODE_POSITIONS",1); CHKERRG(ierr);
    #endif

    /****/
    //build database
    //build field
    ierr = m_field.build_fields(); CHKERRG(ierr);
    #ifdef HOON
    Projection10NodeCoordsOnField ent_method_material(m_field,"MESH_NODE_POSITIONS");
    ierr = m_field.loop_dofs("MESH_NODE_POSITIONS",ent_method_material); CHKERRG(ierr);
    #endif
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

    Mat A;
    ierr = m_field.MatCreateMPIAIJWithArrays("TEST_PROBLEM",&A); CHKERRG(ierr);
    Vec D,F;
    ierr = m_field.getInterface<VecManager>()->vecCreateGhost("TEST_PROBLEM",ROW,&F); CHKERRG(ierr);
    ierr = m_field.getInterface<VecManager>()->vecCreateGhost("TEST_PROBLEM",COL,&D); CHKERRG(ierr);

    std::vector<Vec> vec_F;
    vec_F.push_back(F);

    {
      MyFunApprox function_evaluator;
      FieldApproximationH1 field_approximation(m_field);
      field_approximation.loopMatrixAndVectorVolume(
        "TEST_PROBLEM","TEST_FE","FIELD1",A,vec_F,function_evaluator);
      }

      ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY); CHKERRG(ierr);
      ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY); CHKERRG(ierr);
      ierr = VecGhostUpdateBegin(F,ADD_VALUES,SCATTER_REVERSE); CHKERRG(ierr);
      ierr = VecGhostUpdateEnd(F,ADD_VALUES,SCATTER_REVERSE); CHKERRG(ierr);
      ierr = VecGhostUpdateBegin(F,INSERT_VALUES,SCATTER_FORWARD); CHKERRG(ierr);
      ierr = VecGhostUpdateEnd(F,INSERT_VALUES,SCATTER_FORWARD); CHKERRG(ierr);

      KSP solver;
      ierr = KSPCreate(PETSC_COMM_WORLD,&solver); CHKERRG(ierr);
      ierr = KSPSetOperators(solver,A,A); CHKERRG(ierr);
      ierr = KSPSetFromOptions(solver); CHKERRG(ierr);
      ierr = KSPSetUp(solver); CHKERRG(ierr);

      ierr = KSPSolve(solver,F,D); CHKERRG(ierr);
      ierr = VecGhostUpdateBegin(D,INSERT_VALUES,SCATTER_FORWARD); CHKERRG(ierr);
      ierr = VecGhostUpdateEnd(D,INSERT_VALUES,SCATTER_FORWARD); CHKERRG(ierr);

      ierr = m_field.getInterface<VecManager>()->setGlobalGhostVector("TEST_PROBLEM",COL,D,INSERT_VALUES,SCATTER_REVERSE); CHKERRG(ierr);

      ierr = KSPDestroy(&solver); CHKERRG(ierr);
      ierr = VecDestroy(&D); CHKERRG(ierr);
      ierr = VecDestroy(&F); CHKERRG(ierr);
      ierr = MatDestroy(&A); CHKERRG(ierr);

      EntityHandle fe_meshset = m_field.get_finite_element_meshset("TEST_FE");
      Range tets;
      rval = moab.get_entities_by_type(fe_meshset,MBTET,tets,true); CHKERRG(rval);
      Range tets_edges;
      rval = moab.get_adjacencies(tets,1,false,tets_edges,moab::Interface::UNION); CHKERRG(rval);
      EntityHandle edges_meshset;
      rval = moab.create_meshset(MESHSET_SET,edges_meshset); CHKERRG(rval);
      rval = moab.add_entities(edges_meshset,tets); CHKERRG(rval);
      rval = moab.add_entities(edges_meshset,tets_edges); CHKERRG(rval);
      rval = moab.convert_entities(edges_meshset,true,false,false); CHKERRG(rval);

      ProjectionFieldOn10NodeTet ent_method_field1_on_10nodeTet(m_field,"FIELD1",true,false,"FIELD1");
      ierr = m_field.loop_dofs("FIELD1",ent_method_field1_on_10nodeTet); CHKERRG(ierr);
      ent_method_field1_on_10nodeTet.setNodes = false;
      ierr = m_field.loop_dofs("FIELD1",ent_method_field1_on_10nodeTet); CHKERRG(ierr);

      if(pcomm->rank()==0) {
        EntityHandle out_meshset;
        rval = moab.create_meshset(MESHSET_SET,out_meshset); CHKERRG(rval);
        ierr = m_field.get_problem_finite_elements_entities("TEST_PROBLEM","TEST_FE",out_meshset); CHKERRG(ierr);
        rval = moab.write_file("out.vtk","VTK","",&out_meshset,1); CHKERRG(rval);
        rval = moab.delete_entities(&out_meshset,1); CHKERRG(rval);
      }

      typedef tee_device<std::ostream, std::ofstream> TeeDevice;
      typedef stream<TeeDevice> TeeStream;

      std::ofstream ofs("field_approximation.txt");
      TeeDevice tee(cout, ofs);
      TeeStream my_split(tee);

      Range nodes;
      rval = moab.get_entities_by_type(0,MBVERTEX,nodes,true); CHKERRG(rval);
      MatrixDouble nodes_vals;
      nodes_vals.resize(nodes.size(),3);
      rval = moab.tag_get_data(
        ent_method_field1_on_10nodeTet.th,nodes,&*nodes_vals.data().begin()); CHKERRG(rval);

        const double eps = 1e-4;

        my_split.precision(3);
        my_split.setf(std::ios::fixed);
        for(
          DoubleAllocator::iterator it = nodes_vals.data().begin();
          it!=nodes_vals.data().end();it++) {
            *it = fabs(*it)<eps ? 0.0 : *it;
          }
          my_split << nodes_vals << std::endl;

          const Problem *problemPtr;
          ierr = m_field.get_problem("TEST_PROBLEM",&problemPtr); CHKERRG(ierr);
          std::map<EntityHandle,double> m0,m1,m2;
          for(_IT_NUMEREDDOF_ROW_FOR_LOOP_(problemPtr,dit)) {

            my_split.precision(3);
            my_split.setf(std::ios::fixed);
            double val = fabs(dit->get()->getFieldData())<eps ? 0.0 : dit->get()->getFieldData();
            my_split << dit->get()->getPetscGlobalDofIdx() << " " << val << std::endl;

          }


        } CATCH_ERRORS;

  MoFEM::Core::Finalize(); CHKERRG(ierr);

  return 0;

}
