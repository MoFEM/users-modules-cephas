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

    const char *option;
    option = "";
    rval = moab.load_file(mesh_file_name, 0, option); CHKERRG(rval);

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
    ierr = m_field.add_field("DISPLACEMENT",H1,AINSWORTH_LEGENDRE_BASE,3); CHKERRG(ierr);
    ierr = m_field.add_field("MESH_NODE_POSITIONS",H1,AINSWORTH_LEGENDRE_BASE,3); CHKERRG(ierr);

    //Problem
    ierr = m_field.add_problem("TEST_PROBLEM"); CHKERRG(ierr);
    //set refinement level for problem
    ierr = m_field.modify_problem_ref_level_add_bit("TEST_PROBLEM",bit_level0); CHKERRG(ierr);

    //meshset consisting all entities in mesh
    EntityHandle root_set = moab.get_root_set();
    //add entities to field
    ierr = m_field.add_ents_to_field_by_type(root_set,MBTET,"DISPLACEMENT"); CHKERRG(ierr);
    ierr = m_field.add_ents_to_field_by_type(root_set,MBTET,"MESH_NODE_POSITIONS"); CHKERRG(ierr);

    for(_IT_CUBITMESHSETS_BY_BCDATA_TYPE_FOR_LOOP_(m_field,NODESET|FORCESET,it)) {

      std::ostringstream fe_name;
      fe_name << "FORCE_FE_" << it->getMeshsetId();
      ierr = m_field.add_finite_element(fe_name.str()); CHKERRG(ierr);
      ierr = m_field.modify_finite_element_add_field_row(fe_name.str(),"DISPLACEMENT"); CHKERRG(ierr);
      ierr = m_field.modify_finite_element_add_field_col(fe_name.str(),"DISPLACEMENT"); CHKERRG(ierr);
      ierr = m_field.modify_finite_element_add_field_data(fe_name.str(),"DISPLACEMENT"); CHKERRG(ierr);
      ierr = m_field.modify_problem_add_finite_element("TEST_PROBLEM",fe_name.str()); CHKERRG(ierr);

      Range tris;
      rval = moab.get_entities_by_type(it->meshset,MBTRI,tris,true); CHKERRG(rval);
      ierr = m_field.add_ents_to_finite_element_by_type(tris,MBTRI,fe_name.str()); CHKERRG(ierr);

    }

    for(_IT_CUBITMESHSETS_BY_BCDATA_TYPE_FOR_LOOP_(m_field,SIDESET|PRESSURESET,it)) {

      std::ostringstream fe_name;
      fe_name << "PRESSURE_FE_" << it->getMeshsetId();
      ierr = m_field.add_finite_element(fe_name.str()); CHKERRG(ierr);
      ierr = m_field.modify_finite_element_add_field_row(fe_name.str(),"DISPLACEMENT"); CHKERRG(ierr);
      ierr = m_field.modify_finite_element_add_field_col(fe_name.str(),"DISPLACEMENT"); CHKERRG(ierr);
      ierr = m_field.modify_finite_element_add_field_data(fe_name.str(),"DISPLACEMENT"); CHKERRG(ierr);
      ierr = m_field.modify_finite_element_add_field_data(fe_name.str(),"MESH_NODE_POSITIONS"); CHKERRG(ierr);
      ierr = m_field.modify_problem_add_finite_element("TEST_PROBLEM",fe_name.str()); CHKERRG(ierr);

      Range tris;
      rval = moab.get_entities_by_type(it->meshset,MBTRI,tris,true); CHKERRG(rval);
      ierr = m_field.add_ents_to_finite_element_by_type(tris,MBTRI,fe_name.str()); CHKERRG(ierr);

    }

    //set app. order
    //see Hierarchic Finite Element Bases on Unstructured Tetrahedral Meshes (Mark Ainsworth & Joe Coyle)
    int order = 2;
    ierr = m_field.set_field_order(root_set,MBTET,"DISPLACEMENT",order); CHKERRG(ierr);
    ierr = m_field.set_field_order(root_set,MBTRI,"DISPLACEMENT",order); CHKERRG(ierr);
    ierr = m_field.set_field_order(root_set,MBEDGE,"DISPLACEMENT",order); CHKERRG(ierr);
    ierr = m_field.set_field_order(root_set,MBVERTEX,"DISPLACEMENT",1); CHKERRG(ierr);
    ierr = m_field.set_field_order(root_set,MBTET,"MESH_NODE_POSITIONS",2); CHKERRG(ierr);
    ierr = m_field.set_field_order(root_set,MBTRI,"MESH_NODE_POSITIONS",2); CHKERRG(ierr);
    ierr = m_field.set_field_order(root_set,MBEDGE,"MESH_NODE_POSITIONS",2); CHKERRG(ierr);
    ierr = m_field.set_field_order(root_set,MBVERTEX,"MESH_NODE_POSITIONS",1); CHKERRG(ierr);

    /****/
    //build database
    //build field
    ierr = m_field.build_fields(); CHKERRG(ierr);
    //set FIELD1 from positions of 10 node tets
    Projection10NodeCoordsOnField ent_method(m_field,"MESH_NODE_POSITIONS");
    ierr = m_field.loop_dofs("MESH_NODE_POSITIONS",ent_method); CHKERRG(ierr);

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

    typedef tee_device<std::ostream, std::ofstream> TeeDevice;
    typedef stream<TeeDevice> TeeStream;
    std::ostringstream txt_name;
    txt_name << "forces_and_sources_" << mesh_file_name << ".txt";
    std::ofstream ofs(txt_name.str().c_str());
    TeeDevice my_tee(std::cout, ofs);
    TeeStream my_split(my_tee);

    ierr = VecZeroEntries(F); CHKERRG(ierr);
    boost::ptr_map<std::string,NeumannForcesSurface> neumann_forces;
    for(_IT_CUBITMESHSETS_BY_BCDATA_TYPE_FOR_LOOP_(m_field,NODESET|FORCESET,it)) {
      std::ostringstream fe_name;
      fe_name << "FORCE_FE_" << it->getMeshsetId();
      string fe_name_str = fe_name.str();
      neumann_forces.insert(fe_name_str,new NeumannForcesSurface(m_field));
      neumann_forces.at(fe_name_str).addForce("DISPLACEMENT",F,it->getMeshsetId()); CHKERRG(ierr);
      ForceCubitBcData data;
      ierr = it->getBcDataStructure(data); CHKERRG(ierr);
      my_split << *it << std::endl;
      my_split << data << std::endl;
    }
    for(_IT_CUBITMESHSETS_BY_BCDATA_TYPE_FOR_LOOP_(m_field,SIDESET|PRESSURESET,it)) {
      std::ostringstream fe_name;
      fe_name << "PRESSURE_FE_" << it->getMeshsetId();
      string fe_name_str = fe_name.str();
      neumann_forces.insert(fe_name_str,new NeumannForcesSurface(m_field));
      neumann_forces.at(fe_name_str).addPressure("DISPLACEMENT",F,it->getMeshsetId()); CHKERRG(ierr);
      PressureCubitBcData data;
      ierr = it->getBcDataStructure(data); CHKERRG(ierr);
      my_split << *it << std::endl;
      my_split << data << std::endl;
    }
    boost::ptr_map<std::string,NeumannForcesSurface>::iterator mit = neumann_forces.begin();
    for(;mit!=neumann_forces.end();mit++) {
      ierr = m_field.loop_finite_elements("TEST_PROBLEM",mit->first,mit->second->getLoopFe()); CHKERRG(ierr);
    }
    ierr = VecAssemblyBegin(F); CHKERRG(ierr);
    ierr = VecAssemblyEnd(F); CHKERRG(ierr);

    ierr = m_field.getInterface<VecManager>()->setGlobalGhostVector("TEST_PROBLEM",ROW,F,INSERT_VALUES,SCATTER_REVERSE); CHKERRG(ierr);

    const double eps = 1e-4;

    const Problem *problemPtr;
    ierr = m_field.get_problem("TEST_PROBLEM",&problemPtr); CHKERRG(ierr);
    for(_IT_NUMEREDDOF_ROW_FOR_LOOP_(problemPtr,dit)) {

      my_split.precision(3);
      my_split.setf(std::ios::fixed);
      double val = fabs(dit->get()->getFieldData())<eps ? 0.0 : dit->get()->getFieldData();
      my_split << dit->get()->getPetscGlobalDofIdx() << " " << val << std::endl;

    }

    double sum = 0;
    ierr = VecSum(F,&sum); CHKERRG(ierr);
    sum = fabs(sum)<eps ? 0.0 : sum;
    my_split << std::endl << "Sum : " << std::setprecision(3) << sum << std::endl;

    ierr = VecDestroy(&F); CHKERRG(ierr);

  } 
  CATCH_ERRORS;


  MoFEM::Core::Finalize();

  return 0;

}
