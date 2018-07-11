/** \file fluid_pressure_element.cpp

  \brief Atom test for fluid pressure element

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
using bio::tee_device;
using bio::stream;

static char help[] = "...\n\n";

int main(int argc, char *argv[]) {

  MoFEM::Core::Initialize(&argc,&argv,(char *)0,help);

  try {

    moab::Core mb_instance;
    moab::Interface& moab = mb_instance;

    PetscBool flg = PETSC_TRUE;
    char mesh_file_name[255];
    ierr = PetscOptionsGetString(PETSC_NULL,PETSC_NULL,"-my_file",mesh_file_name,255,&flg); CHKERRG(ierr);
    if(flg != PETSC_TRUE) {
      SETERRQ(PETSC_COMM_SELF,1,"*** ERROR -my_file (MESH FILE NEEDED)");
    }

    const char *option;
    option = "";//"PARALLEL=BCAST;";//;DEBUG_IO";
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

    //Definitions

    //add DISPLACEMENT field, Hilbert space H1, veror field rank 3 (displacemnt
    //has three components ux,uy,uz)
    ierr = m_field.add_field("DISPLACEMENT",H1,AINSWORTH_LEGENDRE_BASE,3); CHKERRG(ierr);


    //add entities on wich DISPLACEMENT field is approximated, you can add
    //entities form several approximation levels at onec. You can as well
    //approximate field only on some mesh subdomain, in that case displacements
    //are approximated on roor moab mesh.
    ierr = m_field.add_ents_to_field_by_type(0,MBTET,"DISPLACEMENT"); CHKERRG(ierr);

    //set app. order for displacement fiedl. it is set uniform approximation
    //order. in genreal evry entity can have arbitraty approximation level,
    //ranging from 1 to 10 and more.
    int order = 1;
    ierr = m_field.set_field_order(0,MBTET,"DISPLACEMENT",order); CHKERRG(ierr);
    ierr = m_field.set_field_order(0,MBTRI,"DISPLACEMENT",order); CHKERRG(ierr);
    ierr = m_field.set_field_order(0,MBEDGE,"DISPLACEMENT",order); CHKERRG(ierr);
    ierr = m_field.set_field_order(0,MBVERTEX,"DISPLACEMENT",1); CHKERRG(ierr);

    //define fluid pressure finite elements
    FluidPressure fluid_pressure_fe(m_field);
    fluid_pressure_fe.addNeumannFluidPressureBCElements("DISPLACEMENT");

    ///add probelem which will be solved, could be more than one problem
    //operating on some subset of defined approximatons spces
    ierr = m_field.add_problem("TEST_PROBLEM"); CHKERRG(ierr);
    //mesh could have several Refinement levels which share some subset of entities between them.
    //below defines on which set of entities (on Refinement level 0) build approximation spaces for TEST_PROBLEM
    ierr = m_field.modify_problem_ref_level_add_bit("TEST_PROBLEM",bit_level0); CHKERRG(ierr);

    //add finite element to test problem
    ierr = m_field.modify_problem_add_finite_element("TEST_PROBLEM","FLUID_PRESSURE_FE"); CHKERRG(ierr);

    //construct data structures for fields and finite elements. at that points
    //entities, finite elements or dofs have unique uid, but are not partitioned
    //or numbered. user can add entities to mesh, add dofs or elements if
    //necessaery. in case of modifications data structures are updated.
    ierr = m_field.build_fields(); CHKERRG(ierr);
    ierr = m_field.build_finite_elements(); CHKERRG(ierr);
    ierr = m_field.build_adjacencies(bit_level0); CHKERRG(ierr);


    ProblemsManager *prb_mng_ptr;
    ierr = m_field.getInterface(prb_mng_ptr); CHKERRG(ierr);
    ierr = prb_mng_ptr->buildProblem("TEST_PROBLEM",true); CHKERRG(ierr);
    //to solve problem it need to be represented in matrix vector form. this
    //demand numeration of dofs and problem  partitioning.
    ierr = prb_mng_ptr->partitionSimpleProblem("TEST_PROBLEM"); CHKERRG(ierr);
    ierr = prb_mng_ptr->partitionFiniteElements("TEST_PROBLEM"); CHKERRG(ierr);
    //what are ghost nodes, see Petsc Manual
    ierr = prb_mng_ptr->partitionGhostDofs("TEST_PROBLEM"); CHKERRG(ierr);

    //create vector for problem
    Vec F;
    ierr = m_field.getInterface<VecManager>()->vecCreateGhost("TEST_PROBLEM",ROW,&F); CHKERRG(ierr);
    ierr = fluid_pressure_fe.setNeumannFluidPressureFiniteElementOperators("DISPLACEMENT",F,false,false); CHKERRG(ierr);

    ierr = VecZeroEntries(F); CHKERRG(ierr);
    ierr = m_field.loop_finite_elements("TEST_PROBLEM","FLUID_PRESSURE_FE",fluid_pressure_fe.getLoopFe()); CHKERRG(ierr);
    ierr = VecAssemblyBegin(F); CHKERRG(ierr);
    ierr = VecAssemblyEnd(F); CHKERRG(ierr);
    ierr = m_field.getInterface<VecManager>()->setGlobalGhostVector("TEST_PROBLEM",ROW,F,INSERT_VALUES,SCATTER_REVERSE); CHKERRG(ierr);

    //ierr = VecView(F,PETSC_VIEWER_STDOUT_WORLD); CHKERRG(ierr);

    // PetscViewer viewer;
    // ierr = PetscViewerASCIIOpen(PETSC_COMM_WORLD,"fluid_pressure_element.txt",&viewer); CHKERRG(ierr);
    // ierr = VecChop(F,1e-4); CHKERRG(ierr);
    // ierr = VecView(F,viewer); CHKERRG(ierr);
    // ierr = PetscViewerDestroy(&viewer); CHKERRG(ierr);

    double sum = 0;
    ierr = VecSum(F,&sum); CHKERRG(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"sum  = %4.3e\n",sum); CHKERRG(ierr);
    if(fabs(sum-1.0)>1e-8) {
      SETERRQ(PETSC_COMM_WORLD,MOFEM_ATOM_TEST_INVALID,"Failed to pass test");
    }
    double fnorm;
    ierr = VecNorm(F,NORM_2,&fnorm); CHKERRG(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"fnorm  = %9.8e\n",fnorm); CHKERRG(ierr);
    if(fabs(fnorm-6.23059402e-01)>1e-6) {
      SETERRQ(PETSC_COMM_WORLD,MOFEM_ATOM_TEST_INVALID,"Failed to pass test");
    }


    // std::map<EntityHandle,VectorDouble > tags_vals;
    // for(_IT_GET_DOFS_FIELD_BY_NAME_FOR_LOOP_(m_field,"DISPLACEMENT",dof)) {
    //   tags_vals[dof->getEnt()].resize(3);
    //   tags_vals[dof->getEnt()][dof->getDofCoeffIdx()] = dof->getFieldData();
    // }
    // std::vector<EntityHandle> ents;
    // ents.resize(tags_vals.size());
    // std::vector<double> vals(3*tags_vals.size());
    // int idx = 0;
    // for(std::map<EntityHandle,VectorDouble >::iterator mit = tags_vals.begin();
    //   mit!=tags_vals.end();mit++,idx++) {
    //   ents[idx] = mit->first;
    //   vals[3*idx + 0] = mit->second[0];
    //   vals[3*idx + 1] = mit->second[1];
    //   vals[3*idx + 2] = mit->second[2];
    // }
    //
    // double def_VAL[3] = {0,0,0};
    // Tag th_vals;
    // rval = moab.tag_get_handle("FLUID_PRESURE_FORCES",3,MB_TYPE_DOUBLE,th_vals,MB_TAG_CREAT|MB_TAG_SPARSE,def_VAL); CHKERRG(rval);
    // rval = moab.tag_set_data(th_vals,&ents[0],ents.size(),&vals[0]); CHKERRG(rval);
    //
    // EntityHandle out_meshset;
    // rval = moab.create_meshset(MESHSET_SET,out_meshset); CHKERRG(rval);
    // ierr = m_field.get_problem_finite_elements_entities("TEST_PROBLEM","FLUID_PRESSURE_FE",out_meshset); CHKERRG(ierr);
    // rval = moab.write_file("out.vtk","VTK","",&out_meshset,1); CHKERRG(rval);
    // rval = moab.delete_entities(&out_meshset,1); CHKERRG(rval);

    //destroy vector
    ierr = VecDestroy(&F); CHKERRG(ierr);

  } 
  CATCH_ERRORS;


  MoFEM::Core::Finalize(); 

  return 0;

}
