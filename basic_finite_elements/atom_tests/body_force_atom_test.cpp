/** \file body_force_atom_test.cpp

  \brief Atom test for (linear) body forces implementation

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

    ParallelComm *pcomm = ParallelComm::get_pcomm(&moab, MYPCOMM_INDEX);
    if (pcomm == NULL)
      pcomm = new ParallelComm(&moab, PETSC_COMM_WORLD);

    const char *option;
    option = ""; //"PARALLEL=BCAST;";//;DEBUG_IO";
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
    CHKERR m_field.add_field("DISPLACEMENT", H1, AINSWORTH_LEGENDRE_BASE, 3);
    CHKERR m_field.add_field("MESH_NODE_POSITIONS", H1, AINSWORTH_LEGENDRE_BASE,
                             3);

    // FE
    CHKERR m_field.add_finite_element("TEST_FE");

    // Define rows/cols and element data
    CHKERR m_field.modify_finite_element_add_field_row("TEST_FE",
                                                       "DISPLACEMENT");
    CHKERR m_field.modify_finite_element_add_field_col("TEST_FE",
                                                       "DISPLACEMENT");
    CHKERR m_field.modify_finite_element_add_field_data("TEST_FE",
                                                        "DISPLACEMENT");
    CHKERR m_field.modify_finite_element_add_field_data("TEST_FE",
                                                        "MESH_NODE_POSITIONS");

    // Problem
    CHKERR m_field.add_problem("TEST_PROBLEM");

    // set finite elements for problem
    CHKERR m_field.modify_problem_add_finite_element("TEST_PROBLEM", "TEST_FE");
    // set refinement level for problem
    CHKERR m_field.modify_problem_ref_level_add_bit("TEST_PROBLEM", bit_level0);

    // meshset consisting all entities in mesh
    EntityHandle root_set = moab.get_root_set();
    // add entities to field
    CHKERR m_field.add_ents_to_field_by_type(root_set, MBTET, "DISPLACEMENT");
    CHKERR m_field.add_ents_to_field_by_type(root_set, MBTET,
                                             "MESH_NODE_POSITIONS");
    // add entities to finite element
    CHKERR m_field.add_ents_to_finite_element_by_type(root_set, MBTET,
                                                      "TEST_FE");

    // set app. order
    // see Hierarchic Finite Element Bases on Unstructured Tetrahedral Meshes
    // (Mark Ainsworth & Joe Coyle)
    int order = 2;
    CHKERR m_field.set_field_order(root_set, MBTET, "DISPLACEMENT", order);
    CHKERR m_field.set_field_order(root_set, MBTRI, "DISPLACEMENT", order);
    CHKERR m_field.set_field_order(root_set, MBEDGE, "DISPLACEMENT", order);
    CHKERR m_field.set_field_order(root_set, MBVERTEX, "DISPLACEMENT", 1);

    CHKERR m_field.set_field_order(root_set, MBTET, "MESH_NODE_POSITIONS", 2);
    CHKERR m_field.set_field_order(root_set, MBTRI, "MESH_NODE_POSITIONS", 2);
    CHKERR m_field.set_field_order(root_set, MBEDGE, "MESH_NODE_POSITIONS", 2);
    CHKERR m_field.set_field_order(root_set, MBVERTEX, "MESH_NODE_POSITIONS",
                                   1);

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

    // set from positions of 10 node tets
    Projection10NodeCoordsOnField ent_method(m_field, "MESH_NODE_POSITIONS");
    CHKERR m_field.loop_dofs("MESH_NODE_POSITIONS", ent_method);

    Vec F;
    CHKERR m_field.getInterface<VecManager>()->vecCreateGhost("TEST_PROBLEM",
                                                              ROW, &F);

    typedef tee_device<std::ostream, std::ofstream> TeeDevice;
    typedef stream<TeeDevice> TeeStream;
    std::ofstream ofs("body_force_atom_test.txt");
    TeeDevice my_tee(std::cout, ofs);
    TeeStream my_split(my_tee);

    CHKERR VecZeroEntries(F);
    BodyForceConstantField body_forces_methods(m_field);

    for (_IT_CUBITMESHSETS_BY_BCDATA_TYPE_FOR_LOOP_(
             m_field, BLOCKSET | BODYFORCESSET, it)) {
      Block_BodyForces mydata;
      CHKERR it->getAttributeDataStructure(mydata);
      my_split << mydata << std::endl;
      CHKERR body_forces_methods.addBlock("DISPLACEMENT", F,
                                          it->getMeshsetId());
    }
    CHKERR m_field.loop_finite_elements("TEST_PROBLEM", "TEST_FE",
                                        body_forces_methods.getLoopFe());
    CHKERR VecAssemblyBegin(F);
    CHKERR VecAssemblyEnd(F);

    CHKERR m_field.getInterface<VecManager>()->setGlobalGhostVector(
        "TEST_PROBLEM", COL, F, INSERT_VALUES, SCATTER_REVERSE);

    const Problem *problemPtr;
    CHKERR m_field.get_problem("TEST_PROBLEM", &problemPtr);
    std::map<EntityHandle, double> m0, m1, m2;
    for (_IT_NUMEREDDOF_ROW_FOR_LOOP_(problemPtr, dit)) {

      if (dit->get()->getDofCoeffIdx() != 1)
        continue;

      my_split.precision(3);
      my_split.setf(std::ios::fixed);
      my_split << dit->get()->getPetscGlobalDofIdx() << " "
               << dit->get()->getFieldData() << std::endl;
    }

    double sum = 0;
    CHKERR VecSum(F, &sum);
    my_split << std::endl
             << "Sum : " << std::setprecision(3) << sum << std::endl;

    CHKERR VecDestroy(&F);
  }
  CATCH_ERRORS;

  MoFEM::Core::Finalize();

  return 0;
}
