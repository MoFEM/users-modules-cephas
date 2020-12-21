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
using bio::stream;
using bio::tee_device;

#define HOON

static char help[] = "...\n\n";

/// Example approx. function
struct MyFunApprox {

  std::vector<VectorDouble> result;

  std::vector<VectorDouble> &operator()(double x, double y, double z) {
    result.resize(1);
    result[0].resize(3);
    (result[0])[0] = x;
    (result[0])[1] = y;
    (result[0])[2] = z * z;
    return result;
  }
};

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
      SETERRQ(PETSC_COMM_SELF, MOFEM_INVALID_DATA,
              "*** ERROR -my_file (MESH FILE NEEDED)");
    }

    char petsc_options[] = {"-ksp_monitor"
                            "-ksp_type fgmres"
                            "-pc_type bjacobi"
                            "-ksp_atol 0"
                            "-ksp_rtol 1e-12"};
    CHKERR PetscOptionsInsertString(NULL, petsc_options);

    const char *option;
    option = "";
    CHKERR moab.load_file(mesh_file_name, 0, option);
    ParallelComm *pcomm = ParallelComm::get_pcomm(&moab, MYPCOMM_INDEX);
    if (pcomm == NULL)
      pcomm = new ParallelComm(&moab, PETSC_COMM_WORLD);

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
    CHKERR m_field.add_field("FIELD1", H1, AINSWORTH_LEGENDRE_BASE, 3);
#ifdef HOON
    CHKERR m_field.add_field("MESH_NODE_POSITIONS", H1, AINSWORTH_LEGENDRE_BASE,
                             3, MB_TAG_SPARSE, MF_ZERO);
#endif

    // FE
    CHKERR m_field.add_finite_element("TEST_FE");

    // Define rows/cols and element data
    CHKERR m_field.modify_finite_element_add_field_row("TEST_FE", "FIELD1");
    CHKERR m_field.modify_finite_element_add_field_col("TEST_FE", "FIELD1");
    CHKERR m_field.modify_finite_element_add_field_data("TEST_FE", "FIELD1");
#ifdef HOON
    CHKERR m_field.modify_finite_element_add_field_data("TEST_FE",
                                                        "MESH_NODE_POSITIONS");
#endif

    // Problem
    CHKERR m_field.add_problem("TEST_PROBLEM");

    // set finite elements for problem
    CHKERR m_field.modify_problem_add_finite_element("TEST_PROBLEM", "TEST_FE");
    // set refinement level for problem
    CHKERR m_field.modify_problem_ref_level_add_bit("TEST_PROBLEM", bit_level0);

    // meshset consisting all entities in mesh
    EntityHandle root_set = moab.get_root_set();
    // add entities to field
    CHKERR m_field.add_ents_to_field_by_type(root_set, MBTET, "FIELD1");
#ifdef HOON
    CHKERR m_field.add_ents_to_field_by_type(root_set, MBTET,
                                             "MESH_NODE_POSITIONS");
#endif
    // add entities to finite element
    CHKERR
        m_field.add_ents_to_finite_element_by_type(root_set, MBTET, "TEST_FE");

    // set app. order
    // see Hierarchic Finite Element Bases on Unstructured Tetrahedral Meshes
    // (Mark Ainsworth & Joe Coyle)
    int order = 3;
    CHKERR
        PetscOptionsGetInt(PETSC_NULL, PETSC_NULL, "-my_order", &order, &flg);
    if (flg != PETSC_TRUE) {
      order = 3;
    }
    CHKERR m_field.set_field_order(root_set, MBTET, "FIELD1", order);
    CHKERR m_field.set_field_order(root_set, MBTRI, "FIELD1", order);
    CHKERR m_field.set_field_order(root_set, MBEDGE, "FIELD1", order);
    CHKERR m_field.set_field_order(root_set, MBVERTEX, "FIELD1", 1);
#ifdef HOON
    CHKERR m_field.set_field_order(root_set, MBTET, "MESH_NODE_POSITIONS", 2);
    CHKERR m_field.set_field_order(root_set, MBTRI, "MESH_NODE_POSITIONS", 2);
    CHKERR m_field.set_field_order(root_set, MBEDGE, "MESH_NODE_POSITIONS", 2);
    CHKERR
        m_field.set_field_order(root_set, MBVERTEX, "MESH_NODE_POSITIONS", 1);
#endif

    /****/
    // build database
    // build field
    CHKERR m_field.build_fields();
#ifdef HOON
    Projection10NodeCoordsOnField ent_method_material(m_field,
                                                      "MESH_NODE_POSITIONS");
    CHKERR m_field.loop_dofs("MESH_NODE_POSITIONS", ent_method_material);
#endif
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

    Mat A;
    CHKERR
        m_field.getInterface<MatrixManager>()
            ->createMPIAIJWithArrays<PetscGlobalIdx_mi_tag>("TEST_PROBLEM", &A);
    Vec D, F;
    CHKERR m_field.getInterface<VecManager>()->vecCreateGhost("TEST_PROBLEM",
                                                              ROW, &F);
    CHKERR m_field.getInterface<VecManager>()->vecCreateGhost("TEST_PROBLEM",
                                                              COL, &D);

    std::vector<Vec> vec_F;
    vec_F.push_back(F);

    {
      MyFunApprox function_evaluator;
      FieldApproximationH1 field_approximation(m_field);
      field_approximation.loopMatrixAndVectorVolume(
          "TEST_PROBLEM", "TEST_FE", "FIELD1", A, vec_F, function_evaluator);
    }

    CHKERR MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);
    CHKERR MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);
    CHKERR VecGhostUpdateBegin(F, ADD_VALUES, SCATTER_REVERSE);
    CHKERR VecGhostUpdateEnd(F, ADD_VALUES, SCATTER_REVERSE);
    CHKERR VecGhostUpdateBegin(F, INSERT_VALUES, SCATTER_FORWARD);
    CHKERR VecGhostUpdateEnd(F, INSERT_VALUES, SCATTER_FORWARD);

    KSP solver;
    CHKERR KSPCreate(PETSC_COMM_WORLD, &solver);
    CHKERR KSPSetOperators(solver, A, A);
    CHKERR KSPSetFromOptions(solver);
    CHKERR KSPSetUp(solver);

    CHKERR KSPSolve(solver, F, D);
    CHKERR VecGhostUpdateBegin(D, INSERT_VALUES, SCATTER_FORWARD);
    CHKERR VecGhostUpdateEnd(D, INSERT_VALUES, SCATTER_FORWARD);

    CHKERR m_field.getInterface<VecManager>()->setGlobalGhostVector(
        "TEST_PROBLEM", COL, D, INSERT_VALUES, SCATTER_REVERSE);

    CHKERR KSPDestroy(&solver);
    CHKERR VecDestroy(&D);
    CHKERR VecDestroy(&F);
    CHKERR MatDestroy(&A);

    EntityHandle fe_meshset = m_field.get_finite_element_meshset("TEST_FE");
    Range tets;
    CHKERR moab.get_entities_by_type(fe_meshset, MBTET, tets, true);
    Range tets_edges;
    CHKERR moab.get_adjacencies(tets, 1, false, tets_edges,
                                moab::Interface::UNION);
    EntityHandle edges_meshset;
    CHKERR moab.create_meshset(MESHSET_SET, edges_meshset);
    CHKERR moab.add_entities(edges_meshset, tets);
    CHKERR moab.add_entities(edges_meshset, tets_edges);
    CHKERR moab.convert_entities(edges_meshset, true, false, false);

    ProjectionFieldOn10NodeTet ent_method_field1_on_10nodeTet(
        m_field, "FIELD1", true, false, "FIELD1");
    CHKERR m_field.loop_dofs("FIELD1", ent_method_field1_on_10nodeTet);
    ent_method_field1_on_10nodeTet.setNodes = false;
    CHKERR m_field.loop_dofs("FIELD1", ent_method_field1_on_10nodeTet);

    if (pcomm->rank() == 0) {
      EntityHandle out_meshset;
      CHKERR moab.create_meshset(MESHSET_SET, out_meshset);
      CHKERR m_field.get_problem_finite_elements_entities(
          "TEST_PROBLEM", "TEST_FE", out_meshset);
      CHKERR moab.write_file("out.vtk", "VTK", "", &out_meshset, 1);
      CHKERR moab.delete_entities(&out_meshset, 1);
    }

    typedef tee_device<std::ostream, std::ofstream> TeeDevice;
    typedef stream<TeeDevice> TeeStream;

    std::ofstream ofs("field_approximation.txt");
    TeeDevice tee(cout, ofs);
    TeeStream my_split(tee);

    Range nodes;
    CHKERR moab.get_entities_by_type(0, MBVERTEX, nodes, true);
    MatrixDouble nodes_vals;
    nodes_vals.resize(nodes.size(), 3);
    CHKERR moab.tag_get_data(ent_method_field1_on_10nodeTet.th, nodes,
                             &*nodes_vals.data().begin());

    const double eps = 1e-4;

    my_split.precision(3);
    my_split.setf(std::ios::fixed);
    for (DoubleAllocator::iterator it = nodes_vals.data().begin();
         it != nodes_vals.data().end(); it++) {
      *it = fabs(*it) < eps ? 0.0 : *it;
    }
    my_split << nodes_vals << std::endl;

    const Problem *problemPtr;
    CHKERR m_field.get_problem("TEST_PROBLEM", &problemPtr);
    std::map<EntityHandle, double> m0, m1, m2;
    for (_IT_NUMEREDDOF_ROW_FOR_LOOP_(problemPtr, dit)) {

      my_split.precision(3);
      my_split.setf(std::ios::fixed);
      double val = fabs(dit->get()->getFieldData()) < eps
                       ? 0.0
                       : dit->get()->getFieldData();
      my_split << dit->get()->getPetscGlobalDofIdx() << " " << val << std::endl;
    }
  }
  CATCH_ERRORS;

  MoFEM::Core::Finalize();

  return 0;
}
