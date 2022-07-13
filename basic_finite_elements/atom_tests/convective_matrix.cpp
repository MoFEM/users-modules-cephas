/** \file convective_matrix.cpp

 * \ingroup convective_mass_elem
 * \ingroup nonlinear_elastic_elem
 *
 * Atom test for convective mass element
 *
 */



#include <BasicFiniteElements.hpp>
using namespace MoFEM;

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
    option = ""; //"PARALLEL=BCAST;";//;DEBUG_IO";
    CHKERR moab.load_file(mesh_file_name, 0, option);

    MoFEM::Core core(moab);
    MoFEM::Interface &m_field = core;

    // ref meshset ref level 0
    BitRefLevel bit_level0;
    bit_level0.set(0);
    EntityHandle meshset_level0;
    CHKERR moab.create_meshset(MESHSET_SET, meshset_level0);
    CHKERR m_field.getInterface<BitRefManager>()->setBitRefLevelByDim(
        0, 3, bit_level0);
    CHKERR m_field.getInterface<BitRefManager>()->getEntitiesByRefLevel(
        bit_level0, BitRefLevel().set(), meshset_level0);

    // Fields
    CHKERR m_field.add_field("SPATIAL_POSITION", H1, AINSWORTH_LEGENDRE_BASE,
                             3);

    // FE
    CHKERR m_field.add_finite_element("ELASTIC");

    // Define rows/cols and element data
    CHKERR m_field.modify_finite_element_add_field_row("ELASTIC",
                                                       "SPATIAL_POSITION");
    CHKERR m_field.modify_finite_element_add_field_col("ELASTIC",
                                                       "SPATIAL_POSITION");
    CHKERR m_field.modify_finite_element_add_field_data("ELASTIC",
                                                        "SPATIAL_POSITION");

    // define problems
    CHKERR m_field.add_problem("ELASTIC_MECHANICS");

    // set finite elements for problems
    CHKERR m_field.modify_problem_add_finite_element("ELASTIC_MECHANICS",
                                                     "ELASTIC");

    // set refinement level for problem
    CHKERR m_field.modify_problem_ref_level_add_bit("ELASTIC_MECHANICS",
                                                    bit_level0);

    // add entitities (by tets) to the field
    CHKERR m_field.add_ents_to_field_by_type(0, MBTET, "SPATIAL_POSITION");

    // add finite elements entities
    CHKERR m_field.add_ents_to_finite_element_by_bit_ref(
        bit_level0, BitRefLevel().set(), "ELASTIC", MBTET);

    // set app. order
    PetscInt order;
    CHKERR PetscOptionsGetInt(PETSC_NULL, PETSC_NULL, "-my_order", &order,
                              &flg);
    if (flg != PETSC_TRUE) {
      order = 1;
    }
    CHKERR m_field.set_field_order(0, MBTET, "SPATIAL_POSITION", order);
    CHKERR m_field.set_field_order(0, MBTRI, "SPATIAL_POSITION", order);
    CHKERR m_field.set_field_order(0, MBEDGE, "SPATIAL_POSITION", order);
    CHKERR m_field.set_field_order(0, MBVERTEX, "SPATIAL_POSITION", 1);

    CHKERR m_field.add_finite_element("NEUAMNN_FE");
    CHKERR m_field.modify_finite_element_add_field_row("NEUAMNN_FE",
                                                       "SPATIAL_POSITION");
    CHKERR m_field.modify_finite_element_add_field_col("NEUAMNN_FE",
                                                       "SPATIAL_POSITION");
    CHKERR m_field.modify_finite_element_add_field_data("NEUAMNN_FE",
                                                        "SPATIAL_POSITION");
    CHKERR m_field.modify_problem_add_finite_element("ELASTIC_MECHANICS",
                                                     "NEUAMNN_FE");
    for (_IT_CUBITMESHSETS_BY_BCDATA_TYPE_FOR_LOOP_(m_field, NODESET | FORCESET,
                                                    it)) {
      Range tris;
      CHKERR moab.get_entities_by_type(it->meshset, MBTRI, tris, true);
      CHKERR m_field.add_ents_to_finite_element_by_type(tris, MBTRI,
                                                        "NEUAMNN_FE");
    }
    for (_IT_CUBITMESHSETS_BY_BCDATA_TYPE_FOR_LOOP_(
             m_field, SIDESET | PRESSURESET, it)) {
      Range tris;
      CHKERR moab.get_entities_by_type(it->meshset, MBTRI, tris, true);
      CHKERR m_field.add_ents_to_finite_element_by_type(tris, MBTRI,
                                                        "NEUAMNN_FE");
    }

    // Velocity
    CHKERR m_field.add_field("SPATIAL_VELOCITY", H1, AINSWORTH_LEGENDRE_BASE,
                             3);
    CHKERR m_field.add_ents_to_field_by_type(0, MBTET, "SPATIAL_VELOCITY");
    int order_velocity = 1;
    CHKERR m_field.set_field_order(0, MBTET, "SPATIAL_VELOCITY",
                                   order_velocity);
    CHKERR m_field.set_field_order(0, MBTRI, "SPATIAL_VELOCITY",
                                   order_velocity);
    CHKERR m_field.set_field_order(0, MBEDGE, "SPATIAL_VELOCITY",
                                   order_velocity);
    CHKERR m_field.set_field_order(0, MBVERTEX, "SPATIAL_VELOCITY", 1);

    CHKERR m_field.add_field("DOT_SPATIAL_POSITION", H1,
                             AINSWORTH_LEGENDRE_BASE, 3);
    CHKERR m_field.add_ents_to_field_by_type(0, MBTET, "DOT_SPATIAL_POSITION");
    CHKERR m_field.set_field_order(0, MBTET, "DOT_SPATIAL_POSITION", order);
    CHKERR m_field.set_field_order(0, MBTRI, "DOT_SPATIAL_POSITION", order);
    CHKERR m_field.set_field_order(0, MBEDGE, "DOT_SPATIAL_POSITION", order);
    CHKERR m_field.set_field_order(0, MBVERTEX, "DOT_SPATIAL_POSITION", 1);
    CHKERR m_field.add_field("DOT_SPATIAL_VELOCITY", H1,
                             AINSWORTH_LEGENDRE_BASE, 3);
    CHKERR m_field.add_ents_to_field_by_type(0, MBTET, "DOT_SPATIAL_VELOCITY");
    CHKERR m_field.set_field_order(0, MBTET, "DOT_SPATIAL_VELOCITY",
                                   order_velocity);
    CHKERR m_field.set_field_order(0, MBTRI, "DOT_SPATIAL_VELOCITY",
                                   order_velocity);
    CHKERR m_field.set_field_order(0, MBEDGE, "DOT_SPATIAL_VELOCITY",
                                   order_velocity);
    CHKERR m_field.set_field_order(0, MBVERTEX, "DOT_SPATIAL_VELOCITY", 1);

    ConvectiveMassElement inertia(m_field, 1);
    CHKERR inertia.setBlocks();
    CHKERR inertia.addConvectiveMassElement("MASS_ELEMENT", "SPATIAL_VELOCITY",
                                            "SPATIAL_POSITION");
    CHKERR inertia.addVelocityElement("VELOCITY_ELEMENT", "SPATIAL_VELOCITY",
                                      "SPATIAL_POSITION");
    CHKERR m_field.modify_problem_add_finite_element("ELASTIC_MECHANICS",
                                                     "MASS_ELEMENT");
    CHKERR m_field.modify_problem_add_finite_element("ELASTIC_MECHANICS",
                                                     "VELOCITY_ELEMENT");

    // build field
    CHKERR m_field.build_fields();

    double scale_positions = 2;
    {
      EntityHandle node = 0;
      double coords[3];
      for (_IT_GET_DOFS_FIELD_BY_NAME_FOR_LOOP_(m_field, "SPATIAL_POSITION",
                                                dof_ptr)) {
        if (dof_ptr->get()->getEntType() != MBVERTEX)
          continue;
        EntityHandle ent = dof_ptr->get()->getEnt();
        int dof_rank = dof_ptr->get()->getDofCoeffIdx();
        double &fval = dof_ptr->get()->getFieldData();
        if (node != ent) {
          CHKERR moab.get_coords(&ent, 1, coords);
          node = ent;
        }
        fval = scale_positions * coords[dof_rank];
      }
    }

    double scale_velocities = 4;
    {
      EntityHandle node = 0;
      double coords[3];
      for (_IT_GET_DOFS_FIELD_BY_NAME_FOR_LOOP_(m_field, "DOT_SPATIAL_POSITION",
                                                dof_ptr)) {
        if (dof_ptr->get()->getEntType() != MBVERTEX)
          continue;
        EntityHandle ent = dof_ptr->get()->getEnt();
        int dof_rank = dof_ptr->get()->getDofCoeffIdx();
        double &fval = dof_ptr->get()->getFieldData();
        if (node != ent) {
          CHKERR moab.get_coords(&ent, 1, coords);
          node = ent;
        }
        fval = scale_velocities * coords[dof_rank];
      }
    }

    // build finite elemnts
    CHKERR m_field.build_finite_elements();

    // build adjacencies
    CHKERR m_field.build_adjacencies(bit_level0);

    // build problem

    ProblemsManager *prb_mng_ptr;
    CHKERR m_field.getInterface(prb_mng_ptr);

    CHKERR prb_mng_ptr->buildProblem("ELASTIC_MECHANICS", true);

    // partition
    CHKERR prb_mng_ptr->partitionProblem("ELASTIC_MECHANICS");
    CHKERR prb_mng_ptr->partitionFiniteElements("ELASTIC_MECHANICS");
    CHKERR prb_mng_ptr->partitionGhostDofs("ELASTIC_MECHANICS");

    // create matrices
    Vec F;
    CHKERR m_field.getInterface<VecManager>()->vecCreateGhost(
        "ELASTIC_MECHANICS", COL, &F);
    Vec D;
    CHKERR VecDuplicate(F, &D);
    Mat Aij;
    CHKERR m_field.getInterface<MatrixManager>()
        ->createMPIAIJWithArrays<PetscGlobalIdx_mi_tag>("ELASTIC_MECHANICS",
                                                        &Aij);

    CHKERR inertia.setConvectiveMassOperators("SPATIAL_VELOCITY",
                                              "SPATIAL_POSITION");
    CHKERR inertia.setVelocityOperators("SPATIAL_VELOCITY", "SPATIAL_POSITION");

    inertia.getLoopFeMassRhs().ts_F = F;
    inertia.getLoopFeMassRhs().ts_a = 1;
    inertia.getLoopFeMassLhs().ts_B = Aij;
    inertia.getLoopFeMassLhs().ts_a = 1;

    inertia.getLoopFeVelRhs().ts_F = F;
    inertia.getLoopFeVelRhs().ts_a = 1;
    inertia.getLoopFeVelLhs().ts_B = Aij;
    inertia.getLoopFeVelLhs().ts_a = 1;

    CHKERR m_field.loop_finite_elements("ELASTIC_MECHANICS", "MASS_ELEMENT",
                                        inertia.getLoopFeMassRhs());
    CHKERR m_field.loop_finite_elements("ELASTIC_MECHANICS", "VELOCITY_ELEMENT",
                                        inertia.getLoopFeVelRhs());
    CHKERR VecGhostUpdateBegin(F, ADD_VALUES, SCATTER_REVERSE);
    CHKERR VecGhostUpdateEnd(F, ADD_VALUES, SCATTER_REVERSE);
    CHKERR VecAssemblyBegin(F);
    CHKERR VecAssemblyEnd(F);

    CHKERR m_field.loop_finite_elements("ELASTIC_MECHANICS", "MASS_ELEMENT",
                                        inertia.getLoopFeMassLhs());
    CHKERR m_field.loop_finite_elements("ELASTIC_MECHANICS", "VELOCITY_ELEMENT",
                                        inertia.getLoopFeVelLhs());
    CHKERR MatAssemblyBegin(Aij, MAT_FINAL_ASSEMBLY);
    CHKERR MatAssemblyEnd(Aij, MAT_FINAL_ASSEMBLY);

    double sum = 0;
    CHKERR VecSum(F, &sum);
    CHKERR PetscPrintf(PETSC_COMM_WORLD, "sum  = %9.8e\n", sum);
    double fnorm;
    CHKERR VecNorm(F, NORM_2, &fnorm);
    CHKERR PetscPrintf(PETSC_COMM_WORLD, "fnorm  = %9.8e\n", fnorm);

    double mnorm;
    CHKERR MatNorm(Aij, NORM_1, &mnorm);
    CHKERR PetscPrintf(PETSC_COMM_WORLD, "mnorm  = %9.8e\n", mnorm);

    if (fabs(sum - 6.27285463e+00) > 1e-8) {
      SETERRQ(PETSC_COMM_WORLD, MOFEM_ATOM_TEST_INVALID, "Failed to pass test");
    }
    if (fabs(fnorm - 1.28223353e+00) > 1e-6) {
      SETERRQ(PETSC_COMM_WORLD, MOFEM_ATOM_TEST_INVALID, "Failed to pass test");
    }
    if (fabs(mnorm - 1.31250000e+00) > 1e-6) {
      SETERRQ(PETSC_COMM_WORLD, MOFEM_ATOM_TEST_INVALID, "Failed to pass test");
    }

    CHKERR VecDestroy(&F);
    CHKERR VecDestroy(&D);
    CHKERR MatDestroy(&Aij);
  }
  CATCH_ERRORS;

  MoFEM::Core::Finalize();

  return 0;
}
