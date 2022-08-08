

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
    CHKERR m_field.add_field("DISP", H1, AINSWORTH_LEGENDRE_BASE, 3);
    CHKERR m_field.add_field("TEMP", H1, AINSWORTH_LEGENDRE_BASE, 1);

    // Problem
    CHKERR m_field.add_problem("PROB");

    // set refinement level for problem
    CHKERR m_field.modify_problem_ref_level_add_bit("PROB", bit_level0);

    // meshset consisting all entities in mesh
    EntityHandle root_set = moab.get_root_set();
    // add entities to field
    CHKERR m_field.add_ents_to_field_by_type(root_set, MBTET, "TEMP");
    CHKERR m_field.add_ents_to_field_by_type(root_set, MBTET, "DISP");

    // set app. order
    // see Hierarchic Finite Element Bases on Unstructured Tetrahedral Meshes
    // (Mark Ainsworth & Joe Coyle)
    int order_temp = 2;
    CHKERR m_field.set_field_order(root_set, MBTET, "TEMP", order_temp);
    CHKERR m_field.set_field_order(root_set, MBTRI, "TEMP", order_temp);
    CHKERR m_field.set_field_order(root_set, MBEDGE, "TEMP", order_temp);
    CHKERR m_field.set_field_order(root_set, MBVERTEX, "TEMP", 1);

    int order_disp = 3;
    CHKERR m_field.set_field_order(root_set, MBTET, "DISP", order_disp);
    CHKERR m_field.set_field_order(root_set, MBTRI, "DISP", order_disp);
    CHKERR m_field.set_field_order(root_set, MBEDGE, "DISP", order_disp);
    CHKERR m_field.set_field_order(root_set, MBVERTEX, "DISP", 1);

    ThermalStressElement thermal_stress_elem(m_field);
    CHKERR thermal_stress_elem.addThermalStressElement("ELAS", "DISP", "TEMP");
    CHKERR m_field.modify_problem_add_finite_element("PROB", "ELAS");

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
    CHKERR prb_mng_ptr->buildProblem("PROB", true);
    // mesh partitioning
    // partition
    CHKERR prb_mng_ptr->partitionSimpleProblem("PROB");
    CHKERR prb_mng_ptr->partitionFiniteElements("PROB");
    // what are ghost nodes, see Petsc Manual
    CHKERR prb_mng_ptr->partitionGhostDofs("PROB");

    // set temerature at nodes
    for (_IT_GET_DOFS_FIELD_BY_NAME_AND_TYPE_FOR_LOOP_(m_field, "TEMP",
                                                       MBVERTEX, dof)) {
      EntityHandle ent = dof->get()->getEnt();
      VectorDouble coords(3);
      CHKERR moab.get_coords(&ent, 1, &coords[0]);
      dof->get()->getFieldData() = 1;
    }

    Vec F;
    CHKERR m_field.getInterface<VecManager>()->vecCreateGhost("PROB", ROW, &F);
    CHKERR thermal_stress_elem.setThermalStressRhsOperators("DISP", "TEMP", F,
                                                            1);

    CHKERR m_field.loop_finite_elements(
        "PROB", "ELAS", thermal_stress_elem.getLoopThermalStressRhs());
    CHKERR VecAssemblyBegin(F);
    CHKERR VecAssemblyEnd(F);

    // PetscViewer viewer;
    // CHKERR
    // PetscViewerASCIIOpen(PETSC_COMM_WORLD,"forces_and_sources_thermal_stress_elem.txt",&viewer);
    // CHKERR VecChop(F,1e-4);
    // CHKERR VecView(F,viewer);
    // CHKERR PetscViewerDestroy(&viewer);

    double sum = 0;
    CHKERR VecSum(F, &sum);
    CHKERR PetscPrintf(PETSC_COMM_WORLD, "sum  = %9.8f\n", sum);
    double fnorm;
    CHKERR VecNorm(F, NORM_2, &fnorm);
    CHKERR PetscPrintf(PETSC_COMM_WORLD, "fnorm  = %9.8e\n", fnorm);
    if (fabs(sum) > 1e-7) {
      SETERRQ(PETSC_COMM_WORLD, MOFEM_ATOM_TEST_INVALID, "Failed to pass test");
    }
    if (fabs(fnorm - 2.64638118e+00) > 1e-7) {
      SETERRQ(PETSC_COMM_WORLD, MOFEM_ATOM_TEST_INVALID, "Failed to pass test");
    }

    CHKERR VecZeroEntries(F);
    CHKERR VecDestroy(&F);
  }
  CATCH_ERRORS;

  MoFEM::Core::Finalize();

  return 0;
}
