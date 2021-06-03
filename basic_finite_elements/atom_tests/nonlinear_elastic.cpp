/** \file nonlinear_elastic.cpp

 \brief Atom test for linear elastic dynamics.

 This is not exactly procedure for linear elastic dynamics, since jacobian is
 evaluated at every time step and snes procedure is involved. However it is
 implemented like that, to test methodology for general nonlinear problem.

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
    // add entitities (by tets) to the field
    CHKERR m_field.add_ents_to_field_by_type(0, MBTET, "SPATIAL_POSITION");
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

    NonlinearElasticElement elastic(m_field, 1);
    boost::shared_ptr<
        NonlinearElasticElement::FunctionsToCalculatePiolaKirchhoffI<double>>
    double_kirchhoff_material_ptr(
        new NonlinearElasticElement::FunctionsToCalculatePiolaKirchhoffI<
            double>());
    boost::shared_ptr<
        NonlinearElasticElement::FunctionsToCalculatePiolaKirchhoffI<adouble>>
        adouble_kirchhoff_material_ptr(
            new NonlinearElasticElement::FunctionsToCalculatePiolaKirchhoffI<
                adouble>());
    CHKERR elastic.setBlocks(double_kirchhoff_material_ptr,
                             adouble_kirchhoff_material_ptr);
    CHKERR elastic.addElement("ELASTIC", "SPATIAL_POSITION");
    CHKERR elastic.setOperators("SPATIAL_POSITION");

    // define problems
    CHKERR m_field.add_problem("ELASTIC_MECHANICS");
    // set refinement level for problem
    CHKERR m_field.modify_problem_ref_level_add_bit("ELASTIC_MECHANICS",
                                                    bit_level0);
    // set finite elements for problems
    CHKERR m_field.modify_problem_add_finite_element("ELASTIC_MECHANICS",
                                                     "ELASTIC");

    // build field
    CHKERR m_field.build_fields();

    // use this to apply some strain field to the body (testing only)
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

    // build finite elemnts
    CHKERR m_field.build_finite_elements();
    // build adjacencies
    CHKERR m_field.build_adjacencies(bit_level0);

    ProblemsManager *prb_mng_ptr;
    CHKERR m_field.getInterface(prb_mng_ptr);
    // build problem
    CHKERR prb_mng_ptr->buildProblem("ELASTIC_MECHANICS", true);
    // partition
    CHKERR prb_mng_ptr->partitionProblem("ELASTIC_MECHANICS");
    CHKERR prb_mng_ptr->partitionFiniteElements("ELASTIC_MECHANICS");
    CHKERR prb_mng_ptr->partitionGhostDofs("ELASTIC_MECHANICS");

    // create matrices
    Vec F;
    CHKERR m_field.getInterface<VecManager>()->vecCreateGhost(
        "ELASTIC_MECHANICS", COL, &F);
    Mat Aij;
    CHKERR m_field.getInterface<MatrixManager>()
        ->createMPIAIJWithArrays<PetscGlobalIdx_mi_tag>("ELASTIC_MECHANICS",
                                                        &Aij);

    elastic.getLoopFeRhs().snes_f = F;
    elastic.getLoopFeLhs().snes_B = Aij;

    CHKERR m_field.loop_finite_elements("ELASTIC_MECHANICS", "ELASTIC",
                                        elastic.getLoopFeRhs());
    CHKERR VecGhostUpdateBegin(F, ADD_VALUES, SCATTER_REVERSE);
    CHKERR VecGhostUpdateEnd(F, ADD_VALUES, SCATTER_REVERSE);
    CHKERR VecAssemblyBegin(F);
    CHKERR VecAssemblyEnd(F);

    CHKERR m_field.loop_finite_elements("ELASTIC_MECHANICS", "ELASTIC",
                                        elastic.getLoopFeLhs());
    CHKERR MatAssemblyBegin(Aij, MAT_FINAL_ASSEMBLY);
    CHKERR MatAssemblyEnd(Aij, MAT_FINAL_ASSEMBLY);

    double sum = 0;
    CHKERR VecSum(F, &sum);
    CHKERR PetscPrintf(PETSC_COMM_WORLD, "sum  = %4.3e\n", sum);
    double fnorm;
    CHKERR VecNorm(F, NORM_2, &fnorm);
    CHKERR PetscPrintf(PETSC_COMM_WORLD, "fnorm  = %9.8e\n", fnorm);

    double mnorm;
    CHKERR MatNorm(Aij, NORM_1, &mnorm);
    CHKERR PetscPrintf(PETSC_COMM_WORLD, "mnorm  = %9.8e\n", mnorm);

    if (fabs(sum) > 1e-8) {
      SETERRQ(PETSC_COMM_WORLD, MOFEM_ATOM_TEST_INVALID, "Failed to pass test");
    }
    if (fabs(fnorm - 5.12196914e+00) > 1e-6) {
      SETERRQ(PETSC_COMM_WORLD, MOFEM_ATOM_TEST_INVALID, "Failed to pass test");
    }
    if (fabs(mnorm - 5.48280139e+01) > 1e-6) {
      SETERRQ(PETSC_COMM_WORLD, MOFEM_ATOM_TEST_INVALID, "Failed to pass test");
    }

    CHKERR VecDestroy(&F);
    CHKERR MatDestroy(&Aij);
  }
  CATCH_ERRORS;

  MoFEM::Core::Finalize();

  return 0;
}
