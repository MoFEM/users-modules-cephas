/**
 * \file main_snippet.cpp
 * \example main_snippet.cpp
 *
 * Using Basic interface calculate the divergence of base functions, and
 * integral of flux on the boundary. Since the h-div space is used, volume
 * integral and boundary integral should give the same result.
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

#include <MoFEM.hpp>

using namespace MoFEM;

static char help[] = "...\n\n";

struct CommonData {
  boost::shared_ptr<VectorDouble> rhoAtIntegrationPts;
  enum VecElements {
    ZERO = 0,
    FIRST_X,
    FIRST_Y,
    FIRST_Z,
    SECOND_XX,
    SECOND_XY,
    SECOND_XZ,
    SECOND_YY,
    SECOND_YZ,
    SECOND_ZZ,
    LAST_ELEMENT
  };
  SmartPetscObj<Vec> petscVec;
};

#include <second_moment_of_inertia.hpp>

int main(int argc, char *argv[]) {

  MoFEM::Core::Initialize(&argc, &argv, (char *)0, help);

  try {

    //! [Register MoFEM discrete manager in PETSc]
    DMType dm_name = "DMMOFEM";
    CHKERR DMRegister_MoFEM(dm_name);
    //! [Register MoFEM discrete manager in PETSc

    //! [Create MoAB]
    moab::Core mb_instance;              ///< mesh database
    moab::Interface &moab = mb_instance; ///< mesh database interface
    //! [Create MoAB]

    //! [Create MoFEM]
    MoFEM::Core core(moab);           ///< finite element database
    MoFEM::Interface &m_field = core; ///< finite element database insterface
    //! [Create MoFEM]

    //! [Set up problem]
    Simple *simple_interface = m_field.getInterface<Simple>();
    CHKERR simple_interface->getOptions();
    CHKERR simple_interface->loadFile("");
    // Add field
    CHKERR simple_interface->addDomainField("rho", H1, AINSWORTH_LEGENDRE_BASE,
                                            1);
    constexpr int order = 1;
    CHKERR simple_interface->setFieldOrder("rho", order);
    CHKERR simple_interface->setUp();
    //! [Set up problem]

    //! [Distributions mass]
    auto set_distance = [&](VectorAdaptor &&field_data, double *xcoord,
                            double *ycoord, double *zcoord) {
      MoFEMFunctionBeginHot;
      field_data[0] = 1;
      MoFEMFunctionReturnHot(0);
    };
    FieldBlas *field_blas;
    CHKERR m_field.getInterface(field_blas);
    CHKERR field_blas->setVertexDofs(set_distance, "rho");
    //! [Distributions mass]

    //! [Create common data]
    auto common_data_ptr = boost::make_shared<CommonData>();
    common_data_ptr->petscVec = createSmartVectorMPI(
        m_field.get_comm(),
        (!m_field.get_comm_rank()) ? CommonData::LAST_ELEMENT : 0,
        CommonData::LAST_ELEMENT);
    CHKERR VecZeroEntries(common_data_ptr->petscVec);
    common_data_ptr->rhoAtIntegrationPts =
        boost::make_shared<VectorDouble>();
    //! [Create common data]

    //! [Push operators to pipeline]
    Basic *basic_interface = m_field.getInterface<Basic>();
    basic_interface->getOpDomainRhsPipeline().push_back(
        new OpCalculateScalarFieldValues(
            "rho", common_data_ptr->rhoAtIntegrationPts));
    basic_interface->getOpDomainRhsPipeline().push_back(
        new OpZero(common_data_ptr));
    basic_interface->getOpDomainRhsPipeline().push_back(
        new OpFirst(common_data_ptr));
    basic_interface->getOpDomainRhsPipeline().push_back(
        new OpSecond(common_data_ptr));
    //! [Push operators to pipeline]

    //! [Do calculations]
    auto integration_rule = [](int, int, int p_data) { return p_data + 2; };
    CHKERR basic_interface->setDomainRhsIntegrationRule(integration_rule);
    CHKERR basic_interface->loopFiniteElements();
    CHKERR VecAssemblyBegin(common_data_ptr->petscVec);
    CHKERR VecAssemblyEnd(common_data_ptr->petscVec);
    //! [Do calculations]

    const double *array;
    CHKERR VecGetArrayRead(common_data_ptr->petscVec, &array);

    //! [Print results]
    if (m_field.get_comm_rank() == 0) {
      PetscPrintf(PETSC_COMM_SELF, "Mass %6.4e\n", array[CommonData::ZERO]);
      PetscPrintf(PETSC_COMM_SELF,
                  "First moment of inertia [ %6.4e, %6.4e, %6.4e ] \n",
                  array[CommonData::FIRST_X], array[CommonData::FIRST_Y],
                  array[CommonData::FIRST_Z]);
      PetscPrintf(
          PETSC_COMM_SELF,
          "Second moment of inertia [ %6.4e, %6.4e, %6.4e; %6.4e %6.4e; "
          "%6.4e ]\n",
          array[CommonData::SECOND_XX], array[CommonData::SECOND_XY],
          array[CommonData::SECOND_XZ], array[CommonData::SECOND_YY],
          array[CommonData::SECOND_YZ], array[CommonData::SECOND_ZZ]);
    }
    //! [Print results]

    //! [Test example]
    PetscBool test = PETSC_FALSE;
    CHKERR PetscOptionsGetBool(PETSC_NULL, "", "-test", &test, PETSC_NULL);
    if(m_field.get_comm_rank() == 0 && test) {
      constexpr double eps = 1e-8;
      constexpr double expected_area = 1.;
      constexpr double expected_first_moment = 0.;
      constexpr double expected_second_moment = 1. / 12.;
      if (std::abs(array[CommonData::ZERO] - expected_area) > eps)
        SETERRQ2(PETSC_COMM_SELF, MOFEM_ATOM_TEST_INVALID,
                "Wrong area %6.4e !+ %6.4e", expected_area,
                array[CommonData::ZERO]);
      for (auto i :
           {CommonData::FIRST_X, CommonData::FIRST_Y, CommonData::FIRST_Z}) {
        if (std::abs(array[i] - expected_first_moment) > eps)
          SETERRQ2(PETSC_COMM_SELF, MOFEM_ATOM_TEST_INVALID,
                  "Wrong first moment %6.4e !+ %6.4e", expected_first_moment,
                  array[i]);
      }
      for (auto i : {CommonData::SECOND_XX, CommonData::SECOND_YY,
                     CommonData::SECOND_ZZ}) {
        if (std::abs(array[i] - expected_second_moment) > eps)
          SETERRQ2(PETSC_COMM_SELF, MOFEM_ATOM_TEST_INVALID,
                  "Wrong second moment %6.4e !+ %6.4e", expected_second_moment,
                  array[i]);
      }
    }
    //! [Test example]

    CHKERR VecRestoreArrayRead(common_data_ptr->petscVec, &array);


  }
  CATCH_ERRORS;

  CHKERR MoFEM::Core::Finalize();
}
