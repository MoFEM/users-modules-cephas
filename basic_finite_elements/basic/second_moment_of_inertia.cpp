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

using Element = MoFEM::VolumeElementForcesAndSourcesCoreBase;
using OpElement = Element::UserDataOperator;
using EntData = DataForcesAndSourcesCore::EntData;

struct CommonData {
  boost::shared_ptr<VectorDouble> rho_at_integration_points;
  SmartPetscObj<Vec> vec_mass;
  SmartPetscObj<Vec> vec_second_moment_of_inertia;
};

static CommonData common_data = CommonData();

struct OpCalcMass : public OpElement {
  OpCalcMass() : OpElement("rho", OPROW) {}
  MoFEMErrorCode doWork(int side, EntityType type, EntData &data);
};

struct OpCalcSecondMomentOfInertia : public OpElement {
  OpCalcSecondMomentOfInertia() : OpElement("rho", OPROW) {}
  MoFEMErrorCode doWork(int side, EntityType type, EntData &data);
};

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
    common_data.vec_mass = createSmartVectorMPI(
        m_field.get_comm(), (!m_field.get_comm_rank()) ? 1 : 0, 1);
    common_data.vec_second_moment_of_inertia = createSmartVectorMPI(
        m_field.get_comm(), (!m_field.get_comm_rank()) ? 1 : 0, 1);
    CHKERR VecZeroEntries(common_data.vec_mass);
    CHKERR VecZeroEntries(common_data.vec_second_moment_of_inertia);
    common_data.rho_at_integration_points = boost::make_shared<VectorDouble>();
    //! [Create common data]

    //! [Push operators to pipeline]
    Basic *basic_interface = m_field.getInterface<Basic>();
    basic_interface->getOpDomainRhsPipeline().push_back(
        new OpCalculateScalarFieldValues(
            "rho", common_data.rho_at_integration_points));
    basic_interface->getOpDomainRhsPipeline().push_back(new OpCalcMass());
    basic_interface->getOpDomainRhsPipeline().push_back(
        new OpCalcSecondMomentOfInertia());
    //! [Push operators to pipeline]

    //! [Do calculations]
    auto integration_rule = [](int, int, int p_data) { return p_data; };
    CHKERR basic_interface->loopFiniteElements();
    CHKERR VecAssemblyBegin(common_data.vec_mass);
    CHKERR VecAssemblyBegin(common_data.vec_second_moment_of_inertia);
    CHKERR VecAssemblyEnd(common_data.vec_mass);
    CHKERR VecAssemblyEnd(common_data.vec_second_moment_of_inertia);
    //! [Do calculations]

    //! [Print results]
    double mass, second_moment_of_inertia;
    CHKERR VecSum(common_data.vec_mass, &mass);
    CHKERR VecSum(common_data.vec_second_moment_of_inertia, &mass);
    PetscPrintf(PETSC_COMM_WORLD, "Mass %6.4e\n", mass);
    PetscPrintf(PETSC_COMM_WORLD, "Second moment of area %6.4e\n",
                second_moment_of_inertia);
    //! [Print results]
  }
  CATCH_ERRORS;

  CHKERR MoFEM::Core::Finalize();
}

MoFEMErrorCode OpCalcMass::doWork(int side, EntityType type, EntData &data) {
  MoFEMFunctionBegin;
  if (type == MBVERTEX) {
    const int nb_integration_pts = getGaussPts().size2();
    auto t_w = getFTensor0IntegrationWeight();
    auto t_rho = getFTensor0FromVec(*common_data.rho_at_integration_points);
    FTensor::Index<'i', 3> i;
    const double volume = getVolume();
    double element_local_value = 0;
    for (int gg = 0; gg != nb_integration_pts; ++gg) {
      element_local_value += t_w * t_rho * volume;
      ++t_w;
      ++t_rho;
    }
    CHKERR VecSetValue(common_data.vec_mass, 0, element_local_value,
                       ADD_VALUES);
  }
  MoFEMFunctionReturn(0);
}

MoFEMErrorCode OpCalcSecondMomentOfInertia::doWork(int side, EntityType type,
                                                   EntData &data) {
  MoFEMFunctionBegin;
  if (type == MBVERTEX) {
    const int nb_integration_pts = getGaussPts().size2();
    auto t_w = getFTensor0IntegrationWeight();
    auto t_rho = getFTensor0FromVec(*common_data.rho_at_integration_points);
    auto t_coords = getFTensor1CoordsAtGaussPts();
    const double volume = getVolume();
    double element_local_value = 0;
    for (int gg = 0; gg != nb_integration_pts; ++gg) {
      element_local_value += t_w * t_rho * volume;
      ++t_w;
      ++t_rho;
      ++t_coords;
    }
    cerr << element_local_value << endl;
    CHKERR VecSetValue(common_data.vec_mass, 0, element_local_value,
                       ADD_VALUES);
  }
  MoFEMFunctionReturn(0);
}