/**
 * \file lesson1_moment_of_inertia.cpp
 * \example lesson1_moment_of_inertia.cpp
 *
 * \brief Calculate zero, first and second moments of inertia.
 *
 * Example intended to show how to write user data operator and integrate
 * scalar field.
 *
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

//! [Example]
struct Example {

  Example(MoFEM::Interface &m_field) : mField(m_field) {}

  MoFEMErrorCode runProblem();

private:
  MoFEM::Interface &mField;

  MoFEMErrorCode setUp();
  MoFEMErrorCode createCommonData();
  MoFEMErrorCode setFieldValues();
  MoFEMErrorCode pushOperators();
  MoFEMErrorCode integrateElements();
  MoFEMErrorCode postProcess();
  MoFEMErrorCode checkResults();

  struct CommonData;

  boost::shared_ptr<CommonData> commonDataPtr;

  struct OpZero;

  struct OpFirst;

  struct OpSecond;
};
//! [Example]

//! [Common data]
struct Example::CommonData
    : public boost::enable_shared_from_this<Example::CommonData> {

  VectorDouble rhoAtIntegrationPts; ///< Storing density at integration points

  inline boost::shared_ptr<VectorDouble> getRhoAtIntegrationPtsPtr() {
    return boost::shared_ptr<VectorDouble>(shared_from_this(),
                                           &rhoAtIntegrationPts);
  }

  /**
   * @brief Vector to indicate indices for storing zero, first and second
   * moments of inertia.
   *
   */
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

  SmartPetscObj<Vec>
      petscVec; ///< Smart pointer which stores PETSc distributed vector
};
//! [Common data]

struct Example::OpZero : public OpElement {
  OpZero(boost::shared_ptr<CommonData> &common_data_ptr)
      : OpElement("rho", OPROW), commonDataPtr(common_data_ptr) {
    std::fill(&doEntities[MBEDGE], &doEntities[MBMAXTYPE], false);
  }
  MoFEMErrorCode doWork(int side, EntityType type, EntData &data);

private:
  boost::shared_ptr<CommonData> commonDataPtr;
};

struct Example::OpFirst : public OpElement {
  OpFirst(boost::shared_ptr<CommonData> &common_data_ptr)
      : OpElement("rho", OPROW), commonDataPtr(common_data_ptr) {
    std::fill(&doEntities[MBEDGE], &doEntities[MBMAXTYPE], false);
  }
  MoFEMErrorCode doWork(int side, EntityType type, EntData &data);

private:
  boost::shared_ptr<CommonData> commonDataPtr;
};

//! [Operator]
struct Example::OpSecond : public OpElement {
  OpSecond(boost::shared_ptr<CommonData> &common_data_ptr)
      : OpElement("rho", OPROW), commonDataPtr(common_data_ptr) {
    std::fill(&doEntities[MBEDGE], &doEntities[MBMAXTYPE], false);
  }
  MoFEMErrorCode doWork(int side, EntityType type, EntData &data);

private:
  boost::shared_ptr<CommonData> commonDataPtr;
};
//! [Operator]

//! [Run all]
MoFEMErrorCode Example::runProblem() {
  MoFEMFunctionBegin;
  CHKERR setUp();
  CHKERR createCommonData();
  CHKERR setFieldValues();
  CHKERR pushOperators();
  CHKERR integrateElements();
  CHKERR postProcess();
  CHKERR checkResults();
  MoFEMFunctionReturn(0);
}
//! [Run all]

//! [Set up problem]
MoFEMErrorCode Example::setUp() {
  MoFEMFunctionBegin;
  Simple *simple = mField.getInterface<Simple>();
  CHKERR simple->getOptions();
  CHKERR simple->loadFile();
  // Add field
  CHKERR simple->addDomainField("rho", H1, AINSWORTH_LEGENDRE_BASE, 1);
  constexpr int order = 1;
  CHKERR simple->setFieldOrder("rho", order);
  CHKERR simple->setUp();
  MoFEMFunctionReturn(0);
}
//! [Set up problem]

//! [Create common data]
MoFEMErrorCode Example::createCommonData() {
  MoFEMFunctionBegin;
  commonDataPtr = boost::make_shared<CommonData>();

  int local_size;
  if (mField.get_comm_rank() == 0) // get_comm_rank() gets processor number
    // processor 0
    local_size = CommonData::LAST_ELEMENT; // last element gives size of vector
  else
    // other processors (e.g. 1, 2, 3, etc.)
    local_size = 0; // local size of vector is zero on other processors

  commonDataPtr->petscVec = createSmartVectorMPI(mField.get_comm(), local_size,
                                                 CommonData::LAST_ELEMENT);

  MoFEMFunctionReturn(0);
}
//! [Create common data]

//! [Set density distribution]
MoFEMErrorCode Example::setFieldValues() {
  MoFEMFunctionBegin;
  auto set_density = [&](VectorAdaptor &&field_data, double *xcoord,
                         double *ycoord, double *zcoord) {
    MoFEMFunctionBeginHot;
    field_data[0] = 1;
    MoFEMFunctionReturnHot(0);
  };
  FieldBlas *field_blas;
  CHKERR mField.getInterface(field_blas);
  CHKERR field_blas->setVertexDofs(set_density, "rho");
  MoFEMFunctionReturn(0);
}
//! [Set density distribution]

//! [Push operators to pipeline]
MoFEMErrorCode Example::pushOperators() {
  MoFEMFunctionBegin;
  PipelineManager *pipeline_mng = mField.getInterface<PipelineManager>();

  // Push an operator which calculates values of density at integration points
  pipeline_mng->getOpDomainRhsPipeline().push_back(
      new OpCalculateScalarFieldValues(
          "rho", commonDataPtr->getRhoAtIntegrationPtsPtr()));

  // Push an operator to pipeline to calculate zero moment of inertia (mass)
  pipeline_mng->getOpDomainRhsPipeline().push_back(new OpZero(commonDataPtr));

  // Push an operator to the pipeline to calculate first moment of inertaia
  pipeline_mng->getOpDomainRhsPipeline().push_back(new OpFirst(commonDataPtr));

  // Push an operator to the pipeline to calculate second moment of inertaia
  pipeline_mng->getOpDomainRhsPipeline().push_back(new OpSecond(commonDataPtr));

  // Set integration rule. Integration rule is equal to the polynomial order of
  // the density field plus 2, since under the integral of the second moment of
  // inertia term x*x is present
  auto integration_rule = [](int, int, int p_data) { return p_data + 2; };

  // Add integration rule to the element
  CHKERR pipeline_mng->setDomainRhsIntegrationRule(integration_rule);
  MoFEMFunctionReturn(0);
}
//! [Push operators to pipeline]

//! [Integrate]
MoFEMErrorCode Example::integrateElements() {
  MoFEMFunctionBegin;
  // Zero global vector
  CHKERR VecZeroEntries(commonDataPtr->petscVec);

  // Integrate elements by executing operators in the pipeline
  PipelineManager *pipeline_mng = mField.getInterface<PipelineManager>();
  CHKERR pipeline_mng->loopFiniteElements();

  // Assemble MPI vector
  CHKERR VecAssemblyBegin(commonDataPtr->petscVec);
  CHKERR VecAssemblyEnd(commonDataPtr->petscVec);
  MoFEMFunctionReturn(0);
}
//! [Integrate]

//! [Print results]
MoFEMErrorCode Example::postProcess() {
  MoFEMFunctionBegin;
  const double *array;
  CHKERR VecGetArrayRead(commonDataPtr->petscVec, &array);
  if (mField.get_comm_rank() == 0) {
    MOFEM_LOG_C("SELF", Sev::inform, "Mass %6.4e", array[CommonData::ZERO]);
    MOFEM_LOG_C("SELF", Sev::inform,
                "First moment of inertia [ %6.4e, %6.4e, %6.4e ]",
                array[CommonData::FIRST_X], array[CommonData::FIRST_Y],
                array[CommonData::FIRST_Z]);
    MOFEM_LOG_C("SELF", Sev::inform,
                "Second moment of inertia [ %6.4e, %6.4e, %6.4e; %6.4e %6.4e; "
                "%6.4e ]",
                array[CommonData::SECOND_XX], array[CommonData::SECOND_XY],
                array[CommonData::SECOND_XZ], array[CommonData::SECOND_YY],
                array[CommonData::SECOND_YZ], array[CommonData::SECOND_ZZ]);
  }
  CHKERR VecRestoreArrayRead(commonDataPtr->petscVec, &array);
  MoFEMFunctionReturn(0);
}
//! [Print results]

//! [Test example]
MoFEMErrorCode Example::checkResults() {
  MoFEMFunctionBegin;
  const double *array;
  CHKERR VecGetArrayRead(commonDataPtr->petscVec, &array);

  PetscBool test = PETSC_FALSE;
  CHKERR PetscOptionsGetBool(PETSC_NULL, "", "-test", &test, PETSC_NULL);
  if (mField.get_comm_rank() == 0 && test) {
    constexpr double eps = 1e-8;
    constexpr double expected_volume = 1.;
    constexpr double expected_first_moment = 0.;
    constexpr double expected_second_moment = 1. / 12.;
    if (std::abs(array[CommonData::ZERO] - expected_volume) > eps)
      SETERRQ2(PETSC_COMM_SELF, MOFEM_ATOM_TEST_INVALID,
               "Wrong area %6.4e != %6.4e", expected_volume,
               array[CommonData::ZERO]);
    for (auto i :
         {CommonData::FIRST_X, CommonData::FIRST_Y, CommonData::FIRST_Z}) {
      if (std::abs(array[i] - expected_first_moment) > eps)
        SETERRQ2(PETSC_COMM_SELF, MOFEM_ATOM_TEST_INVALID,
                 "Wrong first moment %6.4e != %6.4e", expected_first_moment,
                 array[i]);
    }
    for (auto i : {CommonData::SECOND_XX, CommonData::SECOND_YY,
                   CommonData::SECOND_ZZ}) {
      if (std::abs(array[i] - expected_second_moment) > eps)
        SETERRQ2(PETSC_COMM_SELF, MOFEM_ATOM_TEST_INVALID,
                 "Wrong second moment %6.4e != %6.4e", expected_second_moment,
                 array[i]);
    }
  }
  CHKERR VecRestoreArrayRead(commonDataPtr->petscVec, &array);

  MoFEMFunctionReturn(0);
}
//! [Test example]

//! [main]
int main(int argc, char *argv[]) {

  // Initialisation of MoFEM/PETSc and MoAB data structures
  MoFEM::Core::Initialize(&argc, &argv, (char *)0, help);

  // Error handling
  try {

    //! [Register MoFEM discrete manager in PETSc]
    DMType dm_name = "DMMOFEM";
    CHKERR DMRegister_MoFEM(dm_name);
    //! [Register MoFEM discrete manager in PETSc]

    //! [Create MoAB]
    moab::Core mb_instance;              ///< mesh database
    moab::Interface &moab = mb_instance; ///< mesh database interface
    //! [Create MoAB]

    //! [Create MoFEM]
    MoFEM::Core core(moab);           ///< finite element database
    MoFEM::Interface &m_field = core; ///< finite element database interface
    //! [Create MoFEM]

    Example ex(m_field);
    CHKERR ex.runProblem();
  }
  CATCH_ERRORS;

  CHKERR MoFEM::Core::Finalize();
}
//! [main]

//! [ZeroOp]
MoFEMErrorCode Example::OpZero::doWork(int side, EntityType type,
                                       EntData &data) {
  MoFEMFunctionBegin;
  if (type == MBVERTEX) {

    const int nb_integration_pts =
        getGaussPts().size2();                 // Number of integration points
    auto t_w = getFTensor0IntegrationWeight(); // Integration weights
    auto t_rho = getFTensor0FromVec(
        commonDataPtr->rhoAtIntegrationPts); // Density at integration weights
    const double volume = getMeasure();

    // Integrate area of the element
    double element_local_value = 0;
    for (int gg = 0; gg != nb_integration_pts; ++gg) {
      element_local_value += t_w * t_rho * volume;
      ++t_w;
      ++t_rho;
    }

    // Assemble area of the element into global PETSc vector
    const int index = CommonData::ZERO;
    CHKERR VecSetValue(commonDataPtr->petscVec, index, element_local_value,
                       ADD_VALUES);
  }
  MoFEMFunctionReturn(0);
}
//! [ZeroOp]

//! [FirstOp]
MoFEMErrorCode Example::OpFirst::doWork(int side, EntityType type,
                                        EntData &data) {
  MoFEMFunctionBegin;
  const int nb_integration_pts = getGaussPts().size2();
  auto t_w = getFTensor0IntegrationWeight(); ///< Integration weight
  auto t_rho = getFTensor0FromVec(
      commonDataPtr->rhoAtIntegrationPts); ///< Density at integration points
  auto t_x =
      getFTensor1CoordsAtGaussPts();  ///< Coordinates at integration points
  const double volume = getMeasure(); ///< Get Volume of element

  FTensor::Index<'i', 3> i;

  FTensor::Tensor1<double, 3>
      t_s;    ///< First moment of inertia is tensor of rank 1, i.e. vector.
  t_s(i) = 0; // Zero entries

  // Integrate
  for (int gg = 0; gg != nb_integration_pts; ++gg) {

    t_s(i) += t_w * t_rho * volume * t_x(i);

    ++t_w;   // move weight to next integration pts
    ++t_rho; // move density
    ++t_x;   // move coordinate
  }

  // Set array of indices
  constexpr std::array<int, 3> indices = {
      CommonData::FIRST_X, CommonData::FIRST_Y, CommonData::FIRST_Z};

  // Assemble first moment of inertia
  CHKERR VecSetValues(commonDataPtr->petscVec, 3, indices.data(), &t_s(0),
                      ADD_VALUES);
  MoFEMFunctionReturn(0);
}
//! [FirstOp]

//! [SecondOp]
MoFEMErrorCode Example::OpSecond::doWork(int side, EntityType type,
                                         EntData &data) {
  MoFEMFunctionBegin;

  const int nb_integration_pts = getGaussPts().size2();
  auto t_w = getFTensor0IntegrationWeight();
  auto t_rho = getFTensor0FromVec(commonDataPtr->rhoAtIntegrationPts);
  auto t_x = getFTensor1CoordsAtGaussPts();
  const double volume = getMeasure();

  // Create storage for a symmetric tensor
  std::array<double, 6> element_local_value;
  std::fill(element_local_value.begin(), element_local_value.end(), 0.0);

  // Create symmetric tensor using pointers to the storage
  FTensor::Tensor2_symmetric<FTensor::PackPtr<double *, 0>, 3> t_I(
      &element_local_value[0], &element_local_value[1], &element_local_value[2],
      &element_local_value[3], &element_local_value[4],
      &element_local_value[5]);

  FTensor::Index<'i', 3> i;
  FTensor::Index<'j', 3> j;

  // Integrate
  for (int gg = 0; gg != nb_integration_pts; ++gg) {

    // Symbol "^" indicates outer product which yields a symmetric tensor
    t_I(i, j) += (t_w * t_rho * volume) * (t_x(i) ^ t_x(j));

    ++t_w;   // move weight pointer to the next integration point
    ++t_rho; // move density pointer
    ++t_x;   // move coordinate pointer
  }

  // Set array of indices
  constexpr std::array<int, 6> indices = {
      CommonData::SECOND_XX, CommonData::SECOND_XY, CommonData::SECOND_XZ,
      CommonData::SECOND_YY, CommonData::SECOND_YZ, CommonData::SECOND_ZZ};

  // Assemble second moment of inertia
  CHKERR VecSetValues(commonDataPtr->petscVec, 6, indices.data(),
                      &element_local_value[0], ADD_VALUES);

  MoFEMFunctionReturn(0);
}
//! [SecondOp]