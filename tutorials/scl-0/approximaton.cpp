/**
 * \file lesson2_approximaton.cpp
 * \example lesson2_approximaton.cpp
 *
 * Using PipelineManager interface calculate the divergence of base functions,
 * and integral of flux on the boundary. Since the h-div space is used, volume
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

#include <BasicFiniteElements.hpp>

using DomainEle = FaceElementForcesAndSourcesCoreBase;
using DomainEleOp = DomainEle::UserDataOperator;
using EntData = DataForcesAndSourcesCore::EntData;

#include <BaseOps.hpp>

using OpDomainMass = OpTools<DomainEleOp>::OpMass;
using OpDomainSource = OpTools<DomainEleOp>::OpSource<2>;

struct Example {

  Example(MoFEM::Interface &m_field) : mField(m_field) {}

  MoFEMErrorCode runProblem();

private:
  MoFEM::Interface &mField;

  static double approxFunction(const double x, const double y, const double z) {
    return sin(x * 10.) * cos(y * 10.);
  }
  static int integrationRule(int, int, int p_data) { return 2 * p_data; };

  MoFEMErrorCode setUP();
  MoFEMErrorCode createCommonData();
  MoFEMErrorCode bC();
  MoFEMErrorCode OPs();
  MoFEMErrorCode kspSolve();
  MoFEMErrorCode postProcess();
  MoFEMErrorCode checkResults();

  struct CommonData {
    boost::shared_ptr<VectorDouble> approxVals;
    SmartPetscObj<Vec> L2Vec;
    SmartPetscObj<Vec> resVec;
  };
  boost::shared_ptr<CommonData> commonDataPtr;

  struct OpError : public DomainEleOp {
    boost::shared_ptr<CommonData> commonDataPtr;
    OpError(boost::shared_ptr<CommonData> &common_data_ptr)
        : DomainEleOp("U", OPROW), commonDataPtr(common_data_ptr) {}
    MoFEMErrorCode doWork(int side, EntityType type, EntData &data);
  };
};

MoFEMErrorCode Example::runProblem() {
  MoFEMFunctionBegin;
  CHKERR setUP();
  CHKERR createCommonData();
  CHKERR bC();
  CHKERR OPs();
  CHKERR kspSolve();
  CHKERR postProcess();
  CHKERR checkResults();
  MoFEMFunctionReturn(0);
}

//! [Set up problem]
MoFEMErrorCode Example::setUP() {
  MoFEMFunctionBegin;
  Simple *simple = mField.getInterface<Simple>();
  // Add field
  CHKERR simple->addDomainField("U", H1, AINSWORTH_BERNSTEIN_BEZIER_BASE, 1);
  constexpr int order = 4;
  CHKERR simple->setFieldOrder("U", order);
  CHKERR simple->setUp();
  MoFEMFunctionReturn(0);
}
//! [Set up problem]

//! [Create common data]
MoFEMErrorCode Example::createCommonData() {
  MoFEMFunctionBegin;
  Simple *simple = mField.getInterface<Simple>();
  commonDataPtr = boost::make_shared<CommonData>();
  commonDataPtr->resVec = smartCreateDMVector(simple->getDM());
  commonDataPtr->L2Vec = createSmartVectorMPI(
      mField.get_comm(), (!mField.get_comm_rank()) ? 1 : 0, 1);
  commonDataPtr->approxVals = boost::make_shared<VectorDouble>();
  MoFEMFunctionReturn(0);
}
//! [Create common data]

//! [Boundary condition]
MoFEMErrorCode Example::bC() { return 0; }
//! [Boundary condition]

//! [Push operators to pipeline]
MoFEMErrorCode Example::OPs() {
  MoFEMFunctionBegin;
  PipelineManager *pipeline_mng = mField.getInterface<PipelineManager>();
  auto beta = [](const double, const double, const double) { return 1; };
  pipeline_mng->getOpDomainLhsPipeline().push_back(
      new OpDomainMass("U", "U", beta));
  pipeline_mng->getOpDomainRhsPipeline().push_back(
      new OpDomainSource("U", Example::approxFunction));
  CHKERR pipeline_mng->setDomainRhsIntegrationRule(integrationRule);
  CHKERR pipeline_mng->setDomainLhsIntegrationRule(integrationRule);
  MoFEMFunctionReturn(0);
}
//! [Push operators to pipeline]

//! [Solve]
MoFEMErrorCode Example::kspSolve() {
  MoFEMFunctionBegin;
  Simple *simple = mField.getInterface<Simple>();
  PipelineManager *pipeline_mng = mField.getInterface<PipelineManager>();
  auto solver = pipeline_mng->createKSP();
  CHKERR KSPSetFromOptions(solver);
  CHKERR KSPSetUp(solver);

  auto dm = simple->getDM();
  auto D = smartCreateDMVector(dm);
  auto F = smartVectorDuplicate(D);

  CHKERR KSPSolve(solver, F, D);
  CHKERR VecGhostUpdateBegin(D, INSERT_VALUES, SCATTER_FORWARD);
  CHKERR VecGhostUpdateEnd(D, INSERT_VALUES, SCATTER_FORWARD);
  CHKERR DMoFEMMeshToLocalVector(dm, D, INSERT_VALUES, SCATTER_REVERSE);
  MoFEMFunctionReturn(0);
}

//! [Solve]
MoFEMErrorCode Example::postProcess() {
  MoFEMFunctionBegin;
  PipelineManager *pipeline_mng = mField.getInterface<PipelineManager>();
  pipeline_mng->getDomainLhsFE().reset();
  auto post_proc_fe = boost::make_shared<PostProcFaceOnRefinedMesh>(mField);
  post_proc_fe->generateReferenceElementMesh();
  post_proc_fe->addFieldValuesPostProc("U");
  pipeline_mng->getDomainRhsFE() = post_proc_fe;
  CHKERR pipeline_mng->loopFiniteElements();
  CHKERR post_proc_fe->writeFile("out_approx.h5m");
  MoFEMFunctionReturn(0);
}
//! [Postprocess results]

//! [Solve]
MoFEMErrorCode Example::checkResults() {
  MoFEMFunctionBegin;
  PipelineManager *pipeline_mng = mField.getInterface<PipelineManager>();
  pipeline_mng->getDomainLhsFE().reset();
  pipeline_mng->getDomainRhsFE().reset();
  pipeline_mng->getOpDomainRhsPipeline().clear();
  pipeline_mng->getOpDomainRhsPipeline().push_back(
      new OpCalculateScalarFieldValues("U", commonDataPtr->approxVals));
  pipeline_mng->getOpDomainRhsPipeline().push_back(new OpError(commonDataPtr));
  CHKERR pipeline_mng->setDomainRhsIntegrationRule(integrationRule);
  CHKERR pipeline_mng->loopFiniteElements();
  CHKERR VecAssemblyBegin(commonDataPtr->L2Vec);
  CHKERR VecAssemblyEnd(commonDataPtr->L2Vec);
  CHKERR VecAssemblyBegin(commonDataPtr->resVec);
  CHKERR VecAssemblyEnd(commonDataPtr->resVec);
  double nrm2;
  CHKERR VecNorm(commonDataPtr->resVec, NORM_2, &nrm2);
  const double *array;
  CHKERR VecGetArrayRead(commonDataPtr->L2Vec, &array);
  if (mField.get_comm_rank() == 0)
    PetscPrintf(PETSC_COMM_SELF, "Error %6.4e Vec norm %6.4e\n", sqrt(array[0]),
                nrm2);
  CHKERR VecRestoreArrayRead(commonDataPtr->L2Vec, &array);
  constexpr double eps = 1e-8;
  if (nrm2 > eps)
    SETERRQ(PETSC_COMM_SELF, MOFEM_DATA_INCONSISTENCY,
            "Not converged solution");
  MoFEMFunctionReturn(0);
}
//! [Solver]

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

    //! [Load mesh]
    Simple *simple = m_field.getInterface<Simple>();
    CHKERR simple->getOptions();
    CHKERR simple->loadFile("");
    //! [Load mesh]

    //! [Example]
    Example ex(m_field);
    CHKERR ex.runProblem();
    //! [Example]
  }
  CATCH_ERRORS;

  CHKERR MoFEM::Core::Finalize();
}

MoFEMErrorCode Example::OpError::doWork(int side, EntityType type,
                                        EntData &data) {
  MoFEMFunctionBegin;

  if (const size_t nb_dofs = data.getIndices().size()) {

    const int nb_integration_pts = getGaussPts().size2();
    auto t_w = getFTensor0IntegrationWeight();
    auto t_val = getFTensor0FromVec(*(commonDataPtr->approxVals));
    auto t_coords = getFTensor1CoordsAtGaussPts();

    VectorDouble nf(nb_dofs, false);
    nf.clear();

    FTensor::Index<'i', 3> i;
    const double volume = getMeasure();

    auto t_row_base = data.getFTensor0N();
    double error = 0;
    for (int gg = 0; gg != nb_integration_pts; ++gg) {

      const double alpha = t_w * volume;
      double diff = t_val - Example::approxFunction(t_coords(0), t_coords(1),
                                                    t_coords(2));
      error += alpha * pow(diff, 2);

      for (size_t r = 0; r != nb_dofs; ++r) {
        nf[r] += alpha * t_row_base * diff;
        ++t_row_base;
      }

      ++t_w;
      ++t_val;
      ++t_coords;
    }

    const int index = 0;
    CHKERR VecSetValue(commonDataPtr->L2Vec, index, error, ADD_VALUES);
    CHKERR VecSetValues(commonDataPtr->resVec, data, &nf[0], ADD_VALUES);
  }

  MoFEMFunctionReturn(0);
}