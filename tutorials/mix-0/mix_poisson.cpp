/**
 * \file mix-poisson.cpp
 * \example mix-poisson.cpp
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

using DomainEle = PipelineManager::FaceEle2D;
using DomainEleOp = DomainEle::UserDataOperator;
using EntData = DataForcesAndSourcesCore::EntData;

using OpHdivHdiv = FormsIntegrators<DomainEleOp>::Assembly<PETSC>::BiLinearForm<
    GAUSS>::OpMass<3, 1>;
using OpHdivU = FormsIntegrators<DomainEleOp>::Assembly<PETSC>::BiLinearForm<
    GAUSS>::OpMixDivTimesScalar<2>;

using OpQQ = FormsIntegrators<DomainEleOp>::Assembly<PETSC>::LinearForm<
    GAUSS>::OpBaseTimesVector<3, 3, 1>;
using OpDivQU = FormsIntegrators<DomainEleOp>::Assembly<PETSC>::LinearForm<
    GAUSS>::OpMixDivTimesU<1, 2>;
using OpUDivQ = FormsIntegrators<DomainEleOp>::Assembly<PETSC>::LinearForm<
    GAUSS>::OpBaseTimesScalarField<1>;

using OpDomainSource = FormsIntegrators<DomainEleOp>::Assembly<
    PETSC>::LinearForm<GAUSS>::OpSource<1, 1>;

struct Example {

  Example(MoFEM::Interface &m_field) : mField(m_field) {}

  MoFEMErrorCode runProblem();

private:
  MoFEM::Interface &mField;
  Simple *simpleInterface;

  MatrixDouble invJac;
  MatrixDouble jAC;

  //! [Approximated function]
  static double approxFunction(const double x, const double y, const double z) {
    return sin(x * 10.) * cos(y * 10.);
  }
  //! [Approximated function]

  MoFEMErrorCode readMesh();
  MoFEMErrorCode setupProblem();
  MoFEMErrorCode setIntegrationRules();
  MoFEMErrorCode createCommonData();
  MoFEMErrorCode boundaryCondition();
  MoFEMErrorCode assembleSystem();
  MoFEMErrorCode solveSystem();
  MoFEMErrorCode outputResults();
  MoFEMErrorCode checkResults();

};

//! [Run programme]
MoFEMErrorCode Example::runProblem() {
  MoFEMFunctionBegin;
  CHKERR readMesh();
  CHKERR setupProblem();
  CHKERR setIntegrationRules();
  CHKERR createCommonData();
  CHKERR boundaryCondition();
  CHKERR assembleSystem();
  CHKERR solveSystem();
  CHKERR outputResults();
  CHKERR checkResults();
  MoFEMFunctionReturn(0);
}
//! [Run programme]

//! [Read mesh]
MoFEMErrorCode Example::readMesh() {
  MoFEMFunctionBegin;

  CHKERR mField.getInterface(simpleInterface);
  CHKERR simpleInterface->getOptions();
  CHKERR simpleInterface->loadFile();

  MoFEMFunctionReturn(0);
}
//! [Read mesh]

//! [Set up problem]
MoFEMErrorCode Example::setupProblem() {
  MoFEMFunctionBegin;
  // Add field
  CHKERR simpleInterface->addDomainField("FLUX", HCURL, DEMKOWICZ_JACOBI_BASE,
                                         1);
  // We using AINSWORTH_LEGENDRE_BASE because DEMKOWICZ_JACOBI_BASE for triangle
  // and tet is not yet implemented for L2 space. DEMKOWICZ_JACOBI_BASE and
  // AINSWORTH_LEGENDRE_BASE are construcreed in the same way on quad.
  CHKERR simpleInterface->addDomainField("U", L2, AINSWORTH_LEGENDRE_BASE, 1);
  constexpr int order = 6;
  CHKERR simpleInterface->setFieldOrder("FLUX", order);
  CHKERR simpleInterface->setFieldOrder("U", order - 1);
  CHKERR simpleInterface->setUp();
  MoFEMFunctionReturn(0);
}
//! [Set up problem]

//! [Set integration rule]
MoFEMErrorCode Example::setIntegrationRules() {
  MoFEMFunctionBegin;

  auto rule = [](int, int, int p) -> int { return 2 * p + 1; };

  PipelineManager *pipeline_mng = mField.getInterface<PipelineManager>();
  CHKERR pipeline_mng->setDomainLhsIntegrationRule(rule);
  CHKERR pipeline_mng->setDomainRhsIntegrationRule(rule);

  MoFEMFunctionReturn(0);
}
//! [Set integration rule]

//! [Create common data]
MoFEMErrorCode Example::createCommonData() {
  MoFEMFunctionBegin;
  MoFEMFunctionReturn(0);
}
//! [Create common data]

//! [Boundary condition]
MoFEMErrorCode Example::boundaryCondition() { return 0; }
//! [Boundary condition]

//! [Push operators to pipeline]
MoFEMErrorCode Example::assembleSystem() {
  MoFEMFunctionBegin;
  PipelineManager *pipeline_mng = mField.getInterface<PipelineManager>();

  pipeline_mng->getOpDomainLhsPipeline().push_back(
      new OpCalculateJacForFace(jAC));
  pipeline_mng->getOpDomainLhsPipeline().push_back(
      new OpCalculateInvJacForFace(invJac));
  pipeline_mng->getOpDomainLhsPipeline().push_back(new OpMakeHdivFromHcurl());
  pipeline_mng->getOpDomainLhsPipeline().push_back(
      new OpSetContravariantPiolaTransformFace(jAC));
  pipeline_mng->getOpDomainLhsPipeline().push_back(
      new OpSetInvJacHcurlFace(invJac));
  pipeline_mng->getOpDomainLhsPipeline().push_back(
      new OpMakeHighOrderGeometryWeightsOnFace());

  auto beta = [](const double, const double, const double) { return 1; };
  pipeline_mng->getOpDomainLhsPipeline().push_back(
      new OpHdivHdiv("FLUX", "FLUX", beta));

  auto minus_one = []() { return -1; };

  pipeline_mng->getOpDomainLhsPipeline().push_back(
      new OpHdivU("FLUX", "U", minus_one, true));
  pipeline_mng->getOpDomainRhsPipeline().push_back(
      new OpDomainSource("U", Example::approxFunction));

  MoFEMFunctionReturn(0);
}
//! [Push operators to pipeline]

//! [Solve]
MoFEMErrorCode Example::solveSystem() {
  MoFEMFunctionBegin;
  PipelineManager *pipeline_mng = mField.getInterface<PipelineManager>();
  auto solver = pipeline_mng->createKSP();
  CHKERR KSPSetFromOptions(solver);
  CHKERR KSPSetUp(solver);

  auto dm = simpleInterface->getDM();
  auto D = smartCreateDMVector(dm);
  auto F = smartVectorDuplicate(D);
  CHKERR VecZeroEntries(D);

  CHKERR KSPSolve(solver, F, D);
  CHKERR VecGhostUpdateBegin(D, INSERT_VALUES, SCATTER_FORWARD);
  CHKERR VecGhostUpdateEnd(D, INSERT_VALUES, SCATTER_FORWARD);
  CHKERR DMoFEMMeshToLocalVector(dm, D, INSERT_VALUES, SCATTER_REVERSE);
  MoFEMFunctionReturn(0);
}

//! [Solve]
MoFEMErrorCode Example::outputResults() {
  MoFEMFunctionBegin;
  PipelineManager *pipeline_mng = mField.getInterface<PipelineManager>();
  pipeline_mng->getDomainLhsFE().reset();
  auto post_proc_fe =
      boost::make_shared<PostProcFaceOnRefinedMeshFor2D>(mField);
  post_proc_fe->generateReferenceElementMesh();
  
  post_proc_fe->getOpPtrVector().push_back(new OpCalculateJacForFace(jAC));
  post_proc_fe->getOpPtrVector().push_back(
      new OpCalculateInvJacForFace(invJac));
  post_proc_fe->getOpPtrVector().push_back(new OpMakeHdivFromHcurl());
  post_proc_fe->getOpPtrVector().push_back(
      new OpSetContravariantPiolaTransformFace(jAC));
  post_proc_fe->getOpPtrVector().push_back(new OpSetInvJacHcurlFace(invJac));

  post_proc_fe->addFieldValuesPostProc("FLUX");
  post_proc_fe->addFieldValuesPostProc("U");
  pipeline_mng->getDomainRhsFE() = post_proc_fe;
  CHKERR pipeline_mng->loopFiniteElements();
  CHKERR post_proc_fe->writeFile("out_mix_poisson.h5m");
  MoFEMFunctionReturn(0);
}
//! [Postprocess results]

//! [Check results]
MoFEMErrorCode Example::checkResults() {
  MoFEMFunctionBegin;
  auto *pipeline_mng = mField.getInterface<PipelineManager>();
  auto *simple = mField.getInterface<Simple>();

  pipeline_mng->getDomainLhsFE().reset();
  pipeline_mng->getDomainRhsFE() = boost::make_shared<DomainEle>(mField);

  auto dm = simple->getDM();
  auto F = smartCreateDMVector(dm);
  pipeline_mng->getDomainRhsFE()->f = F;

  pipeline_mng->getOpDomainRhsPipeline().push_back(
      new OpCalculateJacForFace(jAC));
  pipeline_mng->getOpDomainRhsPipeline().push_back(
      new OpCalculateInvJacForFace(invJac));
  pipeline_mng->getOpDomainRhsPipeline().push_back(new OpMakeHdivFromHcurl());
  pipeline_mng->getOpDomainRhsPipeline().push_back(
      new OpSetContravariantPiolaTransformFace(jAC));
  pipeline_mng->getOpDomainRhsPipeline().push_back(
      new OpSetInvJacHcurlFace(invJac));
  pipeline_mng->getOpDomainRhsPipeline().push_back(
      new OpMakeHighOrderGeometryWeightsOnFace());

  auto res_source = [&](const double x, const double y, const double z) {
    return -approxFunction(x, y, z);
  };
  pipeline_mng->getOpDomainRhsPipeline().push_back(
      new OpDomainSource("U", res_source));

  auto q_ptr = boost::make_shared<MatrixDouble>();
  auto div_ptr = boost::make_shared<VectorDouble>();
  auto u_ptr = boost::make_shared<VectorDouble>();

  pipeline_mng->getOpDomainRhsPipeline().push_back(
      new OpCalculateScalarFieldValues("U", u_ptr));
  pipeline_mng->getOpDomainRhsPipeline().push_back(
      new OpCalculateHdivVectorField<3>("FLUX", q_ptr));
  pipeline_mng->getOpDomainRhsPipeline().push_back(
      new OpCalculateHdivVectorDivergence<3, 2>("FLUX", div_ptr));

  pipeline_mng->getOpDomainRhsPipeline().push_back(new OpQQ("FLUX", q_ptr));
  pipeline_mng->getOpDomainRhsPipeline().push_back(
      new OpDivQU("FLUX", u_ptr, -1));
  pipeline_mng->getOpDomainRhsPipeline().push_back(
      new OpUDivQ("U", div_ptr, -1));

  CHKERR setIntegrationRules();
  CHKERR pipeline_mng->loopFiniteElements();
  CHKERR VecAssemblyBegin(F);
  CHKERR VecAssemblyEnd(F);

  double nrm2;
  CHKERR VecNorm(F, NORM_2, &nrm2);

  MOFEM_LOG("WORLD", Sev::inform) << "Residual norm " << nrm2;

  constexpr double eps = 1e-8;
  if(std::abs(nrm2) < eps)
    nrm2 = 0;

  if (nrm2 != 0)
    SETERRQ(PETSC_COMM_WORLD, MOFEM_DATA_INCONSISTENCY, "Nonzero residual");

  MoFEMFunctionReturn(0);
}
//! [Check results]

int main(int argc, char *argv[]) {

  // Initialisation of MoFEM/PETSc and MOAB data structures
  const char param_file[] = "param_file.petsc";
  MoFEM::Core::Initialize(&argc, &argv, param_file, help);

  // Add logging channel for example
  auto core_log = logging::core::get();
  core_log->add_sink(
      LogManager::createSink(LogManager::getStrmWorld(), "EXAMPLE"));
  LogManager::setLog("EXAMPLE");
  MOFEM_LOG_TAG("EXAMPLE", "example");

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

    //! [Example]
    Example ex(m_field);
    CHKERR ex.runProblem();
    //! [Example]
  }
  CATCH_ERRORS;

  CHKERR MoFEM::Core::Finalize();
}

