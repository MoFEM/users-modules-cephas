/**
 * \file helmholtz.cpp
 * \example helmholtz.cpp
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

using EntData = DataForcesAndSourcesCore::EntData;

using DomainEle = FaceElementForcesAndSourcesCoreBase;
using DomainEleOp = DomainEle::UserDataOperator;
using EdgeEle = EdgeElementForcesAndSourcesCoreBase;
using EdgeEleOp = EdgeEle::UserDataOperator;

using OpDomainGradGrad = OpDiffOps<DomainEleOp>::OpGradGrad<2>;
using OpDomainMass = OpDiffOps<DomainEleOp>::OpMass;
using OpDomainSource = OpDiffOps<DomainEleOp>::OpSource<2>;
using OpBoundaryMass = OpDiffOps<EdgeEleOp>::OpMass;

struct Example {

  Example(MoFEM::Interface &m_field) : mField(m_field) {}

  MoFEMErrorCode runProblem();

private:
  MoFEM::Interface &mField;
  MoFEM::Interface &mField;
  Simple *simpleInterface;

  MoFEMErrorCode readMesh();
  MoFEMErrorCode setupProblem();
  MoFEMErrorCode boundaryCondition();
  MoFEMErrorCode assembleSystem();
  MoFEMErrorCode solveSystem();
  MoFEMErrorCode outputResults();

  MatrixDouble invJac;
};

MoFEMErrorCode Example::runProblem() {
  MoFEMFunctionBegin;
  CHKERR readMesh();
  CHKERR setupProblem();
  CHKERR boundaryCondition();
  CHKERR assembleSystem();
  CHKERR solveSystem();
  CHKERR outputResults();
  MoFEMFunctionReturn(0);
}

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
  CHKERR simpleInterface->addDomainField("U_REAL", H1,
                                         AINSWORTH_BERNSTEIN_BEZIER_BASE, 1);
  CHKERR simpleInterface->addDomainField("U_IMAG", H1,
                                         AINSWORTH_BERNSTEIN_BEZIER_BASE, 1);
  CHKERR simpleInterface->addBoundaryField("U_REAL", H1,
                                           AINSWORTH_BERNSTEIN_BEZIER_BASE, 1);
  CHKERR simpleInterface->addBoundaryField("U_IMAG", H1,
                                           AINSWORTH_BERNSTEIN_BEZIER_BASE, 1);
  constexpr int order = 10;
  CHKERR simpleInterface->setFieldOrder("U_REAL", order);
  CHKERR simpleInterface->setFieldOrder("U_IMAG", order);
  CHKERR simpleInterface->setUp();
  MoFEMFunctionReturn(0);
}
//! [Set up problem]

//! [Applying essential BC]
MoFEMErrorCode Example::boundaryCondition() {
  MoFEMFunctionBegin;
  MoFEMFunctionReturn(0);
}
//! [Applying essential BC]

//! [Push operators to pipeline]
MoFEMErrorCode Example::assembleSystem() {
  MoFEMFunctionBegin;
  PipelineManager *pipeline_mng = mField.getInterface<PipelineManager>();

  auto vol_source_function = [](const double x, const double y,
                                const double z) {
    const double xc = 0;
    const double yc = -1.25;
    const double xs = x - xc;
    const double ys = y - yc;
    constexpr double eps = 60;
    return exp(-pow(eps * sqrt(xs * xs + ys * ys), 2));
  };

  constexpr double k = 90;

  auto beta = [](const double, const double, const double) { return -1; };
  auto k2 = [k](const double, const double, const double) { return pow(k, 2); };
  auto kp = [k](const double, const double, const double) { return k; };
  auto km = [k](const double, const double, const double) { return -k; };
  auto integration_rule = [](int, int, int p_data) { return 2 * p_data; };

  auto set_domain = [&]() {
    MoFEMFunctionBegin;
    pipeline_mng->getOpDomainLhsPipeline().push_back(
        new OpCalculateInvJacForFace(invJac));
    pipeline_mng->getOpDomainLhsPipeline().push_back(
        new OpSetInvJacH1ForFace(invJac));
    pipeline_mng->getOpDomainLhsPipeline().push_back(
        new OpDomainGradGrad("U_REAL", "U_REAL", beta));
    pipeline_mng->getOpDomainLhsPipeline().push_back(
        new OpDomainGradGrad("U_IMAG", "U_IMAG", beta));

    pipeline_mng->getOpDomainLhsPipeline().push_back(
        new OpDomainMass("U_REAL", "U_REAL", k2));
    pipeline_mng->getOpDomainLhsPipeline().push_back(
        new OpDomainMass("U_IMAG", "U_IMAG", k2));

    pipeline_mng->getOpDomainRhsPipeline().push_back(
        new OpDomainSource("U_REAL", vol_source_function));

    CHKERR pipeline_mng->setDomainRhsIntegrationRule(integration_rule);
    CHKERR pipeline_mng->setDomainLhsIntegrationRule(integration_rule);
    MoFEMFunctionReturn(0);
  };

  auto set_boundary = [&]() {
    MoFEMFunctionBegin;
    pipeline_mng->getOpBoundaryLhsPipeline().push_back(
        new OpBoundaryMass("U_IMAG", "U_REAL", kp));
    pipeline_mng->getOpBoundaryLhsPipeline().push_back(
        new OpBoundaryMass("U_REAL", "U_IMAG", km));
    CHKERR pipeline_mng->setBoundaryLhsIntegrationRule(integration_rule);
    MoFEMFunctionReturn(0);
  };

  CHKERR set_domain();
  CHKERR set_boundary();

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

  CHKERR KSPSolve(solver, F, D);
  CHKERR VecGhostUpdateBegin(D, INSERT_VALUES, SCATTER_FORWARD);
  CHKERR VecGhostUpdateEnd(D, INSERT_VALUES, SCATTER_FORWARD);
  CHKERR DMoFEMMeshToLocalVector(dm, D, INSERT_VALUES, SCATTER_REVERSE);
  MoFEMFunctionReturn(0);
}
//! [Solve]

//! [Postprocess results]
MoFEMErrorCode Example::outputResults() {
  MoFEMFunctionBegin;
  PipelineManager *pipeline_mng = mField.getInterface<PipelineManager>();
  pipeline_mng->getDomainLhsFE().reset();
  pipeline_mng->getDomainRhsFE().reset();
  pipeline_mng->getBoundaryLhsFE().reset();
  pipeline_mng->getBoundaryRhsFE().reset();
  auto post_proc_fe = boost::make_shared<PostProcFaceOnRefinedMesh>(mField);
  post_proc_fe->generateReferenceElementMesh();
  post_proc_fe->addFieldValuesPostProc("U_REAL");
  post_proc_fe->addFieldValuesPostProc("U_IMAG");
  pipeline_mng->getDomainRhsFE() = post_proc_fe;
  CHKERR pipeline_mng->loopFiniteElements();
  CHKERR post_proc_fe->writeFile("out_helmholtz.h5m");
  MoFEMFunctionReturn(0);
}
//! [Postprocess results]

int main(int argc, char *argv[]) {

  // Initialisation of MoFEM/PETSc and MOAB data structures
  const char param_file[] = "param_file.petsc";
  MoFEM::Core::Initialize(&argc, &argv, param_file, help);

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
