/**
 * \file basic_helmholtz.cpp
 * \example basic_helmholtz.cpp
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

#include <BasicFiniteElements.hpp>

using EntData = DataForcesAndSourcesCore::EntData;

using DomainEle = FaceElementForcesAndSourcesCoreBase;
using DomainEleOp = DomainEle::UserDataOperator;
using EdgeEle = EdgeElementForcesAndSourcesCoreBase;
using EdgeEleOp = EdgeEle::UserDataOperator;

#include <BaseOps.hpp>

using OpDomainGradGrad = OpTools<DomainEleOp>::OpGradGrad<2>;
using OpDomainMass = OpTools<DomainEleOp>::OpMass;
using OpDomainSource = OpTools<DomainEleOp>::OpSource<2>;
using OpBoundaryMass = OpTools<EdgeEleOp>::OpMass;

struct Example {

  Example(MoFEM::Interface &m_field) : mField(m_field) {}

  MoFEMErrorCode runProblem();

private:
  MoFEM::Interface &mField;

  MoFEMErrorCode setUP();
  MoFEMErrorCode bC();
  MoFEMErrorCode OPs();
  MoFEMErrorCode kspSolve();
  MoFEMErrorCode postProcess();

  MatrixDouble invJac;
};

MoFEMErrorCode Example::runProblem() {
  MoFEMFunctionBegin;
  CHKERR setUP();
  CHKERR bC();
  CHKERR OPs();
  CHKERR kspSolve();
  CHKERR postProcess();
  MoFEMFunctionReturn(0);
}

//! [Set up problem]
MoFEMErrorCode Example::setUP() {
  MoFEMFunctionBegin;
  Simple *simple = mField.getInterface<Simple>();
  // Add field
  CHKERR simple->addDomainField("U_REAL", H1, AINSWORTH_BERNSTEIN_BEZIER_BASE,
                                1);
  CHKERR simple->addDomainField("U_IMAG", H1, AINSWORTH_BERNSTEIN_BEZIER_BASE,
                                1);
  CHKERR simple->addBoundaryField("U_REAL", H1, AINSWORTH_BERNSTEIN_BEZIER_BASE,
                                1);
  CHKERR simple->addBoundaryField("U_IMAG", H1, AINSWORTH_BERNSTEIN_BEZIER_BASE,
                                1);
  constexpr int order = 10;
  CHKERR simple->setFieldOrder("U_REAL", order);
  CHKERR simple->setFieldOrder("U_IMAG", order);
  CHKERR simple->setUp();
  MoFEMFunctionReturn(0);
}
//! [Set up problem]

//! [Applying essential BC]
MoFEMErrorCode Example::bC() {
  MoFEMFunctionBegin;
  MoFEMFunctionReturn(0);
}
//! [Applying essential BC]

//! [Push operators to pipeline]
MoFEMErrorCode Example::OPs() {
  MoFEMFunctionBegin;
  Basic *basic = mField.getInterface<Basic>();

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
    basic->getOpDomainLhsPipeline().push_back(
        new OpCalculateInvJacForFace(invJac));
    basic->getOpDomainLhsPipeline().push_back(new OpSetInvJacH1ForFace(invJac));
    basic->getOpDomainLhsPipeline().push_back(
        new OpDomainGradGrad("U_REAL", "U_REAL", beta));
    basic->getOpDomainLhsPipeline().push_back(
        new OpDomainGradGrad("U_IMAG", "U_IMAG", beta));

    basic->getOpDomainLhsPipeline().push_back(
        new OpDomainMass("U_REAL", "U_REAL", k2));
    basic->getOpDomainLhsPipeline().push_back(
        new OpDomainMass("U_IMAG", "U_IMAG", k2));

    basic->getOpDomainRhsPipeline().push_back(
        new OpDomainSource("U_REAL", vol_source_function));

    CHKERR basic->setDomainRhsIntegrationRule(integration_rule);
    CHKERR basic->setDomainLhsIntegrationRule(integration_rule);
    MoFEMFunctionReturn(0);
  };

  auto set_boundary = [&]() {
    MoFEMFunctionBegin;
    basic->getOpBoundaryLhsPipeline().push_back(
        new OpBoundaryMass("U_IMAG", "U_REAL", kp));
    basic->getOpBoundaryLhsPipeline().push_back(
        new OpBoundaryMass("U_REAL", "U_IMAG", km));
    CHKERR basic->setBoundaryLhsIntegrationRule(integration_rule);
    MoFEMFunctionReturn(0);
  };

  CHKERR set_domain();
  CHKERR set_boundary();

  MoFEMFunctionReturn(0);
}
//! [Push operators to pipeline]

//! [Solve]
MoFEMErrorCode Example::kspSolve() {
  MoFEMFunctionBegin;
  Simple *simple = mField.getInterface<Simple>();
  Basic *basic = mField.getInterface<Basic>();
  auto solver = basic->createKSP();
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

//! [Postprocess results]
MoFEMErrorCode Example::postProcess() {
  MoFEMFunctionBegin;
  Basic *basic = mField.getInterface<Basic>();
  basic->getDomainLhsFE().reset();
  basic->getDomainRhsFE().reset();
  basic->getBoundaryLhsFE().reset();
  basic->getBoundaryRhsFE().reset();
  auto post_proc_fe = boost::make_shared<PostProcFaceOnRefinedMesh>(mField);
  post_proc_fe->generateReferenceElementMesh();
  post_proc_fe->addFieldValuesPostProc("U_REAL");
  post_proc_fe->addFieldValuesPostProc("U_IMAG");
  basic->getDomainRhsFE() = post_proc_fe;
  CHKERR basic->loopFiniteElements();
  CHKERR post_proc_fe->writeFile("out_helmholtz.h5m");
  MoFEMFunctionReturn(0);
}
//! [Postprocess results]

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
    CHKERR simple->loadFile();
    //! [Load mesh]

    //! [Example]
    Example ex(m_field);
    CHKERR ex.runProblem();
    //! [Example]
  }
  CATCH_ERRORS;

  CHKERR MoFEM::Core::Finalize();
}
