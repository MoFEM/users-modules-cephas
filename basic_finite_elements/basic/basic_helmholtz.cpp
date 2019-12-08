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

using VolEle = FaceElementForcesAndSourcesCoreBase;
using VolEleOp = VolEle::UserDataOperator;
using EntData = DataForcesAndSourcesCore::EntData;

#include <BaseOps.hpp>

using OpVolGradGrad = OpTools<VolEleOp>::OpGradGrad<2>;
using OpVolMass = OpTools<VolEleOp>::OpMass;
using OpVolSource = OpTools<VolEleOp>::OpSource<2>;

struct Example {

  Example(MoFEM::Interface &m_field): mField(m_field) {}

  MoFEMErrorCode runProblem();

  private:

  MoFEM::Interface &mField;

  static double sourceFunction(const double x, const double y, const double z) {
    const double xc = 0;
    const double yc = 0;
    const double xs = x - xc;
    const double ys = y - yc;
    constexpr double eps = 50;
    return exp(-pow(eps * sqrt(xs * xs + ys * ys), 2));
  };

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
  constexpr int order = 1;
  CHKERR simple->setFieldOrder("U_REAL", order);
  CHKERR simple->setFieldOrder("U_IMAG", order);
  CHKERR simple->setUp();
  MoFEMFunctionReturn(0);
}
//! [Set up problem]

//! [Applying essential BC]
MoFEMErrorCode Example::bC() {
  MoFEMFunctionBegin;
  Simple *simple = mField.getInterface<Simple>();
  // Get skin on the body, i.e. body boundary, and apply homogenous
  // Dirichlet conditions on that boundary.
  Range surface;
  CHKERR mField.get_moab().get_entities_by_dimension(0, 2, surface, false);
  Skinner skin(&mField.get_moab());
  Range edges;
  CHKERR skin.find_skin(0, surface, false, edges);
  Range edges_part;
  ParallelComm *pcomm = ParallelComm::get_pcomm(&mField.get_moab(), MYPCOMM_INDEX);
  CHKERR pcomm->filter_pstatus(edges, PSTATUS_SHARED | PSTATUS_MULTISHARED,
                               PSTATUS_NOT, -1, &edges_part);
  Range edges_verts;
  CHKERR mField.get_moab().get_connectivity(edges_part, edges_verts, false);
  // Since Dirichlet b.c. are essential boundary conditions, remove DOFs
  // from the problem.
  CHKERR mField.getInterface<ProblemsManager>()->removeDofsOnEntities(
      simple->getProblemName(), "U", unite(edges_verts, edges_part));
  MoFEMFunctionReturn(0);
}
//! [Applying essential BC]


//! [Push operators to pipeline]
MoFEMErrorCode Example::OPs() {
  MoFEMFunctionBegin;
  Basic *basic = mField.getInterface<Basic>();
  basic->getOpDomainLhsPipeline().push_back(
      new OpCalculateInvJacForFace(invJac));
  basic->getOpDomainLhsPipeline().push_back(
      new OpSetInvJacH1ForFace(invJac));
  auto beta = [](const double, const double, const double) { return -1; };
  basic->getOpDomainLhsPipeline().push_back(new OpVolGradGrad("U", beta));
  auto k = [](const double, const double, const double) { return pow(50, 2); };
  basic->getOpDomainLhsPipeline().push_back(new OpVolMass("U", k));

  basic->getOpDomainRhsPipeline().push_back(
      new OpVolSource("U", sourceFunction));

  auto integration_rule = [](int, int, int p_data) { return 2 * p_data; };
  CHKERR basic->setDomainRhsIntegrationRule(integration_rule);
  CHKERR basic->setDomainLhsIntegrationRule(integration_rule);
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
  auto D = smartCreateDMDVector(dm);
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
    auto post_proc_fe = boost::make_shared<PostProcFaceOnRefinedMesh>(mField);
    post_proc_fe->generateReferenceElementMesh();
    post_proc_fe->addFieldValuesPostProc("U");
    basic->getDomainRhsFE() = post_proc_fe;
    CHKERR basic->loopFiniteElements();
    CHKERR post_proc_fe->writeFile(
        "out_basic_helmholtz.h5m");
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

