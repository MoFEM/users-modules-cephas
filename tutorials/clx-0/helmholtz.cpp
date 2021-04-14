/**
 * \file helmholtz.cpp
 * \example helmholtz.cpp
 *
 * Using PipelineManager interface calculate Helmholtz problem. Example show how
 * to manage complex variable fields.
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
using EdgeEle = EdgeElementForcesAndSourcesCoreBase;
using EdgeEleOp = EdgeEle::UserDataOperator;

using OpDomainGradGrad = FormsIntegrators<DomainEleOp>::Assembly<
    PETSC>::BiLinearForm<GAUSS>::OpGradGrad<1, 1, 2>;
using OpDomainMass = FormsIntegrators<DomainEleOp>::Assembly<
    PETSC>::BiLinearForm<GAUSS>::OpMass<1, 1>;
using OpBoundaryMass = FormsIntegrators<EdgeEleOp>::Assembly<
    PETSC>::BiLinearForm<GAUSS>::OpMass<1, 1>;
using OpBoundarySource = FormsIntegrators<EdgeEleOp>::Assembly<
    PETSC>::LinearForm<GAUSS>::OpSource<1, 1>;

struct Example {

  Example(MoFEM::Interface &m_field) : mField(m_field) {}

  MoFEMErrorCode runProblem();

private:
  MoFEM::Interface &mField;
  Simple *simpleInterface;
  boost::shared_ptr<std::vector<unsigned char>> boundaryMarker;

  MoFEMErrorCode readMesh();
  MoFEMErrorCode setupProblem();
  MoFEMErrorCode boundaryCondition();
  MoFEMErrorCode assembleSystem();
  MoFEMErrorCode solveSystem();
  MoFEMErrorCode outputResults();
  MoFEMErrorCode checkResults();

  MatrixDouble invJac;
};

//! [run problem]
MoFEMErrorCode Example::runProblem() {
  MoFEMFunctionBegin;
  CHKERR readMesh();
  CHKERR setupProblem();
  CHKERR boundaryCondition();
  CHKERR assembleSystem();
  CHKERR solveSystem();
  CHKERR outputResults();
  CHKERR checkResults();
  MoFEMFunctionReturn(0);
}
//! [run problem]

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
  CHKERR simpleInterface->addDomainField("P_REAL", H1,
                                         AINSWORTH_BERNSTEIN_BEZIER_BASE, 1);
  CHKERR simpleInterface->addDomainField("P_IMAG", H1,
                                         AINSWORTH_BERNSTEIN_BEZIER_BASE, 1);
  CHKERR simpleInterface->addBoundaryField("P_REAL", H1,
                                           AINSWORTH_BERNSTEIN_BEZIER_BASE, 1);
  CHKERR simpleInterface->addBoundaryField("P_IMAG", H1,
                                           AINSWORTH_BERNSTEIN_BEZIER_BASE, 1);
  int order = 6;
  CHKERR PetscOptionsGetInt(PETSC_NULL, "", "-order", &order, PETSC_NULL);
  CHKERR simpleInterface->setFieldOrder("P_REAL", order);
  CHKERR simpleInterface->setFieldOrder("P_IMAG", order);
  CHKERR simpleInterface->setUp();
  MoFEMFunctionReturn(0);
}
//! [Set up problem]

//! [Applying essential BC]
MoFEMErrorCode Example::boundaryCondition() {
  MoFEMFunctionBegin;

  auto get_ents_on_mesh_skin = [&]() {
    Range boundary_entities;
    for (_IT_CUBITMESHSETS_BY_SET_TYPE_FOR_LOOP_(mField, BLOCKSET, it)) {
      std::string entity_name = it->getName();
      if (entity_name.compare(0, 2, "BC") == 0) {
        CHKERR it->getMeshsetIdEntitiesByDimension(mField.get_moab(), 1,
                                                   boundary_entities, true);
      }
    }
    // Add vertices to boundary entities
    Range boundary_vertices;
    CHKERR mField.get_moab().get_connectivity(boundary_entities,
                                              boundary_vertices, true);
    boundary_entities.merge(boundary_vertices);

    return boundary_entities;
  };

  auto mark_boundary_dofs = [&](Range &&skin_edges) {
    auto problem_manager = mField.getInterface<ProblemsManager>();
    auto marker_ptr = boost::make_shared<std::vector<unsigned char>>();
    problem_manager->markDofs(simpleInterface->getProblemName(), ROW,
                              skin_edges, *marker_ptr);
    return marker_ptr;
  };

  auto remove_dofs_from_problem = [&](Range &&skin_edges) {
    MoFEMFunctionBegin;
    auto problem_manager = mField.getInterface<ProblemsManager>();
    CHKERR problem_manager->removeDofsOnEntities(
        simpleInterface->getProblemName(), "P_IMAG", skin_edges, 0, 1);
    MoFEMFunctionReturn(0);
  };

  // Get global local vector of marked DOFs. Is global, since is set for all
  // DOFs on processor. Is local since only DOFs on processor are in the
  // vector. To access DOFs use local indices.
  boundaryMarker = mark_boundary_dofs(get_ents_on_mesh_skin());
  CHKERR remove_dofs_from_problem(get_ents_on_mesh_skin());

  MoFEMFunctionReturn(0);
}
//! [Applying essential BC]

//! [Push operators to pipeline]
MoFEMErrorCode Example::assembleSystem() {
  MoFEMFunctionBegin;
  PipelineManager *pipeline_mng = mField.getInterface<PipelineManager>();

  double k = 90;
  CHKERR PetscOptionsGetScalar(PETSC_NULL, "", "-k", &k, PETSC_NULL);

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
        new OpSetBc("P_REAL", true, boundaryMarker));

    pipeline_mng->getOpDomainLhsPipeline().push_back(
        new OpDomainGradGrad("P_REAL", "P_REAL", beta));
    pipeline_mng->getOpDomainLhsPipeline().push_back(
        new OpDomainGradGrad("P_IMAG", "P_IMAG", beta));

    pipeline_mng->getOpDomainLhsPipeline().push_back(
        new OpDomainMass("P_REAL", "P_REAL", k2));
    pipeline_mng->getOpDomainLhsPipeline().push_back(
        new OpDomainMass("P_IMAG", "P_IMAG", k2));

    pipeline_mng->getOpDomainLhsPipeline().push_back(new OpUnSetBc("P_REAL"));

    CHKERR pipeline_mng->setDomainLhsIntegrationRule(integration_rule);
    MoFEMFunctionReturn(0);
  };

  auto set_boundary = [&]() {
    MoFEMFunctionBegin;
    pipeline_mng->getOpBoundaryLhsPipeline().push_back(
        new OpSetBc("P_REAL", true, boundaryMarker));
    pipeline_mng->getOpBoundaryLhsPipeline().push_back(
        new OpBoundaryMass("P_IMAG", "P_REAL", kp));
    pipeline_mng->getOpBoundaryLhsPipeline().push_back(
        new OpBoundaryMass("P_REAL", "P_IMAG", km));
    pipeline_mng->getOpBoundaryLhsPipeline().push_back(new OpUnSetBc("P_REAL"));

    pipeline_mng->getOpBoundaryLhsPipeline().push_back(
        new OpSetBc("P_REAL", false, boundaryMarker));
    pipeline_mng->getOpBoundaryLhsPipeline().push_back(
        new OpBoundaryMass("P_REAL", "P_REAL", beta));
    pipeline_mng->getOpBoundaryLhsPipeline().push_back(new OpUnSetBc("P_REAL"));

    pipeline_mng->getOpBoundaryRhsPipeline().push_back(
        new OpSetBc("P_REAL", false, boundaryMarker));
    pipeline_mng->getOpBoundaryRhsPipeline().push_back(
        new OpBoundarySource("P_REAL", beta));
    pipeline_mng->getOpBoundaryRhsPipeline().push_back(new OpUnSetBc("P_REAL"));

    CHKERR pipeline_mng->setBoundaryRhsIntegrationRule(integration_rule);
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
  post_proc_fe->addFieldValuesPostProc("P_REAL");
  post_proc_fe->addFieldValuesPostProc("P_IMAG");
  pipeline_mng->getDomainRhsFE() = post_proc_fe;
  CHKERR pipeline_mng->loopFiniteElements();
  CHKERR post_proc_fe->writeFile("out_helmholtz.h5m");
  MoFEMFunctionReturn(0);
}
//! [Postprocess results]

//! [Check results]
MoFEMErrorCode Example::checkResults() {
  MoFEMFunctionBegin;
  PipelineManager *pipeline_mng = mField.getInterface<PipelineManager>();

  auto dm = simpleInterface->getDM();
  auto D = smartCreateDMVector(dm);
  CHKERR DMoFEMMeshToLocalVector(dm, D, INSERT_VALUES, SCATTER_FORWARD);
  double nrm2;
  CHKERR VecNorm(D, NORM_2, &nrm2);
  MOFEM_LOG("WORLD", Sev::inform)
      << std::setprecision(9) << "Solution norm " << nrm2;

  PetscBool test_flg = PETSC_FALSE;
  CHKERR PetscOptionsGetBool(PETSC_NULL, "", "-test", &test_flg, PETSC_NULL);
  if (test_flg) {
    constexpr double regression_test = 97.261672;
    constexpr double eps = 1e-6;
    if (std::abs(nrm2 - regression_test) / regression_test > eps)
      SETERRQ(PETSC_COMM_SELF, MOFEM_DATA_INCONSISTENCY,
              "Not converged solution");
  }
  MoFEMFunctionReturn(0);
}
//! [Check results]

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
