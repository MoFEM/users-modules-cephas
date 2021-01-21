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

constexpr double dt = 1;

using DomainEle = MoFEM::FaceElementForcesAndSourcesCoreSwitch<
    FaceElementForcesAndSourcesCore::NO_HO_GEOMETRY |
    FaceElementForcesAndSourcesCore::NO_CONTRAVARIANT_TRANSFORM_HDIV |
    FaceElementForcesAndSourcesCore::NO_COVARIANT_TRANSFORM_HCURL>;
using DomainEleOp = DomainEle::UserDataOperator;
using EntData = DataForcesAndSourcesCore::EntData;

using OpHdivHdiv = FormsIntegrators<DomainEleOp>::Assembly<PETSC>::BiLinearForm<
    GAUSS>::OpMass<3, 1>;
using OpHdivU = FormsIntegrators<DomainEleOp>::Assembly<PETSC>::BiLinearForm<
    GAUSS>::OpMixDivTimesScalar<3>;
using OpMass = FormsIntegrators<DomainEleOp>::Assembly<PETSC>::BiLinearForm<
    GAUSS>::OpMass<1, 1>;

// using OpDomainSource = FormsIntegrators<DomainEleOp>::Assembly<
//     PETSC>::LinearForm<GAUSS>::OpSource<1, 1>;

struct Example {

  Example(MoFEM::Interface &m_field) : mField(m_field) {}

  MoFEMErrorCode runProblem();

private:
  MoFEM::Interface &mField;
  Simple *simpleInterface;

  MatrixDouble invJac;
  MatrixDouble jAC;

  MoFEMErrorCode readMesh();
  MoFEMErrorCode setupProblem();
  MoFEMErrorCode setIntegrationRules();
  MoFEMErrorCode createCommonData();
  MoFEMErrorCode boundaryCondition();
  MoFEMErrorCode assembleSystem(bool first);
  MoFEMErrorCode solveSystem(bool first);
  MoFEMErrorCode outputResults();
  MoFEMErrorCode checkResults();

  struct OpGrad : public DomainEleOp {
    OpGrad(boost::shared_ptr<MatrixDouble> grad_ptr);
    MoFEMErrorCode doWork(int side, EntityType type,
                          DataForcesAndSourcesCore::EntData &data);

  private:
    boost::shared_ptr<MatrixDouble> gradPtr;
  };

  struct OpRhs : public DomainEleOp {

    OpRhs(boost::shared_ptr<MatrixDouble> q_ptr,
          boost::shared_ptr<MatrixDouble> grad_ptr);

    MoFEMErrorCode doWork(int side, EntityType type,
                          DataForcesAndSourcesCore::EntData &data);

  private:
    boost::shared_ptr<MatrixDouble> qPtr;
    boost::shared_ptr<MatrixDouble> gradPtr;
  };
};

//! [Run programme]
MoFEMErrorCode Example::runProblem() {
  MoFEMFunctionBegin;
  CHKERR readMesh();
  CHKERR setupProblem();
  CHKERR setIntegrationRules();
  CHKERR createCommonData();
  CHKERR boundaryCondition();
  CHKERR assembleSystem(true);
  CHKERR solveSystem(true);
  CHKERR assembleSystem(false);
  CHKERR solveSystem(false);
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
  // We using AINSWORTH_LEGENDRE_BASE because DEMKOWICZ_JACOBI_BASE for triangle
  // and tet is not yet implemented for L2 space. DEMKOWICZ_JACOBI_BASE and
  // AINSWORTH_LEGENDRE_BASE are construcreed in the same way on quad.
  CHKERR simpleInterface->addDomainField("U", L2, AINSWORTH_LEGENDRE_BASE, 1);
  // Add field
  CHKERR simpleInterface->addDomainField("FLUX", HCURL, DEMKOWICZ_JACOBI_BASE,
                                         1);
  constexpr int order = 1;
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
MoFEMErrorCode Example::assembleSystem(bool first) {
  MoFEMFunctionBegin;
  PipelineManager *pipeline_mng = mField.getInterface<PipelineManager>();
  auto beta = [](const double, const double, const double) { return 1; };

  pipeline_mng->getDomainLhsFE().reset();
  pipeline_mng->getDomainRhsFE().reset();

  pipeline_mng->getOpDomainLhsPipeline().push_back(
      new OpCalculateJacForFaceEmbeddedIn3DSpace(jAC));
  pipeline_mng->getOpDomainLhsPipeline().push_back(
      new OpCalculateInvJacForFaceEmbeddedIn3DSpace(invJac));
  pipeline_mng->getOpDomainLhsPipeline().push_back(new OpMakeHdivFromHcurl());
  pipeline_mng->getOpDomainLhsPipeline().push_back(
      new OpSetContravariantPiolaTransformFaceEmbeddedIn3DSpace(jAC));
  pipeline_mng->getOpDomainLhsPipeline().push_back(
      new OpSetInvJacHcurlFaceEmbeddedIn3DSpace(invJac));
  pipeline_mng->getOpDomainLhsPipeline().push_back(
      new OpHdivHdiv("FLUX", "FLUX", beta));
  pipeline_mng->getOpDomainLhsPipeline().push_back(
      new OpHdivU("FLUX", "U", -1, true));

  if (first) {
    auto rho = [](const double, const double, const double) {
      return -1. / dt;
    };
    pipeline_mng->getOpDomainLhsPipeline().push_back(new OpMass("U", "U", rho));

  } else {

    pipeline_mng->getOpDomainRhsPipeline().push_back(
        new OpCalculateJacForFaceEmbeddedIn3DSpace(jAC));
    pipeline_mng->getOpDomainRhsPipeline().push_back(
        new OpCalculateInvJacForFaceEmbeddedIn3DSpace(invJac));
    pipeline_mng->getOpDomainRhsPipeline().push_back(new OpMakeHdivFromHcurl());
    pipeline_mng->getOpDomainRhsPipeline().push_back(
        new OpSetContravariantPiolaTransformFaceEmbeddedIn3DSpace(jAC));
    pipeline_mng->getOpDomainRhsPipeline().push_back(
        new OpSetInvJacHcurlFaceEmbeddedIn3DSpace(invJac));

    auto q_ptr = boost::make_shared<MatrixDouble>();
    auto grad_ptr = boost::make_shared<MatrixDouble>();
    pipeline_mng->getOpDomainRhsPipeline().push_back(
        new OpCalculateHdivVectorField<3>("FLUX", q_ptr));
    pipeline_mng->getOpDomainRhsPipeline().push_back(new OpGrad(grad_ptr));
    pipeline_mng->getOpDomainRhsPipeline().push_back(
        new OpRhs(q_ptr, grad_ptr));
  }

  MoFEMFunctionReturn(0);
}
//! [Push operators to pipeline]

//! [Solve]
MoFEMErrorCode Example::solveSystem(bool first) {
  MoFEMFunctionBegin;

  auto *pipeline_mng = mField.getInterface<PipelineManager>();
  auto solver = pipeline_mng->createKSP();
  CHKERR KSPSetFromOptions(solver);
  CHKERR KSPSetUp(solver);

  auto dm = simpleInterface->getDM();
  auto D = smartCreateDMVector(dm);
  auto F = smartVectorDuplicate(D);
  CHKERR VecZeroEntries(D);
  CHKERR VecZeroEntries(F);

  if (first) {
    CHKERR DMKSPSetComputeRHS(dm, PETSC_NULL, PETSC_NULL);
    // Here set value to first element, that depends on node numeration, and
    // other things. Smarter way would be apply some source term which
    // approximate singularity on some point on the surface.
    CHKERR VecSetValue(F, 0, -1, INSERT_VALUES);
    CHKERR VecAssemblyBegin(F);
    CHKERR VecAssemblyEnd(F);

    CHKERR KSPSolve(solver, F, D);
    CHKERR VecGhostUpdateBegin(D, INSERT_VALUES, SCATTER_FORWARD);
    CHKERR VecGhostUpdateEnd(D, INSERT_VALUES, SCATTER_FORWARD);
    CHKERR DMoFEMMeshToLocalVector(dm, D, INSERT_VALUES, SCATTER_REVERSE);
  } else {
    CHKERR KSPSolve(solver, F, D);
    CHKERR VecGhostUpdateBegin(D, INSERT_VALUES, SCATTER_FORWARD);
    CHKERR VecGhostUpdateEnd(D, INSERT_VALUES, SCATTER_FORWARD);
    CHKERR DMoFEMMeshToLocalVector(dm, D, INSERT_VALUES, SCATTER_REVERSE);
  }

  double d_nrm;
  CHKERR VecNorm(D, NORM_2, &d_nrm);
  MOFEM_LOG("EXAMPLE", Sev::inform) << "Solution norm " << d_nrm;

  MoFEMFunctionReturn(0);
}
//! [Solve]

//! [Postprocess results]
MoFEMErrorCode Example::outputResults() {
  MoFEMFunctionBegin;
  PipelineManager *pipeline_mng = mField.getInterface<PipelineManager>();
  pipeline_mng->getDomainLhsFE().reset();
  auto post_proc_fe =
      boost::make_shared<PostProcFaceOnRefinedMeshFor2D>(mField);
  post_proc_fe->generateReferenceElementMesh();

  post_proc_fe->getOpPtrVector().push_back(
      new OpCalculateJacForFaceEmbeddedIn3DSpace(jAC));
  post_proc_fe->getOpPtrVector().push_back(new OpMakeHdivFromHcurl());
  post_proc_fe->getOpPtrVector().push_back(
      new OpSetContravariantPiolaTransformFaceEmbeddedIn3DSpace(jAC));

  post_proc_fe->addFieldValuesPostProc("FLUX");
  post_proc_fe->addFieldValuesPostProc("U");
  pipeline_mng->getDomainRhsFE() = post_proc_fe;
  CHKERR pipeline_mng->loopFiniteElements();
  CHKERR post_proc_fe->writeFile("out_heat_method.h5m");
  MoFEMFunctionReturn(0);
}
//! [Postprocess results]

//! [Check results]
MoFEMErrorCode Example::checkResults() {
  MoFEMFunctionBegin;
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

Example::OpGrad::OpGrad(boost::shared_ptr<MatrixDouble> grad_ptr)
    : DomainEleOp("FLUX", DomainEleOp::OPROW), gradPtr(grad_ptr) {}

MoFEMErrorCode
Example::OpGrad::doWork(int side, EntityType type,
                        DataForcesAndSourcesCore::EntData &data) {
  MoFEMFunctionBegin;

  const auto nb_gauss_pts = getGaussPts().size2();
  if (side == 0 && type == MBEDGE) {
    gradPtr->resize(9, nb_gauss_pts, false);
    gradPtr->clear();
  }

  FTensor::Index<'i', 3> i;
  FTensor::Index<'j', 3> j;

  const auto nb_dofs = data.getFieldData().size();
  if (nb_dofs) {

    auto t_grad = getFTensor2FromMat<3, 3>(*gradPtr);
    auto t_diff_base = data.getFTensor2DiffN<3, 3>();
    auto t_w = getFTensor0IntegrationWeight();

    for (size_t gg = 0; gg != nb_gauss_pts; ++gg) {
      auto t_data = data.getFTensor0FieldData();
      for (size_t bb = 0; bb != nb_dofs; ++bb) {
        t_grad(i, j) += t_data * t_diff_base(i, j);
        ++t_data;
        ++t_diff_base;
      }
      ++t_grad;
    }
  }

  MoFEMFunctionReturn(0);
}

Example::OpRhs::OpRhs(boost::shared_ptr<MatrixDouble> q_ptr,
                      boost::shared_ptr<MatrixDouble> grad_ptr)
    : DomainEleOp("U", DomainEleOp::OPROW), qPtr(q_ptr), gradPtr(grad_ptr) {}

MoFEMErrorCode Example::OpRhs::doWork(int side, EntityType type,
                                      DataForcesAndSourcesCore::EntData &data) {
  MoFEMFunctionBegin;

  FTensor::Index<'i', 3> i;
  FTensor::Index<'j', 3> j;

  auto nb_dofs = data.getFieldData().size();
  if (nb_dofs) {

    auto t_q = getFTensor1FromMat<3>(*qPtr);
    auto t_grad = getFTensor2FromMat<3, 3>(*gradPtr);

    auto nb_gauss_pts = getGaussPts().size2();
    std::array<double, MAX_DOFS_ON_ENTITY> nf;
    std::fill(nf.begin(), nf.end(), 0);

    auto t_base = data.getFTensor0N();
    auto t_w = getFTensor0IntegrationWeight();
    auto a = getMeasure();

    for (size_t gg = 0; gg != nb_gauss_pts; ++gg) {
      double alpha = t_w * a;

      const double l2 = t_q(i) * t_q(i);
      const double l = sqrt(l2);

      const double div =
          t_grad(i, i) / l - (t_q(i) * (t_grad(j, i) * t_q(j)) / pow(l, 3));

      for (size_t bb = 0; bb != nb_dofs; ++bb) {
        nf[bb] += alpha * div;
        ++t_base;
      }

      ++t_w;
      ++t_q;
      ++t_grad;
    }

    CHKERR VecSetValues(getKSPf(), data, &nf[0], ADD_VALUES);
  }

  MoFEMFunctionReturn(0);
}
