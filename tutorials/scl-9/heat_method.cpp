/**
 * \file heat_method.cpp \example heat_method.cpp
 *
 * Calculating geodetic distances using heat method. For details see,
 *
 * K. Crane, C. Weischedel, M. Wardetzky, Geodesics in heat: A new approach to
 * computing distance based on heat flow, ACM Transactions on Graphics , vol.
 * 32, no. 5, pp. 152:1-152:11, 2013.
 *
 * Interent resources:
 * http://www.cs.cmu.edu/~kmcrane/Projects/HeatMethod/
 * http://www.numerical-tours.com/matlab/meshproc_7_geodesic_poisson/
 *
 *
 * Solving two problems in sequence. Show hot to use form integrators and how to
 * implement user data operator.
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

double dt = 2; // relative to edge length

#include <BasicFiniteElements.hpp>

using DomainEle = PipelineManager::FaceEle;
using DomainEleOp = DomainEle::UserDataOperator;
using EntData = DataForcesAndSourcesCore::EntData;

// Use forms iterators for Grad-Grad term
using OpGradGrad = FormsIntegrators<DomainEleOp>::Assembly<PETSC>::BiLinearForm<
    GAUSS>::OpGradGrad<1, 1, 3>;

// Use forms for Mass term
using OpMass = FormsIntegrators<DomainEleOp>::Assembly<PETSC>::BiLinearForm<
    GAUSS>::OpMass<1, 1>;

struct Example {

  Example(MoFEM::Interface &m_field) : mField(m_field) {}

  MoFEMErrorCode runProblem();

private:
  MoFEM::Interface &mField;
  Simple *simpleInterface;

  Range pinchNodes;

  MoFEMErrorCode readMesh();
  MoFEMErrorCode setupProblem();
  MoFEMErrorCode setIntegrationRules();
  MoFEMErrorCode createCommonData();
  MoFEMErrorCode boundaryCondition();
  MoFEMErrorCode assembleSystem();
  MoFEMErrorCode solveSystem();
  MoFEMErrorCode outputResults();
  MoFEMErrorCode checkResults();

  /**
   * Use problem specific implementation for second stage of heat methid
   */
  struct OpRhs : public DomainEleOp {

    OpRhs(boost::shared_ptr<MatrixDouble> u_grad_ptr);

    MoFEMErrorCode doWork(int side, EntityType type,
                          DataForcesAndSourcesCore::EntData &data);

  protected:
    boost::shared_ptr<MatrixDouble> uGradPtr;
  };
};

//! [Run programme]
MoFEMErrorCode Example::runProblem() {
  MoFEMFunctionBegin;
  CHKERR readMesh();
  CHKERR setupProblem();
  CHKERR createCommonData();
  CHKERR boundaryCondition();
  CHKERR assembleSystem();
  CHKERR setIntegrationRules();
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

  // FIXME: THis part of algorithm is not working in parallel. If you have bit
  // of free time, feel free to generalise code below.

  Range nodes;
  CHKERR mField.get_moab().get_entities_by_type(0, MBVERTEX, nodes);
  // pinchNodes could be get from BLOCKSET
  pinchNodes.insert(nodes[0]);

  Range edges;
  CHKERR mField.get_moab().get_adjacencies(pinchNodes, 1, false, edges,
                                           moab::Interface::UNION);
  double l2;
  for (auto e : edges) {
    Range nodes;
    CHKERR mField.get_moab().get_connectivity(Range(e, e), nodes, false);
    MatrixDouble coords(nodes.size(), 3);
    CHKERR mField.get_moab().get_coords(nodes, &coords(0, 0));
    double l2e = 0;
    for (int j = 0; j != 3; ++j) {
      l2e += pow(coords(0, j) - coords(1, j), 2);
    }
    l2 = std::max(l2, l2e);
  }

  dt *= std::sqrt(l2);

  MoFEMFunctionReturn(0);
}
//! [Read mesh]

//! [Set up problem]
MoFEMErrorCode Example::setupProblem() {
  MoFEMFunctionBegin;

  // Only one field
  CHKERR simpleInterface->addDomainField("U", H1, AINSWORTH_LEGENDRE_BASE, 1);
  constexpr int order = 1;
  CHKERR simpleInterface->setFieldOrder("U", order);
  CHKERR simpleInterface->setUp();

  MoFEMFunctionReturn(0);
}
//! [Set up problem]

//! [Set integration rule]
MoFEMErrorCode Example::setIntegrationRules() {
  MoFEMFunctionBegin;

  // Set integration order. To 2p is enough to integrate mass matrix exactly.
  auto rule = [](int, int, int p) -> int { return 2 * p; };

  // Set integration rule for elements assembling matrix and vector. Note that
  // rule for vector could be 2p-1, but that not change computation time
  // significantly. So shorter code is better.
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
MoFEMErrorCode Example::boundaryCondition() {
  MoFEMFunctionBegin;
  MoFEMFunctionReturn(0);
}
//! [Boundary condition]

//! [Push operators to pipeline]
MoFEMErrorCode Example::assembleSystem() {
  MoFEMFunctionBegin;
  PipelineManager *pipeline_mng = mField.getInterface<PipelineManager>();

  // This will store gradients at integration points on element
  auto grad_u_ptr = boost::make_shared<MatrixDouble>();

  // Push element from reference configuration to current configuration in 3d
  // space
  auto set_domain_general = [&](auto &pipeline) {
    auto det_ptr = boost::make_shared<VectorDouble>();
    auto jac_ptr = boost::make_shared<MatrixDouble>();
    auto inv_jac_ptr = boost::make_shared<MatrixDouble>();
    pipeline.push_back(new OpCalculateHOJacForFaceEmbeddedIn3DSpace(jac_ptr));
    pipeline.push_back(new OpInvertMatrix<3>(jac_ptr, det_ptr, inv_jac_ptr));
    pipeline.push_back(new OpSetInvJacH1ForFaceEmbeddedIn3DSpace(inv_jac_ptr));
  };

  // Operators for assembling matrix for first stage
  auto set_domain_lhs_first = [&](auto &pipeline) {
    auto one = [](double, double, double) -> double { return 1.; };
    pipeline.push_back(new OpGradGrad("U", "U", one));
    auto rho = [](double, double, double) -> double { return 1. / dt; };
    pipeline.push_back(new OpMass("U", "U", rho));
  };

  // Nothing is assembled for vector for first stage of heat method. Later
  // simply value of one is set to elements of right hand side vector at pinch
  // nodes.
  auto set_domain_rhs_first = [&](auto &pipeline) {};

  // Operators for assembly of left hand side. This time only Grad-Grad
  // operator.
  auto set_domain_lhs_second = [&](auto &pipeline) {
    auto one = [](double, double, double) { return 1.; };
    pipeline.push_back(new OpGradGrad("U", "U", one));
  };

  // Now apply on the right hand side vector X = Grad/||Grad||
  auto set_domain_rhs_second = [&](auto &pipeline) {
    pipeline.push_back(new OpCalculateScalarFieldGradient<3>("U", grad_u_ptr));
    pipeline.push_back(new OpRhs(grad_u_ptr));
  };

  // Solver first problem
  auto solve_first = [&]() {
    MoFEMFunctionBegin;
    auto simple = mField.getInterface<Simple>();
    auto pipeline_mng = mField.getInterface<PipelineManager>();

    auto solver = pipeline_mng->createKSP();
    CHKERR KSPSetFromOptions(solver);
    CHKERR KSPSetUp(solver);

    auto dm = simpleInterface->getDM();
    auto D = smartCreateDMVector(dm);
    auto F = smartVectorDuplicate(D);

    // Note add one at nodes which are pinch nodes
    CHKERR VecZeroEntries(F);
    auto problem_ptr = mField.get_problem(simple->getProblemName());
    auto bit_number = mField.get_field_bit_number("U");
    auto dofs_ptr = problem_ptr->getNumeredRowDofsPtr();
    for (auto v : pinchNodes) {
      const auto uid = DofEntity::getUniqueIdCalculate(
          0, FieldEntity::getLocalUniqueIdCalculate(bit_number, v));
      auto dof = dofs_ptr->get<Unique_mi_tag>().find(uid);
      if (dof != dofs_ptr->end())
        VecSetValue(F, (*dof)->getPetscGlobalDofIdx(), 1, INSERT_VALUES);
    }
    CHKERR VecAssemblyBegin(F);
    CHKERR VecAssemblyEnd(F);

    // Solve problem
    CHKERR KSPSolve(solver, F, D);
    CHKERR VecGhostUpdateBegin(D, INSERT_VALUES, SCATTER_FORWARD);
    CHKERR VecGhostUpdateEnd(D, INSERT_VALUES, SCATTER_FORWARD);
    CHKERR DMoFEMMeshToLocalVector(dm, D, INSERT_VALUES, SCATTER_REVERSE);
    MoFEMFunctionReturn(0);
  };

  // Solve second problem
  auto solve_second = [&]() {
    MoFEMFunctionBegin;
    auto simple = mField.getInterface<Simple>();

    // Note that DOFs on pinch nodes are removed from the problem
    auto prb_mng = mField.getInterface<ProblemsManager>();
    CHKERR prb_mng->removeDofsOnEntities(simple->getProblemName(), "U",
                                         pinchNodes, 0, 1);

    auto *pipeline_mng = mField.getInterface<PipelineManager>();

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
  };

  // Post-process results
  auto post_proc = [&](const std::string out_name) {
    PipelineManager *pipeline_mng = mField.getInterface<PipelineManager>();
    MoFEMFunctionBegin;
    auto tmp_lhs_fe = pipeline_mng->getDomainLhsFE();
    auto tmp_rhs_fe = pipeline_mng->getDomainRhsFE();
    auto det_ptr = boost::make_shared<VectorDouble>();
    auto jac_ptr = boost::make_shared<MatrixDouble>();
    auto inv_jac_ptr = boost::make_shared<MatrixDouble>();
    pipeline_mng->getDomainLhsFE().reset();
    pipeline_mng->getDomainRhsFE().reset();
    auto post_proc_fe =
        boost::make_shared<PostProcFaceOnRefinedMeshFor2D>(mField);
    post_proc_fe->generateReferenceElementMesh();
    post_proc_fe->getOpPtrVector().push_back(
        new OpCalculateHOJacForFaceEmbeddedIn3DSpace(jac_ptr));
    post_proc_fe->getOpPtrVector().push_back(
        new OpInvertMatrix<3>(jac_ptr, det_ptr, inv_jac_ptr));
    post_proc_fe->getOpPtrVector().push_back(
        new OpSetInvJacH1ForFaceEmbeddedIn3DSpace(inv_jac_ptr));
    post_proc_fe->addFieldValuesPostProc("U");
    post_proc_fe->addFieldValuesGradientPostProc("U");
    pipeline_mng->getDomainRhsFE() = post_proc_fe;
    CHKERR pipeline_mng->loopFiniteElements();
    CHKERR post_proc_fe->writeFile(out_name);
    tmp_lhs_fe = pipeline_mng->getDomainLhsFE() = tmp_lhs_fe;
    tmp_rhs_fe = pipeline_mng->getDomainRhsFE() = tmp_rhs_fe;
    MoFEMFunctionReturn(0);
  };

  set_domain_general(pipeline_mng->getOpDomainLhsPipeline());
  set_domain_general(pipeline_mng->getOpDomainRhsPipeline());
  set_domain_lhs_first(pipeline_mng->getOpDomainLhsPipeline());
  set_domain_rhs_first(pipeline_mng->getOpDomainRhsPipeline());

  CHKERR solve_first();
  CHKERR post_proc("out_heat_method_first.h5m");

  pipeline_mng->getOpDomainLhsPipeline().clear();
  pipeline_mng->getOpDomainRhsPipeline().clear();

  set_domain_general(pipeline_mng->getOpDomainLhsPipeline());
  set_domain_general(pipeline_mng->getOpDomainRhsPipeline());
  set_domain_lhs_second(pipeline_mng->getOpDomainLhsPipeline());
  set_domain_rhs_second(pipeline_mng->getOpDomainRhsPipeline());

  CHKERR solve_second();
  CHKERR post_proc("out_heat_method_second.h5m");

  MoFEMFunctionReturn(0);
};
//! [Push operators to pipeline]

//! [Solve]
MoFEMErrorCode Example::solveSystem() {
  MoFEMFunctionBegin;
  MoFEMFunctionReturn(0);
}
//! [Solve]

//! [Postprocess results]
MoFEMErrorCode Example::outputResults() {
  MoFEMFunctionBegin;
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

Example::OpRhs::OpRhs(boost::shared_ptr<MatrixDouble> u_grad_ptr)
    : DomainEleOp("U", DomainEleOp::OPROW), uGradPtr(u_grad_ptr) {}

MoFEMErrorCode Example::OpRhs::doWork(int side, EntityType type,
                                      DataForcesAndSourcesCore::EntData &data) {
  MoFEMFunctionBegin;

  FTensor::Index<'i', 3> i;

  auto nb_dofs = data.getIndices().size();
  if (nb_dofs) {

    auto t_grad = getFTensor1FromMat<3>(*uGradPtr);

    auto nb_base_functions = data.getN().size2();
    auto nb_gauss_pts = getGaussPts().size2();
    std::array<double, MAX_DOFS_ON_ENTITY> nf;
    std::fill(nf.begin(), &nf[nb_dofs], 0);

    auto t_diff_base = data.getFTensor1DiffN<3>();
    auto t_w = getFTensor0IntegrationWeight();
    auto a = getMeasure();

    for (size_t gg = 0; gg != nb_gauss_pts; ++gg) {
      double alpha = t_w * a;

      const auto l2 = t_grad(i) * t_grad(i);
      FTensor::Tensor1<double, 3> t_one;
      if (l2 > std::numeric_limits<double>::epsilon())
        t_one(i) = t_grad(i) / std::sqrt(l2);
      else
        t_one(i) = 0;

      size_t bb = 0;
      for (; bb != nb_dofs; ++bb) {
        nf[bb] -= alpha * t_diff_base(i) * t_one(i);
        ++t_diff_base;
      }

      for (; bb < nb_base_functions; ++bb) {
        ++t_diff_base;
      }

      ++t_grad;
    }

    CHKERR VecSetValues<MoFEM::EssentialBcStorage>(getKSPf(), data, &nf[0],
                                                   ADD_VALUES);
  }

  MoFEMFunctionReturn(0);
}
