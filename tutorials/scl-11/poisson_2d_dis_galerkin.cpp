/**
 * \file poisson_2d_dis_galerkin.cpp
 * \example poisson_2d_dis_galerkin.cpp
 *
 * Example of implementation for discontinuous Galerkin.
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

#include <BasicFiniteElements.hpp>

template <int DIM> struct ElementsAndOps {};

template <> struct ElementsAndOps<2> {
  using DomainEle = PipelineManager::FaceEle;
  using DomainEleOp = DomainEle::UserDataOperator;
  using BoundaryEle = PipelineManager::EdgeEle;
  using BoundaryEleOp = BoundaryEle::UserDataOperator;
  using PostProcEle = PostProcFaceOnRefinedMesh;

  using SkeletonEleOp = DomainEle;
  using FaceSideEle = MoFEM::FaceElementForcesAndSourcesCoreOnSideSwitch<0>;
  using FaceSideOp = FaceSideEle::UserDataOperator;
};

constexpr int BASE_DIM = 1;
constexpr int FIELD_DIM = 1;
constexpr int SPACE_DIM = 2;

using EntData = DataForcesAndSourcesCore::EntData;
using DomainEle = ElementsAndOps<SPACE_DIM>::DomainEle;
using DomainEleOp = ElementsAndOps<SPACE_DIM>::DomainEleOp;
using BoundaryEle = ElementsAndOps<SPACE_DIM>::BoundaryEle;
using BoundaryEleOp = ElementsAndOps<SPACE_DIM>::BoundaryEleOp;
using FaceSideEle = ElementsAndOps<SPACE_DIM>::FaceSideEle;
using SkeletonEleOp = ElementsAndOps<SPACE_DIM>::SkeletonEleOp;
using FaceSideOp = ElementsAndOps<SPACE_DIM>::FaceSideOp;
using PostProcEle = ElementsAndOps<SPACE_DIM>::PostProcEle;

static double penalty = 1e6;
static double phi =
    -1; // 1 - symmetric Nitsche, 0 - nonsymmetric, -1 antisymmetrica
static double nitsche = 1;

#include <PoissonDiscontinousGalerkin.hpp>

using OpDomainGradGrad = FormsIntegrators<DomainEleOp>::Assembly<
    PETSC>::BiLinearForm<GAUSS>::OpGradGrad<BASE_DIM, FIELD_DIM, SPACE_DIM>;
using OpDomainSource = FormsIntegrators<DomainEleOp>::Assembly<
    PETSC>::LinearForm<GAUSS>::OpSource<BASE_DIM, FIELD_DIM>;

auto u_exact = [](const double x, const double y, const double) {
  return x * x * y * y;
  // return cos(2 * x * M_PI) * cos(2 * y * M_PI);
};

auto u_grad_exact = [](const double x, const double y, const double) {
  return FTensor::Tensor1<double, 2>{2 * x * y * y, 2 * x * x * y};

  // return FTensor::Tensor1<double, 2>{

  //     -2 * M_PI * cos(2 * M_PI * y) * sin(2 * M_PI * x),
  //     -2 * M_PI * cos(2 * M_PI * x) * sin(2 * M_PI * y)

  // };
};

auto source = [](const double x, const double y, const double) {
  return -(2 * x * x + 2 * y * y);
  // return 8 * M_PI * M_PI * cos(2 * x * M_PI) * cos(2 * y * M_PI);
};

using namespace MoFEM;
using namespace Poisson2DiscontGalerkinOperators;

static char help[] = "...\n\n";

struct Poisson2DiscontGalerkin {
public:
  Poisson2DiscontGalerkin(MoFEM::Interface &m_field);

  // Declaration of the main function to run analysis
  MoFEMErrorCode runProgram();

private:
  // Declaration of other main functions called in runProgram()
  MoFEMErrorCode readMesh();
  MoFEMErrorCode setupProblem();
  MoFEMErrorCode boundaryCondition();
  MoFEMErrorCode assembleSystem();
  MoFEMErrorCode setIntegrationRules();
  MoFEMErrorCode solveSystem();
  MoFEMErrorCode checkResults();
  MoFEMErrorCode outputResults();

  // MoFEM interfaces
  MoFEM::Interface &mField;
  Simple *simpleInterface;

  // Field name and approximation order
  std::string domainField;
  int oRder;
};

Poisson2DiscontGalerkin::Poisson2DiscontGalerkin(MoFEM::Interface &m_field)
    : domainField("U"), mField(m_field), oRder(4) {}

//! [Read mesh]
MoFEMErrorCode Poisson2DiscontGalerkin::readMesh() {
  MoFEMFunctionBegin;

  CHKERR mField.getInterface(simpleInterface);
  CHKERR simpleInterface->getOptions();
  CHKERR simpleInterface->loadFile();

  MoFEMFunctionReturn(0);
}
//! [Read mesh]

//! [Setup problem]
MoFEMErrorCode Poisson2DiscontGalerkin::setupProblem() {
  MoFEMFunctionBegin;

  CHKERR PetscOptionsGetInt(PETSC_NULL, "", "-order", &oRder, PETSC_NULL);
  CHKERR PetscOptionsGetScalar(PETSC_NULL, "", "-penalty", &penalty,
                               PETSC_NULL);
  CHKERR PetscOptionsGetScalar(PETSC_NULL, "", "-phi", &phi, PETSC_NULL);
  CHKERR PetscOptionsGetScalar(PETSC_NULL, "", "-nitsche", &nitsche, PETSC_NULL);

  MOFEM_LOG("WORLD", Sev::inform) << "Set order: " << oRder;
  MOFEM_LOG("WORLD", Sev::inform) << "Set penalty: " << penalty;
  MOFEM_LOG("WORLD", Sev::inform) << "Set phi: " << phi;
  MOFEM_LOG("WORLD", Sev::inform) << "Set nitche: " << nitsche;

  CHKERR simpleInterface->addDomainField(domainField, L2,
                                         AINSWORTH_BERNSTEIN_BEZIER_BASE, 1);
  simpleInterface->getAddSkeletonFE() = true;
  simpleInterface->getAddBoundaryFE() = true;
  CHKERR simpleInterface->setFieldOrder(domainField, oRder);
  CHKERR simpleInterface->setUp();

  MoFEMFunctionReturn(0);
}
//! [Setup problem]

//! [Boundary condition]
MoFEMErrorCode Poisson2DiscontGalerkin::boundaryCondition() {
  MoFEMFunctionBegin;
  MoFEMFunctionReturn(0);
}

//! [Boundary condition]

//! [Assemble system]
MoFEMErrorCode Poisson2DiscontGalerkin::assembleSystem() {
  MoFEMFunctionBegin;

  auto pipeline_mng = mField.getInterface<PipelineManager>();

  auto add_base_ops = [&](auto &pipeline) {
    auto det_ptr = boost::make_shared<VectorDouble>();
    auto jac_ptr = boost::make_shared<MatrixDouble>();
    auto inv_jac_ptr = boost::make_shared<MatrixDouble>();
    pipeline.push_back(new OpCalculateHOJacForFace(jac_ptr));
    pipeline.push_back(new OpInvertMatrix<2>(jac_ptr, det_ptr, inv_jac_ptr));
    pipeline.push_back(new OpSetInvJacL2ForFace(inv_jac_ptr));
  };

  add_base_ops(pipeline_mng->getOpDomainLhsPipeline());
  pipeline_mng->getOpDomainLhsPipeline().push_back(new OpDomainGradGrad(
      domainField, domainField,
      [](const double, const double, const double) { return 1; }));
  pipeline_mng->getOpDomainRhsPipeline().push_back(
      new OpDomainSource(domainField, source));

  // Push operators to the Pipeline for Skeleton
  auto side_fe_ptr = boost::make_shared<FaceSideEle>(mField);
  add_base_ops(side_fe_ptr->getOpPtrVector());
  side_fe_ptr->getOpPtrVector().push_back(
      new OpCalculateSideData(domainField, domainField));

  // Push operators to the Pipeline for Skeleton
  pipeline_mng->getOpSkeletonLhsPipeline().push_back(
      new OpDomainLhsPenalty(side_fe_ptr));

  // Push operators to the Pipeline for Boundary
  pipeline_mng->getOpBoundaryLhsPipeline().push_back(
      new OpL2BoundaryLhs(side_fe_ptr));
  pipeline_mng->getOpBoundaryRhsPipeline().push_back(
      new OpL2BoundaryRhs(side_fe_ptr, u_exact));

  MoFEMFunctionReturn(0);
}
//! [Assemble system]

//! [Set integration rules]
MoFEMErrorCode Poisson2DiscontGalerkin::setIntegrationRules() {
  MoFEMFunctionBegin;

  auto rule_lhs = [](int, int, int p) -> int { return 2 * p; };
  auto rule_rhs = [](int, int, int p) -> int { return 2 * p; };
  auto rule_2 = [this](int, int, int) { return 2 * oRder; };

  auto pipeline_mng = mField.getInterface<PipelineManager>();
  CHKERR pipeline_mng->setDomainLhsIntegrationRule(rule_lhs);
  CHKERR pipeline_mng->setDomainRhsIntegrationRule(rule_rhs);

  CHKERR pipeline_mng->setSkeletonLhsIntegrationRule(rule_2);
  CHKERR pipeline_mng->setSkeletonRhsIntegrationRule(rule_2);
  CHKERR pipeline_mng->setBoundaryLhsIntegrationRule(rule_2);
  CHKERR pipeline_mng->setBoundaryRhsIntegrationRule(rule_2);

  MoFEMFunctionReturn(0);
}
//! [Set integration rules]

//! [Solve system]
MoFEMErrorCode Poisson2DiscontGalerkin::solveSystem() {
  MoFEMFunctionBegin;

  auto pipeline_mng = mField.getInterface<PipelineManager>();

  auto ksp_solver = pipeline_mng->createKSP();
  CHKERR KSPSetFromOptions(ksp_solver);
  CHKERR KSPSetUp(ksp_solver);

  // Create RHS and solution vectors
  auto dm = simpleInterface->getDM();
  auto F = smartCreateDMVector(dm);
  auto D = smartVectorDuplicate(F);

  CHKERR KSPSolve(ksp_solver, F, D);

  // Scatter result data on the mesh
  CHKERR VecGhostUpdateBegin(D, INSERT_VALUES, SCATTER_FORWARD);
  CHKERR VecGhostUpdateEnd(D, INSERT_VALUES, SCATTER_FORWARD);
  CHKERR DMoFEMMeshToLocalVector(dm, D, INSERT_VALUES, SCATTER_REVERSE);

  MoFEMFunctionReturn(0);
}
//! [Solve system]

MoFEMErrorCode Poisson2DiscontGalerkin::checkResults() {
  MoFEMFunctionBegin;

  auto pipeline_mng = mField.getInterface<PipelineManager>();
  pipeline_mng->getDomainRhsFE().reset();
  pipeline_mng->getDomainLhsFE().reset();
  pipeline_mng->getSkeletonRhsFE().reset();
  pipeline_mng->getSkeletonLhsFE().reset();
  pipeline_mng->getBoundaryRhsFE().reset();
  pipeline_mng->getBoundaryLhsFE().reset();

  auto rule = [](int, int, int p) -> int { return 2 * p; };
  CHKERR pipeline_mng->setDomainRhsIntegrationRule(rule);

  auto add_base_ops = [&](auto &pipeline) {
    auto det_ptr = boost::make_shared<VectorDouble>();
    auto jac_ptr = boost::make_shared<MatrixDouble>();
    auto inv_jac_ptr = boost::make_shared<MatrixDouble>();
    pipeline.push_back(new OpCalculateHOJacForFace(jac_ptr));
    pipeline.push_back(new OpInvertMatrix<2>(jac_ptr, det_ptr, inv_jac_ptr));
    pipeline.push_back(new OpSetInvJacL2ForFace(inv_jac_ptr));
  };

  auto u_vals_ptr = boost::make_shared<VectorDouble>();
  auto u_grad_ptr = boost::make_shared<MatrixDouble>();

  add_base_ops(pipeline_mng->getOpDomainRhsPipeline());
  pipeline_mng->getOpDomainRhsPipeline().push_back(
      new OpCalculateScalarFieldValues(domainField, u_vals_ptr));
  pipeline_mng->getOpDomainRhsPipeline().push_back(
      new OpCalculateScalarFieldGradient<2>(domainField, u_grad_ptr));

  enum NORMS { L2 = 0, SEMI_NORM, H1, LAST_NORM };
  std::array<int, LAST_NORM> error_indices;
  for (int i = 0; i != LAST_NORM; ++i)
    error_indices[i] = i;
  auto l2_vec = createSmartVectorMPI(
      mField.get_comm(), (!mField.get_comm_rank()) ? LAST_NORM : 0, LAST_NORM);

  auto error_op = new DomainEleOp(domainField, DomainEleOp::OPROW);
  error_op->doWorkRhsHook = [&](DataOperator *op_ptr, int side, EntityType type,
                                DataForcesAndSourcesCore::EntData &data) {
    MoFEMFunctionBegin;
    auto o = static_cast<DomainEleOp *>(op_ptr);

    FTensor::Index<'i', 2> i;

    if (const size_t nb_dofs = data.getIndices().size()) {

      const int nb_integration_pts = o->getGaussPts().size2();
      auto t_w = o->getFTensor0IntegrationWeight();
      auto t_val = getFTensor0FromVec(*u_vals_ptr);
      auto t_grad = getFTensor1FromMat<2>(*u_grad_ptr);
      auto t_coords = o->getFTensor1CoordsAtGaussPts();

      auto t_row_base = data.getFTensor0N();
      std::array<double, LAST_NORM> error;
      std::fill(error.begin(), error.end(), 0);

      for (int gg = 0; gg != nb_integration_pts; ++gg) {

        const double alpha = t_w * o->getMeasure();
        const double diff =
            t_val - u_exact(t_coords(0), t_coords(1), t_coords(2));

        auto t_exact_grad = u_grad_exact(t_coords(0), t_coords(1), t_coords(2));

        const double diff_grad =
            (t_grad(i) - t_exact_grad(i)) * (t_grad(i) - t_exact_grad(i));

        error[L2] += alpha * pow(diff, 2);
        error[SEMI_NORM] += alpha * diff_grad;
        error[H1] += error[L2] + error[SEMI_NORM];

        ++t_w;
        ++t_val;
        ++t_grad;
        ++t_coords;
      }

      CHKERR VecSetValues(l2_vec, LAST_NORM, error_indices.data(), error.data(),
                          ADD_VALUES);
    }

    MoFEMFunctionReturn(0);
  };

  pipeline_mng->getOpDomainRhsPipeline().push_back(error_op);

  CHKERR pipeline_mng->loopFiniteElements();

  CHKERR VecAssemblyBegin(l2_vec);
  CHKERR VecAssemblyEnd(l2_vec);

  if (mField.get_comm_rank() == 0) {
    const double *array;
    CHKERR VecGetArrayRead(l2_vec, &array);
    MOFEM_LOG_C("SELF", Sev::inform, "Error Norm L2 %6.4e",
                std::sqrt(array[L2]));
    MOFEM_LOG_C("SELF", Sev::inform, "Error Norm Energetic %6.4e",
                std::sqrt(array[SEMI_NORM]));
    MOFEM_LOG_C("SELF", Sev::inform, "Error Norm H1 %6.4e",
                std::sqrt(array[H1]));
    CHKERR VecRestoreArrayRead(l2_vec, &array);
    const MoFEM::Problem *problem_ptr;
    CHKERR DMMoFEMGetProblemPtr(simpleInterface->getDM(), &problem_ptr);
    MOFEM_LOG_C("SELF", Sev::inform, "Nb. DOFs %d",
                problem_ptr->getNbDofsRow());
  }

  MoFEMFunctionReturn(0);
}

//! [Output results]
MoFEMErrorCode Poisson2DiscontGalerkin::outputResults() {
  MoFEMFunctionBegin;

  auto pipeline_mng = mField.getInterface<PipelineManager>();
  pipeline_mng->getDomainLhsFE().reset();
  pipeline_mng->getSkeletonRhsFE().reset();
  pipeline_mng->getSkeletonLhsFE().reset();
  pipeline_mng->getBoundaryRhsFE().reset();
  pipeline_mng->getBoundaryLhsFE().reset();

  auto post_proc_fe = boost::make_shared<PostProcEle>(mField);
  post_proc_fe->generateReferenceElementMesh();
  post_proc_fe->addFieldValuesPostProc(domainField);
  pipeline_mng->getDomainRhsFE() = post_proc_fe;
  CHKERR pipeline_mng->loopFiniteElements();
  CHKERR post_proc_fe->writeFile("out_result.h5m");

  MoFEMFunctionReturn(0);
}
//! [Output results]

//! [Run program]
MoFEMErrorCode Poisson2DiscontGalerkin::runProgram() {
  MoFEMFunctionBegin;

  CHKERR readMesh();
  CHKERR setupProblem();
  CHKERR boundaryCondition();
  CHKERR assembleSystem();
  CHKERR setIntegrationRules();
  CHKERR solveSystem();
  CHKERR checkResults();
  CHKERR outputResults();

  MoFEMFunctionReturn(0);
}
//! [Run program]

//! [Main]
int main(int argc, char *argv[]) {

  // Initialisation of MoFEM/PETSc and MOAB data structures
  const char param_file[] = "param_file.petsc";
  MoFEM::Core::Initialize(&argc, &argv, param_file, help);

  // Error handling
  try {
    // Register MoFEM discrete manager in PETSc
    DMType dm_name = "DMMOFEM";
    CHKERR DMRegister_MoFEM(dm_name);

    // Create MOAB instance
    moab::Core mb_instance;              // mesh database
    moab::Interface &moab = mb_instance; // mesh database interface

    // Create MoFEM instance
    MoFEM::Core core(moab);           // finite element database
    MoFEM::Interface &m_field = core; // finite element interface

    // Run the main analysis
    Poisson2DiscontGalerkin poisson_problem(m_field);
    CHKERR poisson_problem.runProgram();
  }
  CATCH_ERRORS;

  // Finish work: cleaning memory, getting statistics, etc.
  MoFEM::Core::Finalize();

  return 0;
}
//! [Main]