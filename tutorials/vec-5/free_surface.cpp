/**
 * \file approximation.cpp
 * \FreeSurface approximation.cpp
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

constexpr int BASE_DIM = 1;
constexpr int SPACE_DIM = 2;
constexpr int U_FIELD_DIM = SPACE_DIM;
constexpr int H_FIELD_DIM = 1;

template <int DIM> struct ElementsAndOps {};

template <> struct ElementsAndOps<2> {
  using DomainEle = PipelineManager::FaceEle;
  using DomainEleOp = DomainEle::UserDataOperator;
  using PostProcEle = PostProcFaceOnRefinedMesh;
};

template <> struct ElementsAndOps<3> {
  using DomainEle = VolumeElementForcesAndSourcesCore;
  using DomainEleOp = DomainEle::UserDataOperator;
  using PostProcEle = PostProcVolumeOnRefinedMesh;
};

using DomainEle = ElementsAndOps<SPACE_DIM>::DomainEle;
using DomainEleOp = DomainEle::UserDataOperator;
using EntData = DataForcesAndSourcesCore::EntData;

using PostProcEle = ElementsAndOps<SPACE_DIM>::PostProcEle;

using AssemblyDomainEleOp =
    FormsIntegrators<DomainEleOp>::Assembly<PETSC>::OpBase;

using OpDomainMassU = FormsIntegrators<DomainEleOp>::Assembly<
    PETSC>::BiLinearForm<GAUSS>::OpMass<BASE_DIM, U_FIELD_DIM>;
using OpDomainSourceU = FormsIntegrators<DomainEleOp>::Assembly<
    PETSC>::LinearForm<GAUSS>::OpSource<BASE_DIM, U_FIELD_DIM>;
using OpDomainSourceH = FormsIntegrators<DomainEleOp>::Assembly<
    PETSC>::LinearForm<GAUSS>::OpSource<BASE_DIM, H_FIELD_DIM>;

using OpConvectiveU =
    FormsIntegrators<DomainEleOp>::Assembly<PETSC>::LinearForm<
        GAUSS>::OpConvectiveTermRhs<BASE_DIM, U_FIELD_DIM, SPACE_DIM>;
using OpConvectiveU_dU =
    FormsIntegrators<DomainEleOp>::Assembly<PETSC>::BiLinearForm<
        GAUSS>::OpConvectiveTermLhsDu<BASE_DIM, U_FIELD_DIM, SPACE_DIM>;
using OpConvectiveU_dGradU =
    FormsIntegrators<DomainEleOp>::Assembly<PETSC>::BiLinearForm<
        GAUSS>::OpConvectiveTermLhsDy<BASE_DIM, U_FIELD_DIM, SPACE_DIM>;

using OpDomainMassH = FormsIntegrators<DomainEleOp>::Assembly<
    PETSC>::BiLinearForm<GAUSS>::OpMass<BASE_DIM, H_FIELD_DIM>;
using OpDomainSourceJ = FormsIntegrators<DomainEleOp>::Assembly<
    PETSC>::LinearForm<GAUSS>::OpSource<BASE_DIM, H_FIELD_DIM>;

using OpDomainGradHTimesU =
    FormsIntegrators<DomainEleOp>::Assembly<PETSC>::LinearForm<
        GAUSS>::OpGradTimesTensor<BASE_DIM, H_FIELD_DIM, SPACE_DIM>;
using OpDinianMixVectorTimesGradH =
    FormsIntegrators<DomainEleOp>::Assembly<PETSC>::BiLinearForm<
        GAUSS>::OpMixVectorTimesGrad<BASE_DIM, H_FIELD_DIM, SPACE_DIM>;
using OpDomainGradHTimesGardH = OpDomainGradHTimesU;
using OpDinianGradHGradH = FormsIntegrators<DomainEleOp>::Assembly<
    PETSC>::BiLinearForm<GAUSS>::OpGradGrad<BASE_DIM, H_FIELD_DIM, SPACE_DIM>;

FTensor::Index<'i', SPACE_DIM> i;
FTensor::Index<'j', SPACE_DIM> j;
FTensor::Index<'k', SPACE_DIM> k;

constexpr double Re_p = 1e1;
constexpr double Re_m = 1e-1;
constexpr double Ri_p = 1e1;
constexpr double Ri_m = 1e-1;
constexpr double lambda = 1;

/**
 * @brief Rhs for U
 *
 */
struct OpRhsU : public AssemblyDomainEleOp {

  OpRhsU(const std::string field_name,
         boost::shared_ptr<MatrixDouble> dot_u_ptr,
         boost::shared_ptr<MatrixDouble> u_ptr,
         boost::shared_ptr<MatrixDouble> grad_u_ptr,
         boost::shared_ptr<VectorDouble> phi_ptr,
         boost::shared_ptr<MatrixDouble> grad_phi_ptr)
      : AssemblyDomainEleOp(field_name, field_name, AssemblyDomainEleOp::OPROW),
        dotUPtr(dot_u_ptr), uPtr(u_ptr), gradUPtr(grad_u_ptr), phiPtr(phi_ptr),
        gradPhiPtr(grad_phi_ptr) {}

  MoFEMErrorCode iNtegrate(DataForcesAndSourcesCore::EntData &data) {
    MoFEMFunctionBegin;

    const double vol = getMeasure();
    auto t_dot_u = getFTensor1FromMat<U_FIELD_DIM>(*dotUPtr);
    auto t_u = getFTensor1FromMat<U_FIELD_DIM>(*uPtr);
    auto t_grad_u = getFTensor2FromMat<U_FIELD_DIM, SPACE_DIM>(*gradUPtr);
    auto t_phi = getFTensor0FromVec(*phiPtr);
    auto t_grad_phi = getFTensor1FromMat<SPACE_DIM>(*dotUPtr);

    auto t_base = data.getFTensor0N();
    auto t_diff_base = data.getFTensor1DiffN<SPACE_DIM>();

    auto t_w = getFTensor0IntegrationWeight();

    for (int gg = 0; gg != nbIntegrationPts; gg++) {

      const double alpha = t_w * vol;

      const double Re = 0.5 * ((1 + t_phi) * Re_p + (1 - t_phi) * Re_m);
      const double buoyancy =
          t_phi * (0.5 * ((1 + t_phi) * Ri_p + (1 - t_phi) * Ri_m));

      FTensor::Tensor2_symmetric<double, SPACE_DIM> t_stress;
      t_stress(i, j) = (1 / Re) * (t_grad_u(i, j) || t_grad_u(j, i));
      FTensor::Tensor2_symmetric<double, SPACE_DIM> t_tension;
      t_tension(i, j) = (lambda * t_grad_phi(i)) ^ t_grad_phi(j);

      FTensor::Tensor1<double, SPACE_DIM> t_convection;
      t_convection(i) = t_u(j) * t_grad_u(j, i);
      FTensor::Tensor1<double, SPACE_DIM> t_buoyancy;
      t_buoyancy(i) = 0;
      t_buoyancy(SPACE_DIM - 1) = buoyancy;

      auto t_nf = getFTensor1FromArray<U_FIELD_DIM, U_FIELD_DIM>(locF);

      int bb = 0;
      for (; bb != nbRows / U_FIELD_DIM; ++bb) {

        t_nf(i) +=
            (t_base * alpha) * (t_dot_u(i) + t_convection(i) - t_buoyancy(i));
        t_nf(i) +=
            (t_diff_base(i) * alpha) * (t_stress(i, j) + t_tension(i, j));

        ++t_base;
        ++t_diff_base;
        ++t_nf;
      }

      for (; bb < nbRowBaseFunctions; ++bb) {
        ++t_diff_base;
        ++t_base;
      }

      ++t_dot_u;
      ++t_u;
      ++t_grad_u;
      ++t_phi;
      ++t_grad_phi;

      ++t_base;
      ++t_diff_base;

      ++t_w;
    }

    MoFEMFunctionReturn(0);
  }

private:
  boost::shared_ptr<MatrixDouble> dotUPtr;
  boost::shared_ptr<MatrixDouble> uPtr;
  boost::shared_ptr<MatrixDouble> gradUPtr;
  boost::shared_ptr<VectorDouble> phiPtr;
  boost::shared_ptr<MatrixDouble> gradPhiPtr;
};

/**
 * @brief Explict term for IMEX method
 *
 */
struct OpRhsExplicitTermH : public AssemblyDomainEleOp {

  OpRhsExplicitTermH(const std::string field_name,
                     boost::shared_ptr<MatrixDouble> u_ptr,
                     boost::shared_ptr<MatrixDouble> grad_phi_ptr,
                     boost::shared_ptr<VectorDouble> f_ptr,
                     boost::shared_ptr<double> ksi_ptr)
      : AssemblyDomainEleOp(field_name, field_name, AssemblyDomainEleOp::OPROW),
        fPtr(f_ptr), ksiPtr(ksi_ptr) {}

  MoFEMErrorCode iNtegrate(DataForcesAndSourcesCore::EntData &data) {
    MoFEMFunctionBegin;

    const double vol = getMeasure();
    auto t_w = getFTensor0IntegrationWeight();
    auto t_base = data.getFTensor0N();
    auto t_f = getFTensor0FromVec(*fPtr);
    const double ksi = *ksiPtr;

    for (int gg = 0; gg != nbIntegrationPts; gg++) {

      const double alpha = t_w * vol;
      auto nf_ptr = &locF[0];

      int rr = 0;
      for (; rr != nbRows; ++rr) {

        (*nf_ptr) += (alpha * lambda) * (-t_f + ksi) * t_base;
        (*nf_ptr) += (alpha * lambda) * (-t_f + ksi) * t_base;
        (*nf_ptr) -= alpha * (t_u(i) * t_grad_phi(i)) * t_base;

        ++nf_ptr;
        ++t_base;
      }

      ++t_f;
      ++t_u;
      ++t_grad_phi;

      ++t_w;
    }

    MoFEMFunctionReturn(0);
  }

private:
  boost::shared_ptr<MatrixDouble> uPtr;
  boost::shared_ptr<MatrixDouble> gradPhiPtr;
  boost::shared_ptr<VectorDouble> fPtr;
  boost::shared_ptr<double> ksiPtr;
};

struct OpCalculateKsi : public DomainEleOp {

  OpCalculateKsi(const std::string field_name,
                 boost::shared_ptr<VectorDouble> phi_ptr,
                 SmartPetscObj<Vec> ksi_vec)
      : DomainEleOp(field_name, DomainEleOp::OPROW), phiPtr(phi_ptr),
        ksiVec(ksi_vec, true) {
    doEntities[MBVERTEX] = true;
    std::fill(&doEntities[MBEDGE], &doEntities[MBMAXTYPE], false);
  }

  MoFEMErrorCode doWork(int side, EntityType type,
                        DataForcesAndSourcesCore::EntData &data) {
    MoFEMFunctionBegin;

    const double vol = getMeasure();
    auto t_w = getFTensor0IntegrationWeight();
    auto t_row_base = data.getFTensor0N();
    auto t_phi = getFTensor0FromVec(*phiPtr);
    auto nb_gauss_pts = getGaussPts().size2();

    double ksi = 0;
    for (int gg = 0; gg != nb_gauss_pts; gg++) {
      const double alpha = t_w * vol;
      ksi += alpha * t_phi;
      ++t_phi;
    }

    CHKERR VecSetValue(ksiVec, 0, ksi, ADD_VALUES);

    MoFEMFunctionReturn(0);
  }

private:
  SmartPetscObj<Vec> ksiVec;
  boost::shared_ptr<VectorDouble> phiPtr;
};

struct FreeSurface {

  FreeSurface(MoFEM::Interface &m_field) : mField(m_field) {}

  MoFEMErrorCode runProblem();

private:
  MoFEMErrorCode readMesh();
  MoFEMErrorCode setupProblem();
  MoFEMErrorCode boundaryCondition();
  MoFEMErrorCode assembleSystem();
  MoFEMErrorCode solveSystem();

  MoFEM::Interface &mField;
};

//! [Run programme]
MoFEMErrorCode FreeSurface::runProblem() {
  MoFEMFunctionBegin;
  CHKERR readMesh();
  CHKERR setupProblem();
  CHKERR boundaryCondition();
  CHKERR assembleSystem();
  CHKERR solveSystem();
  MoFEMFunctionReturn(0);
}
//! [Run programme]

//! [Read mesh]
MoFEMErrorCode FreeSurface::readMesh() {
  MoFEMFunctionBegin;

  auto simple = mField.getInterface<Simple>();
  CHKERR simple->getOptions();
  CHKERR simple->loadFile();

  MoFEMFunctionReturn(0);
}
//! [Read mesh]

//! [Set up problem]
MoFEMErrorCode FreeSurface::setupProblem() {
  MoFEMFunctionBegin;

  auto simple = mField.getInterface<Simple>();
  CHKERR simple->addDomainField("U", H1, AINSWORTH_LEGENDRE_BASE, U_FIELD_DIM);
  CHKERR simple->addDomainField("H", H1, AINSWORTH_LEGENDRE_BASE, 1);

  constexpr int order = 2;
  CHKERR simple->setFieldOrder("U", order);
  CHKERR simple->setFieldOrder("H", order);
  CHKERR simple->setUp();

  MoFEMFunctionReturn(0);
}
//! [Set up problem]

//! [Boundary condition]
MoFEMErrorCode FreeSurface::boundaryCondition() {
  MoFEMFunctionBegin;

  auto init_h = [](double, double y, double) { return tanh((y - 0.5) * 10); };

  auto set_domain_general = [&](auto &pipeline) {
    pipeline.push_back(new OpSetHOWeigthsOnFace());
  };

  auto set_domain_rhs = [&](auto &pipeline) {
    pipeline.push_back(new OpDomainSourceH("H", init_h));
  };

  auto set_domain_lhs = [&](auto &pipeline) {
    pipeline.push_back(
        new OpDomainMassU("U", "U", [](double, double, double) { return 1; }));
    pipeline.push_back(
        new OpDomainMassH("H", "H", [](double, double, double) { return 1; }));
  };

  auto post_proc = [&]() {
    MoFEMFunctionBegin;
    auto simple = mField.getInterface<Simple>();
    auto dm = simple->getDM();

    auto post_proc_fe = boost::make_shared<PostProcEle>(mField);
    post_proc_fe->generateReferenceElementMesh();

    auto det_ptr = boost::make_shared<VectorDouble>();
    auto jac_ptr = boost::make_shared<MatrixDouble>();
    auto inv_jac_ptr = boost::make_shared<MatrixDouble>();

    post_proc_fe->addFieldValuesPostProc("U");
    post_proc_fe->addFieldValuesPostProc("H");

    CHKERR DMoFEMLoopFiniteElements(dm, "dFE", post_proc_fe);
    CHKERR post_proc_fe->writeFile("out_init.h5m");

    MoFEMFunctionReturn(0);
  };

  auto solve_init = [&]() {
    MoFEMFunctionBegin;
    auto simple = mField.getInterface<Simple>();
    auto pipeline_mng = mField.getInterface<PipelineManager>();

    auto solver = pipeline_mng->createKSP();
    CHKERR KSPSetFromOptions(solver);
    PC pc;
    CHKERR KSPGetPC(solver, &pc);
    PetscBool is_pcfs = PETSC_FALSE;
    PetscObjectTypeCompare((PetscObject)pc, PCFIELDSPLIT, &is_pcfs);
    if (is_pcfs == PETSC_TRUE) {
      auto bc_mng = mField.getInterface<BcManager>();
      auto name_prb = simple->getProblemName();
      SmartPetscObj<IS> is_u;
      CHKERR mField.getInterface<ISManager>()->isCreateProblemFieldAndRank(
          name_prb, ROW, "U", 0, 3, is_u);
      CHKERR PCFieldSplitSetIS(pc, PETSC_NULL, is_u);
      CHKERR PCFieldSplitSetType(pc, PC_COMPOSITE_ADDITIVE);
    }

    CHKERR KSPSetUp(solver);

    auto dm = simple->getDM();
    auto D = smartCreateDMVector(dm);
    auto F = smartVectorDuplicate(D);

    CHKERR KSPSolve(solver, F, D);
    CHKERR VecGhostUpdateBegin(D, INSERT_VALUES, SCATTER_FORWARD);
    CHKERR VecGhostUpdateEnd(D, INSERT_VALUES, SCATTER_FORWARD);
    CHKERR DMoFEMMeshToLocalVector(dm, D, INSERT_VALUES, SCATTER_REVERSE);

    MoFEMFunctionReturn(0);
  };

  auto pipeline_mng = mField.getInterface<PipelineManager>();

  auto integration_rule = [](int, int, int approx_order) {
    return 2 * approx_order + 4;
  };
  CHKERR pipeline_mng->setDomainRhsIntegrationRule(integration_rule);
  CHKERR pipeline_mng->setDomainLhsIntegrationRule(integration_rule);

  set_domain_general(pipeline_mng->getOpDomainRhsPipeline());
  set_domain_rhs(pipeline_mng->getOpDomainRhsPipeline());
  set_domain_general(pipeline_mng->getOpDomainLhsPipeline());
  set_domain_lhs(pipeline_mng->getOpDomainLhsPipeline());

  CHKERR solve_init();
  CHKERR post_proc();

  // Clear pipelines
  pipeline_mng->getOpDomainRhsPipeline().clear();
  pipeline_mng->getOpDomainLhsPipeline().clear();

  MoFEMFunctionReturn(0);
}
//! [Boundary condition]

//! [Push operators to pipeline]
MoFEMErrorCode FreeSurface::assembleSystem() {
  MoFEMFunctionBegin;
  PipelineManager *pipeline_mng = mField.getInterface<PipelineManager>();

  // Push element from reference configuration to current configuration in 3d
  // space
  auto set_domain_general = [&](auto &pipeline) {
    auto det_ptr = boost::make_shared<VectorDouble>();
    auto jac_ptr = boost::make_shared<MatrixDouble>();
    auto inv_jac_ptr = boost::make_shared<MatrixDouble>();
    pipeline.push_back(new OpSetHOWeigthsOnFace());
    pipeline.push_back(new OpCalculateHOJacForFace(jac_ptr));
    pipeline.push_back(
        new OpInvertMatrix<SPACE_DIM>(jac_ptr, det_ptr, inv_jac_ptr));
    pipeline.push_back(new OpSetInvJacH1ForFace(inv_jac_ptr));
  };

  auto set_domain_rhs = [&](auto &pipeline) {

  };

  MoFEMFunctionReturn(0);
}
//! [Push operators to pipeline]

//! [Solve]
MoFEMErrorCode FreeSurface::solveSystem() {
  MoFEMFunctionBegin;
  PipelineManager *pipeline_mng = mField.getInterface<PipelineManager>();

  MoFEMFunctionReturn(0);
}

int main(int argc, char *argv[]) {

  // Initialisation of MoFEM/PETSc and MOAB data structures
  const char param_file[] = "param_file.petsc";
  MoFEM::Core::Initialize(&argc, &argv, param_file, help);

  // Add logging channel for example
  auto core_log = logging::core::get();
  core_log->add_sink(LogManager::createSink(LogManager::getStrmWorld(), "SW"));
  LogManager::setLog("FS");
  MOFEM_LOG_TAG("FS", "free surface");

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

    //! [FreeSurface]
    FreeSurface ex(m_field);
    CHKERR ex.runProblem();
    //! [FreeSurface]
  }
  CATCH_ERRORS;

  CHKERR MoFEM::Core::Finalize();
}
