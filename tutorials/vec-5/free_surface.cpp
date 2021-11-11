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

using AssemblyDomainEleOp =
    FormsIntegrators<DomainEleOp>::Assembly<PETSC>::OpBase;

using OpDomainMassU = FormsIntegrators<DomainEleOp>::Assembly<
    PETSC>::BiLinearForm<GAUSS>::OpMass<BASE_DIM, U_FIELD_DIM>;
using OpDomainSourceU = FormsIntegrators<DomainEleOp>::Assembly<
    PETSC>::LinearForm<GAUSS>::OpSource<BASE_DIM, U_FIELD_DIM>;

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

using OpDomainGradHTimesU = FormsIntegrators<DomainEleOp>::Assembly<
    PETSC>::LinearForm<GAUSS>::OpGradTimesTensor<BASE, H_FIELD_DIM, SPACE_DIM>;
using OpDinianMixVectorTimesGradH =
    FormsIntegrators<DomainEleOp>::Assembly<PETSC>::BiLinearForm<
        GAUSS>::OpMixVectorTimesGrad<BASE, U_DIELD_DIM, SPACE_DIM>;
using OpDomainGradHTimesGardH = OpDomainGradHTimesU;
using OpDinianGradHGradH = FormsIntegrators<DomainEleOp>::Assembly<
    PETSC>::BiLinearForm<GAUSS>::OpGradGrad<BASE, H_DIELD_DIM, SPACE_DIM>;

constexpr gamma = 1;

/**
 * @brief Explict term for IMEX method
 * 
 */
struct OpRhsExplicitTermH : public AssemblyDomainEleOp {

  OpURhs(const std::string field_name, boost::shared_ptr<VectorDouble> f_ptr,
         boost::shared_ptr<double> ksi_ptr)
      : AssemblyDomainEleOp(field_name, field_name, AssemblyDomainEleOp::OPROW),
        fPtr(f_ptr), ksiPtr(ksi_ptr) {}

  MoFEMErrorCode iNtegrate(DataForcesAndSourcesCore::EntData &row_data) {
    MoFEMFunctionBegin;

    const double vol = getMeasure();
    auto t_w = getFTensor0IntegrationWeight();
    auto t_row_base = row_data.getFTensor0N();
    auto t_f = getFTensor0FromVec(*fPtr);

    for (int gg = 0; gg != nbIntegrationPts; gg++) {

      const double alpha = t_w * vol;
      auto nf_ptr = &*locF[0];

      int rr = 0;
      for (; rr != nbRows; ++rr) {

        (*nf_ptr) += alpha * (t_f - ksi) * t_row_base;

        ++nf_ptr;
        ++t_row_base;
        ++t_row_diff_base;
      }

      ++t_f;
    };

    MoFEMFuncturnReturn(0);
  }

private:
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
    auto t_row_base = row_data.getFTensor0N();
    auto t_phi = getFTensor0FromVec(*phiPtr);

    double ksi = 0;
    for (int gg = 0; gg != nbIntegrationPts; gg++) {
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

  CHKERR mField.getInterface(simpleInterface);
  CHKERR simpleInterface->getOptions();
  CHKERR simpleInterface->loadFile();

  MoFEMFunctionReturn(0);
}
//! [Read mesh]

//! [Set up problem]
MoFEMErrorCode FreeSurface::setupProblem() {
  MoFEMFunctionBegin;

  CHKERR simpleInterface->addDomainField("U", H1, AINSWORTH_LEGENDRE_BASE,
                                         U_FIELD_DIM);
  CHKERR simpleInterface->addDomainField("H", H1, AINSWORTH_LEGENDRE_BASE, 1);

  constexpr int order = 2;
  CHKERR simpleInterface->setFieldOrder("U", order);
  CHKERR simpleInterface->setFieldOrder("H", order);
  CHKERR simpleInterface->setUp();

  MoFEMFunctionReturn(0);
}
//! [Set up problem]

//! [Boundary condition]
MoFEMErrorCode FreeSurface::boundaryCondition() { 
  MoFEMFunctionBegin;

  auto int_u = [](double, double, double) { return 0; };
  auto int_h = [](double, double y, double) { return tanh((y-0.5)*  ); };

  auto set_domain_general =
      [&](auto &pipeline) { pipeline.push_back(new OpSetHOWeigthsOnFace()); };

  auto set_domain_rhs = [&](auto &pipeline) {
    pipeline.push_back(new OpSourceU("U", init_u));
    pipeline.push_back(new OpSourceH("H", init_h));
  };

  auto set_domain_lhs = [&](auto &pipeline) {
    pipeline.push_back(
        new OpMassUU("U", "U", [](double, double, double) { return 1; }));
    pipeline.push_back(
        new OpMassHH("H", "H", [](double, double, double) { return 1; }));
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

    post_proc_fe->getOpPtrVector().push_back(
        new OpGetHONormalsOnFace("HO_POSITIONS"));
    post_proc_fe->getOpPtrVector().push_back(
        new OpCalculateHOCoords("HO_POSITIONS"));
    post_proc_fe->getOpPtrVector().push_back(
        new OpCalculateHOJacForFaceEmbeddedIn3DSpace(jac_ptr));
    post_proc_fe->getOpPtrVector().push_back(
        new OpInvertMatrix<3>(jac_ptr, det_ptr, inv_jac_ptr));
    post_proc_fe->addFieldValuesPostProc("U");
    post_proc_fe->addFieldValuesPostProc("H");
    post_proc_fe->addFieldValuesPostProc("HO_POSITIONS");

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
}
//! [Boundary condition]

//! [Push operators to pipeline]
MoFEMErrorCode FreeSurface::assembleSystem() {
  MoFEMFunctionBegin;
  PipelineManager *pipeline_mng = mField.getInterface<PipelineManager>();
  auto beta = [](const double, const double, const double) { return 1; };
  pipeline_mng->getOpDomainLhsPipeline().push_back(
      new OpDomainMassU(FIELD_NAME, FIELD_NAME, beta));
  pipeline_mng->getOpDomainRhsPipeline().push_back(
      new OpDomainSoUurce(FIELD_NAME, approxFunction));
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
