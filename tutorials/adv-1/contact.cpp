/**
 * \file contact.cpp
 * \example contact.cpp
 *
 * Example of contact problem
 *
 */



#ifndef EXECUTABLE_DIMENSION
#define EXECUTABLE_DIMENSION 3
#endif

#include <MoFEM.hpp>
#include <MatrixFunction.hpp>

using namespace MoFEM;

template <int DIM> struct ElementsAndOps {};

template <> struct ElementsAndOps<2> : PipelineManager::ElementsAndOpsByDim<2> {
  static constexpr FieldSpace CONTACT_SPACE = HCURL;
  using OpSetPiolaTransformOnBoundary =
      OpSetContravariantPiolaTransformOnEdge2D;
};

template <> struct ElementsAndOps<3> : PipelineManager::ElementsAndOpsByDim<3> {
  static constexpr FieldSpace CONTACT_SPACE = HDIV;
  using OpSetPiolaTransformOnBoundary =
      OpHOSetContravariantPiolaTransformOnFace3D;
};

constexpr FieldSpace ElementsAndOps<2>::CONTACT_SPACE;
constexpr FieldSpace ElementsAndOps<3>::CONTACT_SPACE;

constexpr int SPACE_DIM =
    EXECUTABLE_DIMENSION; //< Space dimension of problem, mesh

constexpr EntityType boundary_ent = SPACE_DIM == 3 ? MBTRI : MBEDGE;
using EntData = EntitiesFieldData::EntData;
using DomainEle = ElementsAndOps<SPACE_DIM>::DomainEle;
using DomainEleOp = DomainEle::UserDataOperator;
using BoundaryEle = ElementsAndOps<SPACE_DIM>::BoundaryEle;
using BoundaryEleOp = BoundaryEle::UserDataOperator;
using PostProcEle = PostProcBrokenMeshInMoab<DomainEle>;

using AssemblyDomainEleOp =
    FormsIntegrators<DomainEleOp>::Assembly<PETSC>::OpBase;
using AssemblyBoundaryEleOp =
    FormsIntegrators<BoundaryEleOp>::Assembly<PETSC>::OpBase;
using OpSetPiolaTransformOnBoundary =
    ElementsAndOps<SPACE_DIM>::OpSetPiolaTransformOnBoundary;
constexpr FieldSpace CONTACT_SPACE = ElementsAndOps<SPACE_DIM>::CONTACT_SPACE;

//! [Operators used for contact]
using OpMixDivULhs = FormsIntegrators<DomainEleOp>::Assembly<
    PETSC>::BiLinearForm<GAUSS>::OpMixDivTimesVec<SPACE_DIM>;
using OpLambdaGraULhs = FormsIntegrators<DomainEleOp>::Assembly<
    PETSC>::BiLinearForm<GAUSS>::OpMixTensorTimesGrad<SPACE_DIM>;
using OpMixDivURhs = FormsIntegrators<DomainEleOp>::Assembly<PETSC>::LinearForm<
    GAUSS>::OpMixDivTimesU<3, SPACE_DIM, SPACE_DIM>;
using OpMixLambdaGradURhs = FormsIntegrators<DomainEleOp>::Assembly<
    PETSC>::LinearForm<GAUSS>::OpMixTensorTimesGradU<SPACE_DIM>;
using OpMixUTimesDivLambdaRhs = FormsIntegrators<DomainEleOp>::Assembly<
    PETSC>::LinearForm<GAUSS>::OpMixVecTimesDivLambda<SPACE_DIM>;
using OpMixUTimesLambdaRhs = FormsIntegrators<DomainEleOp>::Assembly<
    PETSC>::LinearForm<GAUSS>::OpGradTimesTensor<1, SPACE_DIM, SPACE_DIM>;
using OpSpringLhs = FormsIntegrators<BoundaryEleOp>::Assembly<
    PETSC>::BiLinearForm<GAUSS>::OpMass<1, SPACE_DIM>;
using OpSpringRhs = FormsIntegrators<BoundaryEleOp>::Assembly<
    PETSC>::LinearForm<GAUSS>::OpBaseTimesVector<1, SPACE_DIM, 1>;
//! [Operators used for contact]

//! [Body force]
using OpBodyForce = FormsIntegrators<DomainEleOp>::Assembly<PETSC>::LinearForm<
    GAUSS>::OpSource<1, SPACE_DIM>;
//! [Body force]

//! [Only used with Hooke equation (linear material model)]
using OpKCauchy = FormsIntegrators<DomainEleOp>::Assembly<PETSC>::BiLinearForm<
    GAUSS>::OpGradSymTensorGrad<1, SPACE_DIM, SPACE_DIM, 0>;
using OpInternalForceCauchy = FormsIntegrators<DomainEleOp>::Assembly<
    PETSC>::LinearForm<GAUSS>::OpGradTimesSymTensor<1, SPACE_DIM, SPACE_DIM>;
//! [Only used with Hooke equation (linear material model)]

//! [Only used for dynamics]
using OpMass = FormsIntegrators<DomainEleOp>::Assembly<PETSC>::BiLinearForm<
    GAUSS>::OpMass<1, SPACE_DIM>;
using OpInertiaForce = FormsIntegrators<DomainEleOp>::Assembly<
    PETSC>::LinearForm<GAUSS>::OpBaseTimesVector<1, SPACE_DIM, 1>;
//! [Only used for dynamics]

//! [Essential boundary conditions]
using OpBoundaryMass = FormsIntegrators<BoundaryEleOp>::Assembly<
    PETSC>::BiLinearForm<GAUSS>::OpMass<1, SPACE_DIM>;
using OpBoundaryVec = FormsIntegrators<BoundaryEleOp>::Assembly<
    PETSC>::LinearForm<GAUSS>::OpBaseTimesVector<1, SPACE_DIM, 0>;
using OpBoundaryInternal = FormsIntegrators<BoundaryEleOp>::Assembly<
    PETSC>::LinearForm<GAUSS>::OpBaseTimesVector<1, SPACE_DIM, 1>;
//! [Essential boundary conditions]

// Only used with Hencky/nonlinear material
using OpKPiola = FormsIntegrators<DomainEleOp>::Assembly<PETSC>::BiLinearForm<
    GAUSS>::OpGradTensorGrad<1, SPACE_DIM, SPACE_DIM, 1>;
using OpInternalForcePiola = FormsIntegrators<DomainEleOp>::Assembly<
    PETSC>::LinearForm<GAUSS>::OpGradTimesTensor<1, SPACE_DIM, SPACE_DIM>;

constexpr bool is_quasi_static = true;
constexpr bool is_large_strains = true;

constexpr int order = 2;
constexpr double young_modulus = 100;
constexpr double poisson_ratio = 0.25;
constexpr double rho = 0;
constexpr double cn = 0.01;
constexpr double spring_stiffness = 0.1;

#include <ContactOps.hpp>
#include <HenckyOps.hpp>
#include <PostProcContact.hpp>
using namespace ContactOps;
using namespace HenckyOps;

struct Example {

  Example(MoFEM::Interface &m_field) : mField(m_field) {}

  MoFEMErrorCode runProblem();

private:
  MoFEM::Interface &mField;

  MoFEMErrorCode setupProblem();
  MoFEMErrorCode createCommonData();
  MoFEMErrorCode bC();
  MoFEMErrorCode OPs();
  MoFEMErrorCode tsSolve();
  MoFEMErrorCode postProcess();
  MoFEMErrorCode checkResults();

  boost::shared_ptr<ContactOps::CommonData> commonDataPtr;
  std::tuple<SmartPetscObj<Vec>, SmartPetscObj<VecScatter>> uXScatter;
  std::tuple<SmartPetscObj<Vec>, SmartPetscObj<VecScatter>> uYScatter;
  std::tuple<SmartPetscObj<Vec>, SmartPetscObj<VecScatter>> uZScatter;

  template <int DIM> Range getEntsOnMeshSkin();
};

//! [Run problem]
MoFEMErrorCode Example::runProblem() {
  MoFEMFunctionBegin;
  CHKERR setupProblem();
  CHKERR createCommonData();
  CHKERR bC();
  CHKERR OPs();
  CHKERR tsSolve();
  CHKERR postProcess();
  CHKERR checkResults();
  MoFEMFunctionReturn(0);
}
//! [Run problem]

//! [Set up problem]
MoFEMErrorCode Example::setupProblem() {
  MoFEMFunctionBegin;
  Simple *simple = mField.getInterface<Simple>();

  // Select base
  enum bases { AINSWORTH, DEMKOWICZ, LASBASETOPT };
  const char *list_bases[LASBASETOPT] = {"ainsworth", "demkowicz"};
  PetscInt choice_base_value = AINSWORTH;
  CHKERR PetscOptionsGetEList(PETSC_NULL, NULL, "-base", list_bases,
                              LASBASETOPT, &choice_base_value, PETSC_NULL);

  FieldApproximationBase base;
  switch (choice_base_value) {
  case AINSWORTH:
    base = AINSWORTH_LEGENDRE_BASE;
    MOFEM_LOG("EXAMPLE", Sev::inform)
        << "Set AINSWORTH_LEGENDRE_BASE for displacements";
    break;
  case DEMKOWICZ:
    base = DEMKOWICZ_JACOBI_BASE;
    MOFEM_LOG("EXAMPLE", Sev::inform)
        << "Set DEMKOWICZ_JACOBI_BASE for displacements";
    break;
  default:
    base = LASTBASE;
    break;
  }

  // Note: For tets we have only H1 Ainsworth base, for Hex we have only H1
  // Demkowicz base. We need to implement Demkowicz H1 base on tet.
  CHKERR simple->addDomainField("U", H1, base, SPACE_DIM);
  CHKERR simple->addBoundaryField("U", H1, base, SPACE_DIM);

  CHKERR simple->addDomainField("SIGMA", CONTACT_SPACE, DEMKOWICZ_JACOBI_BASE,
                                SPACE_DIM);
  CHKERR simple->addBoundaryField("SIGMA", CONTACT_SPACE, DEMKOWICZ_JACOBI_BASE,
                                  SPACE_DIM);

  CHKERR simple->setFieldOrder("U", order);
  CHKERR simple->setFieldOrder("SIGMA", 0);

  auto skin_edges = getEntsOnMeshSkin<SPACE_DIM>();

  // filter not owned entities, those are not on boundary
  Range boundary_ents;
  ParallelComm *pcomm =
      ParallelComm::get_pcomm(&mField.get_moab(), MYPCOMM_INDEX);
  if (pcomm == NULL) {
    SETERRQ(PETSC_COMM_WORLD, MOFEM_DATA_INCONSISTENCY,
            "Communicator not created");
  }

  CHKERR pcomm->filter_pstatus(skin_edges, PSTATUS_SHARED | PSTATUS_MULTISHARED,
                               PSTATUS_NOT, -1, &boundary_ents);

  CHKERR simple->setFieldOrder("SIGMA", order - 1, &boundary_ents);
  // Range adj_edges;
  // CHKERR mField.get_moab().get_adjacencies(boundary_ents, 1, false,
  // adj_edges,
  //                                          moab::Interface::UNION);
  // adj_edges.merge(boundary_ents);
  // CHKERR simple->setFieldOrder("U", order, &adj_edges);

  CHKERR simple->setUp();
  MoFEMFunctionReturn(0);
}
//! [Set up problem]

//! [Create common data]
MoFEMErrorCode Example::createCommonData() {
  MoFEMFunctionBegin;

  auto set_matrial_stiffness = [&]() {
    MoFEMFunctionBegin;
    FTensor::Index<'i', SPACE_DIM> i;
    FTensor::Index<'j', SPACE_DIM> j;
    FTensor::Index<'k', SPACE_DIM> k;
    FTensor::Index<'l', SPACE_DIM> l;
    constexpr auto t_kd = FTensor::Kronecker_Delta_symmetric<int>();
    constexpr double bulk_modulus_K =
        young_modulus / (3 * (1 - 2 * poisson_ratio));
    constexpr double shear_modulus_G =
        young_modulus / (2 * (1 + poisson_ratio));
    constexpr double A =
        (SPACE_DIM == 2) ? 2 * shear_modulus_G /
                               (bulk_modulus_K + (4. / 3.) * shear_modulus_G)
                         : 1;
    auto t_D =
        getFTensor4DdgFromMat<SPACE_DIM, SPACE_DIM, 0>(*commonDataPtr->mDPtr);
    t_D(i, j, k, l) = 2 * shear_modulus_G * ((t_kd(i, k) ^ t_kd(j, l)) / 4.) +
                      A * (bulk_modulus_K - (2. / 3.) * shear_modulus_G) *
                          t_kd(i, j) * t_kd(k, l);
    MoFEMFunctionReturn(0);
  };

  commonDataPtr = boost::make_shared<ContactOps::CommonData>();

  commonDataPtr->mGradPtr = boost::make_shared<MatrixDouble>();
  commonDataPtr->mStrainPtr = boost::make_shared<MatrixDouble>();
  commonDataPtr->mStressPtr = boost::make_shared<MatrixDouble>();
  commonDataPtr->contactStressPtr = boost::make_shared<MatrixDouble>();
  commonDataPtr->contactStressDivergencePtr =
      boost::make_shared<MatrixDouble>();
  commonDataPtr->contactTractionPtr = boost::make_shared<MatrixDouble>();
  commonDataPtr->contactDispPtr = boost::make_shared<MatrixDouble>();
  commonDataPtr->curlContactStressPtr = boost::make_shared<MatrixDouble>();

  constexpr auto size_symm = (SPACE_DIM * (SPACE_DIM + 1)) / 2;

  commonDataPtr->mDPtr = boost::make_shared<MatrixDouble>();
  commonDataPtr->mDPtr->resize(size_symm * size_symm, 1);

  CHKERR set_matrial_stiffness();
  MoFEMFunctionReturn(0);
}
//! [Create common data]

//! [Boundary condition]
MoFEMErrorCode Example::bC() {
  MoFEMFunctionBegin;
  auto bc_mng = mField.getInterface<BcManager>();
  auto simple = mField.getInterface<Simple>();

  CHKERR bc_mng->removeBlockDOFsOnEntities(simple->getProblemName(), "REMOVE_X",
                                           "U", 0, 0);
  CHKERR bc_mng->removeBlockDOFsOnEntities(simple->getProblemName(), "REMOVE_Y",
                                           "U", 1, 1);
  CHKERR bc_mng->removeBlockDOFsOnEntities(simple->getProblemName(), "REMOVE_Z",
                                           "U", 2, 2);
  CHKERR bc_mng->removeBlockDOFsOnEntities(simple->getProblemName(),
                                           "REMOVE_ALL", "U", 0, 3);

  CHKERR bc_mng->removeBlockDOFsOnEntities(simple->getProblemName(),
                                           "NO_CONTACT", "SIGMA", 0, 3);
  CHKERR bc_mng->removeBlockDOFsOnEntities<DisplacementCubitBcData>(
      simple->getProblemName(), "U");

  MoFEMFunctionReturn(0);
}
//! [Boundary condition]

//! [Push operators to pip]
MoFEMErrorCode Example::OPs() {
  MoFEMFunctionBegin;
  PipelineManager *pip_mng = mField.getInterface<PipelineManager>();
  auto bc_mng = mField.getInterface<BcManager>();

  auto add_domain_base_ops = [&](auto &pip) {
    auto det_ptr = boost::make_shared<VectorDouble>();
    auto jac_ptr = boost::make_shared<MatrixDouble>();
    auto inv_jac_ptr = boost::make_shared<MatrixDouble>();

    pip.push_back(new OpCalculateHOJac<SPACE_DIM>(jac_ptr));
    pip.push_back(
        new OpInvertMatrix<SPACE_DIM>(jac_ptr, det_ptr, inv_jac_ptr));
    pip.push_back(
        new OpSetHOInvJacToScalarBases<SPACE_DIM>(H1, inv_jac_ptr));

    if (SPACE_DIM == 2) {
      pip.push_back(new OpMakeHdivFromHcurl());
      pip.push_back(new OpSetContravariantPiolaTransformOnFace2D(jac_ptr));
      pip.push_back(new OpSetInvJacHcurlFace(inv_jac_ptr));
      pip.push_back(new OpSetHOWeightsOnFace());
    } else {
      pip.push_back(
          new OpSetHOContravariantPiolaTransform(HDIV, det_ptr, jac_ptr));
      pip.push_back(new OpSetHOInvJacVectorBase(HDIV, inv_jac_ptr));
      pip.push_back(new OpSetHOWeights(det_ptr));
    }
  };

  auto henky_common_data_ptr = boost::make_shared<HenckyOps::CommonData>();
  henky_common_data_ptr->matGradPtr = commonDataPtr->mGradPtr;
  henky_common_data_ptr->matDPtr = commonDataPtr->mDPtr;

  auto add_domain_ops_lhs = [&](auto &pip) {

    if (is_large_strains) {
      pip_mng->getOpDomainLhsPipeline().push_back(
          new OpCalculateVectorFieldGradient<SPACE_DIM, SPACE_DIM>(
              "U", commonDataPtr->mGradPtr));
      pip_mng->getOpDomainLhsPipeline().push_back(
          new OpCalculateEigenVals<SPACE_DIM>("U", henky_common_data_ptr));
      pip_mng->getOpDomainLhsPipeline().push_back(
          new OpCalculateLogC<SPACE_DIM>("U", henky_common_data_ptr));
      pip_mng->getOpDomainLhsPipeline().push_back(
          new OpCalculateLogC_dC<SPACE_DIM>("U", henky_common_data_ptr));
      pip_mng->getOpDomainLhsPipeline().push_back(
          new OpCalculateHenckyStress<SPACE_DIM>("U", henky_common_data_ptr));
      pip_mng->getOpDomainLhsPipeline().push_back(
          new OpCalculatePiolaStress<SPACE_DIM>("U", henky_common_data_ptr));
      pip_mng->getOpDomainLhsPipeline().push_back(
          new OpHenckyTangent<SPACE_DIM>("U", henky_common_data_ptr));
      pip_mng->getOpDomainLhsPipeline().push_back(
          new OpKPiola("U", "U", henky_common_data_ptr->getMatTangent()));
    } else {
      pip.push_back(new OpKCauchy("U", "U", commonDataPtr->mDPtr));
    }

    if (!is_quasi_static) {
      // Get pointer to U_tt shift in domain element
      auto get_rho = [this](const double, const double, const double) {
        auto *pip_mng = mField.getInterface<PipelineManager>();
        auto &fe_domain_lhs = pip_mng->getDomainLhsFE();
        return rho * fe_domain_lhs->ts_aa;
      };
      pip_mng->getOpDomainLhsPipeline().push_back(
          new OpMass("U", "U", get_rho));
    }

    auto unity = []() { return 1; };
    pip.push_back(new OpMixDivULhs("SIGMA", "U", unity, true));
    pip.push_back(new OpLambdaGraULhs("SIGMA", "U", unity, true));

  };

  auto add_domain_ops_rhs = [&](auto &pip) {
    auto get_body_force = [this](const double, const double, const double) {
      auto *pip_mng = mField.getInterface<PipelineManager>();
      FTensor::Index<'i', SPACE_DIM> i;
      FTensor::Tensor1<double, SPACE_DIM> t_source;
      auto fe_domain_rhs = pip_mng->getDomainRhsFE();
      const auto time = fe_domain_rhs->ts_t;

      // hardcoded gravity load
      t_source(i) = 0;
      t_source(1) = 1.0 * time;
      return t_source;
    };

    pip.push_back(new OpBodyForce("U", get_body_force));
    pip.push_back(new OpCalculateVectorFieldGradient<SPACE_DIM, SPACE_DIM>(
        "U", commonDataPtr->mGradPtr));

    if (is_large_strains) {
      pip_mng->getOpDomainRhsPipeline().push_back(
          new OpCalculateEigenVals<SPACE_DIM>("U", henky_common_data_ptr));
      pip_mng->getOpDomainRhsPipeline().push_back(
          new OpCalculateLogC<SPACE_DIM>("U", henky_common_data_ptr));
      pip_mng->getOpDomainRhsPipeline().push_back(
          new OpCalculateLogC_dC<SPACE_DIM>("U", henky_common_data_ptr));
      pip_mng->getOpDomainRhsPipeline().push_back(
          new OpCalculateHenckyStress<SPACE_DIM>("U", henky_common_data_ptr));
      pip_mng->getOpDomainRhsPipeline().push_back(
          new OpCalculatePiolaStress<SPACE_DIM>("U", henky_common_data_ptr));
      pip_mng->getOpDomainRhsPipeline().push_back(new OpInternalForcePiola(
          "U", henky_common_data_ptr->getMatFirstPiolaStress()));
    } else {
      pip.push_back(new OpSymmetrizeTensor<SPACE_DIM>(
          "U", commonDataPtr->mGradPtr, commonDataPtr->mStrainPtr));
      pip.push_back(new OpTensorTimesSymmetricTensor<SPACE_DIM, SPACE_DIM>(
          "U", commonDataPtr->mStrainPtr, commonDataPtr->mStressPtr,
          commonDataPtr->mDPtr));
      pip.push_back(
          new OpInternalForceCauchy("U", commonDataPtr->mStressPtr));
    }

    pip.push_back(new OpCalculateVectorFieldValues<SPACE_DIM>(
        "U", commonDataPtr->contactDispPtr));

    pip.push_back(new OpCalculateHVecTensorField<SPACE_DIM, SPACE_DIM>(
        "SIGMA", commonDataPtr->contactStressPtr));
    pip.push_back(
        new OpCalculateHVecTensorDivergence<SPACE_DIM, SPACE_DIM>(
            "SIGMA", commonDataPtr->contactStressDivergencePtr));

    pip.push_back(
        new OpMixDivURhs("SIGMA", commonDataPtr->contactDispPtr,
                         [](double, double, double) { return 1; }));
    pip.push_back(
        new OpMixLambdaGradURhs("SIGMA", commonDataPtr->mGradPtr));

    pip.push_back(new OpMixUTimesDivLambdaRhs(
        "U", commonDataPtr->contactStressDivergencePtr));
    pip.push_back(
        new OpMixUTimesLambdaRhs("U", commonDataPtr->contactStressPtr));

    // only in case of dynamics
    if (!is_quasi_static) {
      auto mat_acceleration = boost::make_shared<MatrixDouble>();
      pip_mng->getOpDomainRhsPipeline().push_back(
          new OpCalculateVectorFieldValuesDotDot<SPACE_DIM>("U",
                                                            mat_acceleration));
      pip_mng->getOpDomainRhsPipeline().push_back(new OpInertiaForce(
          "U", mat_acceleration, [](double, double, double) { return rho; }));
    }

  };

  auto add_boundary_base_ops = [&](auto &pip) {
    pip.push_back(new OpSetPiolaTransformOnBoundary(CONTACT_SPACE));
    if (SPACE_DIM == 3)
      pip.push_back(new OpSetHOWeightsOnFace());
    pip.push_back(new OpCalculateVectorFieldValues<SPACE_DIM>(
        "U", commonDataPtr->contactDispPtr));
    pip.push_back(new OpCalculateHVecTensorTrace<SPACE_DIM, BoundaryEleOp>(
        "SIGMA", commonDataPtr->contactTractionPtr));
  };

  auto add_boundary_ops_lhs = [&](auto &pip) {
    MoFEMFunctionBegin;
    auto &bc_map = bc_mng->getBcMapByBlockName();
    for (auto bc : bc_map) {
      if (bc_mng->checkBlock(bc, "FIX_")) {
        MOFEM_LOG("EXAMPLE", Sev::inform)
            << "Set boundary matrix for " << bc.first;
        pip.push_back(
            new OpSetBc("U", false, bc.second->getBcMarkersPtr()));
        pip.push_back(new OpBoundaryMass(
            "U", "U", [](double, double, double) { return 1.; },
            bc.second->getBcEntsPtr()));
      }
    }

    pip.push_back(
        new OpConstrainBoundaryLhs_dU("SIGMA", "U", commonDataPtr));
    pip.push_back(
        new OpConstrainBoundaryLhs_dTraction("SIGMA", "SIGMA", commonDataPtr));
    pip.push_back(new OpSpringLhs(
        "U", "U",

        [this](double, double, double) { return spring_stiffness; }

        ));

    MoFEMFunctionReturn(0);
  };

  auto time_scaled = [&](double, double, double) {
    auto *pip_mng = mField.getInterface<PipelineManager>();
    auto &fe_domain_rhs = pip_mng->getDomainRhsFE();
    return -fe_domain_rhs->ts_t;
  };

  auto add_boundary_ops_rhs = [&](auto &pip) {
    MoFEMFunctionBegin;
    for (auto &bc : mField.getInterface<BcManager>()->getBcMapByBlockName()) {
      if (bc_mng->checkBlock(bc, "FIX_")) {
        MOFEM_LOG("EXAMPLE", Sev::inform)
            << "Set boundary residual for " << bc.first;
        pip.push_back(
            new OpSetBc("U", false, bc.second->getBcMarkersPtr()));
        auto attr_vec = boost::make_shared<MatrixDouble>(SPACE_DIM, 1);
        attr_vec->clear();
        if (bc.second->bcAttributes.size() != SPACE_DIM)
          SETERRQ1(PETSC_COMM_WORLD, MOFEM_DATA_INCONSISTENCY,
                   "Wrong size of boundary attributes vector. Set right block "
                   "size attributes. Size of attributes %d",
                   bc.second->bcAttributes.size());
        std::copy(&bc.second->bcAttributes[0],
                  &bc.second->bcAttributes[SPACE_DIM],
                  attr_vec->data().begin());

        pip.push_back(new OpBoundaryVec("U", attr_vec, time_scaled,
                                             bc.second->getBcEntsPtr()));
        pip.push_back(new OpBoundaryInternal(
            "U", commonDataPtr->contactDispPtr,
            [](double, double, double) { return 1.; },
            bc.second->getBcEntsPtr()));

      }
    }

    pip.push_back(new OpConstrainBoundaryRhs("SIGMA", commonDataPtr));
    pip.push_back(new OpSpringRhs(
        "U", commonDataPtr->contactDispPtr,
        [this](double, double, double) { return spring_stiffness; }));
    MoFEMFunctionReturn(0);
  };

  add_domain_base_ops(pip_mng->getOpDomainLhsPipeline());
  add_domain_base_ops(pip_mng->getOpDomainRhsPipeline());
  add_domain_ops_lhs(pip_mng->getOpDomainLhsPipeline());
  add_domain_ops_rhs(pip_mng->getOpDomainRhsPipeline());

  add_boundary_base_ops(pip_mng->getOpBoundaryLhsPipeline());
  add_boundary_base_ops(pip_mng->getOpBoundaryRhsPipeline());
  CHKERR add_boundary_ops_lhs(pip_mng->getOpBoundaryLhsPipeline());
  CHKERR add_boundary_ops_rhs(pip_mng->getOpBoundaryRhsPipeline());

  auto integration_rule_vol = [](int, int, int approx_order) {
    return 3 * order;
  };
  CHKERR pip_mng->setDomainRhsIntegrationRule(integration_rule_vol);
  CHKERR pip_mng->setDomainLhsIntegrationRule(integration_rule_vol);
  auto integration_rule_boundary = [](int, int, int approx_order) {
    return 3 * order;
  };
  CHKERR pip_mng->setBoundaryRhsIntegrationRule(integration_rule_boundary);
  CHKERR pip_mng->setBoundaryLhsIntegrationRule(integration_rule_boundary);

  // Enforce non-homegonus boundary conditions
  auto get_bc_hook_rhs = [&]() {
    EssentialPreProc<DisplacementCubitBcData> hook(
        mField, pip_mng->getDomainRhsFE(),
        {boost::make_shared<TimeScale>()});
    return hook;
  };

  auto get_bc_hook_lhs = [&]() {
    EssentialPreProc<DisplacementCubitBcData> hook(
        mField, pip_mng->getDomainLhsFE(),
        {boost::make_shared<TimeScale>()});
    return hook;
  };

  pip_mng->getDomainRhsFE()->preProcessHook = get_bc_hook_rhs();
  pip_mng->getDomainLhsFE()->preProcessHook = get_bc_hook_lhs();

  MoFEMFunctionReturn(0);
}
//! [Push operators to pip]

//! [Solve]
MoFEMErrorCode Example::tsSolve() {
  MoFEMFunctionBegin;

  Simple *simple = mField.getInterface<Simple>();
  PipelineManager *pip_mng = mField.getInterface<PipelineManager>();
  ISManager *is_manager = mField.getInterface<ISManager>();

  auto set_section_monitor = [&](auto solver) {
    MoFEMFunctionBegin;
    SNES snes;
    CHKERR TSGetSNES(solver, &snes);
    PetscViewerAndFormat *vf;
    CHKERR PetscViewerAndFormatCreate(PETSC_VIEWER_STDOUT_WORLD,
                                      PETSC_VIEWER_DEFAULT, &vf);
    CHKERR SNESMonitorSet(
        snes,
        (MoFEMErrorCode(*)(SNES, PetscInt, PetscReal, void *))SNESMonitorFields,
        vf, (MoFEMErrorCode(*)(void **))PetscViewerAndFormatDestroy);
    MoFEMFunctionReturn(0);
  };

  auto scatter_create = [&](auto D, auto coeff) {
    SmartPetscObj<IS> is;
    CHKERR is_manager->isCreateProblemFieldAndRank(simple->getProblemName(),
                                                   ROW, "U", coeff, coeff, is);
    int loc_size;
    CHKERR ISGetLocalSize(is, &loc_size);
    Vec v;
    CHKERR VecCreateMPI(mField.get_comm(), loc_size, PETSC_DETERMINE, &v);
    VecScatter scatter;
    CHKERR VecScatterCreate(D, is, v, PETSC_NULL, &scatter);
    return std::make_tuple(SmartPetscObj<Vec>(v),
                           SmartPetscObj<VecScatter>(scatter));
  };

  auto set_time_monitor = [&](auto dm, auto solver) {
    MoFEMFunctionBegin;
    boost::shared_ptr<Monitor> monitor_ptr(
        new Monitor(dm, commonDataPtr, uXScatter, uYScatter, uZScatter));
    boost::shared_ptr<ForcesAndSourcesCore> null;
    CHKERR DMMoFEMTSSetMonitor(dm, solver, simple->getDomainFEName(),
                               monitor_ptr, null, null);
    MoFEMFunctionReturn(0);
  };

  auto dm = simple->getDM();
  auto D = smartCreateDMVector(dm);
  uXScatter = scatter_create(D, 0);
  uYScatter = scatter_create(D, 1);
  if (SPACE_DIM == 3)
    uZScatter = scatter_create(D, 2);

  if (is_quasi_static) {
    auto solver = pip_mng->createTSIM();
    auto D = smartCreateDMVector(dm);
    CHKERR set_section_monitor(solver);
    CHKERR set_time_monitor(dm, solver);
    CHKERR TSSetSolution(solver, D);
    CHKERR TSSetFromOptions(solver);
    CHKERR TSSetUp(solver);
    CHKERR TSSolve(solver, NULL);
  } else {
    auto solver = pip_mng->createTSIM2();
    auto dm = simple->getDM();
    auto D = smartCreateDMVector(dm);
    auto DD = smartVectorDuplicate(D);
    CHKERR set_section_monitor(solver);
    CHKERR set_time_monitor(dm, solver);
    CHKERR TS2SetSolution(solver, D, DD);
    CHKERR TSSetFromOptions(solver);
    CHKERR TSSetUp(solver);
    CHKERR TSSolve(solver, NULL);
  }

  CHKERR VecGhostUpdateBegin(D, INSERT_VALUES, SCATTER_FORWARD);
  CHKERR VecGhostUpdateEnd(D, INSERT_VALUES, SCATTER_FORWARD);
  CHKERR DMoFEMMeshToLocalVector(dm, D, INSERT_VALUES, SCATTER_REVERSE);

  MoFEMFunctionReturn(0);
}
//! [Solve]

//! [Postprocess results]
MoFEMErrorCode Example::postProcess() { return 0; }
//! [Postprocess results]

//! [Check]
MoFEMErrorCode Example::checkResults() { return 0; }
//! [Check]

template <int DIM> Range Example::getEntsOnMeshSkin() {
  Range body_ents;
  CHKERR mField.get_moab().get_entities_by_dimension(0, DIM, body_ents);
  Skinner skin(&mField.get_moab());
  Range skin_ents;
  CHKERR skin.find_skin(0, body_ents, false, skin_ents);

  return skin_ents;
};

static char help[] = "...\n\n";

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
    MoFEM::Interface &m_field = core; ///< finite element database interface
    //! [Create MoFEM]

    //! [Load mesh]
    Simple *simple = m_field.getInterface<Simple>();
    CHKERR simple->getOptions();
    CHKERR simple->loadFile("");
    //! [Load mesh]

    //! [Example]
    Example ex(m_field);
    CHKERR ex.runProblem();
    //! [Example]
  }
  CATCH_ERRORS;

  CHKERR MoFEM::Core::Finalize();
}
