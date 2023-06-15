/**
 * \file plastic.cpp
 * \example plastic.cpp
 *
 * Plasticity in 2d and 3d
 *
 */

/* The above code is a preprocessor directive in C++ that checks if the macro
"EXECUTABLE_DIMENSION" has been defined. If it has not been defined, it replaces
the " */
#ifndef EXECUTABLE_DIMENSION
#define EXECUTABLE_DIMENSION 3
#endif

#include <MoFEM.hpp>
#include <MatrixFunction.hpp>
#include <IntegrationRules.hpp>

using namespace MoFEM;

template <int DIM> struct ElementsAndOps {};

template <> struct ElementsAndOps<2> {
  using DomainEle = PipelineManager::FaceEle;
  using BoundaryEle = PipelineManager::EdgeEle;
};

template <> struct ElementsAndOps<3> {
  using DomainEle = PipelineManager::VolEle;
  using BoundaryEle = PipelineManager::FaceEle;
};

constexpr int SPACE_DIM =
    EXECUTABLE_DIMENSION; //< Space dimension of problem, mesh
constexpr auto size_symm = (SPACE_DIM * (SPACE_DIM + 1)) / 2;

constexpr AssemblyType A = (SCHUR_ASSEMBLE)
                               ? AssemblyType::SCHUR
                               : AssemblyType::PETSC; //< selected assembly type
constexpr IntegrationType G =
    IntegrationType::GAUSS; //< selected integration type

using EntData = EntitiesFieldData::EntData;
using DomainEle = ElementsAndOps<SPACE_DIM>::DomainEle;
using DomainEleOp = DomainEle::UserDataOperator;
using BoundaryEle = ElementsAndOps<SPACE_DIM>::BoundaryEle;
using BoundaryEleOp = BoundaryEle::UserDataOperator;
using PostProcEle = PostProcBrokenMeshInMoab<DomainEle>;

using AssemblyDomainEleOp = FormsIntegrators<DomainEleOp>::Assembly<A>::OpBase;

//! [Only used with Hooke equation (linear material model)]
using OpKCauchy = FormsIntegrators<DomainEleOp>::Assembly<A>::BiLinearForm<
    GAUSS>::OpGradSymTensorGrad<1, SPACE_DIM, SPACE_DIM, 0>;
using OpInternalForceCauchy = FormsIntegrators<DomainEleOp>::Assembly<
    PETSC>::LinearForm<G>::OpGradTimesSymTensor<1, SPACE_DIM, SPACE_DIM>;
//! [Only used with Hooke equation (linear material model)]

//! [Only used for dynamics]
using OpMass = FormsIntegrators<DomainEleOp>::Assembly<A>::BiLinearForm<
    GAUSS>::OpMass<1, SPACE_DIM>;
using OpInertiaForce = FormsIntegrators<DomainEleOp>::Assembly<
    PETSC>::LinearForm<G>::OpBaseTimesVector<1, SPACE_DIM, 1>;
//! [Only used for dynamics]

//! [Only used with Hencky/nonlinear material]
using OpKPiola = FormsIntegrators<DomainEleOp>::Assembly<A>::BiLinearForm<
    GAUSS>::OpGradTensorGrad<1, SPACE_DIM, SPACE_DIM, 1>;
using OpInternalForcePiola = FormsIntegrators<DomainEleOp>::Assembly<
    PETSC>::LinearForm<G>::OpGradTimesTensor<1, SPACE_DIM, SPACE_DIM>;
//! [Only used with Hencky/nonlinear material]

//! [Essential boundary conditions]
using OpBoundaryMass = FormsIntegrators<BoundaryEleOp>::Assembly<
    PETSC>::BiLinearForm<G>::OpMass<1, SPACE_DIM>;
using OpBoundaryVec = FormsIntegrators<BoundaryEleOp>::Assembly<
    PETSC>::LinearForm<G>::OpBaseTimesVector<1, SPACE_DIM, 0>;
using OpBoundaryInternal = FormsIntegrators<BoundaryEleOp>::Assembly<
    PETSC>::LinearForm<G>::OpBaseTimesVector<1, SPACE_DIM, 1>;
//! [Essential boundary conditions]
using OpScaleL2 = MoFEM::OpScaleBaseBySpaceInverseOfMeasure<DomainEleOp>;

using DomainNaturalBC = NaturalBC<DomainEleOp>::Assembly<A>::LinearForm<G>;
using OpBodyForce =
    DomainNaturalBC::OpFlux<NaturalMeshsetType<BLOCKSET>, 1, SPACE_DIM>;

using BoundaryNaturalBC = NaturalBC<BoundaryEleOp>::Assembly<A>::LinearForm<G>;
using OpForce =
    BoundaryNaturalBC::OpFlux<NaturalMeshsetType<BLOCKSET>, 1, SPACE_DIM>;

using OpEssentialLhs = EssentialBC<BoundaryEleOp>::Assembly<A>::BiLinearForm<
    GAUSS>::OpEssentialLhs<DisplacementCubitBcData, 1, SPACE_DIM>;
using OpEssentialRhs = EssentialBC<BoundaryEleOp>::Assembly<A>::LinearForm<
    GAUSS>::OpEssentialRhs<DisplacementCubitBcData, 1, SPACE_DIM>;

PetscBool is_large_strains = PETSC_TRUE;
PetscBool set_timer = PETSC_FALSE;

double scale = 1.;

double young_modulus = 206913;
double poisson_ratio = 0.29;
double rho = 0;
double sigmaY = 450;
double H = 129;
double visH = 0;
double cn0 = 1;
double cn1 = 1;
double zeta = 5e-2;
double Qinf = 265;
double b_iso = 16.93;

int order = 2;      ///< Order if fixed.
int geom_order = 2; ///< Order if fixed.

constexpr size_t activ_history_sise = 1;

inline double hardening_exp(double tau) {
  return std::exp(
      std::max(static_cast<double>(std::numeric_limits<float>::min_exponent10),
               -b_iso * tau));
}

inline double hardening(double tau) {
  return H * tau + Qinf * (1. - hardening_exp(tau)) + sigmaY;
}

inline double hardening_dtau(double tau) {
  auto r = [](auto tau) { return H + Qinf * b_iso * hardening_exp(tau); };
  constexpr double eps = 1e-12;
  return std::max(r(tau), eps * r(0));
}

inline double hardening_dtau2(double tau) {
  return -(Qinf * (b_iso * b_iso)) * hardening_exp(tau);
}

#include <HenckyOps.hpp>
#include <PlasticOps.hpp>

using namespace PlasticOps;
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

  boost::shared_ptr<PlasticOps::CommonData> commonPlasticDataPtr;
  boost::shared_ptr<HenckyOps::CommonData> commonHenckyDataPtr;
  boost::shared_ptr<DomainEle> reactionFe;

  std::tuple<SmartPetscObj<Vec>, SmartPetscObj<VecScatter>> uXScatter;
  std::tuple<SmartPetscObj<Vec>, SmartPetscObj<VecScatter>> uYScatter;
  std::tuple<SmartPetscObj<Vec>, SmartPetscObj<VecScatter>> uZScatter;

  boost::shared_ptr<std::vector<unsigned char>> boundaryMarker;
  boost::shared_ptr<std::vector<unsigned char>> reactionMarker;

  struct PlasticityTimeScale : public MoFEM::TimeScale {
    using MoFEM::TimeScale::TimeScale;
    double getScale(const double time) {
      return scale * MoFEM::TimeScale::getScale(time);
    };
  };
};

//! [Run problem]
MoFEMErrorCode Example::runProblem() {
  MoFEMFunctionBegin;
  CHKERR createCommonData();
  CHKERR setupProblem();
  CHKERR bC();
  CHKERR OPs();
  CHKERR tsSolve();
  MoFEMFunctionReturn(0);
}
//! [Run problem]

//! [Set up problem]
MoFEMErrorCode Example::setupProblem() {
  MoFEMFunctionBegin;
  Simple *simple = mField.getInterface<Simple>();

  Range domain_ents;
  CHKERR mField.get_moab().get_entities_by_dimension(0, SPACE_DIM, domain_ents,
                                                     true);
  auto get_ents_by_dim = [&](const auto dim) {
    if (dim == SPACE_DIM) {
      return domain_ents;
    } else {
      Range ents;
      if (dim == 0)
        CHKERR mField.get_moab().get_connectivity(domain_ents, ents, true);
      else
        CHKERR mField.get_moab().get_entities_by_dimension(0, dim, ents, true);
      return ents;
    }
  };

  auto get_base = [&]() {
    auto domain_ents = get_ents_by_dim(SPACE_DIM);
    if (domain_ents.empty())
      CHK_THROW_MESSAGE(MOFEM_NOT_FOUND, "Empty mesh");
    const auto type = type_from_handle(domain_ents[0]);
    switch (type) {
    case MBQUAD:
      return DEMKOWICZ_JACOBI_BASE;
    case MBHEX:
      return DEMKOWICZ_JACOBI_BASE;
    case MBTRI:
      return AINSWORTH_LEGENDRE_BASE;
    case MBTET:
      return AINSWORTH_LEGENDRE_BASE;
    default:
      CHK_THROW_MESSAGE(MOFEM_NOT_FOUND, "Element type not handled");
    }
    return NOBASE;
  };

  const auto base = get_base();
  MOFEM_LOG("WORLD", Sev::inform) << "Base " << ApproximationBaseNames[base];

  CHKERR simple->addDomainField("U", H1, base, SPACE_DIM);
  CHKERR simple->addDomainField("EP", L2, base, size_symm);
  CHKERR simple->addDomainField("TAU", L2, base, 1);
  CHKERR simple->addBoundaryField("U", H1, base, SPACE_DIM);

  CHKERR simple->addDataField("GEOMETRY", H1, base, SPACE_DIM);

  auto ents = get_ents_by_dim(0);
  ents.merge(get_ents_by_dim(1));
  // ents.merge(get_ents_by_dim(2));
  CHKERR simple->setFieldOrder("U", order, &ents);
  CHKERR simple->setFieldOrder("EP", order - 1);
  CHKERR simple->setFieldOrder("TAU", order - 2);

  CHKERR simple->setFieldOrder("GEOMETRY", geom_order);

  CHKERR simple->setUp();
  CHKERR simple->addFieldToEmptyFieldBlocks("U", "TAU");

  auto project_ho_geometry = [&]() {
    Projection10NodeCoordsOnField ent_method(mField, "GEOMETRY");
    return mField.loop_dofs("GEOMETRY", ent_method);
  };
  CHKERR project_ho_geometry();

  MoFEMFunctionReturn(0);
}
//! [Set up problem]

//! [Create common data]
MoFEMErrorCode Example::createCommonData() {
  MoFEMFunctionBegin;

  auto get_command_line_parameters = [&]() {
    MoFEMFunctionBegin;
    CHKERR PetscOptionsGetScalar(PETSC_NULL, "", "-scale", &scale, PETSC_NULL);
    CHKERR PetscOptionsGetScalar(PETSC_NULL, "", "-rho", &rho, PETSC_NULL);
    CHKERR PetscOptionsGetScalar(PETSC_NULL, "", "-young_modulus",
                                 &young_modulus, PETSC_NULL);
    CHKERR PetscOptionsGetScalar(PETSC_NULL, "", "-poisson_ratio",
                                 &poisson_ratio, PETSC_NULL);
    CHKERR PetscOptionsGetScalar(PETSC_NULL, "", "-hardening", &H, PETSC_NULL);
    CHKERR PetscOptionsGetScalar(PETSC_NULL, "", "-hardening_viscous", &visH,
                                 PETSC_NULL);
    CHKERR PetscOptionsGetScalar(PETSC_NULL, "", "-yield_stress", &sigmaY,
                                 PETSC_NULL);
    CHKERR PetscOptionsGetScalar(PETSC_NULL, "", "-cn0", &cn0, PETSC_NULL);
    CHKERR PetscOptionsGetScalar(PETSC_NULL, "", "-cn1", &cn1, PETSC_NULL);
    CHKERR PetscOptionsGetScalar(PETSC_NULL, "", "-zeta", &zeta, PETSC_NULL);
    CHKERR PetscOptionsGetScalar(PETSC_NULL, "", "-Qinf", &Qinf, PETSC_NULL);
    CHKERR PetscOptionsGetScalar(PETSC_NULL, "", "-b_iso", &b_iso, PETSC_NULL);
    CHKERR PetscOptionsGetBool(PETSC_NULL, "", "-large_strains",
                               &is_large_strains, PETSC_NULL);
    CHKERR PetscOptionsGetBool(PETSC_NULL, "", "-set_timer", &set_timer,
                               PETSC_NULL);

    CHKERR PetscOptionsGetInt(PETSC_NULL, "", "-order", &order, PETSC_NULL);
    CHKERR PetscOptionsGetInt(PETSC_NULL, "", "-geom_order", &geom_order,
                              PETSC_NULL);

    MOFEM_LOG("EXAMPLE", Sev::inform) << "Young modulus " << young_modulus;
    MOFEM_LOG("EXAMPLE", Sev::inform) << "Poisson ratio " << poisson_ratio;
    MOFEM_LOG("EXAMPLE", Sev::inform) << "Yield stress " << sigmaY;
    MOFEM_LOG("EXAMPLE", Sev::inform) << "Hardening " << H;
    MOFEM_LOG("EXAMPLE", Sev::inform) << "Viscous hardening " << visH;
    MOFEM_LOG("EXAMPLE", Sev::inform) << "Saturation yield stress " << Qinf;
    MOFEM_LOG("EXAMPLE", Sev::inform) << "Saturation exponent " << b_iso;
    MOFEM_LOG("EXAMPLE", Sev::inform) << "cn0 " << cn0;
    MOFEM_LOG("EXAMPLE", Sev::inform) << "cn1 " << cn1;
    MOFEM_LOG("EXAMPLE", Sev::inform) << "zeta " << zeta;

    MOFEM_LOG("EXAMPLE", Sev::inform) << "order " << order;
    MOFEM_LOG("EXAMPLE", Sev::inform) << "geom order " << geom_order;

    PetscBool is_scale = PETSC_TRUE;
    CHKERR PetscOptionsGetBool(PETSC_NULL, "", "-is_scale", &is_scale,
                               PETSC_NULL);
    if (is_scale) {
      scale = scale / young_modulus;
      young_modulus *= scale;
      rho *= scale;
      sigmaY *= scale;
      H *= scale;
      Qinf *= scale;
      visH *= scale;

      MOFEM_LOG("EXAMPLE", Sev::inform)
          << "Scaled Young modulus " << young_modulus;
      MOFEM_LOG("EXAMPLE", Sev::inform)
          << "Scaled Poisson ratio " << poisson_ratio;
      MOFEM_LOG("EXAMPLE", Sev::inform) << "Scaled Yield stress " << sigmaY;
      MOFEM_LOG("EXAMPLE", Sev::inform) << "Scaled Hardening " << H;
      MOFEM_LOG("EXAMPLE", Sev::inform) << "Scaled Viscous hardening " << visH;
      MOFEM_LOG("EXAMPLE", Sev::inform)
          << "Scaled Saturation yield stress " << Qinf;
    }

    MoFEMFunctionReturn(0);
  };

  auto set_matrial_stiffness = [&]() {
    MoFEMFunctionBegin;
    FTensor::Index<'i', SPACE_DIM> i;
    FTensor::Index<'j', SPACE_DIM> j;
    FTensor::Index<'k', SPACE_DIM> k;
    FTensor::Index<'l', SPACE_DIM> l;
    constexpr auto t_kd = FTensor::Kronecker_Delta_symmetric<int>();
    const double bulk_modulus_K = young_modulus / (3 * (1 - 2 * poisson_ratio));
    const double shear_modulus_G = young_modulus / (2 * (1 + poisson_ratio));

    // Plane stress or when 1, plane strain or 3d
    const double A = (SPACE_DIM == 2)
                         ? 2 * shear_modulus_G /
                               (bulk_modulus_K + (4. / 3.) * shear_modulus_G)
                         : 1;

    auto t_D = getFTensor4DdgFromMat<SPACE_DIM, SPACE_DIM, 0>(
        *commonPlasticDataPtr->mDPtr);
    auto t_D_axiator = getFTensor4DdgFromMat<SPACE_DIM, SPACE_DIM, 0>(
        *commonPlasticDataPtr->mDPtr_Axiator);
    auto t_D_deviator = getFTensor4DdgFromMat<SPACE_DIM, SPACE_DIM, 0>(
        *commonPlasticDataPtr->mDPtr_Deviator);

    constexpr double third = boost::math::constants::third<double>();
    t_D_axiator(i, j, k, l) = A *
                              (bulk_modulus_K - (2. / 3.) * shear_modulus_G) *
                              t_kd(i, j) * t_kd(k, l);
    t_D_deviator(i, j, k, l) =
        2 * shear_modulus_G * ((t_kd(i, k) ^ t_kd(j, l)) / 4.);
    t_D(i, j, k, l) = t_D_axiator(i, j, k, l) + t_D_deviator(i, j, k, l);

    MoFEMFunctionReturn(0);
  };

  auto make_d_mat = []() {
    return boost::make_shared<MatrixDouble>(size_symm * size_symm, 1);
  };

  commonPlasticDataPtr = boost::make_shared<PlasticOps::CommonData>();
  commonPlasticDataPtr->mDPtr = make_d_mat();
  commonPlasticDataPtr->mDPtr_Axiator = make_d_mat();
  commonPlasticDataPtr->mDPtr_Deviator = make_d_mat();

  commonPlasticDataPtr->mGradPtr = boost::make_shared<MatrixDouble>();
  commonPlasticDataPtr->mStrainPtr = boost::make_shared<MatrixDouble>();
  commonPlasticDataPtr->mStressPtr = boost::make_shared<MatrixDouble>();

  CHKERR get_command_line_parameters();
  CHKERR set_matrial_stiffness();

  if (is_large_strains) {
    commonHenckyDataPtr = boost::make_shared<HenckyOps::CommonData>();
    commonHenckyDataPtr->matGradPtr = commonPlasticDataPtr->mGradPtr;
    commonHenckyDataPtr->matDPtr = commonPlasticDataPtr->mDPtr;
    commonHenckyDataPtr->matLogCPlastic =
        commonPlasticDataPtr->getPlasticStrainPtr();
    commonPlasticDataPtr->mStrainPtr = commonHenckyDataPtr->getMatLogC();
    commonPlasticDataPtr->mStressPtr =
        commonHenckyDataPtr->getMatHenckyStress();
  }

  MoFEMFunctionReturn(0);
}
//! [Create common data]

//! [Boundary condition]
MoFEMErrorCode Example::bC() {
  MoFEMFunctionBegin;

  auto simple = mField.getInterface<Simple>();
  auto bc_mng = mField.getInterface<BcManager>();
  auto prb_mng = mField.getInterface<ProblemsManager>();

  CHKERR bc_mng->removeBlockDOFsOnEntities(simple->getProblemName(), "REMOVE_X",
                                           "U", 0, 0);
  CHKERR bc_mng->removeBlockDOFsOnEntities(simple->getProblemName(), "REMOVE_Y",
                                           "U", 1, 1);
  CHKERR bc_mng->removeBlockDOFsOnEntities(simple->getProblemName(), "REMOVE_Z",
                                           "U", 2, 2);
  CHKERR bc_mng->removeBlockDOFsOnEntities(simple->getProblemName(),
                                           "REMOVE_ALL", "U", 0, 3);

  CHKERR bc_mng->pushMarkDOFsOnEntities<DisplacementCubitBcData>(
      simple->getProblemName(), "U");

  auto &bc_map = bc_mng->getBcMapByBlockName();
  boundaryMarker =
      bc_mng->getMergedBlocksMarker(vector<string>{"FIX_", "ROTATE_"});

  CHKERR bc_mng->pushMarkDOFsOnEntities(simple->getProblemName(), "REACTION",
                                        "U", 0, 3);

  for (auto bc : bc_map)
    MOFEM_LOG("EXAMPLE", Sev::verbose) << "Marker " << bc.first;

  // OK. We have problem with GMesh, it adding empty characters at the end of
  // block. So first block is search by regexp. popMarkDOFsOnEntities should
  // work with regexp.
  std::string reaction_block_set;
  for (auto bc : bc_map) {
    if (bc_mng->checkBlock(bc, "REACTION")) {
      reaction_block_set = bc.first;
      break;
    }
  }

  if (auto bc = bc_mng->popMarkDOFsOnEntities(reaction_block_set)) {
    reactionMarker = bc->getBcMarkersPtr();

    // Only take reaction from nodes
    Range nodes;
    CHKERR mField.get_moab().get_entities_by_type(0, MBVERTEX, nodes, true);
    CHKERR prb_mng->markDofs(simple->getProblemName(), ROW,
                             ProblemsManager::MarkOP::AND, nodes,
                             *reactionMarker);

  } else {
    MOFEM_LOG("EXAMPLE", Sev::warning) << "REACTION blockset does not exist";
  }

  if (!reactionMarker) {
    MOFEM_LOG("EXAMPLE", Sev::warning) << "REACTION blockset does not exist";
  }

  MoFEMFunctionReturn(0);
}
//! [Boundary condition]

//! [Push operators to pipeline]
MoFEMErrorCode Example::OPs() {
  MoFEMFunctionBegin;
  auto pip = mField.getInterface<PipelineManager>();
  auto simple = mField.getInterface<Simple>();
  auto bc_mng = mField.getInterface<BcManager>();

  auto add_domain_ops_lhs_mechanical = [&](auto &pipeline) {
    MoFEMFunctionBegin;

    CHKERR AddHOOps<SPACE_DIM, SPACE_DIM, SPACE_DIM>::add(pipeline, {H1},
                                                          "GEOMETRY");

    pipeline.push_back(new OpSetBc("U", true, boundaryMarker));

    CHKERR PlasticOps::opFactoryDomainLhs<DomainEleOp, A, G>(pipeline, "U",
                                                             "EP", "TAU");

    pipeline.push_back(new OpUnSetBc("U"));
    MoFEMFunctionReturn(0);
  };

  auto add_domain_ops_rhs_mechanical = [&](auto &pipeline) {
    MoFEMFunctionBegin;

    CHKERR AddHOOps<SPACE_DIM, SPACE_DIM, SPACE_DIM>::add(pipeline, {H1},
                                                          "GEOMETRY");

    pipeline.push_back(new OpSetBc("U", true, boundaryMarker));

    CHKERR DomainNaturalBC::AddFluxToPipeline<OpBodyForce>::add(
        pipeline, mField, "U", {boost::make_shared<PlasticityTimeScale>()},
        "BODY_FORCE", Sev::inform);

    CHKERR PlasticOps::opFactoryDomainRhs<DomainEleOp, A, G>(pipeline, "U",
                                                             "EP", "TAU");

    pipeline.push_back(new OpUnSetBc("U"));
    MoFEMFunctionReturn(0);
  };

  auto add_boundary_ops_lhs_mechanical = [&](auto &pipeline) {
    MoFEMFunctionBegin;
    CHKERR EssentialBC<BoundaryEleOp>::Assembly<A>::BiLinearForm<G>::
        AddEssentialToPipeline<OpEssentialLhs>::add(
            mField, pipeline, simple->getProblemName(), "U");
    MoFEMFunctionReturn(0);
  };

  auto add_boundary_ops_rhs_mechanical = [&](auto &pipeline) {
    MoFEMFunctionBegin;

    CHKERR AddHOOps<SPACE_DIM - 1, SPACE_DIM, SPACE_DIM>::add(
        pipeline, {NOSPACE}, "GEOMETRY");

    pipeline.push_back(new OpSetBc("U", true, boundaryMarker));
    CHKERR BoundaryNaturalBC::AddFluxToPipeline<OpForce>::add(
        pipeline, mField, "U", {boost::make_shared<PlasticityTimeScale>()},
        "FORCE", Sev::inform);

    pipeline.push_back(new OpUnSetBc("U"));

    auto u_mat_ptr = boost::make_shared<MatrixDouble>();
    pipeline.push_back(
        new OpCalculateVectorFieldValues<SPACE_DIM>("U", u_mat_ptr));

    CHKERR EssentialBC<BoundaryEleOp>::Assembly<A>::LinearForm<G>::
        AddEssentialToPipeline<OpEssentialRhs>::add(
            mField, pipeline, simple->getProblemName(), "U", u_mat_ptr,
            {boost::make_shared<TimeScale>()}); // note displacements have no
                                                // scaling

    MoFEMFunctionReturn(0);
  };

  // Domain
  CHKERR add_domain_ops_lhs_mechanical(pip->getOpDomainLhsPipeline());
  CHKERR add_domain_ops_rhs_mechanical(pip->getOpDomainRhsPipeline());
  
  // Boundary
  CHKERR add_boundary_ops_lhs_mechanical(pip->getOpBoundaryLhsPipeline());
  CHKERR add_boundary_ops_rhs_mechanical(pip->getOpBoundaryRhsPipeline());

  auto integration_rule_bc = [](int, int, int ao) { return 2 * ao; };

  auto vol_rule = [](int, int, int ao) { return 2 * ao + geom_order - 1; };

  CHKERR pip->setDomainRhsIntegrationRule(vol_rule);
  CHKERR pip->setDomainLhsIntegrationRule(vol_rule);

  CHKERR pip->setBoundaryLhsIntegrationRule(integration_rule_bc);
  CHKERR pip->setBoundaryRhsIntegrationRule(integration_rule_bc);

  auto create_reaction_pipeline = [&](auto &pipeline) {
    MoFEMFunctionBegin;

    if (reactionMarker) {

      CHKERR AddHOOps<SPACE_DIM, SPACE_DIM, SPACE_DIM>::add(pipeline, {H1},
                                                            "GEOMETRY");

      pipeline.push_back(
          new OpCalculateVectorFieldGradient<SPACE_DIM, SPACE_DIM>(
              "U", commonPlasticDataPtr->mGradPtr));
      pipeline.push_back(new OpCalculateTensor2SymmetricFieldValues<SPACE_DIM>(
          "EP", commonPlasticDataPtr->getPlasticStrainPtr()));

      if (is_large_strains) {

        if (commonPlasticDataPtr->mGradPtr != commonHenckyDataPtr->matGradPtr)
          SETERRQ(PETSC_COMM_SELF, MOFEM_DATA_INCONSISTENCY,
                  "Wrong pointer for grad");

        pipeline.push_back(
            new OpCalculateEigenVals<SPACE_DIM>("U", commonHenckyDataPtr));
        pipeline.push_back(
            new OpCalculateLogC<SPACE_DIM>("U", commonHenckyDataPtr));
        pipeline.push_back(
            new OpCalculateLogC_dC<SPACE_DIM>("U", commonHenckyDataPtr));
        pipeline.push_back(new OpCalculateHenckyPlasticStress<SPACE_DIM>(
            "U", commonHenckyDataPtr, commonPlasticDataPtr->mDPtr, scale));
        pipeline.push_back(
            new OpCalculatePiolaStress<SPACE_DIM>("U", commonHenckyDataPtr));

      } else {
        pipeline.push_back(new OpSymmetrizeTensor<SPACE_DIM>(
            "U", commonPlasticDataPtr->mGradPtr,
            commonPlasticDataPtr->mStrainPtr));
        pipeline.push_back(new OpPlasticStress(
            "U", commonPlasticDataPtr, commonPlasticDataPtr->mDPtr, scale));
      }

      pipeline.push_back(new OpSetBc("U", false, reactionMarker));
      // Calculate internal force
      if (is_large_strains) {
        pipeline.push_back(new OpInternalForcePiola(
            "U", commonHenckyDataPtr->getMatFirstPiolaStress()));
      } else {
        pipeline.push_back(
            new OpInternalForceCauchy("U", commonPlasticDataPtr->mStressPtr));
      }
      pipeline.push_back(new OpUnSetBc("U"));
    }

    MoFEMFunctionReturn(0);
  };

  reactionFe = boost::make_shared<DomainEle>(mField);
  reactionFe->getRuleHook = vol_rule;

  CHKERR create_reaction_pipeline(reactionFe->getOpPtrVector());

  MoFEMFunctionReturn(0);
}
//! [Push operators to pipeline]

//! [Solve]
struct SetUpSchur {

  /**
   * @brief Create data structure for handling Schur complement
   *
   * @param m_field
   * @param sub_dm  Schur complement sub dm
   * @param field_split_it IS of Schur block
   * @param ao_map AO map from sub dm to main problem
   * @return boost::shared_ptr<SetUpSchur>
   */
  static boost::shared_ptr<SetUpSchur> createSetUpSchur(

      MoFEM::Interface &m_field, SmartPetscObj<DM> sub_dm,
      SmartPetscObj<IS> field_split_it, SmartPetscObj<AO> ao_map

  );
  virtual MoFEMErrorCode setUp(KSP solver) = 0;

  virtual MoFEMErrorCode preProc() = 0;
  virtual MoFEMErrorCode postProc() = 0;

protected:
  SetUpSchur() = default;
};

MoFEMErrorCode Example::tsSolve() {
  MoFEMFunctionBegin;

  Simple *simple = mField.getInterface<Simple>();
  PipelineManager *pip = mField.getInterface<PipelineManager>();
  ISManager *is_manager = mField.getInterface<ISManager>();

  auto snes_ctx_ptr = getDMSnesCtx(simple->getDM());

  auto set_section_monitor = [&](auto solver) {
    MoFEMFunctionBegin;
    SNES snes;
    CHKERR TSGetSNES(solver, &snes);
    CHKERR SNESMonitorSet(snes,
                          (MoFEMErrorCode(*)(SNES, PetscInt, PetscReal,
                                             void *))MoFEMSNESMonitorFields,
                          (void *)(snes_ctx_ptr.get()), nullptr);
    MoFEMFunctionReturn(0);
  };

  auto create_post_process_element = [&]() {
    auto pp_fe = boost::make_shared<PostProcEle>(mField);
    using OpPPMap = OpPostProcMapInMoab<SPACE_DIM, SPACE_DIM>;

    CHKERR AddHOOps<SPACE_DIM, SPACE_DIM, SPACE_DIM>::add(
        pp_fe->getOpPtrVector(), {H1}, "GEOMETRY");

    auto x_ptr = boost::make_shared<MatrixDouble>();
    pp_fe->getOpPtrVector().push_back(
        new OpCalculateVectorFieldValues<SPACE_DIM>("GEOMETRY", x_ptr));
    auto u_ptr = boost::make_shared<MatrixDouble>();
    pp_fe->getOpPtrVector().push_back(
        new OpCalculateVectorFieldValues<SPACE_DIM>("U", u_ptr));

    pp_fe->getOpPtrVector().push_back(
        new OpCalculateVectorFieldGradient<SPACE_DIM, SPACE_DIM>(
            "U", commonPlasticDataPtr->mGradPtr));
    pp_fe->getOpPtrVector().push_back(new OpCalculateScalarFieldValues(
        "TAU", commonPlasticDataPtr->getPlasticTauPtr()));
    pp_fe->getOpPtrVector().push_back(
        new OpCalculateTensor2SymmetricFieldValues<SPACE_DIM>(
            "EP", commonPlasticDataPtr->getPlasticStrainPtr()));

    if (is_large_strains) {

      if (commonPlasticDataPtr->mGradPtr != commonHenckyDataPtr->matGradPtr)
        CHK_THROW_MESSAGE(MOFEM_DATA_INCONSISTENCY, "Wrong pointer for grad");

      pp_fe->getOpPtrVector().push_back(
          new OpCalculateEigenVals<SPACE_DIM>("U", commonHenckyDataPtr));
      pp_fe->getOpPtrVector().push_back(
          new OpCalculateLogC<SPACE_DIM>("U", commonHenckyDataPtr));
      pp_fe->getOpPtrVector().push_back(
          new OpCalculateLogC_dC<SPACE_DIM>("U", commonHenckyDataPtr));
      pp_fe->getOpPtrVector().push_back(
          new OpCalculateHenckyPlasticStress<SPACE_DIM>(
              "U", commonHenckyDataPtr, commonPlasticDataPtr->mDPtr, scale));
      pp_fe->getOpPtrVector().push_back(
          new OpCalculatePiolaStress<SPACE_DIM>("U", commonHenckyDataPtr));

      pp_fe->getOpPtrVector().push_back(

          new OpPPMap(

              pp_fe->getPostProcMesh(), pp_fe->getMapGaussPts(),

              {},

              {{"U", u_ptr}, {"GEOMETRY", x_ptr}},

              {{"GRAD", commonPlasticDataPtr->mGradPtr},
               {"FIRST_PIOLA", commonHenckyDataPtr->getMatFirstPiolaStress()}},

              {}

              )

      );

    } else {
      pp_fe->getOpPtrVector().push_back(
          new OpSymmetrizeTensor<SPACE_DIM>("U", commonPlasticDataPtr->mGradPtr,
                                            commonPlasticDataPtr->mStrainPtr));
      pp_fe->getOpPtrVector().push_back(new OpPlasticStress(
          "U", commonPlasticDataPtr, commonPlasticDataPtr->mDPtr, scale));

      pp_fe->getOpPtrVector().push_back(

          new OpPPMap(

              pp_fe->getPostProcMesh(), pp_fe->getMapGaussPts(),

              {},

              {{"U", u_ptr}, {"GEOMETRY", x_ptr}},

              {},

              {{"STRAIN", commonPlasticDataPtr->mStrainPtr},
               {"STRESS", commonPlasticDataPtr->mStressPtr}}

              )

      );
    }

    pp_fe->getOpPtrVector().push_back(
        new OpCalculatePlasticSurface("U", commonPlasticDataPtr));

    pp_fe->getOpPtrVector().push_back(

        new OpPPMap(

            pp_fe->getPostProcMesh(), pp_fe->getMapGaussPts(),

            {{"PLASTIC_SURFACE", commonPlasticDataPtr->getPlasticSurfacePtr()},
             {"PLASTIC_MULTIPLIER", commonPlasticDataPtr->getPlasticTauPtr()}},

            {},

            {},

            {{"PLASTIC_STRAIN", commonPlasticDataPtr->getPlasticStrainPtr()},
             {"PLASTIC_FLOW", commonPlasticDataPtr->getPlasticFlowPtr()}}

            )

    );

    return pp_fe;
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
        new Monitor(dm, create_post_process_element(), reactionFe, uXScatter,
                    uYScatter, uZScatter));
    boost::shared_ptr<ForcesAndSourcesCore> null;
    CHKERR DMMoFEMTSSetMonitor(dm, solver, simple->getDomainFEName(),
                               monitor_ptr, null, null);
    MoFEMFunctionReturn(0);
  };

  auto set_fieldsplit_preconditioner = [&](auto solver,
                                           boost::shared_ptr<SetUpSchur>
                                               &schur_ptr) {
    MoFEMFunctionBeginHot;

    SNES snes;
    CHKERR TSGetSNES(solver, &snes);
    KSP ksp;
    CHKERR SNESGetKSP(snes, &ksp);
    PC pc;
    CHKERR KSPGetPC(ksp, &pc);
    PetscBool is_pcfs = PETSC_FALSE;
    PetscObjectTypeCompare((PetscObject)pc, PCFIELDSPLIT, &is_pcfs);

    // Setup fieldsplit (block) solver - optional: yes/no
    if (is_pcfs == PETSC_TRUE) {

      auto bc_mng = mField.getInterface<BcManager>();
      auto name_prb = simple->getProblemName();

      // create sub dm to handle boundary conditions (least square)
      auto create_sub_bc_dm = [&](SmartPetscObj<DM> base_dm,
                                  SmartPetscObj<DM> &dm_sub,
                                  SmartPetscObj<IS> &is_sub,
                                  SmartPetscObj<AO> &ao_sub) {
        MoFEMFunctionBegin;

        dm_sub = createDM(mField.get_comm(), "DMMOFEM");
        CHKERR DMMoFEMCreateSubDM(dm_sub, base_dm, "SUB_BC");
        CHKERR DMMoFEMSetSquareProblem(dm_sub, PETSC_TRUE);
        CHKERR DMMoFEMAddElement(dm_sub, simple->getDomainFEName());
        CHKERR DMMoFEMAddElement(dm_sub, simple->getBoundaryFEName());
        for (auto f : {"U", "EP", "TAU"}) {
          CHKERR DMMoFEMAddSubFieldRow(dm_sub, f);
          CHKERR DMMoFEMAddSubFieldCol(dm_sub, f);
        }
        CHKERR DMSetUp(dm_sub);

        CHKERR bc_mng->removeBlockDOFsOnEntities("SUB_BC", "FIX_X", "U", 0, 0);
        CHKERR bc_mng->removeBlockDOFsOnEntities("SUB_BC", "FIX_Y", "U", 1, 1);
        CHKERR bc_mng->removeBlockDOFsOnEntities("SUB_BC", "FIX_Z", "U", 2, 2);
        CHKERR bc_mng->removeBlockDOFsOnEntities("SUB_BC", "FIX_ALL", "U", 0,
                                                 2);

        auto *prb_ptr = getProblemPtr(dm_sub);
        if (auto sub_data = prb_ptr->getSubData()) {
          is_sub = sub_data->getSmartRowIs();
          ao_sub = sub_data->getSmartRowMap();
          int is_sub_size;
          CHKERR ISGetSize(is_sub, &is_sub_size);
          MOFEM_LOG("EXAMPLE", Sev::inform)
              << "Field split second block size " << is_sub_size;

        } else {
          SETERRQ(PETSC_COMM_WORLD, MOFEM_DATA_INCONSISTENCY, "No sub data");
        }

        MoFEMFunctionReturn(0);
      };

      // create sub dm for Schur complement
      auto create_sub_u_dm = [&](SmartPetscObj<DM> base_dm,
                                 SmartPetscObj<DM> &dm_sub) {
        MoFEMFunctionBegin;
        dm_sub = createDM(mField.get_comm(), "DMMOFEM");
        CHKERR DMMoFEMCreateSubDM(dm_sub, base_dm, "SUB_U");
        CHKERR DMMoFEMSetSquareProblem(dm_sub, PETSC_TRUE);
        CHKERR DMMoFEMAddElement(dm_sub, simple->getDomainFEName());
        CHKERR DMMoFEMAddElement(dm_sub, simple->getBoundaryFEName());
        for (auto f : {"U"}) {
          CHKERR DMMoFEMAddSubFieldRow(dm_sub, f);
          CHKERR DMMoFEMAddSubFieldCol(dm_sub, f);
        }
        CHKERR DMSetUp(dm_sub);

        MoFEMFunctionReturn(0);
      };

      // get IS for all boundary conditions
      auto create_all_bc_is = [&](SmartPetscObj<IS> &is_all_bc) {
        MoFEMFunctionBegin;
        is_all_bc = bc_mng->getBlockIS(name_prb, "FIX_X", "U", 0, 0);
        is_all_bc = bc_mng->getBlockIS(name_prb, "FIX_Y", "U", 1, 1, is_all_bc);
        is_all_bc = bc_mng->getBlockIS(name_prb, "FIX_Z", "U", 2, 2, is_all_bc);
        is_all_bc =
            bc_mng->getBlockIS(name_prb, "FIX_ALL", "U", 0, 2, is_all_bc);
        int is_all_bc_size;
        CHKERR ISGetSize(is_all_bc, &is_all_bc_size);
        MOFEM_LOG("EXAMPLE", Sev::inform)
            << "Field split first block size " << is_all_bc_size;
        MoFEMFunctionReturn(0);
      };

      SmartPetscObj<IS> is_all_bc;
      SmartPetscObj<DM> dm_bc_sub;
      SmartPetscObj<IS> is_bc_sub;
      SmartPetscObj<AO> ao_bc_sub;

      CHKERR create_all_bc_is(is_all_bc);
      // note that Schur dm is sub dm for boundary conditions, i.e. is nested.
      CHKERR create_sub_bc_dm(simple->getDM(), dm_bc_sub, is_bc_sub, ao_bc_sub);

      // Create field split for boundary conditions
      CHKERR PCFieldSplitSetIS(pc, PETSC_NULL,
                               is_all_bc); // boundary block
      CHKERR PCFieldSplitSetIS(pc, PETSC_NULL, is_bc_sub);

      // Create nested (sub BC) Schur DM
      if constexpr (A == AssemblyType::SCHUR) {

        SmartPetscObj<IS> is_epp;
        CHKERR mField.getInterface<ISManager>()->isCreateProblemFieldAndRank(
            "SUB_BC", ROW, "EP", 0, MAX_DOFS_ON_ENTITY, is_epp);
        SmartPetscObj<IS> is_tau;
        CHKERR mField.getInterface<ISManager>()->isCreateProblemFieldAndRank(
            "SUB_BC", ROW, "TAU", 0, MAX_DOFS_ON_ENTITY, is_tau);
        IS is_union_raw;
        CHKERR ISExpand(is_epp, is_tau, &is_union_raw);
        SmartPetscObj<IS> is_union(is_union_raw);

        SmartPetscObj<DM> dm_u_sub;
        CHKERR create_sub_u_dm(dm_bc_sub, dm_u_sub);

        // Indices has to be map fro very to level, while assembling Schur
        // complement.
        auto is_up = getDMSubData(dm_u_sub)->getSmartRowIs();
        CHKERR AOPetscToApplicationIS(ao_bc_sub, is_up);
        auto ao_up = createAOMappingIS(is_up, PETSC_NULL);
        schur_ptr =
            SetUpSchur::createSetUpSchur(mField, dm_u_sub, is_union, ao_up);
        PetscInt n;
        KSP *ksps;
        CHKERR PCFieldSplitGetSubKSP(pc, &n, &ksps);
        CHKERR schur_ptr->setUp(
            ksps[1]); // note that FS is applied in second block of boundary BC
      }
    }

    MoFEMFunctionReturnHot(0);
  };

  auto dm = simple->getDM();
  auto D = createDMVector(dm);
  uXScatter = scatter_create(D, 0);
  uYScatter = scatter_create(D, 1);
  if constexpr (SPACE_DIM == 3)
    uZScatter = scatter_create(D, 2);

  auto solver = pip->createTSIM();

  auto active_pre_lhs = [&]() {
    MoFEMFunctionBegin;
    std::fill(commonPlasticDataPtr->activityData.begin(),
              commonPlasticDataPtr->activityData.end(), 0);
    MoFEMFunctionReturn(0);
  };

  auto active_post_lhs = [&]() {
    MoFEMFunctionBegin;
    auto get_iter = [&]() {
      SNES snes;
      CHK_THROW_MESSAGE(TSGetSNES(solver, &snes), "Can not get SNES");
      int iter;
      CHK_THROW_MESSAGE(SNESGetIterationNumber(snes, &iter),
                        "Can not get iter");
      return iter;
    };

    auto iter = get_iter();
    if (iter >= 0) {

      std::array<int, 5> activity_data;
      std::fill(activity_data.begin(), activity_data.end(), 0);
      MPI_Allreduce(commonPlasticDataPtr->activityData.data(),
                    activity_data.data(), activity_data.size(), MPI_INT,
                    MPI_SUM, mField.get_comm());

      int &active_points = activity_data[0];
      int &avtive_full_elems = activity_data[1];
      int &avtive_elems = activity_data[2];
      int &nb_points = activity_data[3];
      int &nb_elements = activity_data[4];

      if (nb_points) {

        double proc_nb_points =
            100 * static_cast<double>(active_points) / nb_points;
        double proc_nb_active =
            100 * static_cast<double>(avtive_elems) / nb_elements;
        double proc_nb_full_active = 100;
        if (avtive_elems)
          proc_nb_full_active =
              100 * static_cast<double>(avtive_full_elems) / avtive_elems;

        MOFEM_LOG_C("EXAMPLE", Sev::inform,
                    "Iter %d nb pts %d nb avtive pts %d (%3.3f\%) nb active "
                    "elements %d "
                    "(%3.3f\%) nb full active elems %d (%3.3f\%)",
                    iter, nb_points, active_points, proc_nb_points,
                    avtive_elems, proc_nb_active, avtive_full_elems,
                    proc_nb_full_active, iter);
      }
    }

    MoFEMFunctionReturn(0);
  };

  CHKERR TSSetSolution(solver, D);
  CHKERR set_section_monitor(solver);
  CHKERR set_time_monitor(dm, solver);
  CHKERR TSSetSolution(solver, D);
  CHKERR TSSetFromOptions(solver);

  boost::shared_ptr<SetUpSchur> schur_ptr;
  CHKERR set_fieldsplit_preconditioner(solver, schur_ptr);

  // Domain element is run first by TSSolber, thus run Schur pre-proc, which
  // clears Schur complement matrix
  mField.getInterface<PipelineManager>()->getDomainLhsFE()->preProcessHook =
      [&]() {
        MoFEMFunctionBegin;
        if (schur_ptr)
          CHKERR schur_ptr->preProc();
        CHKERR active_pre_lhs();
        MoFEMFunctionReturn(0);
      };
  // Do nothing, assemble after integrating boundary
  mField.getInterface<PipelineManager>()->getDomainLhsFE()->postProcessHook =
      [&]() {
        MoFEMFunctionBegin;
        CHKERR active_post_lhs();
        MoFEMFunctionReturn(0);
      };
  // Assemble matrices in post-proc of boundary pipeline
  mField.getInterface<PipelineManager>()->getBoundaryLhsFE()->postProcessHook =
      [&]() {
        MoFEMFunctionBegin;
        if (schur_ptr)
          CHKERR schur_ptr->postProc();
        MoFEMFunctionReturn(0);
      };

  MOFEM_LOG_CHANNEL("TIMER");
  MOFEM_LOG_TAG("TIMER", "timer");
  if (set_timer)
    BOOST_LOG_SCOPED_THREAD_ATTR("Timeline", attrs::timer());
  MOFEM_LOG("TIMER", Sev::verbose) << "TSSetUp";
  CHKERR TSSetUp(solver);
  MOFEM_LOG("TIMER", Sev::verbose) << "TSSetUp <= done";
  MOFEM_LOG("TIMER", Sev::verbose) << "TSSolve";
  CHKERR TSSolve(solver, NULL);
  MOFEM_LOG("TIMER", Sev::verbose) << "TSSolve <= done";

  CHKERR VecGhostUpdateBegin(D, INSERT_VALUES, SCATTER_FORWARD);
  CHKERR VecGhostUpdateEnd(D, INSERT_VALUES, SCATTER_FORWARD);
  CHKERR DMoFEMMeshToLocalVector(dm, D, INSERT_VALUES, SCATTER_REVERSE);

  MoFEMFunctionReturn(0);
}
//! [Solve]

static char help[] = "...\n\n";

int main(int argc, char *argv[]) {

  // Initialisation of MoFEM/PETSc and MOAB data structures
  const char param_file[] = "param_file.petsc";
  MoFEM::Core::Initialize(&argc, &argv, param_file, help);

  // Add logging channel for example
  auto core_log = logging::core::get();
  core_log->add_sink(
      LogManager::createSink(LogManager::getStrmWorld(), "EXAMPLE"));
  core_log->add_sink(
      LogManager::createSink(LogManager::getStrmWorld(), "TIMER"));
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

struct SetUpSchurImpl : public SetUpSchur {

  SetUpSchurImpl(MoFEM::Interface &m_field, SmartPetscObj<DM> sub_dm,
                 SmartPetscObj<IS> field_split_is, SmartPetscObj<AO> ao_up)
      : SetUpSchur(), mField(m_field), subDM(sub_dm),
        fieldSplitIS(field_split_is), aoUp(ao_up) {
    if (S) {
      CHK_THROW_MESSAGE(
          MOFEM_DATA_INCONSISTENCY,
          "Is expected that schur matrix is not allocated. This is "
          "possible only is PC is set up twice");
    }
  }
  virtual ~SetUpSchurImpl() { S.reset(); }

  MoFEMErrorCode setUp(KSP solver);
  MoFEMErrorCode preProc();
  MoFEMErrorCode postProc();

private:
  SmartPetscObj<Mat> S;

  MoFEM::Interface &mField;
  SmartPetscObj<DM> subDM;        ///< field split sub dm
  SmartPetscObj<IS> fieldSplitIS; ///< IS for split Schur block
  SmartPetscObj<AO> aoUp;         ///> main DM to subDM
};

MoFEMErrorCode SetUpSchurImpl::setUp(KSP solver) {
  MoFEMFunctionBegin;
  auto pip = mField.getInterface<PipelineManager>();
  PC pc;
  CHKERR KSPSetFromOptions(solver);
  CHKERR KSPGetPC(solver, &pc);
  PetscBool is_pcfs = PETSC_FALSE;
  PetscObjectTypeCompare((PetscObject)pc, PCFIELDSPLIT, &is_pcfs);
  if (is_pcfs) {
    if (S) {
      CHK_THROW_MESSAGE(
          MOFEM_DATA_INCONSISTENCY,
          "Is expected that schur matrix is not allocated. This is "
          "possible only is PC is set up twice");
    }
    S = createDMMatrix(subDM);

    auto set_ops = [&]() {
      MoFEMFunctionBegin;
      auto pip = mField.getInterface<PipelineManager>();
      // Boundary
      pip->getOpBoundaryLhsPipeline().push_front(new OpSchurAssembleBegin());
      pip->getOpBoundaryLhsPipeline().push_back(
          new OpSchurAssembleEnd<SCHUR_DGESV>(

              {"EP", "TAU"}, {nullptr, nullptr}, {SmartPetscObj<AO>(), aoUp},
              {SmartPetscObj<Mat>(), S}, {false, false}

              ));
      // Domain
      pip->getOpDomainLhsPipeline().push_front(new OpSchurAssembleBegin());
      pip->getOpDomainLhsPipeline().push_back(
          new OpSchurAssembleEnd<SCHUR_DGESV>(

              {"EP", "TAU"}, {nullptr, nullptr}, {SmartPetscObj<AO>(), aoUp},
              {SmartPetscObj<Mat>(), S}, {false, false}

              ));
      MoFEMFunctionReturn(0);
    };

    auto set_pc = [&]() {
      MoFEMFunctionBegin;
      CHKERR PCFieldSplitSetIS(pc, NULL, fieldSplitIS);
      CHKERR PCFieldSplitSetSchurPre(pc, PC_FIELDSPLIT_SCHUR_PRE_USER, S);
      MoFEMFunctionReturn(0);
    };

    CHKERR set_ops();
    CHKERR set_pc();

  } else {
    pip->getOpBoundaryLhsPipeline().push_front(new OpSchurAssembleBegin());
    pip->getOpBoundaryLhsPipeline().push_back(
        new OpSchurAssembleEnd<SCHUR_DGESV>({}, {}, {}, {}, {}));
    pip->getOpDomainLhsPipeline().push_front(new OpSchurAssembleBegin());
    pip->getOpDomainLhsPipeline().push_back(
        new OpSchurAssembleEnd<SCHUR_DGESV>({}, {}, {}, {}, {}));
  }

  // we do not those anymore
  subDM.reset();
  fieldSplitIS.reset();
  aoUp.reset();
  MoFEMFunctionReturn(0);
}

MoFEMErrorCode SetUpSchurImpl::preProc() {
  MoFEMFunctionBegin;
  if (SetUpSchurImpl::S) {
    CHKERR MatZeroEntries(S);
  }
  MOFEM_LOG("TIMER", Sev::verbose) << "Lhs Assemble Begin";
  MoFEMFunctionReturn(0);
}

MoFEMErrorCode SetUpSchurImpl::postProc() {
  MoFEMFunctionBegin;
  MOFEM_LOG("TIMER", Sev::verbose) << "Lhs Assemble End";
  if (S) {
    CHKERR MatAssemblyBegin(S, MAT_FINAL_ASSEMBLY);
    CHKERR MatAssemblyEnd(S, MAT_FINAL_ASSEMBLY);
  }
  MoFEMFunctionReturn(0);
}

boost::shared_ptr<SetUpSchur>
SetUpSchur::createSetUpSchur(MoFEM::Interface &m_field,
                             SmartPetscObj<DM> sub_dm, SmartPetscObj<IS> is_sub,
                             SmartPetscObj<AO> ao_up) {
  return boost::shared_ptr<SetUpSchur>(
      new SetUpSchurImpl(m_field, sub_dm, is_sub, ao_up));
}