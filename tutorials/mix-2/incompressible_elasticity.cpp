/**
 * @file incompressible_elasticity.cpp
 * @brief Incompressible elasticity problem
 */

#include <MoFEM.hpp>

using namespace MoFEM;

constexpr int SPACE_DIM =
    EXECUTABLE_DIMENSION; //< Space dimension of problem, mesh

constexpr AssemblyType AT =
    (SCHUR_ASSEMBLE) ? AssemblyType::SCHUR
                     : AssemblyType::PETSC; //< selected assembly type

constexpr IntegrationType IT =
    IntegrationType::GAUSS; //< selected integration type
constexpr CoordinateTypes coord_type = CARTESIAN;

using EntData = EntitiesFieldData::EntData;
using DomainEle = PipelineManager::ElementsAndOpsByDim<SPACE_DIM>::DomainEle;
using DomainEleOp = DomainEle::UserDataOperator;
using BoundaryEle =
    PipelineManager::ElementsAndOpsByDim<SPACE_DIM>::BoundaryEle;
using BoundaryEleOp = BoundaryEle::UserDataOperator;
using PostProcEle = PostProcBrokenMeshInMoab<DomainEle>;
using SkinPostProcEle = PostProcBrokenMeshInMoab<BoundaryEle>;

struct MonitorIncompressible : public FEMethod {

  MonitorIncompressible(
      SmartPetscObj<DM> dm,
      std::pair<boost::shared_ptr<PostProcEle>,
                boost::shared_ptr<SkinPostProcEle>>
          pair_post_proc_fe,
      std::tuple<SmartPetscObj<Vec>, SmartPetscObj<VecScatter>> ux_scatter,
      std::tuple<SmartPetscObj<Vec>, SmartPetscObj<VecScatter>> uy_scatter,
      std::tuple<SmartPetscObj<Vec>, SmartPetscObj<VecScatter>> uz_scatter)
      : dM(dm), uXScatter(ux_scatter), uYScatter(uy_scatter),
        uZScatter(uz_scatter) {
    postProcFe = pair_post_proc_fe.first;
    skinPostProcFe = pair_post_proc_fe.second;
  };

  MoFEMErrorCode preProcess() { return 0; }
  MoFEMErrorCode operator()() { return 0; }

  MoFEMErrorCode postProcess() {
    MoFEMFunctionBegin;

    MoFEM::Interface *m_field_ptr;
    CHKERR DMoFEMGetInterfacePtr(dM, &m_field_ptr);

    auto make_vtk = [&]() {
      MoFEMFunctionBegin;
      if (postProcFe) {
        CHKERR DMoFEMLoopFiniteElements(dM, "dFE", postProcFe,
                                        getCacheWeakPtr());
        CHKERR postProcFe->writeFile("out_incomp_elasticity" +
                                     boost::lexical_cast<std::string>(ts_step) +
                                     ".h5m");
      }
      if (skinPostProcFe) {
        CHKERR DMoFEMLoopFiniteElements(dM, "bFE", skinPostProcFe,
                                        getCacheWeakPtr());
        CHKERR skinPostProcFe->writeFile(
            "out_skin_incomp_elasticity_" +
            boost::lexical_cast<std::string>(ts_step) + ".h5m");
      }
      MoFEMFunctionReturn(0);
    };

    auto print_max_min = [&](auto &tuple, const std::string msg) {
      MoFEMFunctionBegin;
      CHKERR VecScatterBegin(std::get<1>(tuple), ts_u, std::get<0>(tuple),
                             INSERT_VALUES, SCATTER_FORWARD);
      CHKERR VecScatterEnd(std::get<1>(tuple), ts_u, std::get<0>(tuple),
                           INSERT_VALUES, SCATTER_FORWARD);
      double max, min;
      CHKERR VecMax(std::get<0>(tuple), PETSC_NULL, &max);
      CHKERR VecMin(std::get<0>(tuple), PETSC_NULL, &min);
      MOFEM_LOG_C("INCOMP_ELASTICITY", Sev::inform,
                  "%s time %3.4e min %3.4e max %3.4e", msg.c_str(), ts_t, min,
                  max);
      MoFEMFunctionReturn(0);
    };

    CHKERR make_vtk();
    CHKERR print_max_min(uXScatter, "Ux");
    CHKERR print_max_min(uYScatter, "Uy");
    if constexpr (SPACE_DIM == 3)
      CHKERR print_max_min(uZScatter, "Uz");

    MoFEMFunctionReturn(0);
  }

private:
  SmartPetscObj<DM> dM;
  boost::shared_ptr<PostProcEle> postProcFe;
  boost::shared_ptr<SkinPostProcEle> skinPostProcFe;
  std::tuple<SmartPetscObj<Vec>, SmartPetscObj<VecScatter>> uXScatter;
  std::tuple<SmartPetscObj<Vec>, SmartPetscObj<VecScatter>> uYScatter;
  std::tuple<SmartPetscObj<Vec>, SmartPetscObj<VecScatter>> uZScatter;
};

// Assemble to A matrix, by default, however, some terms are assembled only to
// preconditioning.

template <>
typename MoFEM::OpBaseImpl<AT, DomainEleOp>::MatSetValuesHook
    MoFEM::OpBaseImpl<AT, DomainEleOp>::matSetValuesHook =
        [](ForcesAndSourcesCore::UserDataOperator *op_ptr,
           const EntitiesFieldData::EntData &row_data,
           const EntitiesFieldData::EntData &col_data, MatrixDouble &m) {
          return MatSetValues<AssemblyTypeSelector<AT>>(
              op_ptr->getKSPA(), row_data, col_data, m, ADD_VALUES);
        };

/**
 * @brief Element used to specialise assembly
 *
 */
struct DomainEleOpStab : public DomainEleOp {
  using DomainEleOp::DomainEleOp;
};

/**
 * @brief Specialise assembly for Stabilised matrix
 *
 * @tparam
 */
template <>
typename MoFEM::OpBaseImpl<AT, DomainEleOpStab>::MatSetValuesHook
    MoFEM::OpBaseImpl<AT, DomainEleOpStab>::matSetValuesHook =
        [](ForcesAndSourcesCore::UserDataOperator *op_ptr,
           const EntitiesFieldData::EntData &row_data,
           const EntitiesFieldData::EntData &col_data, MatrixDouble &m) {
          cerr << op_ptr->getKSPA() << " " << op_ptr->getKSPB() << endl;
          return MatSetValues<AssemblyTypeSelector<AT>>(
              op_ptr->getKSPB(), row_data, col_data, m, ADD_VALUES);
        };
//! [Specialisation for assembly]

int order = 2;
int geom_order = 1;
inline static double young_modulus = 100;
inline static double poisson_ratio = 0.25;
inline static double mu;
inline static double lambda;

PetscBool isDiscontinuousPressure = PETSC_FALSE;

struct Incompressible {

  Incompressible(MoFEM::Interface &m_field) : mField(m_field) {}

  MoFEMErrorCode runProblem();

private:
  MoFEM::Interface &mField;

  MoFEMErrorCode setupProblem();
  MoFEMErrorCode createCommonData();
  MoFEMErrorCode bC();
  MoFEMErrorCode OPs();
  MoFEMErrorCode tsSolve();
  MoFEMErrorCode checkResults();

  std::tuple<SmartPetscObj<Vec>, SmartPetscObj<VecScatter>> uXScatter;
  std::tuple<SmartPetscObj<Vec>, SmartPetscObj<VecScatter>> uYScatter;
  std::tuple<SmartPetscObj<Vec>, SmartPetscObj<VecScatter>> uZScatter;
};

template <int DIM>
struct OpCalculateLameStress : public ForcesAndSourcesCore::UserDataOperator {
  OpCalculateLameStress(double m_u, boost::shared_ptr<MatrixDouble> stress_ptr,
                        boost::shared_ptr<MatrixDouble> strain_ptr,
                        boost::shared_ptr<VectorDouble> pressure_ptr)
      : ForcesAndSourcesCore::UserDataOperator(NOSPACE, OPLAST), mU(m_u),
        stressPtr(stress_ptr), strainPtr(strain_ptr),
        pressurePtr(pressure_ptr) {}

  MoFEMErrorCode doWork(int side, EntityType type,
                        EntitiesFieldData::EntData &data) {
    MoFEMFunctionBegin;
    // Define Indicies
    FTensor::Index<'i', SPACE_DIM> i;
    FTensor::Index<'j', SPACE_DIM> j;

    // Define Kronecker Delta
    constexpr auto t_kd = FTensor::Kronecker_Delta_symmetric<double>();

    // Number of Gauss points
    const size_t nb_gauss_pts = getGaussPts().size2();

    stressPtr->resize((DIM * (DIM + 1)) / 2, nb_gauss_pts);
    auto t_stress = getFTensor2SymmetricFromMat<DIM>(*(stressPtr));
    auto t_strain = getFTensor2SymmetricFromMat<DIM>(*(strainPtr));
    auto t_pressure = getFTensor0FromVec(*(pressurePtr));

    const double l_mu = mU;
    // Extract matrix from data matrix
    for (auto gg = 0; gg != nb_gauss_pts; ++gg) {

      t_stress(i, j) = t_pressure * t_kd(i, j) + 2. * l_mu * t_strain(i, j);

      ++t_strain;
      ++t_stress;
      ++t_pressure;
    }

    MoFEMFunctionReturn(0);
  }

private:
  double mU;
  boost::shared_ptr<MatrixDouble> stressPtr;
  boost::shared_ptr<MatrixDouble> strainPtr;
  boost::shared_ptr<VectorDouble> pressurePtr;
};

//! [Run problem]
MoFEMErrorCode Incompressible::runProblem() {
  MoFEMFunctionBegin;
  CHKERR setupProblem();
  CHKERR createCommonData();
  CHKERR bC();
  CHKERR OPs();
  CHKERR tsSolve();
  CHKERR checkResults();
  MoFEMFunctionReturn(0);
}
//! [Run problem]

//! [Set up problem]
MoFEMErrorCode Incompressible::setupProblem() {
  MoFEMFunctionBegin;
  Simple *simple = mField.getInterface<Simple>();

  CHKERR PetscOptionsGetInt(PETSC_NULL, "", "-order", &order, PETSC_NULL);
  CHKERR PetscOptionsGetInt(PETSC_NULL, "", "-geom_order", &geom_order,
                            PETSC_NULL);
  CHKERR PetscOptionsGetBool(PETSC_NULL, "", "-is_discontinuous_pressure",
                             &isDiscontinuousPressure, PETSC_NULL);

  MOFEM_LOG("INCOMP_ELASTICITY", Sev::inform) << "Order " << order;
  MOFEM_LOG("INCOMP_ELASTICITY", Sev::inform) << "Geom order " << geom_order;

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
    MOFEM_LOG("INCOMP_ELASTICITY", Sev::inform)
        << "Set AINSWORTH_LEGENDRE_BASE for displacements";
    break;
  case DEMKOWICZ:
    base = DEMKOWICZ_JACOBI_BASE;
    MOFEM_LOG("INCOMP_ELASTICITY", Sev::inform)
        << "Set DEMKOWICZ_JACOBI_BASE for displacements";
    break;
  default:
    base = LASTBASE;
    break;
  }

  // Note: For tets we have only H1 Ainsworth base, for Hex we have only H1
  // Demkowicz base. We need to implement Demkowicz H1 base on tet.
  CHKERR simple->addDataField("GEOMETRY", H1, base, SPACE_DIM);
  CHKERR simple->setFieldOrder("GEOMETRY", geom_order);

  // Adding fields related to incompressible elasticity
  // Add displacement domain and boundary fields
  CHKERR simple->addDomainField("U", H1, base, SPACE_DIM);
  CHKERR simple->addBoundaryField("U", H1, base, SPACE_DIM);
  CHKERR simple->setFieldOrder("U", order);

  // Add pressure domain and boundary fields
  // Choose either Crouzeix-Raviart element:
  if (isDiscontinuousPressure) {
    CHKERR simple->addDomainField("P", L2, base, 1);
    CHKERR simple->setFieldOrder("P", order - 2);
  } else {
    // ... or Taylor-Hood element:
    CHKERR simple->addDomainField("P", H1, base, 1);
    CHKERR simple->setFieldOrder("P", order - 1);
  }

  // Add geometry data field
  CHKERR simple->addDataField("GEOMETRY", H1, base, SPACE_DIM);
  CHKERR simple->setFieldOrder("GEOMETRY", geom_order);

  CHKERR simple->setUp();

  auto project_ho_geometry = [&]() {
    Projection10NodeCoordsOnField ent_method(mField, "GEOMETRY");
    return mField.loop_dofs("GEOMETRY", ent_method);
  };
  CHKERR project_ho_geometry();

  MoFEMFunctionReturn(0);
} //! [Set up problem]

//! [Create common data]
MoFEMErrorCode Incompressible::createCommonData() {
  MoFEMFunctionBegin;

  auto get_options = [&]() {
    MoFEMFunctionBegin;
    CHKERR PetscOptionsGetScalar(PETSC_NULL, "", "-young_modulus",
                                 &young_modulus, PETSC_NULL);
    CHKERR PetscOptionsGetScalar(PETSC_NULL, "", "-poisson_ratio",
                                 &poisson_ratio, PETSC_NULL);

    MOFEM_LOG("INCOMP_ELASTICITY", Sev::inform)
        << "Young modulus " << young_modulus;
    MOFEM_LOG("INCOMP_ELASTICITY", Sev::inform)
        << "Poisson_ratio " << poisson_ratio;

    mu = young_modulus / (2. + 2. * poisson_ratio);
    const double lambda_denom =
        (1. + poisson_ratio) * (1. - 2. * poisson_ratio);
    lambda = young_modulus * poisson_ratio / lambda_denom;

    MoFEMFunctionReturn(0);
  };

  CHKERR get_options();

  MoFEMFunctionReturn(0);
}
//! [Create common data]

//! [Boundary condition]
MoFEMErrorCode Incompressible::bC() {
  MoFEMFunctionBegin;

  auto pipeline_mng = mField.getInterface<PipelineManager>();
  auto simple = mField.getInterface<Simple>();
  auto bc_mng = mField.getInterface<BcManager>();
  auto time_scale = boost::make_shared<TimeScale>();

  auto integration_rule = [](int, int, int approx_order) {
    return 2 * (approx_order - 1);
  };

  using DomainNaturalBC =
      NaturalBC<DomainEleOp>::Assembly<PETSC>::LinearForm<GAUSS>;
  using OpBodyForce =
      DomainNaturalBC::OpFlux<NaturalMeshsetType<BLOCKSET>, 1, SPACE_DIM>;
  using BoundaryNaturalBC =
      NaturalBC<BoundaryEleOp>::Assembly<PETSC>::LinearForm<GAUSS>;
  using OpForce = BoundaryNaturalBC::OpFlux<NaturalForceMeshsets, 1, SPACE_DIM>;

  CHKERR pipeline_mng->setBoundaryRhsIntegrationRule(integration_rule);
  CHKERR AddHOOps<SPACE_DIM - 1, SPACE_DIM, SPACE_DIM>::add(
      pipeline_mng->getOpBoundaryRhsPipeline(), {NOSPACE}, "GEOMETRY");
  //! [Natural boundary condition]
  CHKERR BoundaryNaturalBC::AddFluxToPipeline<OpForce>::add(
      pipeline_mng->getOpBoundaryRhsPipeline(), mField, "U", {time_scale},
      "FORCE", Sev::inform);
  //! [Define gravity vector]
  CHKERR DomainNaturalBC::AddFluxToPipeline<OpBodyForce>::add(
      pipeline_mng->getOpDomainRhsPipeline(), mField, "U", {time_scale},
      "BODY_FORCE", Sev::inform);

  // Essential BC
  CHKERR bc_mng->removeBlockDOFsOnEntities(simple->getProblemName(), "REMOVE_X",
                                           "U", 0, 0);
  CHKERR bc_mng->removeBlockDOFsOnEntities(simple->getProblemName(), "REMOVE_Y",
                                           "U", 1, 1);
  CHKERR bc_mng->removeBlockDOFsOnEntities(simple->getProblemName(), "REMOVE_Z",
                                           "U", 2, 2);
  CHKERR bc_mng->pushMarkDOFsOnEntities<DisplacementCubitBcData>(
      simple->getProblemName(), "U");

  MoFEMFunctionReturn(0);
}
//! [Boundary condition]

//! [Push operators to pip]
MoFEMErrorCode Incompressible::OPs() {
  MoFEMFunctionBegin;
  auto simple = mField.getInterface<Simple>();
  auto pip_mng = mField.getInterface<PipelineManager>();
  auto bc_mng = mField.getInterface<BcManager>();

  auto integration_rule_vol = [](int, int, int approx_order) {
    return 2 * approx_order + geom_order - 1;
  };

  auto add_domain_base_ops = [&](auto &pip) {
    MoFEMFunctionBegin;
    CHKERR AddHOOps<SPACE_DIM, SPACE_DIM, SPACE_DIM>::add(pip, {H1, L2},
                                                          "GEOMETRY");
    MoFEMFunctionReturn(0);
  };

  auto add_domain_ops_lhs = [&](auto &pip) {
    MoFEMFunctionBegin;

    // This assemble A-matrix
    using OpMassPressure = FormsIntegrators<DomainEleOp>::Assembly<
        AT>::BiLinearForm<GAUSS>::OpMass<1, 1>;
    // This assemble B-matrix (preconditioned)
    using OpMassPressureStab = FormsIntegrators<DomainEleOpStab>::Assembly<
        AT>::BiLinearForm<GAUSS>::OpMass<1, 1>;
    //! [Operators used for RHS incompressible elasticity]

    //! [Operators used for incompressible elasticity]
    using OpGradSymTensorGrad = FormsIntegrators<DomainEleOp>::Assembly<
        AT>::BiLinearForm<IT>::OpGradSymTensorGrad<1, SPACE_DIM, SPACE_DIM, 0>;
    using OpMixScalarTimesDiv = FormsIntegrators<DomainEleOp>::Assembly<
        AT>::BiLinearForm<IT>::OpMixScalarTimesDiv<SPACE_DIM, coord_type>;
    //! [Operators used for incompressible elasticity]

    auto mat_D_ptr = boost::make_shared<MatrixDouble>();
    constexpr auto size_symm = (SPACE_DIM * (SPACE_DIM + 1)) / 2;
    mat_D_ptr->resize(size_symm * size_symm, 1);

    FTensor::Index<'i', SPACE_DIM> i;
    FTensor::Index<'j', SPACE_DIM> j;
    FTensor::Index<'k', SPACE_DIM> k;
    FTensor::Index<'l', SPACE_DIM> l;

    constexpr auto t_kd = FTensor::Kronecker_Delta_symmetric<int>();
    auto t_mat = getFTensor4DdgFromMat<SPACE_DIM, SPACE_DIM, 0>(*mat_D_ptr);
    t_mat(i, j, k, l) = -2. * mu * ((t_kd(i, k) ^ t_kd(j, l)) / 4.);

    pip.push_back(new OpMixScalarTimesDiv(
        "P", "U",
        [](const double, const double, const double) constexpr { return -1.; },
        true, false));
    pip.push_back(new OpGradSymTensorGrad("U", "U", mat_D_ptr));

    auto get_lambda_reciprocal = [](const double, const double, const double) {
      return 1. / lambda;
    };
    if (poisson_ratio < 0.5)
      pip.push_back(new OpMassPressure("P", "P", get_lambda_reciprocal));

    double eps_stab = 0;
    CHKERR PetscOptionsGetScalar(PETSC_NULL, "", "-eps_stab", &eps_stab,
                                 PETSC_NULL);
    if (eps_stab > 0)
      pip.push_back(new OpMassPressureStab(
          "P", "P", [eps_stab](double, double, double) { return eps_stab; }));

    MoFEMFunctionReturn(0);
  };

  auto add_domain_ops_rhs = [&](auto &pip) {
    MoFEMFunctionBegin;

    //! [Operators used for RHS incompressible elasticity]
    using OpDomainGradTimesTensor = FormsIntegrators<DomainEleOp>::Assembly<
        AT>::LinearForm<GAUSS>::OpGradTimesSymTensor<1, SPACE_DIM, SPACE_DIM>;

    using OpDivDeltaUTimesP = FormsIntegrators<DomainEleOp>::Assembly<
        AT>::LinearForm<GAUSS>::OpMixDivTimesU<1, SPACE_DIM, SPACE_DIM>;

    using OpBaseTimesScalarValues = FormsIntegrators<DomainEleOp>::Assembly<
        AT>::LinearForm<GAUSS>::OpBaseTimesScalar<1>;

    //! [Operators used for RHS incompressible elasticity]

    auto pressure_ptr = boost::make_shared<VectorDouble>();
    pip.push_back(new OpCalculateScalarFieldValues("P", pressure_ptr));

    auto div_u_ptr = boost::make_shared<VectorDouble>();
    pip.push_back(
        new OpCalculateDivergenceVectorFieldValues<SPACE_DIM>("U", div_u_ptr));

    auto grad_u_ptr = boost::make_shared<MatrixDouble>();
    pip.push_back(new OpCalculateVectorFieldGradient<SPACE_DIM, SPACE_DIM>(
        "U", grad_u_ptr));

    auto strain_ptr = boost::make_shared<MatrixDouble>();
    pip.push_back(
        new OpSymmetrizeTensor<SPACE_DIM>("U", grad_u_ptr, strain_ptr));

    auto get_four_mu = [](const double, const double, const double) {
      return -2. * mu;
    };
    auto minus_one = [](const double, const double, const double) constexpr {
      return -1.;
    };

    pip.push_back(new OpDomainGradTimesTensor("U", strain_ptr, get_four_mu));

    pip.push_back(new OpDivDeltaUTimesP("U", pressure_ptr, minus_one));

    pip.push_back(new OpBaseTimesScalarValues("P", div_u_ptr, minus_one));

    auto get_lambda_reciprocal = [](const double, const double, const double) {
      return 1. / lambda;
    };
    if (poisson_ratio < 0.5) {
      pip.push_back(new OpBaseTimesScalarValues("P", pressure_ptr,
                                                get_lambda_reciprocal));
    }
    MoFEMFunctionReturn(0);
  };

  CHKERR add_domain_base_ops(pip_mng->getOpDomainLhsPipeline());
  CHKERR add_domain_base_ops(pip_mng->getOpDomainRhsPipeline());
  CHKERR add_domain_ops_lhs(pip_mng->getOpDomainLhsPipeline());
  CHKERR add_domain_ops_rhs(pip_mng->getOpDomainRhsPipeline());

  CHKERR pip_mng->setDomainRhsIntegrationRule(integration_rule_vol);
  CHKERR pip_mng->setDomainLhsIntegrationRule(integration_rule_vol);

  MoFEMFunctionReturn(0);
}
//! [Push operators to pip]

//! [Solve]
struct SetUpSchur {
  static boost::shared_ptr<SetUpSchur>
  createSetUpSchur(MoFEM::Interface &m_field, SmartPetscObj<Mat> A,
                   SmartPetscObj<Mat> P);
  virtual MoFEMErrorCode setUp(SmartPetscObj<TS> solver) = 0;

protected:
  SetUpSchur() = default;
};

MoFEMErrorCode Incompressible::tsSolve() {
  MoFEMFunctionBegin;

  auto simple = mField.getInterface<Simple>();
  auto pip_mng = mField.getInterface<PipelineManager>();
  auto is_manager = mField.getInterface<ISManager>();

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

  auto create_post_process_elements = [&]() {
    auto pp_fe = boost::make_shared<PostProcEle>(mField);
    auto &pip = pp_fe->getOpPtrVector();

    auto push_vol_ops = [this](auto &pip) {
      MoFEMFunctionBegin;
      CHKERR AddHOOps<SPACE_DIM, SPACE_DIM, SPACE_DIM>::add(pip, {H1},
                                                            "GEOMETRY");

      MoFEMFunctionReturn(0);
    };

    auto push_vol_post_proc_ops = [this](auto &pp_fe, auto &&p) {
      MoFEMFunctionBegin;

      auto &pip = pp_fe->getOpPtrVector();

      using OpPPMap = OpPostProcMapInMoab<SPACE_DIM, SPACE_DIM>;

      auto x_ptr = boost::make_shared<MatrixDouble>();
      pip.push_back(
          new OpCalculateVectorFieldValues<SPACE_DIM>("GEOMETRY", x_ptr));
      auto u_ptr = boost::make_shared<MatrixDouble>();
      pip.push_back(new OpCalculateVectorFieldValues<SPACE_DIM>("U", u_ptr));

      auto pressure_ptr = boost::make_shared<VectorDouble>();
      pip.push_back(new OpCalculateScalarFieldValues("P", pressure_ptr));

      auto div_u_ptr = boost::make_shared<VectorDouble>();
      pip.push_back(new OpCalculateDivergenceVectorFieldValues<SPACE_DIM>(
          "U", div_u_ptr));

      auto grad_u_ptr = boost::make_shared<MatrixDouble>();
      pip.push_back(new OpCalculateVectorFieldGradient<SPACE_DIM, SPACE_DIM>(
          "U", grad_u_ptr));

      auto strain_ptr = boost::make_shared<MatrixDouble>();
      pip.push_back(
          new OpSymmetrizeTensor<SPACE_DIM>("U", grad_u_ptr, strain_ptr));

      auto stress_ptr = boost::make_shared<MatrixDouble>();
      pip.push_back(new OpCalculateLameStress<SPACE_DIM>(
          mu, stress_ptr, strain_ptr, pressure_ptr));

      pip.push_back(

          new OpPPMap(

              pp_fe->getPostProcMesh(), pp_fe->getMapGaussPts(),

              {{"P", pressure_ptr}},

              {{"U", u_ptr}, {"GEOMETRY", x_ptr}},

              {},

              {{"STRAIN", strain_ptr}, {"STRESS", stress_ptr}}

              )

      );

      MoFEMFunctionReturn(0);
    };

    auto vol_post_proc = [this, push_vol_post_proc_ops, push_vol_ops]() {
      PetscBool post_proc_vol = PETSC_FALSE;
      CHKERR PetscOptionsGetBool(PETSC_NULL, "", "-post_proc_vol",
                                 &post_proc_vol, PETSC_NULL);
      if (post_proc_vol == PETSC_FALSE)
        return boost::shared_ptr<PostProcEle>();
      auto pp_fe = boost::make_shared<PostProcEle>(mField);
      CHK_MOAB_THROW(
          push_vol_post_proc_ops(pp_fe, push_vol_ops(pp_fe->getOpPtrVector())),
          "push_vol_post_proc_ops");
      return pp_fe;
    };

    auto skin_post_proc = [this, push_vol_post_proc_ops, push_vol_ops]() {
      PetscBool post_proc_skin = PETSC_TRUE;
      CHKERR PetscOptionsGetBool(PETSC_NULL, "", "-post_proc_skin",
                                 &post_proc_skin, PETSC_NULL);
      if (post_proc_skin == PETSC_FALSE)
        return boost::shared_ptr<SkinPostProcEle>();

      auto simple = mField.getInterface<Simple>();
      auto pp_fe = boost::make_shared<SkinPostProcEle>(mField);
      auto op_side = new OpLoopSide<DomainEle>(
          mField, simple->getDomainFEName(), SPACE_DIM, Sev::verbose);
      pp_fe->getOpPtrVector().push_back(op_side);
      CHK_MOAB_THROW(push_vol_post_proc_ops(
                         pp_fe, push_vol_ops(op_side->getOpPtrVector())),
                     "push_vol_post_proc_ops");
      return pp_fe;
    };

    return std::make_pair(vol_post_proc(), skin_post_proc());
  };

  auto set_time_monitor = [&](auto dm, auto solver) {
    MoFEMFunctionBegin;
    boost::shared_ptr<MonitorIncompressible> monitor_ptr(
        new MonitorIncompressible(dm, create_post_process_elements(), uXScatter,
                                  uYScatter, uZScatter));
    boost::shared_ptr<ForcesAndSourcesCore> null;
    CHKERR DMMoFEMTSSetMonitor(dm, solver, simple->getDomainFEName(),
                               monitor_ptr, null, null);
    MoFEMFunctionReturn(0);
  };

  auto set_essential_bc = [&]() {
    MoFEMFunctionBegin;
    // This is low level pushing finite elements (pipelines) to solver
    auto ts_ctx_ptr = getDMTsCtx(simple->getDM());
    auto pre_proc_ptr = boost::make_shared<FEMethod>();
    auto post_proc_rhs_ptr = boost::make_shared<FEMethod>();

    // Add boundary condition scaling
    auto time_scale = boost::make_shared<TimeScale>();

    pre_proc_ptr->preProcessHook = EssentialPreProc<DisplacementCubitBcData>(
        mField, pre_proc_ptr, {time_scale}, false);
    post_proc_rhs_ptr->postProcessHook =
        EssentialPostProcRhs<DisplacementCubitBcData>(mField, post_proc_rhs_ptr,
                                                      1.);
    ts_ctx_ptr->getPreProcessIFunction().push_front(pre_proc_ptr);
    ts_ctx_ptr->getPreProcessIJacobian().push_front(pre_proc_ptr);
    ts_ctx_ptr->getPostProcessIFunction().push_back(post_proc_rhs_ptr);
    MoFEMFunctionReturn(0);
  };

  auto set_schur_pc = [&](auto solver) {
    SNES snes;
    CHKERR TSGetSNES(solver, &snes);
    KSP ksp;
    CHKERR SNESGetKSP(snes, &ksp);
    PC pc;
    CHKERR KSPGetPC(ksp, &pc);
    PetscBool is_pcfs = PETSC_FALSE;
    PetscObjectTypeCompare((PetscObject)pc, PCFIELDSPLIT, &is_pcfs);
    boost::shared_ptr<SetUpSchur> schur_ptr;
    auto ts_ctx_ptr = getDMTsCtx(simple->getDM());

    if (is_pcfs == PETSC_TRUE) {
      auto A = createDMMatrix(simple->getDM());
      auto B = matDuplicate(A, MAT_DO_NOT_COPY_VALUES);
      CHK_MOAB_THROW(
          TSSetIJacobian(solver, A, B, TsSetIJacobian, ts_ctx_ptr.get()),
          "set operators");
      auto pre_proc_schur_lhs_ptr = boost::make_shared<FEMethod>();
      auto post_proc_schur_lhs_ptr = boost::make_shared<FEMethod>();
      pre_proc_schur_lhs_ptr->preProcessHook = [pre_proc_schur_lhs_ptr]() {
        MoFEMFunctionBegin;
        MOFEM_LOG("INCOMP_ELASTICITY", Sev::verbose) << "Lhs Zero matrices";
        CHKERR MatZeroEntries(pre_proc_schur_lhs_ptr->A);
        CHKERR MatZeroEntries(pre_proc_schur_lhs_ptr->B);
        MoFEMFunctionReturn(0);
      };
      post_proc_schur_lhs_ptr->postProcessHook = [this,
                                                  post_proc_schur_lhs_ptr]() {
        MoFEMFunctionBegin;
        MOFEM_LOG("INCOMP_ELASTICITY", Sev::verbose) << "Lhs Assemble Begin";
        *(post_proc_schur_lhs_ptr->matAssembleSwitch) = false;
        CHKERR MatAssemblyBegin(post_proc_schur_lhs_ptr->A, MAT_FINAL_ASSEMBLY);
        CHKERR MatAssemblyEnd(post_proc_schur_lhs_ptr->A, MAT_FINAL_ASSEMBLY);
        CHKERR MatAssemblyBegin(post_proc_schur_lhs_ptr->B, MAT_FINAL_ASSEMBLY);
        CHKERR MatAssemblyEnd(post_proc_schur_lhs_ptr->B, MAT_FINAL_ASSEMBLY);
        CHKERR EssentialPostProcLhs<DisplacementCubitBcData>(
            mField, post_proc_schur_lhs_ptr, 1.,
            SmartPetscObj<Mat>(post_proc_schur_lhs_ptr->A))();
        CHKERR MatAXPY(post_proc_schur_lhs_ptr->B, 1,
                       post_proc_schur_lhs_ptr->A, SAME_NONZERO_PATTERN);
        MOFEM_LOG("INCOMP_ELASTICITY", Sev::verbose) << "Lhs Assemble End";
        MoFEMFunctionReturn(0);
      };
      ts_ctx_ptr->getPreProcessIJacobian().push_front(pre_proc_schur_lhs_ptr);
      ts_ctx_ptr->getPostProcessIJacobian().push_back(post_proc_schur_lhs_ptr);

      if (AT == AssemblyType::SCHUR) {
        schur_ptr = SetUpSchur::createSetUpSchur(mField, A, B);
        CHK_MOAB_THROW(schur_ptr->setUp(solver), "setup schur preconditioner");
      } else {
        auto set_fieldsplit_preconditioner_ts = [&](auto solver) {
          MoFEMFunctionBegin;
          auto bc_mng = mField.getInterface<BcManager>();
          auto name_prb = simple->getProblemName();
          SmartPetscObj<IS> is_p;
          CHKERR mField.getInterface<ISManager>()->isCreateProblemFieldAndRank(
              name_prb, ROW, "P", 0, 1, is_p);
          CHKERR PCFieldSplitSetIS(pc, PETSC_NULL, is_p);
          MoFEMFunctionReturn(0);
        };
        CHK_MOAB_THROW(set_fieldsplit_preconditioner_ts(solver),
                       "set fieldsplit preconditioner");
      }
      return boost::make_tuple(schur_ptr, A, B);
    }

    auto post_proc_schur_lhs_ptr = boost::make_shared<FEMethod>();
    post_proc_schur_lhs_ptr->postProcessHook =
        EssentialPostProcLhs<DisplacementCubitBcData>(
            mField, post_proc_schur_lhs_ptr, 1.);
    ts_ctx_ptr->getPostProcessIJacobian().push_back(post_proc_schur_lhs_ptr);

    return boost::make_tuple(schur_ptr, SmartPetscObj<Mat>(),
                             SmartPetscObj<Mat>());
  };

  auto dm = simple->getDM();
  auto D = createDMVector(dm);

  uXScatter = scatter_create(D, 0);
  uYScatter = scatter_create(D, 1);
  if (SPACE_DIM == 3)
    uZScatter = scatter_create(D, 2);

  // Add extra finite elements to SNES solver pipelines to resolve essential
  // boundary conditions
  CHKERR set_essential_bc();

  auto solver = pip_mng->createTSIM();
  CHKERR TSSetFromOptions(solver);

  CHKERR set_section_monitor(solver);
  CHKERR set_time_monitor(dm, solver);
  auto [schur_pc_ptr, A, B] = set_schur_pc(solver);

  CHKERR TSSetSolution(solver, D);
  CHKERR TSSetUp(solver);
  CHKERR TSSolve(solver, NULL);

  MoFEMFunctionReturn(0);
}
//! [Solve]

//! [Check]
MoFEMErrorCode Incompressible::checkResults() { return 0; }
//! [Check]

static char help[] = "...\n\n";

int main(int argc, char *argv[]) {

  // Initialisation of MoFEM/PETSc and MOAB data structures
  const char param_file[] = "param_file.petsc";
  MoFEM::Core::Initialize(&argc, &argv, param_file, help);

  // Add logging channel for CONTACT
  auto core_log = logging::core::get();
  core_log->add_sink(
      LogManager::createSink(LogManager::getStrmWorld(), "INCOMP_ELASTICITY"));
  LogManager::setLog("INCOMP_ELASTICITY");
  MOFEM_LOG_TAG("INCOMP_ELASTICITY", "Indent");

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

    //! [CONTACT]
    Incompressible ex(m_field);
    CHKERR ex.runProblem();
    //! [CONTACT]
  }
  CATCH_ERRORS;

  CHKERR MoFEM::Core::Finalize();

  return 0;
}

struct SetUpSchurImpl : public SetUpSchur {

  SetUpSchurImpl(MoFEM::Interface &m_field, SmartPetscObj<Mat> A,
                 SmartPetscObj<Mat> P)
      : SetUpSchur(), mField(m_field), A(A), P(P) {}
  virtual ~SetUpSchurImpl() { S.reset(); }
  MoFEMErrorCode setUp(SmartPetscObj<TS> solver);

private:
  MoFEMErrorCode setOperator();
  MoFEMErrorCode setPC(PC pc);

  SmartPetscObj<DM> createSubDM();

  SmartPetscObj<Mat> A;
  SmartPetscObj<Mat> P;
  SmartPetscObj<Mat> S;

  MoFEM::Interface &mField;

  SmartPetscObj<DM> subDM;
};

MoFEMErrorCode SetUpSchurImpl::setUp(SmartPetscObj<TS> solver) {
  MoFEMFunctionBegin;
  auto simple = mField.getInterface<Simple>();
  auto pip = mField.getInterface<PipelineManager>();
  auto dm = simple->getDM();

  SNES snes;
  CHKERR TSGetSNES(solver, &snes);
  KSP ksp;
  CHKERR SNESGetKSP(snes, &ksp);
  PC pc;
  CHKERR KSPGetPC(ksp, &pc);

  MOFEM_LOG("INCOMP_ELASTICITY", Sev::inform) << "Setup Schur pc";

  if (S) {
    CHK_THROW_MESSAGE(MOFEM_DATA_INCONSISTENCY,
                      "Is expected that schur matrix is not allocated. This is "
                      "possible only is PC is set up twice");
  }

  auto create_sub_dm = [&]() {
    auto sub_dm = createDM(mField.get_comm(), "DMMOFEM");
    auto set_up = [&]() {
      MoFEMFunctionBegin;
      CHKERR DMMoFEMCreateSubDM(sub_dm, simple->getDM(), "SUB");
      CHKERR DMMoFEMSetSquareProblem(sub_dm, PETSC_TRUE);
      CHKERR DMMoFEMAddElement(sub_dm, simple->getDomainFEName());
      CHKERR DMMoFEMAddSubFieldRow(sub_dm, "U");
      CHKERR DMSetUp(sub_dm);
      MoFEMFunctionReturn(0);
    };
    CHK_THROW_MESSAGE(set_up(), "sey up dm");
    return sub_dm;
  };

  subDM = create_sub_dm();
  S = createDMMatrix(subDM);

  auto dm_is = getDMSubData(subDM)->getSmartRowIs();
  auto ao_up = createAOMappingIS(dm_is, PETSC_NULL);
  // Domain
  pip->getOpDomainLhsPipeline().push_front(new OpSchurAssembleBegin());
  pip->getOpDomainLhsPipeline().push_back(new OpSchurAssembleEnd<SCHUR_DSYSV>(
      {"P"}, {nullptr}, {ao_up}, {S}, {false}, false));

  auto pre_proc_schur_lhs_ptr = boost::make_shared<FEMethod>();
  auto post_proc_schur_lhs_ptr = boost::make_shared<FEMethod>();

  pre_proc_schur_lhs_ptr->preProcessHook = [this]() {
    MoFEMFunctionBegin;
    CHKERR MatZeroEntries(S);
    MoFEMFunctionReturn(0);
  };

  post_proc_schur_lhs_ptr->postProcessHook = [this, ao_up,
                                              post_proc_schur_lhs_ptr]() {
    MoFEMFunctionBegin;

    auto print_mat_norm = [this](auto a, std::string prefix) {
      MoFEMFunctionBegin;
      double nrm;
      CHKERR MatNorm(a, NORM_FROBENIUS, &nrm);
      MOFEM_LOG("INCOMP_ELASTICITY", Sev::verbose)
          << prefix << " norm = " << nrm;
      MoFEMFunctionReturn(0);
    };

    CHKERR EssentialPostProcLhs<DisplacementCubitBcData>(
        mField, post_proc_schur_lhs_ptr, 1, S, ao_up)();

    // #ifndef NDEBUG
    CHKERR print_mat_norm(A, "A");
    CHKERR print_mat_norm(P, "P");
    CHKERR print_mat_norm(S, "S");
    // #endif // NDEBUG
    MoFEMFunctionReturn(0);
  };

  auto ts_ctx_ptr = getDMTsCtx(simple->getDM());
  ts_ctx_ptr->getPreProcessIJacobian().push_front(pre_proc_schur_lhs_ptr);
  ts_ctx_ptr->getPostProcessIJacobian().push_back(post_proc_schur_lhs_ptr);

  SmartPetscObj<IS> is_p;
  CHKERR mField.getInterface<ISManager>()->isCreateProblemFieldAndRank(
      simple->getProblemName(), ROW, "P", 0, 1, is_p);
  CHKERR PCFieldSplitSetIS(pc, NULL, is_p);
  CHKERR PCFieldSplitSetSchurPre(pc, PC_FIELDSPLIT_SCHUR_PRE_USER, S);

  MoFEMFunctionReturn(0);
}

boost::shared_ptr<SetUpSchur>
SetUpSchur::createSetUpSchur(MoFEM::Interface &m_field, SmartPetscObj<Mat> A,
                             SmartPetscObj<Mat> P) {
  return boost::shared_ptr<SetUpSchur>(new SetUpSchurImpl(m_field, A, P));
}
