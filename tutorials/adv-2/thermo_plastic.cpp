/**
 * \file thermo_elastic.cpp
 * \example thermo_elastic.cpp
 *
 * Thermo plasticity
 *
 */

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

constexpr EntityType boundary_ent = SPACE_DIM == 3 ? MBTRI : MBEDGE;
using EntData = EntitiesFieldData::EntData;
using DomainEle = ElementsAndOps<SPACE_DIM>::DomainEle;
using DomainEleOp = DomainEle::UserDataOperator;
using BoundaryEle = ElementsAndOps<SPACE_DIM>::BoundaryEle;
using BoundaryEleOp = BoundaryEle::UserDataOperator;
using PostProcEle = PostProcBrokenMeshInMoab<DomainEle>;

using AssemblyDomainEleOp =
    FormsIntegrators<DomainEleOp>::Assembly<PETSC>::OpBase;

//! [Only used with Hooke equation (linear material model)]
using OpKCauchy = FormsIntegrators<DomainEleOp>::Assembly<PETSC>::BiLinearForm<
    GAUSS>::OpGradSymTensorGrad<1, SPACE_DIM, SPACE_DIM, 0>;
using OpInternalForceCauchy = FormsIntegrators<DomainEleOp>::Assembly<
    PETSC>::LinearForm<GAUSS>::OpGradTimesSymTensor<1, SPACE_DIM, SPACE_DIM>;
//! [Only used with Hooke equation (linear material model)]

//! [Essential boundary conditions]
using OpBoundaryMass = FormsIntegrators<BoundaryEleOp>::Assembly<
    PETSC>::BiLinearForm<GAUSS>::OpMass<1, SPACE_DIM>;
using OpBoundaryVec = FormsIntegrators<BoundaryEleOp>::Assembly<
    PETSC>::LinearForm<GAUSS>::OpBaseTimesVector<1, SPACE_DIM, 0>;
using OpBoundaryInternal = FormsIntegrators<BoundaryEleOp>::Assembly<
    PETSC>::LinearForm<GAUSS>::OpBaseTimesVector<1, SPACE_DIM, 1>;
//! [Essential boundary conditions]
using OpScaleL2 = MoFEM::OpScaleBaseBySpaceInverseOfMeasure<DomainEleOp>;

// Thermal operators
/**
 * @brief Integrate Lhs base of flux (1/k) base of flux (FLUX x FLUX)
 *
 */
using OpHdivHdiv = FormsIntegrators<DomainEleOp>::Assembly<PETSC>::BiLinearForm<
    GAUSS>::OpMass<3, 3>;

/**
 * @brief Integrate Lhs div of base of flux time base of temperature (FLUX x T)
 * and transpose of it, i.e. (T x FLAX)
 *
 */
using OpHdivT = FormsIntegrators<DomainEleOp>::Assembly<PETSC>::BiLinearForm<
    GAUSS>::OpMixDivTimesScalar<SPACE_DIM>;

/**
 * @brief Integrate Lhs base of temerature times (heat capacity) times base of
 * temperature (T x T)
 *
 */
using OpCapacity = FormsIntegrators<DomainEleOp>::Assembly<PETSC>::BiLinearForm<
    GAUSS>::OpMass<1, 1>;

/**
 * @brief Integrating Rhs flux base (1/k) flux  (FLUX)
 */
using OpHdivFlux = FormsIntegrators<DomainEleOp>::Assembly<PETSC>::LinearForm<
    GAUSS>::OpBaseTimesVector<3, 3, 1>;

/**
 * @brief  Integrate Rhs div flux base times temperature (T)
 *
 */
using OpHDivTemp = FormsIntegrators<DomainEleOp>::Assembly<PETSC>::LinearForm<
    GAUSS>::OpMixDivTimesU<3, 1, 2>;

/**
 * @brief Integrate Rhs base of temperature time heat capacity times heat rate
 * (T)
 *
 */
using OpBaseDotT = FormsIntegrators<DomainEleOp>::Assembly<PETSC>::LinearForm<
    GAUSS>::OpBaseTimesScalar<1>;

/**
 * @brief Integrate Rhs base of temperature times divergence of flux (T)
 *
 */
using OpBaseDivFlux = OpBaseDotT;

using DomainNaturalBC =
    NaturalBC<DomainEleOp>::Assembly<PETSC>::LinearForm<GAUSS>;
using OpBodyForce =
    DomainNaturalBC::OpFlux<NaturalMeshsetType<BLOCKSET>, 1, SPACE_DIM>;
using OpHeatSource =
    DomainNaturalBC::OpFlux<NaturalMeshsetType<BLOCKSET>, 1, 1>;

using BoundaryNaturalBC =
    NaturalBC<BoundaryEleOp>::Assembly<PETSC>::LinearForm<GAUSS>;
using OpForce =
    BoundaryNaturalBC::OpFlux<NaturalMeshsetType<BLOCKSET>, 1, SPACE_DIM>;
using OpTemperatureBC = BoundaryNaturalBC::OpFlux<NaturalMeshsetType<BLOCKSET>,
                                                  SPACE_DIM, SPACE_DIM>;

double young_modulus = 206913;
double poisson_ratio = 0.29;
double coeff_expansion = 10e-6;
double ref_temp = 0.0;

double heat_conductivity =
    16.2; // Force / (time temperature )  or Power /
          // (length temperature) // Time unit is hour. force unit kN
double heat_capacity = 5961.6; // length^2/(time^2 temperature) // length is
                               // millimeter time is hour

int order = 2;

#include <ThermoElasticOps.hpp>
using namespace ThermoElasticOps;

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

  boost::shared_ptr<PostProcEle> postProcFe;
  boost::shared_ptr<MatrixDouble> mDPtr;
  boost::shared_ptr<MatrixDouble> mDPtr_Axiator;
  boost::shared_ptr<MatrixDouble> mDPtr_Deviator;
};

//! [Run problem]
MoFEMErrorCode Example::runProblem() {
  MoFEMFunctionBegin;
  CHKERR setupProblem();
  CHKERR createCommonData();
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
  // Add field
  constexpr FieldApproximationBase base = AINSWORTH_LEGENDRE_BASE;
  constexpr auto size_symm = (SPACE_DIM * (SPACE_DIM + 1)) / 2;
  // Mechanical fields
  CHKERR simple->addDomainField("U", H1, base, SPACE_DIM);
  CHKERR simple->addBoundaryField("U", H1, base, SPACE_DIM);
  // Temperature
  const auto flux_space = (SPACE_DIM == 2) ? HCURL : HDIV;
  CHKERR simple->addDomainField("T", L2, AINSWORTH_LEGENDRE_BASE, 1);

  CHKERR PetscOptionsGetInt(PETSC_NULL, "", "-order", &order, PETSC_NULL);
  CHKERR simple->setFieldOrder("U", order);
  CHKERR simple->setFieldOrder("FLUX", order);
  CHKERR simple->setFieldOrder("T", order - 1);
  CHKERR simple->setUp();
  MoFEMFunctionReturn(0);
}
//! [Set up problem]

//! [Create common data]
MoFEMErrorCode Example::createCommonData() {
  MoFEMFunctionBegin;

  auto get_command_line_parameters = [&]() {
    MoFEMFunctionBegin;
    CHKERR PetscOptionsGetScalar(PETSC_NULL, "", "-young_modulus",
                                 &young_modulus, PETSC_NULL);
    CHKERR PetscOptionsGetScalar(PETSC_NULL, "", "-poisson_ratio",
                                 &poisson_ratio, PETSC_NULL);
    CHKERR PetscOptionsGetScalar(PETSC_NULL, "", "-coeff_expansion",
                                 &coeff_expansion, PETSC_NULL);
    CHKERR PetscOptionsGetScalar(PETSC_NULL, "", "-ref_temp", &ref_temp,
                                 PETSC_NULL);

    CHKERR PetscOptionsGetScalar(PETSC_NULL, "", "-capacity", &heat_capacity,
                                 PETSC_NULL);
    CHKERR PetscOptionsGetScalar(PETSC_NULL, "", "-conductivity",
                                 &heat_conductivity, PETSC_NULL);

    MOFEM_LOG("EXAMPLE", Sev::inform) << "Young modulus " << young_modulus;
    MOFEM_LOG("EXAMPLE", Sev::inform) << "Poisson ratio " << poisson_ratio;
    MOFEM_LOG("EXAMPLE", Sev::inform) << "Coeff_expansion " << coeff_expansion;
    MOFEM_LOG("EXAMPLE", Sev::inform) << "Reference_temperature  " << ref_temp;

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

    auto t_D = getFTensor4DdgFromMat<SPACE_DIM, SPACE_DIM, 0>(*mDPtr);
    auto t_D_axiator =
        getFTensor4DdgFromMat<SPACE_DIM, SPACE_DIM, 0>(*mDPtr_Axiator);
    auto t_D_deviator =
        getFTensor4DdgFromMat<SPACE_DIM, SPACE_DIM, 0>(*mDPtr_Deviator);

    constexpr double third = boost::math::constants::third<double>();
    t_D_axiator(i, j, k, l) = A *
                              (bulk_modulus_K - (2. / 3.) * shear_modulus_G) *
                              t_kd(i, j) * t_kd(k, l);
    t_D_deviator(i, j, k, l) =
        2 * shear_modulus_G * ((t_kd(i, k) ^ t_kd(j, l)) / 4.);
    t_D(i, j, k, l) = t_D_axiator(i, j, k, l) + t_D_deviator(i, j, k, l);

    MoFEMFunctionReturn(0);
  };

  constexpr auto size_symm = (SPACE_DIM * (SPACE_DIM + 1)) / 2;
  mDPtr = boost::make_shared<MatrixDouble>();
  mDPtr->resize(size_symm * size_symm, 1);
  mDPtr_Axiator = boost::make_shared<MatrixDouble>();
  mDPtr_Axiator->resize(size_symm * size_symm, 1);
  mDPtr_Deviator = boost::make_shared<MatrixDouble>();
  mDPtr_Deviator->resize(size_symm * size_symm, 1);

  CHKERR get_command_line_parameters();
  CHKERR set_matrial_stiffness();

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
  CHKERR bc_mng->removeBlockDOFsOnEntities(simple->getProblemName(),
                                           "ZERO_FLUX", "FLUX", 0, 1);

  PetscBool zero_fix_skin_flux = PETSC_TRUE;
  CHKERR PetscOptionsGetBool(PETSC_NULL, "", "-fix_skin_flux",
                             &zero_fix_skin_flux, PETSC_NULL);
  if (zero_fix_skin_flux) {
    Range faces;
    CHKERR mField.get_moab().get_entities_by_dimension(0, SPACE_DIM, faces);
    Skinner skin(&mField.get_moab());
    Range skin_ents;
    CHKERR skin.find_skin(0, faces, false, skin_ents);
    ParallelComm *pcomm =
        ParallelComm::get_pcomm(&mField.get_moab(), MYPCOMM_INDEX);
    if (pcomm == NULL)
      SETERRQ(PETSC_COMM_WORLD, MOFEM_DATA_INCONSISTENCY,
              "Communicator not created");
    CHKERR pcomm->filter_pstatus(skin_ents,
                                 PSTATUS_SHARED | PSTATUS_MULTISHARED,
                                 PSTATUS_NOT, -1, &skin_ents);
    CHKERR prb_mng->removeDofsOnEntities(simple->getProblemName(), "FLUX",
                                         skin_ents, 0, 1);
  }

  CHKERR bc_mng->removeBlockDOFsOnEntities<DisplacementCubitBcData>(
      simple->getProblemName(), "U");

  MoFEMFunctionReturn(0);
}
//! [Boundary condition]

//! [Push operators to pipeline]
MoFEMErrorCode Example::OPs() {
  MoFEMFunctionBegin;
  auto pipeline_mng = mField.getInterface<PipelineManager>();
  auto simple = mField.getInterface<Simple>();
  auto bc_mng = mField.getInterface<BcManager>();

  auto integration_rule_deviator = [](int o_row, int o_col, int approx_order) {
    return 2 * (approx_order - 1);
  };
  auto integration_rule_bc = [](int, int, int approx_order) {
    return 2 * approx_order;
  };

  auto mat_grad_ptr = boost::make_shared<MatrixDouble>();
  auto mat_strain_ptr = boost::make_shared<MatrixDouble>();
  auto mat_stress_ptr = boost::make_shared<MatrixDouble>();

  auto vec_temp_ptr = boost::make_shared<VectorDouble>();
  auto vec_temp_dot_ptr = boost::make_shared<VectorDouble>();
  auto mat_flux_ptr = boost::make_shared<MatrixDouble>();
  auto vec_temp_div_ptr = boost::make_shared<VectorDouble>();

  auto time_scale = boost::make_shared<TimeScale>();

  auto add_domain_ops = [&](auto &pipeline) {
    MoFEMFunctionBegin;

    auto det_ptr = boost::make_shared<VectorDouble>();
    auto jac_ptr = boost::make_shared<MatrixDouble>();
    auto inv_jac_ptr = boost::make_shared<MatrixDouble>();
    pipeline.push_back(new OpCalculateHOJac<SPACE_DIM>(jac_ptr));
    pipeline.push_back(
        new OpInvertMatrix<SPACE_DIM>(jac_ptr, det_ptr, inv_jac_ptr));
    pipeline.push_back(
        new OpSetHOInvJacToScalarBases<SPACE_DIM>(H1, inv_jac_ptr));

    if (SPACE_DIM == 2) {
      pipeline.push_back(new OpMakeHdivFromHcurl());
      pipeline.push_back(new OpSetContravariantPiolaTransformOnFace2D(jac_ptr));
      pipeline.push_back(new OpSetInvJacHcurlFace(inv_jac_ptr));
      pipeline.push_back(new OpSetInvJacL2ForFace(inv_jac_ptr));
    } else {
      postProcFe->getOpPtrVector().push_back(
          new OpSetHOContravariantPiolaTransform(HDIV, det_ptr, jac_ptr));
      postProcFe->getOpPtrVector().push_back(
          new OpSetHOInvJacVectorBase(HDIV, inv_jac_ptr));
    }

    pipeline.push_back(new OpCalculateScalarFieldValues("T", vec_temp_ptr));
    pipeline.push_back(
        new OpCalculateScalarFieldValuesDot("T", vec_temp_dot_ptr));
    pipeline.push_back(new OpCalculateHdivVectorDivergence<3, SPACE_DIM>(
        "FLUX", vec_temp_div_ptr));
    pipeline.push_back(new OpCalculateHVecVectorField<3>("FLUX", mat_flux_ptr));

    pipeline.getOpDomainRhsPipeline().push_back(
        new OpCalculateVectorFieldGradient<SPACE_DIM, SPACE_DIM>("U",
                                                                 mat_grad_ptr));
    pipeline.push_back(
        new OpSymmetrizeTensor<SPACE_DIM>("U", mat_grad_ptr, mat_strain_ptr));
    pipeline.push_back(new OpStressThermal(
        "U", mat_strain_ptr, vec_temp_ptr, mDPtr, mat_stress_ptr));

    MoFEMFunctionReturn(0);
  };

  auto add_domain_rhs_ops = [&](auto &pipeline) {
    MoFEMFunctionBegin;
    pipeline.push_back(new OpInternalForceCauchy(
        "U", mat_stress_ptr,
        [](double, double, double) constexpr { return 1; }));
    auto resistance = [](const double, const double, const double) {
      return (1. / heat_conductivity);
    };
    auto capacity = [&](const double, const double, const double) {
      return -heat_capacity;
    };
    auto unity = [](const double, const double, const double) { return 1; };
    pipeline.push_back(new OpHdivFlux("FLUX", mat_flux_ptr, resistance));
    pipeline.push_back(new OpHDivTemp("FLUX", vec_temp_ptr, unity));
    pipeline.push_back(new OpBaseDivFlux("T", vec_temp_div_ptr, unity));
    pipeline.push_back(new OpBaseDotT("T", vec_temp_dot_ptr, capacity));

    CHKERR DomainNaturalBC::addFluxToPipeline(
        FluxOpType<OpHeatSource>(), pipeline, mField, "U", {time_scale},
        "HEAT_SOURCE", Sev::inform);
    CHKERR DomainNaturalBC::addFluxToPipeline(
        FluxOpType<OpBodyForce>(), pipeline_mng->getOpDomainRhsPipeline(),
        mField, "U", {time_scale}, "BODY_FORCE", Sev::inform);

    MoFEMFunctionReturn(0);
  };

  auto add_domain_lhs_ops = [&](auto &pipeline) {
    MoFEMFunctionBegin;
    pipeline.push_back(new OpKCauchy("U", "U", mDPtr));
    pipeline.push_back(
        new ThermoElasticOps::OpKCauchyThermoElasticity("U", "T", mDPtr));

    auto resistance = [](const double, const double, const double) {
      return (1. / heat_conductivity);
    };
    auto capacity = [](const double, const double, const double) {
      return -heat_capacity;
    };
    pipeline.push_back(new OpHdivHdiv("FLUX", "FLUX", resistance));
    pipeline.push_back(new OpHdivT(
        "FLUX", "T", []() { return 1; }, true));

    auto op_capacity = new OpCapacity("T", "T", capacity);
    op_capacity->feScalingFun = [](const FEMethod *fe_ptr) {
      return fe_ptr->ts_a;
    };
    pipeline.push_back(op_capacity);

    MoFEMFunctionReturn(0);
  };

  auto add_boundary_rhs_ope = [&](auto &pipeline) {
    MoFEMFunctionBegin;

    CHKERR BoundaryNaturalBC::addFluxToPipeline(
        FluxOpType<OpForce>(), pipeline_mng->getOpBoundaryRhsPipeline(), mField,
        "U", {time_scale}, "FORCE", Sev::inform);

    CHKERR BoundaryNaturalBC::addFluxToPipeline(
        FluxOpType<OpTemperatureBC>(), pipeline_mng->getOpBoundaryRhsPipeline(),
        mField, "FLUX", {time_scale}, "TEMPERATURE", Sev::inform);

    MoFEMFunctionReturn(0);
  };

  // Mechanics

  MoFEMFunctionReturn(0);
}
//! [Push operators to pipeline]

//! [Solve]
MoFEMErrorCode Example::tsSolve() {
  MoFEMFunctionBegin;

  Simple *simple = mField.getInterface<Simple>();
  PipelineManager *pipeline_mng = mField.getInterface<PipelineManager>();
  ISManager *is_manager = mField.getInterface<ISManager>();

  auto dm = simple->getDM();
  auto snes_ctx_ptr = smartGetDMSnesCtx(dm);

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
    MoFEMFunctionBegin;
    postProcFe = boost::make_shared<PostProcEle>(mField);

    auto mat_grad_ptr = boost::make_shared<MatrixDouble>();
    auto mat_strain_ptr = boost::make_shared<MatrixDouble>();
    auto mat_stress_ptr = boost::make_shared<MatrixDouble>();

    auto vec_temp_ptr = boost::make_shared<VectorDouble>();
    auto mat_flux_ptr = boost::make_shared<MatrixDouble>();

    auto det_ptr = boost::make_shared<VectorDouble>();
    auto jac_ptr = boost::make_shared<MatrixDouble>();
    auto inv_jac_ptr = boost::make_shared<MatrixDouble>();
    postProcFe->getOpPtrVector().push_back(
        new OpCalculateHOJac<SPACE_DIM>(jac_ptr));
    postProcFe->getOpPtrVector().push_back(
        new OpInvertMatrix<SPACE_DIM>(jac_ptr, det_ptr, inv_jac_ptr));
    postProcFe->getOpPtrVector().push_back(
        new OpSetHOInvJacToScalarBases<SPACE_DIM>(H1, inv_jac_ptr));

    if (SPACE_DIM == 2) {
      postProcFe->getOpPtrVector().push_back(new OpMakeHdivFromHcurl());
      postProcFe->getOpPtrVector().push_back(
          new OpSetContravariantPiolaTransformOnFace2D(jac_ptr));
      postProcFe->getOpPtrVector().push_back(
          new OpSetInvJacHcurlFace(inv_jac_ptr));
      postProcFe->getOpPtrVector().push_back(
          new OpSetInvJacL2ForFace(inv_jac_ptr));
    } else {
      postProcFe->getOpPtrVector().push_back(
          new OpSetHOContravariantPiolaTransform(HDIV, det_ptr, jac_ptr));
      postProcFe->getOpPtrVector().push_back(
          new OpSetHOInvJacVectorBase(HDIV, inv_jac_ptr));
    }

    postProcFe->getOpPtrVector().push_back(
        new OpCalculateScalarFieldValues("T", vec_temp_ptr));
    postProcFe->getOpPtrVector().push_back(
        new OpCalculateHVecVectorField<3>("FLUX", mat_flux_ptr));

    auto u_ptr = boost::make_shared<MatrixDouble>();
    postProcFe->getOpPtrVector().push_back(
        new OpCalculateVectorFieldValues<SPACE_DIM>("U", u_ptr));
    postProcFe->getOpPtrVector().push_back(
        new OpCalculateVectorFieldGradient<SPACE_DIM, SPACE_DIM>("U",
                                                                 mat_grad_ptr));
    postProcFe->getOpPtrVector().push_back(
        new OpSymmetrizeTensor<SPACE_DIM>("U", mat_grad_ptr, mat_strain_ptr));
    postProcFe->getOpPtrVector().push_back(new OpStressThermal(
        "U", mat_strain_ptr, vec_temp_ptr, mDPtr, mat_stress_ptr));

    using OpPPMap = OpPostProcMapInMoab<SPACE_DIM, SPACE_DIM>;

    postProcFe->getOpPtrVector().push_back(

        new OpPPMap(

            postProcFe->getPostProcMesh(), postProcFe->getMapGaussPts(),

            {{"T", vec_temp_ptr}},

            {{"U", u_ptr}, {"FLUX", mat_flux_ptr}},

            {},

            {{"STRAIN", mat_strain_ptr}, {"STRESS", mat_stress_ptr}}

            )

    );

    MoFEMFunctionReturn(0);
  };

  auto set_time_monitor = [&](auto dm, auto solver) {
    MoFEMFunctionBegin;
    auto monitor_ptr = boost::make_shared<FEMethod>();
    monitor_ptr->preProcessHook = [&]() {
      MoFEMFunctionBegin;
      CHKERR DMoFEMLoopFiniteElements(dm, "dFE", postProcFe);
      CHKERR postProcFe->writeFile(
          "out_" + boost::lexical_cast<std::string>(monitor_ptr->ts_step) +
          ".h5m");
      MoFEMFunctionReturn(0);
    };
    auto null = boost::shared_ptr<FEMethod>();
    CHKERR DMMoFEMTSSetMonitor(dm, solver, simple->getDomainFEName(),
                               monitor_ptr, null, null);
    MoFEMFunctionReturn(0);
  };

  CHKERR create_post_process_element();

  auto solver = pipeline_mng->createTSIM();
  auto D = smartCreateDMVector(dm);

  CHKERR TSSetSolution(solver, D);
  CHKERR set_section_monitor(solver);
  CHKERR set_time_monitor(dm, solver);
  CHKERR TSSetSolution(solver, D);
  CHKERR TSSetFromOptions(solver);
  CHKERR TSSetUp(solver);
  CHKERR TSSolve(solver, NULL);

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
