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

using EntData = EntitiesFieldData::EntData;
using DomainEle = ElementsAndOps<SPACE_DIM>::DomainEle;
using DomainEleOp = DomainEle::UserDataOperator;
using BoundaryEle = ElementsAndOps<SPACE_DIM>::BoundaryEle;
using BoundaryEleOp = BoundaryEle::UserDataOperator;
using PostProcEle = PostProcBrokenMeshInMoab<DomainEle>;

using AssemblyDomainEleOp =
    FormsIntegrators<DomainEleOp>::Assembly<PETSC>::OpBase;

//! [Linear elastic problem]
using OpKCauchy = FormsIntegrators<DomainEleOp>::Assembly<PETSC>::BiLinearForm<
    GAUSS>::OpGradSymTensorGrad<1, SPACE_DIM, SPACE_DIM,
                                0>; //< Elastic stiffness matrix
using OpInternalForceCauchy =
    FormsIntegrators<DomainEleOp>::Assembly<PETSC>::LinearForm<
        GAUSS>::OpGradTimesSymTensor<1, SPACE_DIM,
                                     SPACE_DIM>; //< Elastic internal forces
//! [Linear elastic problem]

//! [Thermal problem]
/**
 * @brief Integrate Lhs base of flux (1/k) base of flux (FLUX x FLUX)
 *
 */
using OpHdivHdiv = FormsIntegrators<DomainEleOp>::Assembly<PETSC>::BiLinearForm<
    GAUSS>::OpMass<3, SPACE_DIM>;

/**
 * @brief Integrate Lhs div of base of flux time base of temperature (FLUX x T)
 * and transpose of it, i.e. (T x FLAX)
 *
 */
using OpHdivT = FormsIntegrators<DomainEleOp>::Assembly<PETSC>::BiLinearForm<
    GAUSS>::OpMixDivTimesScalar<SPACE_DIM>;

/**
 * @brief Integrate Lhs base of temperature times (heat capacity) times base of
 * temperature (T x T)
 *
 */
using OpCapacity = FormsIntegrators<DomainEleOp>::Assembly<PETSC>::BiLinearForm<
    GAUSS>::OpMass<1, 1>;

/**
 * @brief Integrating Rhs flux base (1/k) flux  (FLUX)
 */
using OpHdivFlux = FormsIntegrators<DomainEleOp>::Assembly<PETSC>::LinearForm<
    GAUSS>::OpBaseTimesVector<3, SPACE_DIM, 1>;

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

//! [Thermal problem]

//! [Body and heat source]
using DomainNaturalBC =
    NaturalBC<DomainEleOp>::Assembly<PETSC>::LinearForm<GAUSS>;
using OpBodyForce =
    DomainNaturalBC::OpFlux<NaturalMeshsetType<BLOCKSET>, 1, SPACE_DIM>;
using OpHeatSource =
    DomainNaturalBC::OpFlux<NaturalMeshsetType<BLOCKSET>, 1, 1>;
//! [Body and heat source]

//! [Natural boundary conditions]
using BoundaryNaturalBC =
    NaturalBC<BoundaryEleOp>::Assembly<PETSC>::LinearForm<GAUSS>;
using OpForce = BoundaryNaturalBC::OpFlux<NaturalForceMeshsets, 1, SPACE_DIM>;
using OpTemperatureBC =
    BoundaryNaturalBC::OpFlux<NaturalTemperatureMeshsets, 3, SPACE_DIM>;
//! [Natural boundary conditions]

//! [Essential boundary conditions (Least square approach)]
using OpEssentialFluxRhs =
    EssentialBC<BoundaryEleOp>::Assembly<PETSC>::LinearForm<
        GAUSS>::OpEssentialRhs<HeatFluxCubitBcData, 3, SPACE_DIM>;
using OpEssentialFluxLhs =
    EssentialBC<BoundaryEleOp>::Assembly<PETSC>::BiLinearForm<
        GAUSS>::OpEssentialLhs<HeatFluxCubitBcData, 3, SPACE_DIM>;
//! [Essential boundary conditions (Least square approach)]

double young_modulus = 1;
double poisson_ratio = 0.25;
double coeff_expansion = 1;
double ref_temp = 0.0;

double heat_conductivity =
    1; // Force / (time temperature )  or Power /
       // (length temperature) // Time unit is hour. force unit kN
double heat_capacity = 1; // length^2/(time^2 temperature) // length is
                          // millimeter time is hour

int order = 2; //< default approximation order

#include <ThermoElasticOps.hpp>   //< additional coupling opearyors
using namespace ThermoElasticOps; //< name space of coupling operators

struct ThermoElasticProblem {

  ThermoElasticProblem(MoFEM::Interface &m_field) : mField(m_field) {}

  MoFEMErrorCode runProblem();

private:
  MoFEM::Interface &mField;

  MoFEMErrorCode setupProblem();     ///< add fields
  MoFEMErrorCode createCommonData(); //< read global data from command line
  MoFEMErrorCode bC();               //< read boundary conditions
  MoFEMErrorCode OPs();              //< add operators to pipeline
  MoFEMErrorCode tsSolve();          //< time solver

  boost::shared_ptr<MatrixDouble> getMatDPtr();
};

//! [Run problem]
MoFEMErrorCode ThermoElasticProblem::runProblem() {
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
MoFEMErrorCode ThermoElasticProblem::setupProblem() {
  MoFEMFunctionBegin;
  Simple *simple = mField.getInterface<Simple>();
  // Add field
  constexpr FieldApproximationBase base = AINSWORTH_LEGENDRE_BASE;
  // Mechanical fields
  CHKERR simple->addDomainField("U", H1, base, SPACE_DIM);
  CHKERR simple->addBoundaryField("U", H1, base, SPACE_DIM);
  // Temperature
  const auto flux_space = (SPACE_DIM == 2) ? HCURL : HDIV;
  CHKERR simple->addDomainField("T", L2, AINSWORTH_LEGENDRE_BASE, 1);
  CHKERR simple->addDomainField("FLUX", flux_space, DEMKOWICZ_JACOBI_BASE, 1);
  CHKERR simple->addBoundaryField("FLUX", flux_space, DEMKOWICZ_JACOBI_BASE, 1);

  CHKERR PetscOptionsGetInt(PETSC_NULL, "", "-order", &order, PETSC_NULL);
  CHKERR simple->setFieldOrder("U", order);
  CHKERR simple->setFieldOrder("FLUX", order);
  CHKERR simple->setFieldOrder("T", order - 1);
  CHKERR simple->setUp();
  MoFEMFunctionReturn(0);
}
//! [Set up problem]

//! [Create common data]
MoFEMErrorCode ThermoElasticProblem::createCommonData() {
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

    MOFEM_LOG("ThermoElastic", Sev::inform)
        << "Young modulus " << young_modulus;
    MOFEM_LOG("ThermoElastic", Sev::inform)
        << "Poisson ratio " << poisson_ratio;
    MOFEM_LOG("ThermoElastic", Sev::inform)
        << "Coeff_expansion " << coeff_expansion;
    MOFEM_LOG("ThermoElastic", Sev::inform)
        << "Reference_temperature  " << ref_temp;

    MoFEMFunctionReturn(0);
  };

  CHKERR get_command_line_parameters();

  MoFEMFunctionReturn(0);
}
//! [Create common data]

//! [Boundary condition]
MoFEMErrorCode ThermoElasticProblem::bC() {
  MoFEMFunctionBegin;

  auto simple = mField.getInterface<Simple>();
  auto bc_mng = mField.getInterface<BcManager>();

  CHKERR bc_mng->removeBlockDOFsOnEntities<DisplacementCubitBcData>(
      simple->getProblemName(), "U");
  CHKERR bc_mng->pushMarkDOFsOnEntities<HeatFluxCubitBcData>(
      simple->getProblemName(), "FLUX");
  MoFEMFunctionReturn(0);
}
//! [Boundary condition]

//! [Push operators to pipeline]
MoFEMErrorCode ThermoElasticProblem::OPs() {
  MoFEMFunctionBegin;
  auto pipeline_mng = mField.getInterface<PipelineManager>();
  auto simple = mField.getInterface<Simple>();
  auto bc_mng = mField.getInterface<BcManager>();

  auto boundary_marker =
      bc_mng->getMergedBlocksMarker(vector<string>{"HEATFLUX"});
  auto mDPtr = getMatDPtr();

  auto integration_rule = [](int, int, int approx_order) {
    return 2 * approx_order;
  };
  CHKERR pipeline_mng->setDomainRhsIntegrationRule(integration_rule);
  CHKERR pipeline_mng->setDomainLhsIntegrationRule(integration_rule);
  CHKERR pipeline_mng->setBoundaryRhsIntegrationRule(integration_rule);
  CHKERR pipeline_mng->setBoundaryLhsIntegrationRule(integration_rule);

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

    if constexpr (SPACE_DIM == 2) {
      pipeline.push_back(new OpMakeHdivFromHcurl());
      pipeline.push_back(new OpSetContravariantPiolaTransformOnFace2D(jac_ptr));
      pipeline.push_back(new OpSetInvJacHcurlFace(inv_jac_ptr));
      pipeline.push_back(new OpSetInvJacL2ForFace(inv_jac_ptr));
    } else {
      pipeline.push_back(
          new OpSetHOContravariantPiolaTransform(HDIV, det_ptr, jac_ptr));
      pipeline.push_back(new OpSetHOInvJacVectorBase(HDIV, inv_jac_ptr));
      pipeline.push_back(new OpSetInvJacL2ForFace(inv_jac_ptr));
    }

    MoFEMFunctionReturn(0);
  };

  auto add_domain_rhs_ops = [&](auto &pipeline) {
    MoFEMFunctionBegin;

    auto mat_grad_ptr = boost::make_shared<MatrixDouble>();
    auto mat_strain_ptr = boost::make_shared<MatrixDouble>();
    auto mat_stress_ptr = boost::make_shared<MatrixDouble>();

    auto vec_temp_ptr = boost::make_shared<VectorDouble>();
    auto vec_temp_dot_ptr = boost::make_shared<VectorDouble>();
    auto mat_flux_ptr = boost::make_shared<MatrixDouble>();
    auto vec_temp_div_ptr = boost::make_shared<VectorDouble>();

    pipeline.push_back(new OpCalculateScalarFieldValues("T", vec_temp_ptr));
    pipeline.push_back(
        new OpCalculateScalarFieldValuesDot("T", vec_temp_dot_ptr));
    pipeline.push_back(new OpCalculateHdivVectorDivergence<3, SPACE_DIM>(
        "FLUX", vec_temp_div_ptr));
    pipeline.push_back(
        new OpCalculateHVecVectorField<3, SPACE_DIM>("FLUX", mat_flux_ptr));

    pipeline.push_back(new OpCalculateVectorFieldGradient<SPACE_DIM, SPACE_DIM>(
        "U", mat_grad_ptr));
    pipeline.push_back(
        new OpSymmetrizeTensor<SPACE_DIM>("U", mat_grad_ptr, mat_strain_ptr));
    pipeline.push_back(new OpStressThermal("U", mat_strain_ptr, vec_temp_ptr,
                                           mDPtr, mat_stress_ptr));

    pipeline.push_back(new OpSetBc("FLUX", true, boundary_marker));

    pipeline.push_back(new OpInternalForceCauchy(
        "U", mat_stress_ptr,
        [](double, double, double) constexpr { return 1; }));

    auto resistance = [](const double, const double, const double) {
      return (1. / heat_conductivity);
    };
    auto capacity = [&](const double, const double, const double) {
      return heat_capacity;
    };
    auto unity = [](const double, const double, const double) { return -1.; };
    pipeline.push_back(new OpHdivFlux("FLUX", mat_flux_ptr, resistance));
    pipeline.push_back(new OpHDivTemp("FLUX", vec_temp_ptr, unity));
    pipeline.push_back(new OpBaseDivFlux("T", vec_temp_div_ptr, unity));
    pipeline.push_back(new OpBaseDotT("T", vec_temp_dot_ptr, capacity));

    CHKERR DomainNaturalBC::AddFluxToPipeline<OpHeatSource>::add(
        pipeline, mField, "T", {time_scale}, "HEAT_SOURCE", Sev::inform);
    CHKERR DomainNaturalBC::AddFluxToPipeline<OpBodyForce>::add(
        pipeline_mng->getOpDomainRhsPipeline(), mField, "U", {time_scale},
        "BODY_FORCE", Sev::inform);

    pipeline.push_back(new OpUnSetBc("FLUX"));
    MoFEMFunctionReturn(0);
  };

  auto add_domain_lhs_ops = [&](auto &pipeline) {
    MoFEMFunctionBegin;
    pipeline.push_back(new OpSetBc("FLUX", true, boundary_marker));

    pipeline.push_back(new OpKCauchy("U", "U", mDPtr));
    pipeline.push_back(
        new ThermoElasticOps::OpKCauchyThermoElasticity("U", "T", mDPtr));

    auto resistance = [](const double, const double, const double) {
      return (1. / heat_conductivity);
    };
    auto capacity = [](const double, const double, const double) {
      return heat_capacity;
    };
    pipeline.push_back(new OpHdivHdiv("FLUX", "FLUX", resistance));
    pipeline.push_back(new OpHdivT(
        "FLUX", "T", []() { return -1; }, true));

    auto op_capacity = new OpCapacity("T", "T", capacity);
    op_capacity->feScalingFun = [](const FEMethod *fe_ptr) {
      return fe_ptr->ts_a;
    };
    pipeline.push_back(op_capacity);

    pipeline.push_back(new OpUnSetBc("FLUX"));
    MoFEMFunctionReturn(0);
  };

  auto add_boundary_rhs_ops = [&](auto &pipeline) {
    MoFEMFunctionBegin;

    if constexpr (SPACE_DIM == 2) {
      pipeline.push_back(new OpSetContravariantPiolaTransformOnEdge2D());
    } else {
      pipeline.push_back(new OpHOSetContravariantPiolaTransformOnFace3D(HDIV));
    }

    pipeline.push_back(new OpSetBc("FLUX", true, boundary_marker));

    CHKERR BoundaryNaturalBC::AddFluxToPipeline<OpForce>::add(
        pipeline_mng->getOpBoundaryRhsPipeline(), mField, "U", {time_scale},
        "FORCE", Sev::inform);

    CHKERR BoundaryNaturalBC::AddFluxToPipeline<OpTemperatureBC>::add(
        pipeline_mng->getOpBoundaryRhsPipeline(), mField, "FLUX", {time_scale},
        "TEMPERATURE", Sev::inform);

    pipeline.push_back(new OpUnSetBc("FLUX"));

    auto mat_flux_ptr = boost::make_shared<MatrixDouble>();
    pipeline.push_back(
        new OpCalculateHVecVectorField<3, SPACE_DIM>("FLUX", mat_flux_ptr));
    CHKERR EssentialBC<BoundaryEleOp>::Assembly<PETSC>::LinearForm<GAUSS>::
        AddEssentialToPipeline<OpEssentialFluxRhs>::add(
            mField, pipeline, simple->getProblemName(), "FLUX", mat_flux_ptr,
            {time_scale});

    MoFEMFunctionReturn(0);
  };

  auto add_boundary_lhs_ops = [&](auto &pipeline) {
    MoFEMFunctionBegin;

    if constexpr (SPACE_DIM == 2) {
      pipeline.push_back(new OpSetContravariantPiolaTransformOnEdge2D());
    } else {
      pipeline.push_back(new OpHOSetContravariantPiolaTransformOnFace3D(HDIV));
    }

    CHKERR EssentialBC<BoundaryEleOp>::Assembly<PETSC>::BiLinearForm<GAUSS>::
        AddEssentialToPipeline<OpEssentialFluxLhs>::add(
            mField, pipeline, simple->getProblemName(), "FLUX");

    MoFEMFunctionReturn(0);
  };

  auto get_bc_hook_rhs = [&]() {
    EssentialPreProc<DisplacementCubitBcData> hook(
        mField, pipeline_mng->getDomainRhsFE(), {time_scale});
    return hook;
  };
  auto get_bc_hook_lhs = [&]() {
    EssentialPreProc<DisplacementCubitBcData> hook(
        mField, pipeline_mng->getDomainLhsFE(), {time_scale});
    return hook;
  };

  pipeline_mng->getDomainRhsFE()->preProcessHook = get_bc_hook_rhs();
  pipeline_mng->getDomainLhsFE()->preProcessHook = get_bc_hook_lhs();

  CHKERR add_domain_ops(pipeline_mng->getOpDomainRhsPipeline());
  CHKERR add_domain_rhs_ops(pipeline_mng->getOpDomainRhsPipeline());

  CHKERR add_domain_ops(pipeline_mng->getOpDomainLhsPipeline());
  CHKERR add_domain_lhs_ops(pipeline_mng->getOpDomainLhsPipeline());

  CHKERR add_boundary_rhs_ops(pipeline_mng->getOpBoundaryRhsPipeline());
  CHKERR add_boundary_lhs_ops(pipeline_mng->getOpBoundaryLhsPipeline());

  MoFEMFunctionReturn(0);
}
//! [Push operators to pipeline]

//! [Solve]
MoFEMErrorCode ThermoElasticProblem::tsSolve() {
  MoFEMFunctionBegin;

  Simple *simple = mField.getInterface<Simple>();
  PipelineManager *pipeline_mng = mField.getInterface<PipelineManager>();
  ISManager *is_manager = mField.getInterface<ISManager>();

  auto dm = simple->getDM();
  auto solver = pipeline_mng->createTSIM();
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
    auto post_proc_fe = boost::make_shared<PostProcEle>(mField);

    auto mat_grad_ptr = boost::make_shared<MatrixDouble>();
    auto mat_strain_ptr = boost::make_shared<MatrixDouble>();
    auto mat_stress_ptr = boost::make_shared<MatrixDouble>();

    auto vec_temp_ptr = boost::make_shared<VectorDouble>();
    auto mat_flux_ptr = boost::make_shared<MatrixDouble>();

    auto det_ptr = boost::make_shared<VectorDouble>();
    auto jac_ptr = boost::make_shared<MatrixDouble>();
    auto inv_jac_ptr = boost::make_shared<MatrixDouble>();

    post_proc_fe->getOpPtrVector().push_back(
        new OpCalculateHOJac<SPACE_DIM>(jac_ptr));
    post_proc_fe->getOpPtrVector().push_back(
        new OpInvertMatrix<SPACE_DIM>(jac_ptr, det_ptr, inv_jac_ptr));
    post_proc_fe->getOpPtrVector().push_back(
        new OpSetHOInvJacToScalarBases<SPACE_DIM>(H1, inv_jac_ptr));

    if constexpr (SPACE_DIM == 2) {
      post_proc_fe->getOpPtrVector().push_back(new OpMakeHdivFromHcurl());
      post_proc_fe->getOpPtrVector().push_back(
          new OpSetContravariantPiolaTransformOnFace2D(jac_ptr));
      post_proc_fe->getOpPtrVector().push_back(
          new OpSetInvJacHcurlFace(inv_jac_ptr));
      post_proc_fe->getOpPtrVector().push_back(
          new OpSetInvJacL2ForFace(inv_jac_ptr));
    } else {
      post_proc_fe->getOpPtrVector().push_back(
          new OpSetHOContravariantPiolaTransform(HDIV, det_ptr, jac_ptr));
      post_proc_fe->getOpPtrVector().push_back(
          new OpSetHOInvJacVectorBase(HDIV, inv_jac_ptr));
    }

    post_proc_fe->getOpPtrVector().push_back(
        new OpCalculateScalarFieldValues("T", vec_temp_ptr));
    post_proc_fe->getOpPtrVector().push_back(
        new OpCalculateHVecVectorField<3, SPACE_DIM>("FLUX", mat_flux_ptr));

    auto u_ptr = boost::make_shared<MatrixDouble>();
    post_proc_fe->getOpPtrVector().push_back(
        new OpCalculateVectorFieldValues<SPACE_DIM>("U", u_ptr));
    post_proc_fe->getOpPtrVector().push_back(
        new OpCalculateVectorFieldGradient<SPACE_DIM, SPACE_DIM>("U",
                                                                 mat_grad_ptr));
    post_proc_fe->getOpPtrVector().push_back(
        new OpSymmetrizeTensor<SPACE_DIM>("U", mat_grad_ptr, mat_strain_ptr));
    post_proc_fe->getOpPtrVector().push_back(new OpStressThermal(
        "U", mat_strain_ptr, vec_temp_ptr, getMatDPtr(), mat_stress_ptr));

    using OpPPMap = OpPostProcMapInMoab<SPACE_DIM, SPACE_DIM>;

    post_proc_fe->getOpPtrVector().push_back(

        new OpPPMap(

            post_proc_fe->getPostProcMesh(), post_proc_fe->getMapGaussPts(),

            {{"T", vec_temp_ptr}},

            {{"U", u_ptr}, {"FLUX", mat_flux_ptr}},

            {},

            {{"STRAIN", mat_strain_ptr}, {"STRESS", mat_stress_ptr}}

            )

    );

    return post_proc_fe;
  };

  auto monitor_ptr = boost::make_shared<FEMethod>();
  auto post_proc_fe = create_post_process_element();

  auto set_time_monitor = [&](auto dm, auto solver) {
    MoFEMFunctionBegin;
    monitor_ptr->preProcessHook = [&]() {
      MoFEMFunctionBegin;
      CHKERR DMoFEMLoopFiniteElements(dm, "dFE", post_proc_fe);
      CHKERR post_proc_fe->writeFile(
          "out_" + boost::lexical_cast<std::string>(monitor_ptr->ts_step) +
          ".h5m");
      MoFEMFunctionReturn(0);
    };
    auto null = boost::shared_ptr<FEMethod>();
    CHKERR DMMoFEMTSSetMonitor(dm, solver, simple->getDomainFEName(), null,
                               monitor_ptr, null);
    MoFEMFunctionReturn(0);
  };

  auto set_fieldsplit_preconditioner = [&](auto solver) {
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
      auto is_all_bc = bc_mng->getBlockIS(name_prb, "HEATFLUX", "FLUX", 0, 0);
      int is_all_bc_size;
      CHKERR ISGetSize(is_all_bc, &is_all_bc_size);
      MOFEM_LOG("ThermoElastic", Sev::inform)
          << "Field split block size " << is_all_bc_size;
      CHKERR PCFieldSplitSetIS(pc, PETSC_NULL,
                               is_all_bc); // boundary block
    }

    MoFEMFunctionReturnHot(0);
  };

  auto D = smartCreateDMVector(dm);
  CHKERR TSSetSolution(solver, D);
  CHKERR TSSetFromOptions(solver);
  CHKERR set_section_monitor(solver);
  CHKERR set_time_monitor(dm, solver);
  CHKERR set_fieldsplit_preconditioner(solver);
  CHKERR TSSetUp(solver);
  CHKERR TSSolve(solver, NULL);

  MoFEMFunctionReturn(0);
}
//! [Solve]

boost::shared_ptr<MatrixDouble> ThermoElasticProblem::getMatDPtr() {
  auto set_matrial_stiffness = [&](auto mDPtr) {
    FTensor::Index<'i', SPACE_DIM> i;
    FTensor::Index<'j', SPACE_DIM> j;
    FTensor::Index<'k', SPACE_DIM> k;
    FTensor::Index<'l', SPACE_DIM> l;
    constexpr auto t_kd = FTensor::Kronecker_Delta_symmetric<int>();
    const double bulk_modulus_K = young_modulus / (3 * (1 - 2 * poisson_ratio));
    const double shear_modulus_G = young_modulus / (2 * (1 + poisson_ratio));

    // Plane stress or when 1, plane strain or 3d
    double A = (SPACE_DIM == 2)
                   ? 2 * shear_modulus_G /
                         (bulk_modulus_K + (4. / 3.) * shear_modulus_G)
                   : 1;
    auto t_D = getFTensor4DdgFromMat<SPACE_DIM, SPACE_DIM, 0>(*mDPtr);
    t_D(i, j, k, l) = 2 * shear_modulus_G * ((t_kd(i, k) ^ t_kd(j, l)) / 4.) +
                      A * (bulk_modulus_K - (2. / 3.) * shear_modulus_G) *
                          t_kd(i, j) * t_kd(k, l);
    return mDPtr;
  };

  constexpr auto size_symm = (SPACE_DIM * (SPACE_DIM + 1)) / 2;
  return set_matrial_stiffness(
      boost::make_shared<MatrixDouble>(size_symm * size_symm, 1));
}

static char help[] = "...\n\n";

int main(int argc, char *argv[]) {

  // Initialisation of MoFEM/PETSc and MOAB data structures
  const char param_file[] = "param_file.petsc";
  MoFEM::Core::Initialize(&argc, &argv, param_file, help);

  // Add logging channel for example
  auto core_log = logging::core::get();
  core_log->add_sink(
      LogManager::createSink(LogManager::getStrmWorld(), "ThermoElastic"));
  LogManager::setLog("ThermoElastic");
  MOFEM_LOG_TAG("ThermoElastic", "ThermoElastic");

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

    //! [ThermoElasticProblem]
    ThermoElasticProblem ex(m_field);
    CHKERR ex.runProblem();
    //! [ThermoElasticProblem]
  }
  CATCH_ERRORS;

  CHKERR MoFEM::Core::Finalize();
}
