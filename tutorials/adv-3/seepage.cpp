/**
 * \file thermo_plastic.cpp
 * \example thermo_plastic.cpp
 *
 * Thermo plasticity
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
  using DomainEleOp = DomainEle::UserDataOperator;
  using BoundaryEle = PipelineManager::EdgeEle;
  using BoundaryEleOp = BoundaryEle::UserDataOperator;
  using PostProcEle = PostProcFaceOnRefinedMesh;
};

template <> struct ElementsAndOps<3> {
  using DomainEle = VolumeElementForcesAndSourcesCore;
  using DomainEleOp = DomainEle::UserDataOperator;
  using BoundaryEle = FaceElementForcesAndSourcesCore;
  using BoundaryEleOp = BoundaryEle::UserDataOperator;
  using PostProcEle = PostProcVolumeOnRefinedMesh;
};

constexpr int SPACE_DIM =
    EXECUTABLE_DIMENSION; //< Space dimension of problem, mesh

constexpr EntityType boundary_ent = SPACE_DIM == 3 ? MBTRI : MBEDGE;
using EntData = DataForcesAndSourcesCore::EntData;
using DomainEle = ElementsAndOps<SPACE_DIM>::DomainEle;
using DomainEleOp = ElementsAndOps<SPACE_DIM>::DomainEleOp;
using BoundaryEle = ElementsAndOps<SPACE_DIM>::BoundaryEle;
using BoundaryEleOp = ElementsAndOps<SPACE_DIM>::BoundaryEleOp;
using PostProcEle = ElementsAndOps<SPACE_DIM>::PostProcEle;

using AssemblyDomainEleOp =
    FormsIntegrators<DomainEleOp>::Assembly<PETSC>::OpBase;

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

//! [Only used with Hencky/nonlinear material]
using OpKPiola = FormsIntegrators<DomainEleOp>::Assembly<PETSC>::BiLinearForm<
    GAUSS>::OpGradTensorGrad<1, SPACE_DIM, SPACE_DIM, 1>;
using OpInternalForcePiola = FormsIntegrators<DomainEleOp>::Assembly<
    PETSC>::LinearForm<GAUSS>::OpGradTimesTensor<1, SPACE_DIM, SPACE_DIM>;
//! [Only used with Hencky/nonlinear material]

//! [Essential boundary conditions]
using OpBoundaryMass = FormsIntegrators<BoundaryEleOp>::Assembly<
    PETSC>::BiLinearForm<GAUSS>::OpMass<1, SPACE_DIM>;
using OpBoundaryVec = FormsIntegrators<BoundaryEleOp>::Assembly<
    PETSC>::LinearForm<GAUSS>::OpBaseTimesVector<1, SPACE_DIM, 0>;
using OpBoundaryInternal = FormsIntegrators<BoundaryEleOp>::Assembly<
    PETSC>::LinearForm<GAUSS>::OpBaseTimesVector<1, SPACE_DIM, 1>;
//! [Essential boundary conditions]
using OpScaleL2 = MoFEM::OpScaleBaseBySpaceInverseOfMeasure<DomainEleOp>;

using OpBaseDivU = FormsIntegrators<DomainEleOp>::Assembly<PETSC>::BiLinearForm<
    GAUSS>::OpMixScalarTimesDiv<SPACE_DIM>;

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
using OpHdivQ = FormsIntegrators<DomainEleOp>::Assembly<PETSC>::BiLinearForm<
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
using OpHDivH = FormsIntegrators<DomainEleOp>::Assembly<PETSC>::LinearForm<
    GAUSS>::OpMixDivTimesU<3, 1, 2>;

/**
 * @brief Integrate Rhs base of temerature time heat capacity times heat rate
 * (T)
 *
 */
using OpBaseDotT = FormsIntegrators<DomainEleOp>::Assembly<PETSC>::LinearForm<
    GAUSS>::OpBaseTimesScalarField<1>;

/**
 * @brief Integrate Rhs base of temerature times divergenc of flux (T)
 *
 */
using OpBaseDivFlux = OpBaseDotT;

using OpHeatSource = FormsIntegrators<DomainEleOp>::Assembly<PETSC>::LinearForm<
    GAUSS>::OpSource<1, 1>;
using OpHBC = FormsIntegrators<BoundaryEleOp>::Assembly<
    PETSC>::LinearForm<GAUSS>::OpNormalMixVecTimesScalar<SPACE_DIM>;

double scale = 1.;

double young_modulus = 206913;
double poisson_ratio = 0.29;
double conductivity = 1;
double capacity = 1;
double fluid_density = 1;

#include <OpPostProcElastic.hpp>
#include <SeepageOps.hpp>

struct Monitor : public FEMethod {

  Monitor(SmartPetscObj<DM> &dm, boost::shared_ptr<PostProcEle> &post_proc_fe,
          boost::shared_ptr<DomainEle> &reaction_fe,
          std::tuple<SmartPetscObj<Vec>, SmartPetscObj<VecScatter>> ux_scatter,
          std::tuple<SmartPetscObj<Vec>, SmartPetscObj<VecScatter>> uy_scatter,
          std::tuple<SmartPetscObj<Vec>, SmartPetscObj<VecScatter>> uz_scatter)
      : dM(dm), postProcFe(post_proc_fe), reactionFe(reaction_fe),
        uXScatter(ux_scatter), uYScatter(uy_scatter), uZScatter(uz_scatter){};

  MoFEMErrorCode preProcess() { return 0; }
  MoFEMErrorCode operator()() { return 0; }

  MoFEMErrorCode postProcess() {
    MoFEMFunctionBegin;

    auto make_vtk = [&]() {
      MoFEMFunctionBegin;
      CHKERR DMoFEMLoopFiniteElements(dM, "dFE", postProcFe);
      CHKERR postProcFe->writeFile(
          "out_plastic_" + boost::lexical_cast<std::string>(ts_step) + ".h5m");
      MoFEMFunctionReturn(0);
    };

    auto calculate_reaction = [&]() {
      MoFEMFunctionBegin;
      auto r = smartCreateDMVector(dM);
      reactionFe->f = r;
      CHKERR VecZeroEntries(r);
      CHKERR DMoFEMLoopFiniteElements(dM, "dFE", reactionFe);
      CHKERR VecGhostUpdateBegin(r, ADD_VALUES, SCATTER_REVERSE);
      CHKERR VecGhostUpdateEnd(r, ADD_VALUES, SCATTER_REVERSE);
      CHKERR VecAssemblyBegin(r);
      CHKERR VecAssemblyEnd(r);

      double sum;
      CHKERR VecSum(r, &sum);
      MOFEM_LOG_C("EXAMPLE", Sev::inform, "reaction time %3.4e %3.4e", ts_t,
                  sum);

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
      MOFEM_LOG_C("EXAMPLE", Sev::inform, "%s time %3.4e min %3.4e max %3.4e",
                  msg.c_str(), ts_t, min, max);
      MoFEMFunctionReturn(0);
    };

    CHKERR make_vtk();
    CHKERR calculate_reaction();
    CHKERR print_max_min(uXScatter, "Ux");
    CHKERR print_max_min(uYScatter, "Uy");
    if (SPACE_DIM == 3)
      CHKERR print_max_min(uZScatter, "Uz");

    MoFEMFunctionReturn(0);
  }

private:
  SmartPetscObj<DM> dM;
  boost::shared_ptr<PostProcEle> postProcFe;
  boost::shared_ptr<DomainEle> reactionFe;
  std::tuple<SmartPetscObj<Vec>, SmartPetscObj<VecScatter>> uXScatter;
  std::tuple<SmartPetscObj<Vec>, SmartPetscObj<VecScatter>> uYScatter;
  std::tuple<SmartPetscObj<Vec>, SmartPetscObj<VecScatter>> uZScatter;
};

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
  boost::shared_ptr<DomainEle> reactionFe;

  std::tuple<SmartPetscObj<Vec>, SmartPetscObj<VecScatter>> uXScatter;
  std::tuple<SmartPetscObj<Vec>, SmartPetscObj<VecScatter>> uYScatter;
  std::tuple<SmartPetscObj<Vec>, SmartPetscObj<VecScatter>> uZScatter;

  boost::shared_ptr<std::vector<unsigned char>> boundaryMarker;
  boost::shared_ptr<std::vector<unsigned char>> reactionMarker;

  std::vector<FTensor::Tensor1<double, 3>> bodyForces;

  struct BcHFun {
    BcHFun(double v, FEMethod &fe) : valH(v), fE(fe) {}
    double operator()(const double x, const double y, const double z) {
      return valH * y;// - y; // * fE.ts_t;
    }

  private:
    double valH;
    FEMethod &fE;
  };
  std::vector<BcHFun> bcHFunctions;

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
  // Mechanical fields
  CHKERR simple->addDomainField("U", H1, base, SPACE_DIM);
  CHKERR simple->addBoundaryField("U", H1, base, SPACE_DIM);
  // Temerature
  const auto flux_space = (SPACE_DIM == 2) ? HCURL : HDIV;
  CHKERR simple->addDomainField("H", L2, AINSWORTH_LEGENDRE_BASE, 1);
  CHKERR simple->addDomainField("FLUX", flux_space, DEMKOWICZ_JACOBI_BASE, 1);
  CHKERR simple->addBoundaryField("FLUX", flux_space, DEMKOWICZ_JACOBI_BASE, 1);
  
  int order = 2.;
  CHKERR PetscOptionsGetInt(PETSC_NULL, "", "-order", &order, PETSC_NULL);
  CHKERR simple->setFieldOrder("U", order);
  CHKERR simple->setFieldOrder("H", order - 1);
  CHKERR simple->setFieldOrder("FLUX", order);

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
    CHKERR PetscOptionsGetScalar(PETSC_NULL, "", "-conductivity", &conductivity,
                                 PETSC_NULL);
    CHKERR PetscOptionsGetScalar(PETSC_NULL, "", "-capacity", &capacity,
                                 PETSC_NULL);
    CHKERR PetscOptionsGetScalar(PETSC_NULL, "", "-fluid_density",
                                 &fluid_density, PETSC_NULL);

    MOFEM_LOG("EXAMPLE", Sev::inform) << "Young modulus " << young_modulus;
    MOFEM_LOG("EXAMPLE", Sev::inform) << "Poisson ratio " << poisson_ratio;
    MOFEM_LOG("EXAMPLE", Sev::inform) << "Conductivity " << conductivity;
    MOFEM_LOG("EXAMPLE", Sev::inform) << "Capacity " << capacity;
    MOFEM_LOG("EXAMPLE", Sev::inform) << "Fluid denisty " << fluid_density;

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

    mDPtr = boost::make_shared<MatrixDouble>();
    mDPtr_Axiator = boost::make_shared<MatrixDouble>();
    mDPtr_Deviator = boost::make_shared<MatrixDouble>();
    
    constexpr auto size_symm = (SPACE_DIM * (SPACE_DIM + 1)) / 2;
    mDPtr->resize(size_symm * size_symm, 1);
    mDPtr_Axiator->resize(size_symm * size_symm, 1);
    mDPtr_Deviator->resize(size_symm * size_symm, 1);
  

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

  // undrained means that there will be no fluid flow on the boundaries, essentially all boundaries are fixed...
  PetscBool zero_fix_skin_flux = PETSC_FALSE;
  CHKERR PetscOptionsGetBool(PETSC_NULL, "", "-undrained", &zero_fix_skin_flux,
                             PETSC_NULL);
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

  CHKERR bc_mng->pushMarkDOFsOnEntities(simple->getProblemName(), "FIX_X", "U",
                                        0, 0);
  CHKERR bc_mng->pushMarkDOFsOnEntities(simple->getProblemName(), "FIX_Y", "U",
                                        1, 1);
  CHKERR bc_mng->pushMarkDOFsOnEntities(simple->getProblemName(), "FIX_Z", "U",
                                        2, 2);
  CHKERR bc_mng->pushMarkDOFsOnEntities(simple->getProblemName(), "FIX_ALL",
                                        "U", 0, 3);

  auto &bc_map = bc_mng->getBcMapByBlockName();
  boundaryMarker = bc_mng->getMergedBlocksMarker(vector<string>{"FIX_"});

  CHKERR bc_mng->pushMarkDOFsOnEntities(simple->getProblemName(), "REACTION",
                                        "U", 0, 3);
  for (auto b : bc_map) {
    MOFEM_LOG("EXAMPLE", Sev::verbose) << "Marker " << b.first;
  }

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
  auto pipeline_mng = mField.getInterface<PipelineManager>();
  auto simple = mField.getInterface<Simple>();
  auto bc_mng = mField.getInterface<BcManager>();

  auto u_grad_ptr = boost::make_shared<MatrixDouble>();
  auto dot_u_grad_ptr = boost::make_shared<MatrixDouble>();
  auto trace_dot_u_grad_ptr = boost::make_shared<VectorDouble>();
  auto h_ptr = boost::make_shared<VectorDouble>();
  auto dot_h_ptr = boost::make_shared<VectorDouble>();
  auto flux_ptr = boost::make_shared<MatrixDouble>();
  auto div_flux_ptr = boost::make_shared<VectorDouble>();
  auto strain_ptr = boost::make_shared<MatrixDouble>();
  auto stress_ptr = boost::make_shared<MatrixDouble>();

  auto integration_rule = [](int, int, int approx_order) {
    return 2 * approx_order;
  };

  auto add_domain_base_ops = [&](auto &pipeline) {
    MoFEMFunctionBegin;

    if (SPACE_DIM == 2) {
      auto det_ptr = boost::make_shared<VectorDouble>();
      auto jac_ptr = boost::make_shared<MatrixDouble>();
      auto inv_jac_ptr = boost::make_shared<MatrixDouble>();
      pipeline.push_back(new OpCalculateHOJacForFace(jac_ptr));
      pipeline.push_back(new OpInvertMatrix<2>(jac_ptr, det_ptr, inv_jac_ptr));
      pipeline.push_back(new OpSetInvJacH1ForFace(inv_jac_ptr));
      pipeline.push_back(new OpMakeHdivFromHcurl());
      pipeline.push_back(new OpSetContravariantPiolaTransformOnFace2D(jac_ptr));
      pipeline.push_back(new OpSetInvJacHcurlFace(inv_jac_ptr));
      pipeline.push_back(new OpSetInvJacL2ForFace(inv_jac_ptr));
    }

    pipeline.push_back(new OpCalculateVectorFieldGradient<SPACE_DIM, SPACE_DIM>(
        "U", u_grad_ptr));
    pipeline.push_back(
        new OpSymmetrizeTensor<SPACE_DIM>("U", u_grad_ptr, strain_ptr));
    pipeline.push_back(
        new OpCalculateVectorFieldGradientDot<SPACE_DIM, SPACE_DIM>(
            "U", dot_u_grad_ptr));
    pipeline.push_back(
        new OpCalculateTraceFromMat<SPACE_DIM>(dot_u_grad_ptr,
                                               trace_dot_u_grad_ptr));
                                               
    pipeline.push_back(new OpCalculateScalarFieldValues("H", h_ptr));

    // pipeline.push_back(new OpCalculateScalarFieldValuesDot("H", dot_h_ptr));

    pipeline.push_back(new OpCalculateHVecVectorField<3>("FLUX", flux_ptr));
    pipeline.push_back(new OpCalculateHdivVectorDivergence<3, SPACE_DIM>(
        "FLUX", div_flux_ptr));

    MoFEMFunctionReturn(0);
  };

  auto add_domain_ops_lhs_mechanical = [&](auto &pipeline) {
    MoFEMFunctionBegin;
    pipeline.push_back(new OpSetBc("U", true, boundaryMarker));
    pipeline_mng->getOpDomainLhsPipeline().push_back(
        new OpKCauchy("U", "U", mDPtr));
    pipeline.push_back(new OpUnSetBc("U"));
    pipeline.push_back(new OpBaseDivU(
        "H", "U", []() { return -9.81; }, true, true));
    
    
    MoFEMFunctionReturn(0);
  };

  auto add_domain_ops_rhs_mechanical = [&](auto &pipeline) {
    MoFEMFunctionBegin;
    pipeline.push_back(new OpSetBc("U", true, boundaryMarker));

    for (_IT_CUBITMESHSETS_BY_SET_TYPE_FOR_LOOP_(mField, BLOCKSET, it)) {
      const std::string block_name = "BODY_FORCE";
      if (it->getName().compare(0, block_name.size(), block_name) == 0) {
        std::vector<double> attr;
        CHKERR it->getAttributes(attr);
        if (attr.size() == 3) {
          bodyForces.push_back(
              FTensor::Tensor1<double, 3>{attr[0], attr[1], attr[2]});
        } else {
          SETERRQ1(PETSC_COMM_SELF, MOFEM_INVALID_DATA,
                   "Should be three atributes in BODYFORCE blockset, but is %d",
                   attr.size());
        }
      }
    }

    auto get_body_force = [this](const double, const double, const double) {
      auto *pipeline_mng = mField.getInterface<PipelineManager>();
      FTensor::Index<'i', SPACE_DIM> i;
      FTensor::Tensor1<double, SPACE_DIM> t_source;
      t_source(i) = 0;
      auto fe_domain_rhs = pipeline_mng->getDomainRhsFE();
      const auto time = fe_domain_rhs->ts_t;
      // hardcoded gravity load
      for (auto &t_b : bodyForces)
        t_source(i) += (scale * t_b(i)) * time;
      return t_source;
    };

    pipeline.push_back(new OpBodyForce("U", get_body_force));

    // Calculate internal forece
    pipeline.push_back(
        new OpTensorTimesSymmetricTensor<SPACE_DIM, SPACE_DIM>(
            "U", strain_ptr, stress_ptr, mDPtr));
    pipeline.push_back(
        new OpInternalForceCauchy("U", stress_ptr));

    pipeline.push_back(
        new SeepageOps::OpDomainRhsHydrostaticStress<SPACE_DIM>("U", h_ptr));

    // pipeline.push_back(new OpUnSetBc("U"));
    MoFEMFunctionReturn(0);
  };

  auto add_boundary_ops_lhs_mechanical = [&](auto &pipeline) {
    MoFEMFunctionBegin;
    auto &bc_map = mField.getInterface<BcManager>()->getBcMapByBlockName();
    for (auto bc : bc_map) {
      if (bc_mng->checkBlock(bc, "FIX_")) {
        pipeline.push_back(
            new OpSetBc("U", false, bc.second->getBcMarkersPtr()));
        pipeline.push_back(new OpBoundaryMass(
            "U", "U", [](double, double, double) { return 1.; },
            bc.second->getBcEntsPtr()));
        pipeline.push_back(new OpUnSetBc("U"));
      }
    }
    MoFEMFunctionReturn(0);
  };

  auto add_boundary_ops_rhs_mechanical = [&](auto &pipeline) {
    MoFEMFunctionBegin;

    // auto get_time = [&](double, double, double) {
    //   auto *pipeline_mng = mField.getInterface<PipelineManager>();
    //   auto &fe_domain_rhs = pipeline_mng->getBoundaryRhsFE();
    //   return fe_domain_rhs->ts_t;
    // };

    auto get_time_scaled = [&](double, double, double) {
      auto *pipeline_mng = mField.getInterface<PipelineManager>();
      auto &fe_domain_rhs = pipeline_mng->getBoundaryRhsFE();
      return fe_domain_rhs->ts_t * scale;
    };

    // auto get_minus_time = [&](double, double, double) {
    //   auto *pipeline_mng = mField.getInterface<PipelineManager>();
    //   auto &fe_domain_rhs = pipeline_mng->getBoundaryRhsFE();
    //   return -fe_domain_rhs->ts_t;
    // };

    auto time_scaled = [&](double, double, double) {
      auto *pipeline_mng = mField.getInterface<PipelineManager>();
      auto &fe_domain_rhs = pipeline_mng->getBoundaryRhsFE();
      return fe_domain_rhs->ts_t;
    };

    pipeline.push_back(new OpSetBc("U", true, boundaryMarker));

    for (_IT_CUBITMESHSETS_BY_SET_TYPE_FOR_LOOP_(mField, BLOCKSET, it)) {
      if (it->getName().compare(0, 5, "FORCE") == 0) {
        Range force_edges;
        std::vector<double> attr_vec;
        CHKERR it->getMeshsetIdEntitiesByDimension(
            mField.get_moab(), SPACE_DIM - 1, force_edges, true);
        it->getAttributes(attr_vec);
        if (attr_vec.size() < SPACE_DIM)
          SETERRQ(PETSC_COMM_WORLD, MOFEM_DATA_INCONSISTENCY,
                  "Wrong size of boundary attributes vector. Set right block "
                  "size attributes.");
        auto force_vec_ptr = boost::make_shared<MatrixDouble>(SPACE_DIM, 1);
        std::copy(&attr_vec[0], &attr_vec[SPACE_DIM],
                  force_vec_ptr->data().begin());
        pipeline.push_back(
            new OpBoundaryVec("U", force_vec_ptr, get_time_scaled,
                              boost::make_shared<Range>(force_edges)));
      }
    }

    pipeline.push_back(new OpUnSetBc("U"));

    auto u_mat_ptr = boost::make_shared<MatrixDouble>();
    pipeline.push_back(
        new OpCalculateVectorFieldValues<SPACE_DIM>("U", u_mat_ptr));

    for (auto &bc : mField.getInterface<BcManager>()->getBcMapByBlockName()) {
      if (bc_mng->checkBlock(bc, "FIX_")) {
        pipeline.push_back(
            new OpSetBc("U", false, bc.second->getBcMarkersPtr()));
        auto attr_vec = boost::make_shared<MatrixDouble>(SPACE_DIM, 1);
        attr_vec->clear();
        if (bc.second->bcAttributes.size() < SPACE_DIM)
          SETERRQ1(PETSC_COMM_WORLD, MOFEM_DATA_INCONSISTENCY,
                   "Wrong size of boundary attributes vector. Set right block "
                   "size attributes. Size of attributes %d",
                   bc.second->bcAttributes.size());
        std::copy(&bc.second->bcAttributes[0],
                  &bc.second->bcAttributes[SPACE_DIM],
                  attr_vec->data().begin());

        pipeline.push_back(new OpBoundaryVec("U", attr_vec, time_scaled,
                                             bc.second->getBcEntsPtr()));
        pipeline.push_back(new OpBoundaryInternal(
            "U", u_mat_ptr, [](double, double, double) { return 1.; },
            bc.second->getBcEntsPtr()));

        pipeline.push_back(new OpUnSetBc("U"));
      }
    }

    MoFEMFunctionReturn(0);
  };

  auto add_domain_ops_lhs_seepage = [&](auto &pipeline, auto &fe) {
    MoFEMFunctionBegin;
    auto resistance = [](const double, const double, const double) {
      return (1. / conductivity);
    };
    auto time_pass = [&]() {
      return  -fe->ts_a;
    };

    auto unity = []() { return -1; };
    pipeline.push_back(new OpHdivHdiv("FLUX", "FLUX", resistance));
    pipeline.push_back(new OpHdivQ("FLUX", "H", unity, true));
    pipeline.push_back(new OpBaseDivU(
        "H", "U", time_pass, false, false));

    // pipeline.push_back(new OpCapacity("T", "T", capacity));
    MoFEMFunctionReturn(0);
  };

  auto add_domain_ops_rhs_seepage = [&](auto &pipeline) {
    MoFEMFunctionBegin;
    auto resistance = [](const double, const double, const double) {
      return (1. / conductivity);
    };
    auto get_capacity = [&](const double, const double, const double) {
      return capacity;
    };
    auto minus_one = [](const double, const double, const double) {
      return -1;
    };
    auto plus_one = [](const double, const double, const double) {
      return 1;
    };
    
    // 1
    pipeline.push_back(new OpHdivFlux("FLUX", flux_ptr, resistance));
    
    // 2
    pipeline.push_back(new OpHDivH("FLUX", h_ptr, minus_one));

    // 3
    pipeline.push_back(new OpBaseDotT("H", trace_dot_u_grad_ptr, minus_one));

    // 4
    pipeline.push_back(new OpBaseDivFlux("H", div_flux_ptr, minus_one));

    MoFEMFunctionReturn(0);
  };

  auto add_boundary_ops_rhs_seepage = [&](auto &pipeline, auto &fe) {
    MoFEMFunctionBegin;

    if (SPACE_DIM == 2) {
      pipeline.push_back(new OpSetContravariantPiolaTransformOnEdge2D());
    } else if (SPACE_DIM == 3) {
      pipeline.push_back(new OpHOSetCovariantPiolaTransformOnFace3D(HDIV));
    }

    std::vector<Range> tmp_edges_vec; // Create storage/container for
                                      // edges/facses on which natural boundary
                                      // condition is applied on pressure head.
    bcHFunctions.clear();

    for (_IT_CUBITMESHSETS_BY_SET_TYPE_FOR_LOOP_(mField, BLOCKSET, it)) {
      const std::string block_name = "H_";
      if (it->getName().compare(0, block_name.size(), block_name) == 0) {
        Range temp_edges;
        std::vector<double> attr_vec;
        CHKERR it->getMeshsetIdEntitiesByDimension(
            mField.get_moab(), SPACE_DIM - 1, temp_edges, true);
        it->getAttributes(attr_vec);
        if (attr_vec.size() != 1)
          SETERRQ(PETSC_COMM_SELF, MOFEM_INVALID_DATA,
                  "Should be one attribute");
        MOFEM_LOG("EXAMPLE", Sev::inform)
            << "Set hydraulic head " << attr_vec[0] << " on ents:\n"
            << temp_edges;
        bcHFunctions.push_back(BcHFun(attr_vec[0], *fe));
        tmp_edges_vec.push_back(temp_edges); // Add range to storage.
      }
    }

    if(bcHFunctions.size()!=tmp_edges_vec.size())
      SETERRQ(PETSC_COMM_WORLD, MOFEM_DATA_INCONSISTENCY, "Data inconstency");

    for (int v = 0; v != bcHFunctions.size(); ++v) {
      pipeline.push_back(
          new OpHBC("FLUX", bcHFunctions[v],
                    boost::make_shared<Range>(tmp_edges_vec[v])));
    }
    MoFEMFunctionReturn(0);
  };

  // LHS
  CHKERR add_domain_base_ops(pipeline_mng->getOpDomainLhsPipeline());
  CHKERR add_domain_ops_lhs_mechanical(pipeline_mng->getOpDomainLhsPipeline());
  CHKERR add_domain_ops_lhs_seepage(pipeline_mng->getOpDomainLhsPipeline(), pipeline_mng->getDomainLhsFE());
  CHKERR add_boundary_ops_lhs_mechanical(
      pipeline_mng->getOpBoundaryLhsPipeline());

  // RHS
  CHKERR add_boundary_ops_rhs_mechanical(
      pipeline_mng->getOpBoundaryRhsPipeline());
  CHKERR add_boundary_ops_rhs_seepage(pipeline_mng->getOpBoundaryRhsPipeline(),
                                      pipeline_mng->getBoundaryRhsFE());

  CHKERR add_domain_base_ops(pipeline_mng->getOpDomainRhsPipeline());
  CHKERR add_domain_ops_rhs_mechanical(pipeline_mng->getOpDomainRhsPipeline());
  CHKERR add_domain_ops_rhs_seepage(pipeline_mng->getOpDomainRhsPipeline());

  
  CHKERR pipeline_mng->setDomainRhsIntegrationRule(integration_rule);
  CHKERR pipeline_mng->setDomainLhsIntegrationRule(integration_rule);
  CHKERR pipeline_mng->setBoundaryLhsIntegrationRule(integration_rule);
  CHKERR pipeline_mng->setBoundaryRhsIntegrationRule(integration_rule);

  auto create_reaction_pipeline = [&](auto &pipeline) {
    MoFEMFunctionBegin;

    if (reactionMarker) {

      if (SPACE_DIM == 2) {
        auto det_ptr = boost::make_shared<VectorDouble>();
        auto jac_ptr = boost::make_shared<MatrixDouble>();
        auto inv_jac_ptr = boost::make_shared<MatrixDouble>();
        pipeline.push_back(new OpCalculateHOJacForFace(jac_ptr));
        pipeline.push_back(
            new OpInvertMatrix<2>(jac_ptr, det_ptr, inv_jac_ptr));
        pipeline.push_back(new OpSetInvJacH1ForFace(inv_jac_ptr));
      }

      pipeline.push_back(
          new OpSymmetrizeTensor<SPACE_DIM>("U", u_grad_ptr, strain_ptr));
      pipeline_mng->getOpDomainRhsPipeline().push_back(
          new OpTensorTimesSymmetricTensor<SPACE_DIM, SPACE_DIM>(
              "U", strain_ptr, stress_ptr, mDPtr));
      pipeline_mng->getOpDomainRhsPipeline().push_back(
          new OpInternalForceCauchy("U", stress_ptr));

      pipeline.push_back(new OpUnSetBc("U"));
    }

    MoFEMFunctionReturn(0);
  };

  reactionFe = boost::make_shared<DomainEle>(mField);
  reactionFe->getRuleHook = integration_rule;

  CHKERR create_reaction_pipeline(reactionFe->getOpPtrVector());

  MoFEMFunctionReturn(0);
}
//! [Push operators to pipeline]

//! [Solve]
MoFEMErrorCode Example::tsSolve() {
  MoFEMFunctionBegin;

  Simple *simple = mField.getInterface<Simple>();
  PipelineManager *pipeline_mng = mField.getInterface<PipelineManager>();
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

  auto create_post_process_element = [&]() {
    MoFEMFunctionBegin;
    postProcFe = boost::make_shared<PostProcEle>(mField);
    postProcFe->generateReferenceElementMesh();
    if (SPACE_DIM == 2) {
      auto det_ptr = boost::make_shared<VectorDouble>();
      auto jac_ptr = boost::make_shared<MatrixDouble>();
      auto inv_jac_ptr = boost::make_shared<MatrixDouble>();
      postProcFe->getOpPtrVector().push_back(
          new OpCalculateHOJacForFace(jac_ptr));
      postProcFe->getOpPtrVector().push_back(
          new OpInvertMatrix<2>(jac_ptr, det_ptr, inv_jac_ptr));
      postProcFe->getOpPtrVector().push_back(
          new OpSetInvJacH1ForFace(inv_jac_ptr));
          postProcFe->getOpPtrVector().push_back(new OpMakeHdivFromHcurl());
      postProcFe->getOpPtrVector().push_back(new OpSetContravariantPiolaTransformOnFace2D(jac_ptr));
      postProcFe->getOpPtrVector().push_back(new OpSetInvJacHcurlFace(inv_jac_ptr));
      postProcFe->getOpPtrVector().push_back(new OpSetInvJacL2ForFace(inv_jac_ptr));
    }

    auto u_grad_ptr = boost::make_shared<MatrixDouble>();
    auto strain_ptr = boost::make_shared<MatrixDouble>();
    auto stresses_ptr = boost::make_shared<MatrixDouble>();
    auto h_ptr = boost::make_shared<VectorDouble>();
    auto flux_ptr = boost::make_shared<MatrixDouble>();

    postProcFe->getOpPtrVector().push_back(
        new OpCalculateVectorFieldGradient<SPACE_DIM, SPACE_DIM>("U",
                                                                 u_grad_ptr));
    postProcFe->getOpPtrVector().push_back(
        new OpSymmetrizeTensor<SPACE_DIM>("U", u_grad_ptr, strain_ptr));
    postProcFe->getOpPtrVector().push_back(
        new OpTensorTimesSymmetricTensor<SPACE_DIM, SPACE_DIM>(
            "U", strain_ptr, stresses_ptr, mDPtr));
                                                                 
    postProcFe->getOpPtrVector().push_back(new OpCalculateScalarFieldValues(
        "H", h_ptr));
    postProcFe->getOpPtrVector().push_back(
        new Tutorial::OpPostProcElastic<SPACE_DIM>(
            "U", postProcFe->postProcMesh, postProcFe->mapGaussPts, strain_ptr,
            stresses_ptr));

    postProcFe->addFieldValuesPostProc("U");
    postProcFe->addFieldValuesPostProc("H");
    postProcFe->addFieldValuesPostProc("FLUX");

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
    boost::shared_ptr<Monitor> monitor_ptr(new Monitor(
        dm, postProcFe, reactionFe, uXScatter, uYScatter, uZScatter));
    boost::shared_ptr<ForcesAndSourcesCore> null;
    CHKERR DMMoFEMTSSetMonitor(dm, solver, simple->getDomainFEName(),
                               monitor_ptr, null, null);
    MoFEMFunctionReturn(0);
  };

  auto dm = simple->getDM();
  auto D = smartCreateDMVector(dm);

  CHKERR create_post_process_element();
  uXScatter = scatter_create(D, 0);
  uYScatter = scatter_create(D, 1);
  if (SPACE_DIM == 3)
    uZScatter = scatter_create(D, 2);

  auto solver = pipeline_mng->createTSIM();

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
