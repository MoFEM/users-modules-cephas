/**
 * \file contact.cpp
 * \example contact.cpp
 *
 * Example of contact problem
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

using namespace MoFEM;

template <int DIM> struct ElementsAndOps {};

template <> struct ElementsAndOps<2> {
  static constexpr FieldSpace CONTACT_SPACE = HCURL;
  using DomainEle = PipelineManager::FaceEle;
  using DomainEleOp = DomainEle::UserDataOperator;
  using BoundaryEle = PipelineManager::EdgeEle;
  using BoundaryEleOp = BoundaryEle::UserDataOperator;
  using PostProcEle = PostProcFaceOnRefinedMesh;
  using OpSetPiolaTransformOnBoundary = OpSetContravariantPiolaTransformOnEdge2D;
};

template <> struct ElementsAndOps<3> {
  static constexpr FieldSpace CONTACT_SPACE = HDIV;
  using DomainEle = VolumeElementForcesAndSourcesCore;
  using DomainEleOp = DomainEle::UserDataOperator;
  using BoundaryEle = FaceElementForcesAndSourcesCore;
  using BoundaryEleOp = BoundaryEle::UserDataOperator;
  using PostProcEle = PostProcVolumeOnRefinedMesh;
  using OpSetPiolaTransformOnBoundary =
      OpHOSetContravariantPiolaTransformOnFace3D;
};

constexpr FieldSpace ElementsAndOps<2>::CONTACT_SPACE;
constexpr FieldSpace ElementsAndOps<3>::CONTACT_SPACE;

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
double gravity = 1.0;

PetscBool use_nested_matrix = PETSC_FALSE;

#include <OpPostProcElastic.hpp>
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

  MatrixDouble invJac, jAc;
  boost::shared_ptr<ContactOps::CommonData> commonDataPtr;
  boost::shared_ptr<PostProcEle> postProcFe;
  std::tuple<SmartPetscObj<Vec>, SmartPetscObj<VecScatter>> uXScatter;
  std::tuple<SmartPetscObj<Vec>, SmartPetscObj<VecScatter>> uYScatter;
  std::tuple<SmartPetscObj<Vec>, SmartPetscObj<VecScatter>> uZScatter;

  boost::shared_ptr<std::vector<unsigned char>> boundaryMarker;
  boost::shared_ptr<NestedMatrixContainer> nest_cont;

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
  // Add field
  CHKERR simple->addDomainField("U", H1, AINSWORTH_LEGENDRE_BASE, SPACE_DIM);
  CHKERR simple->addBoundaryField("U", H1, AINSWORTH_LEGENDRE_BASE, SPACE_DIM);

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
  CHKERR PetscOptionsGetScalar(PETSC_NULL, "-gravity", &gravity, PETSC_NULL);
  CHKERR PetscOptionsGetBool(PETSC_NULL, "-use_nested_matrix", &use_nested_matrix, PETSC_NULL);

  char mumps_options[] =
      "-mat_mumps_icntl_14 800 -mat_mumps_icntl_24 1 -mat_mumps_icntl_13 1 "
      "-fieldsplit_0_mat_mumps_icntl_14 800 "
      "-fieldsplit_0_mat_mumps_icntl_24 1 "
      "-fieldsplit_0_mat_mumps_icntl_13 1 "
      "-fieldsplit_1_mat_mumps_icntl_14 800 "
      "-fieldsplit_1_mat_mumps_icntl_24 1 "
      "-fieldsplit_1_mat_mumps_icntl_13 1";
  CHKERR PetscOptionsInsertString(NULL, mumps_options);

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

  jAc.resize(SPACE_DIM, SPACE_DIM, false);
  invJac.resize(SPACE_DIM, SPACE_DIM, false);

  CHKERR set_matrial_stiffness();
  MoFEMFunctionReturn(0);
}
//! [Create common data]

//! [Boundary condition]
MoFEMErrorCode Example::bC() {
  MoFEMFunctionBegin;
  auto bc_mng = mField.getInterface<BcManager>();
  auto simple = mField.getInterface<Simple>();

  CHKERR bc_mng->removeBlockDOFsOnEntities(simple->getProblemName(),
                                           "NO_CONTACT", "SIGMA", 0, 3);

  CHKERR bc_mng->removeBlockDOFsOnEntities(simple->getProblemName(), "REMOVE_X",
                                           "U", 0, 0);
  CHKERR bc_mng->removeBlockDOFsOnEntities(simple->getProblemName(), "REMOVE_Y",
                                           "U", 1, 1);
  CHKERR bc_mng->removeBlockDOFsOnEntities(simple->getProblemName(), "REMOVE_Z",
                                           "U", 2, 2);
  CHKERR bc_mng->removeBlockDOFsOnEntities(simple->getProblemName(),
                                           "REMOVE_ALL", "U", 0, 3);

  CHKERR bc_mng->pushMarkDOFsOnEntities(simple->getProblemName(), "FIX_X", "U",
                                        0, 0);
  CHKERR bc_mng->pushMarkDOFsOnEntities(simple->getProblemName(), "FIX_Y", "U",
                                        1, 1);
  CHKERR bc_mng->pushMarkDOFsOnEntities(simple->getProblemName(), "FIX_Z", "U",
                                        2, 2);
  CHKERR bc_mng->pushMarkDOFsOnEntities(simple->getProblemName(), "FIX_ALL",
                                        "U", 0, 3);
  CHKERR bc_mng->pushMarkDOFsOnEntities(simple->getProblemName(), "ROTATE",
                                        "U", 0, 3);

  auto &bc_map = bc_mng->getBcMapByBlockName();
  if (bc_map.size()) {
    boundaryMarker = boost::make_shared<std::vector<char unsigned>>();
    for (auto b : bc_map) {
      if (std::regex_match(b.first, std::regex("(.*)_FIX_(.*)")) ||
          std::regex_match(b.first, std::regex("(.*)_ROTATE_(.*)"))) {
        boundaryMarker->resize(b.second->bcMarkers.size(), 0);
        for (int i = 0; i != b.second->bcMarkers.size(); ++i) {
          (*boundaryMarker)[i] |= b.second->bcMarkers[i];
        }
      }
    }
  }

  MoFEMFunctionReturn(0);
}
//! [Boundary condition]

//! [Push operators to pipeline]
MoFEMErrorCode Example::OPs() {
  MoFEMFunctionBegin;
  PipelineManager *pipeline_mng = mField.getInterface<PipelineManager>();

  auto add_domain_base_ops = [&](auto &pipeline) {
    if (SPACE_DIM == 2) {
      pipeline.push_back(new OpCalculateJacForFace(jAc));
      pipeline.push_back(new OpCalculateInvJacForFace(invJac));
      pipeline.push_back(new OpSetInvJacH1ForFace(invJac));
      pipeline.push_back(new OpMakeHdivFromHcurl());
      pipeline.push_back(new OpSetContravariantPiolaTransformOnFace2D(jAc));
      pipeline.push_back(new OpSetInvJacHcurlFace(invJac));
    }
  };

  auto henky_common_data_ptr = boost::make_shared<HenckyOps::CommonData>();
  henky_common_data_ptr->matGradPtr = commonDataPtr->mGradPtr;
  henky_common_data_ptr->matDPtr = commonDataPtr->mDPtr;

  auto add_domain_ops_lhs = [&](auto &pipeline) {
    if (boundaryMarker)
      pipeline.push_back(new OpSetBc("U", true, boundaryMarker));

    if (is_large_strains) {
      pipeline_mng->getOpDomainLhsPipeline().push_back(
          new OpCalculateVectorFieldGradient<SPACE_DIM, SPACE_DIM>(
              "U", commonDataPtr->mGradPtr));
      pipeline_mng->getOpDomainLhsPipeline().push_back(
          new OpCalculateEigenVals<SPACE_DIM>("U", henky_common_data_ptr));
      pipeline_mng->getOpDomainLhsPipeline().push_back(
          new OpCalculateLogC<SPACE_DIM>("U", henky_common_data_ptr));
      pipeline_mng->getOpDomainLhsPipeline().push_back(
          new OpCalculateLogC_dC<SPACE_DIM>("U", henky_common_data_ptr));
      pipeline_mng->getOpDomainLhsPipeline().push_back(
          new OpCalculateHenckyStress<SPACE_DIM>("U", henky_common_data_ptr));
      pipeline_mng->getOpDomainLhsPipeline().push_back(
          new OpCalculatePiolaStress<SPACE_DIM>("U", henky_common_data_ptr));
      pipeline_mng->getOpDomainLhsPipeline().push_back(
          new OpHenckyTangent<SPACE_DIM>("U", henky_common_data_ptr));
      pipeline_mng->getOpDomainLhsPipeline().push_back(
          new OpKPiola("U", "U", henky_common_data_ptr->getMatTangent()));
    } else {
      pipeline.push_back(new OpKCauchy("U", "U", commonDataPtr->mDPtr));
    }

    if (!is_quasi_static) {
      // Get pointer to U_tt shift in domain element
      auto get_rho = [this](const double, const double, const double) {
        auto *pipeline_mng = mField.getInterface<PipelineManager>();
        auto &fe_domain_lhs = pipeline_mng->getDomainLhsFE();
        return rho * fe_domain_lhs->ts_aa;
      };
      pipeline_mng->getOpDomainLhsPipeline().push_back(
          new OpMass("U", "U", get_rho));
    }

    pipeline.push_back(new OpMixDivULhs("SIGMA", "U", 1, true));
    pipeline.push_back(new OpLambdaGraULhs("SIGMA", "U", 1, true));

    if (boundaryMarker)
      pipeline.push_back(new OpUnSetBc("U"));
  };

  auto add_domain_ops_rhs = [&](auto &pipeline) {
    auto get_body_force = [this](const double, const double, const double) {
      auto *pipeline_mng = mField.getInterface<PipelineManager>();
      FTensor::Index<'i', SPACE_DIM> i;
      FTensor::Tensor1<double, SPACE_DIM> t_source;
      auto fe_domain_rhs = pipeline_mng->getDomainRhsFE();
      const auto time = fe_domain_rhs->ts_t;

      // hardcoded gravity load
      t_source(i) = 0;
      t_source(1) = gravity * time;
      return t_source;
    };

    if (boundaryMarker)
      pipeline.push_back(new OpSetBc("U", true, boundaryMarker));

    pipeline.push_back(new OpBodyForce("U", get_body_force));
    pipeline.push_back(new OpCalculateVectorFieldGradient<SPACE_DIM, SPACE_DIM>(
        "U", commonDataPtr->mGradPtr));

    if (is_large_strains) {
      pipeline_mng->getOpDomainRhsPipeline().push_back(
          new OpCalculateEigenVals<SPACE_DIM>("U", henky_common_data_ptr));
      pipeline_mng->getOpDomainRhsPipeline().push_back(
          new OpCalculateLogC<SPACE_DIM>("U", henky_common_data_ptr));
      pipeline_mng->getOpDomainRhsPipeline().push_back(
          new OpCalculateLogC_dC<SPACE_DIM>("U", henky_common_data_ptr));
      pipeline_mng->getOpDomainRhsPipeline().push_back(
          new OpCalculateHenckyStress<SPACE_DIM>("U", henky_common_data_ptr));
      pipeline_mng->getOpDomainRhsPipeline().push_back(
          new OpCalculatePiolaStress<SPACE_DIM>("U", henky_common_data_ptr));
      pipeline_mng->getOpDomainRhsPipeline().push_back(new OpInternalForcePiola(
          "U", henky_common_data_ptr->getMatFirstPiolaStress()));
    } else {
      pipeline.push_back(new OpSymmetrizeTensor<SPACE_DIM>(
          "U", commonDataPtr->mGradPtr, commonDataPtr->mStrainPtr));
      pipeline.push_back(new OpTensorTimesSymmetricTensor<SPACE_DIM, SPACE_DIM>(
          "U", commonDataPtr->mStrainPtr, commonDataPtr->mStressPtr,
          commonDataPtr->mDPtr));
      pipeline.push_back(
          new OpInternalForceCauchy("U", commonDataPtr->mStressPtr));
    }

    pipeline.push_back(new OpCalculateVectorFieldValues<SPACE_DIM>(
        "U", commonDataPtr->contactDispPtr));

    pipeline.push_back(new OpCalculateHVecTensorField<SPACE_DIM, SPACE_DIM>(
        "SIGMA", commonDataPtr->contactStressPtr));
    pipeline.push_back(
        new OpCalculateHVecTensorDivergence<SPACE_DIM, SPACE_DIM>(
            "SIGMA", commonDataPtr->contactStressDivergencePtr));

    pipeline.push_back(
        new OpMixDivURhs("SIGMA", commonDataPtr->contactDispPtr,
                         [](double, double, double) { return 1; }));
    pipeline.push_back(
        new OpMixLambdaGradURhs("SIGMA", commonDataPtr->mGradPtr));

    pipeline.push_back(new OpMixUTimesDivLambdaRhs(
        "U", commonDataPtr->contactStressDivergencePtr));
    pipeline.push_back(
        new OpMixUTimesLambdaRhs("U", commonDataPtr->contactStressPtr));

    // only in case of dynamics
    if (!is_quasi_static) {
      auto mat_acceleration = boost::make_shared<MatrixDouble>();
      pipeline_mng->getOpDomainRhsPipeline().push_back(
          new OpCalculateVectorFieldValuesDotDot<SPACE_DIM>("U",
                                                            mat_acceleration));
      pipeline_mng->getOpDomainRhsPipeline().push_back(new OpInertiaForce(
          "U", mat_acceleration, [](double, double, double) { return rho; }));
    }

    if (boundaryMarker)
      pipeline.push_back(new OpUnSetBc("U"));
  };

  auto add_boundary_base_ops = [&](auto &pipeline) {
    pipeline.push_back(new OpSetPiolaTransformOnBoundary(CONTACT_SPACE));
    pipeline.push_back(new OpCalculateVectorFieldValues<SPACE_DIM>(
        "U", commonDataPtr->contactDispPtr));
    pipeline.push_back(new OpCalculateHVecTensorTrace<SPACE_DIM, BoundaryEleOp>(
        "SIGMA", commonDataPtr->contactTractionPtr));
  };

  auto add_boundary_ops_lhs = [&](auto &pipeline) {
    MoFEMFunctionBegin;
    auto &bc_map = mField.getInterface<BcManager>()->getBcMapByBlockName();
    for (auto bc : bc_map) {
      if (std::regex_match(bc.first, std::regex("(.*)_FIX_(.*)")) ||
          std::regex_match(bc.first, std::regex("(.*)_ROTATE_(.*)"))) {
        MOFEM_LOG("EXAMPLE", Sev::inform)
            << "Set boundary matrix for " << bc.first;
        pipeline.push_back(
            new OpSetBc("U", false, bc.second->getBcMarkersPtr()));
        pipeline.push_back(new OpBoundaryMass(
            "U", "U", [](double, double, double) { return 1.; },
            bc.second->getBcEdgesPtr()));
        pipeline.push_back(new OpUnSetBc("U"));
      }
    }

    if (boundaryMarker)
      pipeline.push_back(new OpSetBc("U", true, boundaryMarker));
    pipeline.push_back(
        new OpConstrainBoundaryLhs_dU("SIGMA", "U", commonDataPtr));
    pipeline.push_back(
        new OpConstrainBoundaryLhs_dTraction("SIGMA", "SIGMA", commonDataPtr));
    pipeline.push_back(new OpSpringLhs(
        "U", "U",

        [this](double, double, double) { return spring_stiffness; }

        ));
    if (boundaryMarker)
      pipeline.push_back(new OpUnSetBc("U"));
    MoFEMFunctionReturn(0);
  };

  auto time_scaled = [&](double, double, double) {
    auto *pipeline_mng = mField.getInterface<PipelineManager>();
    auto &fe_domain_rhs = pipeline_mng->getDomainRhsFE();
    return -fe_domain_rhs->ts_t;
  };

  auto add_boundary_ops_rhs = [&](auto &pipeline) {
    MoFEMFunctionBegin;
    for (auto &bc : mField.getInterface<BcManager>()->getBcMapByBlockName()) {
      if (std::regex_match(bc.first, std::regex("(.*)_FIX_(.*)"))) {
        MOFEM_LOG("EXAMPLE", Sev::inform)
            << "Set boundary residual for " << bc.first;
        pipeline.push_back(
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

        pipeline.push_back(new OpBoundaryVec("U", attr_vec, time_scaled,
                                             bc.second->getBcEdgesPtr()));
        pipeline.push_back(new OpBoundaryInternal(
            "U", commonDataPtr->contactDispPtr,
            [](double, double, double) { return 1.; },
            bc.second->getBcEdgesPtr()));

        pipeline.push_back(new OpUnSetBc("U"));
      }
      if (std::regex_match(bc.first, std::regex("(.*)_ROTATE_(.*)"))) {
        MOFEM_LOG("EXAMPLE", Sev::inform)
            << "Set boundary (rotation) residual for " << bc.first;
        pipeline.push_back(
            new OpSetBc("U", false, bc.second->getBcMarkersPtr()));
        auto angles = boost::make_shared<VectorDouble>(3);
        auto c_coords = boost::make_shared<VectorDouble>(3);
        if (bc.second->bcAttributes.size() != 6)
          SETERRQ1(PETSC_COMM_WORLD, MOFEM_DATA_INCONSISTENCY,
                   "Wrong size of boundary attributes vector. Set the correct "
                   "block size attributed (3) angles and (3) coordinates for "
                   "center of rotation. Size of attributes %d",
                   bc.second->bcAttributes.size());
        std::copy(&bc.second->bcAttributes[0],
                  &bc.second->bcAttributes[3],
                  angles->data().begin());
        std::copy(&bc.second->bcAttributes[3],
                  &bc.second->bcAttributes[6],
                  c_coords->data().begin());

        pipeline.push_back(new OpRotate("U", angles, c_coords, time_scaled,
                                        bc.second->getBcEdgesPtr()));
        pipeline.push_back(new OpBoundaryInternal(
            "U", commonDataPtr->contactDispPtr,
            [](double, double, double) { return 1.; },
            bc.second->getBcEdgesPtr()));

        pipeline.push_back(new OpUnSetBc("U"));
      }
    }

    if (boundaryMarker)
      pipeline.push_back(new OpSetBc("U", true, boundaryMarker));
    pipeline.push_back(new OpConstrainBoundaryRhs("SIGMA", commonDataPtr));
    pipeline.push_back(new OpSpringRhs(
        "U", commonDataPtr->contactDispPtr,
        [this](double, double, double) { return spring_stiffness; }));
    if (boundaryMarker)
      pipeline.push_back(new OpUnSetBc("U"));
    MoFEMFunctionReturn(0);
  };

  add_domain_base_ops(pipeline_mng->getOpDomainLhsPipeline());
  add_domain_base_ops(pipeline_mng->getOpDomainRhsPipeline());
  add_domain_ops_lhs(pipeline_mng->getOpDomainLhsPipeline());
  add_domain_ops_rhs(pipeline_mng->getOpDomainRhsPipeline());

  add_boundary_base_ops(pipeline_mng->getOpBoundaryLhsPipeline());
  add_boundary_base_ops(pipeline_mng->getOpBoundaryRhsPipeline());
  CHKERR add_boundary_ops_lhs(pipeline_mng->getOpBoundaryLhsPipeline());
  CHKERR add_boundary_ops_rhs(pipeline_mng->getOpBoundaryRhsPipeline());

  auto integration_rule = [](int, int, int approx_order) {
    return 2 * order + 1;
  };
  CHKERR pipeline_mng->setDomainRhsIntegrationRule(integration_rule);
  CHKERR pipeline_mng->setDomainLhsIntegrationRule(integration_rule);
  CHKERR pipeline_mng->setBoundaryRhsIntegrationRule(integration_rule);
  CHKERR pipeline_mng->setBoundaryLhsIntegrationRule(integration_rule);

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

  auto push_dirichlet_methods = [&]() {
    MoFEMFunctionBeginHot;
    boost::shared_ptr<FEMethod> null;
    auto dirichlet_bc_ptr =
        boost::make_shared<DirichletDisplacementBc>(mField, "U");
    dirichlet_bc_ptr->methodsOp.push_back(new TimeForceScale("-load_history", false));
    CHKERR DMMoFEMTSSetIJacobian(dm, simple->getDomainFEName(), null,
                                 dirichlet_bc_ptr, dirichlet_bc_ptr);
    CHKERR DMMoFEMTSSetIFunction(dm, simple->getDomainFEName(), null,
                                 dirichlet_bc_ptr, dirichlet_bc_ptr);
    MoFEMFunctionReturnHot(0);
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

    Range bc_ents;
    for (auto bc : mField.getInterface<BcManager>()->getBcMapByBlockName()) {
      if (std::regex_match(bc.first, std::regex("(.*)_FIX_(.*)")) ||
          std::regex_match(bc.first, std::regex("(.*)_ROTATE_(.*)"))) {

        bc_ents.merge(*bc.second->getBcEdgesPtr());
      }
    }
    Range nodes;
    CHKERR mField.get_moab().get_connectivity(bc_ents, nodes, true);
    bc_ents.merge(nodes);
    // cout << bc_ents << endl;
    CHKERR mField.getInterface<CommInterface>()->synchroniseEntities(bc_ents);
    // cout << bc_ents << endl;
    // Setup fieldsplit (block) solver - optional: yes/no
    if (is_pcfs == PETSC_TRUE) {
      
      if(bc_ents.empty())
        SETERRQ(PETSC_COMM_SELF, MOFEM_DATA_INCONSISTENCY,
                "Appropriate boundary conditions for fieldsplit has not been "
                "defined. (FIX_ or ROTATE_). Set e.g: -pc_fieldsplit_type lu");

      ublas::vector<IS> nested_is_row(2); // domain - boundary
      ublas::vector<IS> nested_is_col(2); // domain - boundary
      IS is_all_row, is_all_col;

      auto name_prb = simple->getProblemName();
      CHKERR mField.getInterface<ISManager>()->isCreateProblemFieldAndRank(
          name_prb, ROW, "U", 0, 3, &nested_is_row(1), &bc_ents);
      CHKERR mField.getInterface<ISManager>()->isCreateProblemFieldAndRank(
          name_prb, COL, "U", 0, 3, &nested_is_col(1), &bc_ents);
      CHKERR mField.getInterface<ISManager>()->isCreateProblemOrder(
          name_prb, ROW, 0, order, &is_all_row);
      CHKERR mField.getInterface<ISManager>()->isCreateProblemOrder(
          name_prb, COL, 0, order, &is_all_col);

      // CHKERR ISView(nested_is_row(1), PETSC_VIEWER_STDOUT_SELF);

      CHKERR ISDifference(is_all_row, nested_is_row(1), &nested_is_row(0));
      CHKERR ISDifference(is_all_col, nested_is_col(1), &nested_is_col(0));
      
      if (use_nested_matrix) {

        nest_cont = boost::make_shared<NestedMatrixContainer>();
        NestedMatrixContainer::nestContainerPtr = nest_cont;

        // row row - col col
        auto &ao = nest_cont->aO;
        CHKERR AOCreateMappingIS(nested_is_row(0), PETSC_NULL, &ao[0]);
        CHKERR AOCreateMappingIS(nested_is_row(1), PETSC_NULL, &ao[1]);
        CHKERR AOCreateMappingIS(nested_is_col(0), PETSC_NULL, &ao[2]);
        CHKERR AOCreateMappingIS(nested_is_col(1), PETSC_NULL, &ao[3]);

        auto &nested_matrices = nest_cont->nested_matrices;
        ublas::matrix<Mat> nested_matrices_raw(2, 2);
        Mat A, nestA;

        CHKERR DMCreateMatrix(dm, &A);
        CHKERR MatCreateSubMatrix(A, nested_is_row(0), nested_is_col(0),
                                  MAT_INITIAL_MATRIX,
                                  &nested_matrices_raw(0, 0));
        CHKERR MatCreateSubMatrix(A, nested_is_row(0), nested_is_col(1),
                                  MAT_INITIAL_MATRIX,
                                  &nested_matrices_raw(0, 1));
        nested_matrices_raw(1, 0) = PETSC_NULL;
        CHKERR MatCreateSubMatrix(A, nested_is_row(1), nested_is_col(1),
                                  MAT_INITIAL_MATRIX,
                                  &nested_matrices_raw(1, 1));

        CHKERR MatCreateNest(PETSC_COMM_WORLD, 2, &nested_is_row(0), 2,
                             &nested_is_col(0), &nested_matrices_raw(0, 0),
                             &nestA);

        for (int i = 0; i != 2; i++)
          for (int j = 0; j != 2; j++)
            nested_matrices(i, j) =
                SmartPetscObj<Mat>(nested_matrices_raw(i, j), true);

        CHKERR KSPSetOperators(ksp, nestA, nestA); // This does not seem to work
        TsCtx *ts_ctx;
        CHKERR DMMoFEMGetTsCtx(dm, &ts_ctx);
        CHKERR TSSetIJacobian(solver, nestA, nestA, TsSetIJacobian, ts_ctx);

        // set custom operation for nested matrix
        CHKERR MatSetOperation(nestA, MATOP_ZERO_ROWS_COLUMNS,
                               (void (*)(void))CsMatZeroRowsColumns);
        CHKERR MatSetOperation(nestA, MATOP_MISSING_DIAGONAL,
                               (void (*)(void))CsMatMissingDiagonal);
        CHKERR MatSetOperation(nestA, MATOP_SET_VALUES,
                               (void (*)(void))CsMatSetValues);

        CHKERR MatDestroy(&A);
        auto *pip_m = mField.getInterface<PipelineManager>();
        auto fe = pip_m->getDomainLhsFE();
        CHKERR MatDestroy(&fe->ts_B);
        fe->ts_B = nestA;
      }

      CHKERR PCFieldSplitSetIS(pc, PETSC_NULL,
                               nested_is_row(0)); // non-boundary block
      CHKERR PCFieldSplitSetIS(pc, PETSC_NULL,
                               nested_is_row(1)); // boundary block

      CHKERR ISDestroy(&is_all_row);
      CHKERR ISDestroy(&is_all_col);
      for (int i = 0; i < 2; i++) {
        CHKERR ISDestroy(&nested_is_row(i));
        CHKERR ISDestroy(&nested_is_col(i));
      }
    }

    MoFEMFunctionReturnHot(0);
  };

  if (is_quasi_static) {
    auto solver = pipeline_mng->createTS();
    CHKERR push_dirichlet_methods();
    auto D = smartCreateDMVector(dm);
    CHKERR set_section_monitor(solver);
    CHKERR set_time_monitor(dm, solver);
    CHKERR TSSetSolution(solver, D);
    CHKERR TSSetFromOptions(solver);
    CHKERR TSSetUp(solver);
    CHKERR set_fieldsplit_preconditioner(solver);
    CHKERR TSSolve(solver, NULL);
  } else {
    auto solver = pipeline_mng->createTS2();
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
  Range faces;
  EntityType type = MBTRI;
  if (DIM == 3)
    type = MBTET;
  CHKERR mField.get_moab().get_entities_by_type(0, type, faces);
  Skinner skin(&mField.get_moab());
  Range skin_ents;
  CHKERR skin.find_skin(0, faces, false, skin_ents);
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
    MoFEM::Interface &m_field = core; ///< finite element database insterface
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
