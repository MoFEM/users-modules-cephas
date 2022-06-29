/**
 * \file photon_diffusion.cpp
 * \example photon_diffusion.cpp
 *
 **/

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

#include <stdlib.h>
#include <cmath>
#include <BasicFiniteElements.hpp>

using namespace MoFEM;

static char help[] = "...\n\n";

template <int DIM> struct ElementsAndOps {};

//! [Define dimension]
constexpr int SPACE_DIM = 3; //< Space dimension of problem, mesh
//! [Define dimension]

using DomainEle = VolumeElementForcesAndSourcesCore;
using DomainEleOp = DomainEle::UserDataOperator;
using BoundaryEle = FaceElementForcesAndSourcesCore;
using BoundaryEleOp = BoundaryEle::UserDataOperator;
using PostProcEle = PostProcVolumeOnRefinedMesh;

using VolSideFe = VolumeElementForcesAndSourcesCoreOnSide;

using OpDomainMass = FormsIntegrators<DomainEleOp>::Assembly<
    PETSC>::BiLinearForm<GAUSS>::OpMass<1, 1>;
using OpDomainGradGrad = FormsIntegrators<DomainEleOp>::Assembly<
    PETSC>::BiLinearForm<GAUSS>::OpGradGrad<1, 1, SPACE_DIM>;
using OpDomainTimesScalarField = FormsIntegrators<DomainEleOp>::Assembly<
    PETSC>::LinearForm<GAUSS>::OpBaseTimesScalarField<1>;
using OpDomainGradTimesVec = FormsIntegrators<DomainEleOp>::Assembly<
    PETSC>::LinearForm<GAUSS>::OpGradTimesTensor<1, 1, SPACE_DIM>;
using OpDomainSource = FormsIntegrators<DomainEleOp>::Assembly<
    PETSC>::LinearForm<GAUSS>::OpSource<1, 1>;

using OpBoundaryMass = FormsIntegrators<BoundaryEleOp>::Assembly<
    PETSC>::BiLinearForm<GAUSS>::OpMass<1, 1>;
using OpBoundaryTimeScalarField = FormsIntegrators<BoundaryEleOp>::Assembly<
    PETSC>::LinearForm<GAUSS>::OpBaseTimesScalarField<1>;
using OpBoundarySource = FormsIntegrators<BoundaryEleOp>::Assembly<
    PETSC>::LinearForm<GAUSS>::OpSource<1, 1>;

const double c = 30.; ///< speed of light (cm/ns)

double n; ///< refractive index of diffusive medium
double v; ///< phase velocity of light in medium (cm/ns)
double inv_v;

double A;
double h;

double beam_magnitude;
double beam_max_time;
double beam_std_dev;

PetscBool from_initial = PETSC_FALSE;
PetscBool output_volume = PETSC_FALSE;
PetscBool output_camera = PETSC_FALSE;

int order = 2;
int save_every_nth_step = 1;

char init_data_file_name[255] = "init_file.dat";
int numHoLevels = 1;

inline double sqr(double x) { return x * x; };

struct PhotonDiffusion {
public:
  PhotonDiffusion(MoFEM::Interface &m_field);

  // Declaration of the main function to run analysis
  MoFEMErrorCode runProgram();

private:
  // Declaration of other main functions called in runProgram()
  MoFEMErrorCode readMesh();
  MoFEMErrorCode createCommonData();
  MoFEMErrorCode setupProblem();
  MoFEMErrorCode setIntegrationRules();
  MoFEMErrorCode initialCondition();
  MoFEMErrorCode boundaryCondition();
  MoFEMErrorCode assembleSystem();
  MoFEMErrorCode solveSystem();
  MoFEMErrorCode outputResults();

  // Main interfaces
  MoFEM::Interface &mField;

  // Object to mark boundary entities for the assembling of domain elements
  boost::shared_ptr<std::vector<unsigned char>> boundaryMarker;

  boost::shared_ptr<FEMethod> domainLhsFEPtr;
  boost::shared_ptr<FEMethod> boundaryLhsFEPtr;
  boost::shared_ptr<FEMethod> boundaryRhsFEPtr;

  struct CommonData {

    double muA;  ///< absorption coefficient (cm^-1)
    double muSp; ///< scattering coefficient (cm^-1)
    double coeffD;

    boost::shared_ptr<VectorDouble> approxVals;
    boost::shared_ptr<MatrixDouble> approxValsGrad;
    SmartPetscObj<Vec> petscVec;

    enum VecElements {
      VALUES_INTEG = 0,
      FLUX_NORM_INTEG,
      TOTAL_AREA,
      LAST_ELEMENT
    };
  };

  struct BlockData {
    int iD;
    Range tEts; ///< constrains elements in block set
    double muA;
    double muSp;
  };

  boost::shared_ptr<CommonData> commonDataPtr;
  boost::shared_ptr<std::map<int, BlockData>> blockSetPtr;

  struct OpGetBlockData : public DomainEleOp {

    OpGetBlockData(boost::shared_ptr<CommonData> common_data_ptr,
                   boost::shared_ptr<map<int, BlockData>> block_sets_ptr)
        : DomainEleOp("PHOTON_FLUENCE_RATE", OPROW),
          commonDataPtr(common_data_ptr), blockSetsPtr(block_sets_ptr) {
      std::fill(&doEntities[MBVERTEX], &doEntities[MBMAXTYPE], false);
      doEntities[MBVERTEX] = true;
    }

    MoFEMErrorCode doWork(int side, EntityType type,
                          EntitiesFieldData::EntData &data) {
      MoFEMFunctionBegin;
      for (auto &m : (*blockSetsPtr)) {
        if (m.second.tEts.find(getFEEntityHandle()) != m.second.tEts.end()) {
          commonDataPtr->muA = m.second.muA;
          commonDataPtr->muSp = m.second.muSp;
          commonDataPtr->coeffD = 1. / (3. * (m.second.muA + m.second.muSp));
        }
      }
      MoFEMFunctionReturn(0);
    }

  protected:
    boost::shared_ptr<map<int, BlockData>> blockSetsPtr;
    boost::shared_ptr<CommonData> commonDataPtr;
  };

  struct OpCameraInteg : public BoundaryEleOp {
    boost::shared_ptr<CommonData> commonDataPtr;
    OpCameraInteg(boost::shared_ptr<CommonData> common_data_ptr)
        : BoundaryEleOp("PHOTON_FLUENCE_RATE", OPROW),
          commonDataPtr(common_data_ptr) {
      std::fill(&doEntities[MBVERTEX], &doEntities[MBMAXTYPE], false);
      doEntities[MBVERTEX] = true;
    }
    MoFEMErrorCode doWork(int side, EntityType type, EntData &data);
  };

  struct OpGetScalarFieldGradientValuesOnSkin : public BoundaryEleOp {

    boost::shared_ptr<VolSideFe> sideOpFe;

    OpGetScalarFieldGradientValuesOnSkin(boost::shared_ptr<VolSideFe> side_fe)
        : BoundaryEleOp("PHOTON_FLUENCE_RATE", OPROW), sideOpFe(side_fe) {}

    MoFEMErrorCode doWork(int side, EntityType type,
                          DataForcesAndSourcesCore::EntData &data) {
      MoFEMFunctionBegin;
      if (type != MBVERTEX)
        MoFEMFunctionReturnHot(0);
      CHKERR loopSideVolumes("dFE", *sideOpFe);
      MoFEMFunctionReturn(0);
    }
  };

  struct Monitor : public FEMethod {

    Monitor(SmartPetscObj<DM> dm, boost::shared_ptr<PostProcEle> post_proc,
            boost::shared_ptr<PostProcFaceOnRefinedMesh> skin_post_proc,
            boost::shared_ptr<BoundaryEle> skin_post_proc_integ,
            boost::shared_ptr<CommonData> common_data_ptr)
        : dM(dm), postProc(post_proc), skinPostProc(skin_post_proc),
          skinPostProcInteg(skin_post_proc_integ),
          commonDataPtr(common_data_ptr){};

    MoFEMErrorCode preProcess() {
      MoFEMFunctionBegin;
      MoFEMFunctionReturn(0);
    }
    MoFEMErrorCode operator()() {
      MoFEMFunctionBegin;
      MoFEMFunctionReturn(0);
    }

    MoFEMErrorCode postProcess() {
      MoFEMFunctionBegin;

      CHKERR VecZeroEntries(commonDataPtr->petscVec);
      CHKERR VecGhostUpdateBegin(commonDataPtr->petscVec, INSERT_VALUES,
                                 SCATTER_FORWARD);
      CHKERR VecGhostUpdateEnd(commonDataPtr->petscVec, INSERT_VALUES,
                               SCATTER_FORWARD);
      CHKERR DMoFEMLoopFiniteElements(dM, "CAMERA_FE", skinPostProcInteg);
      CHKERR VecAssemblyBegin(commonDataPtr->petscVec);
      CHKERR VecAssemblyEnd(commonDataPtr->petscVec);
      CHKERR VecGhostUpdateBegin(commonDataPtr->petscVec, ADD_VALUES,
                                 SCATTER_REVERSE);
      CHKERR VecGhostUpdateEnd(commonDataPtr->petscVec, ADD_VALUES,
                               SCATTER_REVERSE);
      CHKERR VecGhostUpdateBegin(commonDataPtr->petscVec, INSERT_VALUES,
                                 SCATTER_FORWARD);
      CHKERR VecGhostUpdateEnd(commonDataPtr->petscVec, INSERT_VALUES,
                               SCATTER_FORWARD);
      const double *array;
      CHKERR VecGetArrayRead(commonDataPtr->petscVec, &array);
      MOFEM_LOG("PHOTON", Sev::inform) << "Fluence rate integral: " << array[0];
      MOFEM_LOG("PHOTON", Sev::inform)
          << "Normal flux integral: " << array[1];

      if (ts_step == 0) {
        MOFEM_LOG("PHOTON", Sev::inform) << "Total area: " << array[2];
      }

      if (ts_step % save_every_nth_step == 0) {
        if (output_volume) {
          CHKERR DMoFEMLoopFiniteElements(dM, "dFE", postProc);
          CHKERR postProc->writeFile("out_volume_" +
                                     boost::lexical_cast<std::string>(ts_step) +
                                     ".h5m");
        }
        if (output_camera && skinPostProc) {
          CHKERR DMoFEMLoopFiniteElements(dM, "CAMERA_FE", skinPostProc);
          CHKERR skinPostProc->writeFile(
              "out_camera_" + boost::lexical_cast<std::string>(ts_step) +
              ".h5m");
        }
      }
      MoFEMFunctionReturn(0);
    }

  private:
    SmartPetscObj<DM> dM;
    boost::shared_ptr<PostProcEle> postProc;
    boost::shared_ptr<PostProcFaceOnRefinedMesh> skinPostProc;
    boost::shared_ptr<BoundaryEle> skinPostProcInteg;
    boost::shared_ptr<CommonData> commonDataPtr;
  };
};

PhotonDiffusion::PhotonDiffusion(MoFEM::Interface &m_field) : mField(m_field) {}

MoFEMErrorCode PhotonDiffusion::readMesh() {
  MoFEMFunctionBegin;

  auto *simple = mField.getInterface<Simple>();
  CHKERR mField.getInterface(simple);
  CHKERR simple->getOptions();
  CHKERR simple->loadFile();

  blockSetPtr = boost::make_shared<std::map<int, BlockData>>();

  for (_IT_CUBITMESHSETS_BY_SET_TYPE_FOR_LOOP_(mField, BLOCKSET, bit)) {
    if (bit->getName().compare(0, 11, "MAT_DIFFUSE") == 0) {
      const int id = bit->getMeshsetId();
      auto &block_data = (*blockSetPtr)[id];

      CHKERR mField.get_moab().get_entities_by_dimension(bit->getMeshset(), 3,
                                                         block_data.tEts, true);

      std::vector<double> attributes;
      bit->getAttributes(attributes);
      if (attributes.size() < 2) {
        SETERRQ1(PETSC_COMM_WORLD, MOFEM_DATA_INCONSISTENCY,
                 "should be at least 2 attributes but is %d",
                 attributes.size());
      }
      block_data.iD = id;
      block_data.muA = attributes[0];
      block_data.muSp = attributes[1];

      MOFEM_LOG("PHOTON", Sev::inform) << "------------------------------";
      MOFEM_LOG("PHOTON", Sev::inform) << "Block name: " << bit->getName();
      MOFEM_LOG("PHOTON", Sev::inform) << "Block id: " << block_data.iD;
      MOFEM_LOG("PHOTON", Sev::inform) << "mu_a value: " << block_data.muA;
      MOFEM_LOG("PHOTON", Sev::inform) << "mu_sp value: " << block_data.muSp;
    }
  }

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode PhotonDiffusion::createCommonData() {
  MoFEMFunctionBegin;
  commonDataPtr = boost::make_shared<CommonData>();
  PetscInt ghosts[3] = {0, 1, 2};
  if (!mField.get_comm_rank())
    commonDataPtr->petscVec =
        createSmartGhostVector(mField.get_comm(), 3, 3, 0, ghosts);
  else
    commonDataPtr->petscVec =
        createSmartGhostVector(mField.get_comm(), 0, 3, 3, ghosts);
  commonDataPtr->approxVals = boost::make_shared<VectorDouble>();
  commonDataPtr->approxValsGrad = boost::make_shared<MatrixDouble>();
  MoFEMFunctionReturn(0);
}

MoFEMErrorCode PhotonDiffusion::setupProblem() {
  MoFEMFunctionBegin;

  CHKERR PetscOptionsGetString(PETSC_NULL, "", "-initial_file",
                               init_data_file_name, 255, PETSC_NULL);

  CHKERR PetscOptionsGetBool(PETSC_NULL, "", "-from_initial", &from_initial,
                             PETSC_NULL);
  CHKERR PetscOptionsGetBool(PETSC_NULL, "", "-output_volume", &output_volume,
                             PETSC_NULL);
  CHKERR PetscOptionsGetBool(PETSC_NULL, "", "-output_camera", &output_camera,
                             PETSC_NULL);

  CHKERR PetscOptionsGetScalar(PETSC_NULL, "", "-beam_magnitude",
                               &beam_magnitude, PETSC_NULL);
  CHKERR PetscOptionsGetScalar(PETSC_NULL, "", "-beam_max_time", &beam_max_time,
                               PETSC_NULL);
  CHKERR PetscOptionsGetScalar(PETSC_NULL, "", "-beam_std_dev", &beam_std_dev,
                               PETSC_NULL);

  CHKERR PetscOptionsGetScalar(PETSC_NULL, "", "-refract_index", &n,
                               PETSC_NULL);
  CHKERR PetscOptionsGetScalar(PETSC_NULL, "", "-refract_coeff", &A,
                               PETSC_NULL);

  CHKERR PetscOptionsGetInt(PETSC_NULL, "", "-order", &order, PETSC_NULL);
  CHKERR PetscOptionsGetInt(PETSC_NULL, "", "-save_step", &save_every_nth_step,
                            PETSC_NULL);

  v = c / n; ///< phase velocity of light in medium (cm/ns)
  inv_v = 1. / v;
  h = 0.5 / A;

  MOFEM_LOG("PHOTON", Sev::inform) << "------------------------------";
  MOFEM_LOG("PHOTON", Sev::inform) << "Refractive index: " << n;
  MOFEM_LOG("PHOTON", Sev::inform) << "Speed of light (cm/ns): " << c;
  MOFEM_LOG("PHOTON", Sev::inform) << "Phase velocity in medium (cm/ns): " << v;
  // MOFEM_LOG("PHOTON", Sev::inform) << "Inverse velocity inv_v = 1 / v: " <<
  // inv_v;
  MOFEM_LOG("PHOTON", Sev::inform) << "Refractive coefficient A : " << A;
  // MOFEM_LOG("PHOTON", Sev::inform) << "Coefficient h = 0.5 / A: " << h;
  MOFEM_LOG("PHOTON", Sev::inform)
      << std::boolalpha << "From initial file: " << from_initial;
  MOFEM_LOG("PHOTON", Sev::inform) << "Beam magnitude: " << beam_magnitude;
  MOFEM_LOG("PHOTON", Sev::inform) << "Beam max time (ns): " << beam_max_time;
  MOFEM_LOG("PHOTON", Sev::inform) << "Beam std dev: " << beam_std_dev;
  MOFEM_LOG("PHOTON", Sev::inform) << "Approximation order: " << order;
  MOFEM_LOG("PHOTON", Sev::inform)
      << "Save every nth step: " << save_every_nth_step;
  MOFEM_LOG("PHOTON", Sev::inform)
      << std::boolalpha << "Output volume: " << output_volume;
  MOFEM_LOG("PHOTON", Sev::inform)
      << std::boolalpha << "Output camera: " << output_camera;
  MOFEM_LOG("PHOTON", Sev::inform) << "------------------------------";

  auto *simple = mField.getInterface<Simple>();
  CHKERR simple->addDomainField("PHOTON_FLUENCE_RATE", H1,
                                AINSWORTH_LEGENDRE_BASE, 1);
  CHKERR simple->addBoundaryField("PHOTON_FLUENCE_RATE", H1,
                                  AINSWORTH_LEGENDRE_BASE, 1);

  CHKERR simple->setFieldOrder("PHOTON_FLUENCE_RATE", order);

  auto set_camera_skin_fe = [&]() {
    MoFEMFunctionBegin;
    auto meshset_mng = mField.getInterface<MeshsetsManager>();

    Range camera_surface;
    const std::string block_name = "CAM";
    bool add_fe = false;

    for (_IT_CUBITMESHSETS_BY_SET_TYPE_FOR_LOOP_(mField, BLOCKSET, bit)) {
      if (bit->getName().compare(0, block_name.size(), block_name) == 0) {
        MOFEM_LOG("PHOTON", Sev::inform) << "Found CAM block";
        CHKERR mField.get_moab().get_entities_by_dimension(
            bit->getMeshset(), 2, camera_surface, true);
        add_fe = true;
      }
    }

    MOFEM_LOG("PHOTON", Sev::noisy) << "CAM block entities:\n"
                                    << camera_surface;

    if (add_fe) {
      CHKERR mField.add_finite_element("CAMERA_FE");
      CHKERR mField.modify_finite_element_add_field_data("CAMERA_FE",
                                                         "PHOTON_FLUENCE_RATE");
      CHKERR mField.add_ents_to_finite_element_by_dim(camera_surface, 2,
                                                      "CAMERA_FE");
    }
    MoFEMFunctionReturn(0);
  };

  auto my_simple_set_up = [&]() {
    MoFEMFunctionBegin;
    CHKERR simple->defineFiniteElements();
    CHKERR simple->defineProblem(PETSC_TRUE);
    CHKERR simple->buildFields();
    CHKERR simple->buildFiniteElements();

    if (mField.check_finite_element("CAMERA_FE")) {
      CHKERR mField.build_finite_elements("CAMERA_FE");
      CHKERR DMMoFEMAddElement(simple->getDM(), "CAMERA_FE");
    }

    CHKERR simple->buildProblem();
    MoFEMFunctionReturn(0);
  };

  CHKERR set_camera_skin_fe();
  CHKERR my_simple_set_up();

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode PhotonDiffusion::setIntegrationRules() {
  MoFEMFunctionBegin;

  auto integration_rule = [](int o_row, int o_col, int approx_order) {
    return 2 * approx_order;
  };

  auto *pipeline_mng = mField.getInterface<PipelineManager>();
  CHKERR pipeline_mng->setDomainRhsIntegrationRule(integration_rule);
  CHKERR pipeline_mng->setDomainLhsIntegrationRule(integration_rule);
  CHKERR pipeline_mng->setBoundaryLhsIntegrationRule(integration_rule);
  CHKERR pipeline_mng->setBoundaryRhsIntegrationRule(integration_rule);

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode PhotonDiffusion::initialCondition() {
  MoFEMFunctionBegin;

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode PhotonDiffusion::boundaryCondition() {
  MoFEMFunctionBegin;
  auto bc_mng = mField.getInterface<BcManager>();
  auto *simple = mField.getInterface<Simple>();
  CHKERR bc_mng->pushMarkDOFsOnEntities(simple->getProblemName(), "EXT",
                                        "PHOTON_FLUENCE_RATE", 0, 0, false);
  CHKERR bc_mng->pushMarkDOFsOnEntities(simple->getProblemName(), "LASER",
                                        "PHOTON_FLUENCE_RATE", 0, 0, false);

  // Get boundary edges marked in block named "BOUNDARY_CONDITION"
  Range boundary_ents;
  for (_IT_CUBITMESHSETS_BY_SET_TYPE_FOR_LOOP_(mField, BLOCKSET, it)) {
    std::string entity_name = it->getName();
    if (entity_name.compare(0, 3, "INT") == 0) {
      CHKERR it->getMeshsetIdEntitiesByDimension(mField.get_moab(), 1,
                                                 boundary_ents, true);
    }
  }
  // Add vertices to boundary entities
  Range boundary_verts;
  CHKERR mField.get_moab().get_connectivity(boundary_ents, boundary_verts,
                                            true);
  boundary_ents.merge(boundary_verts);

  // Remove DOFs as homogeneous boundary condition is used
  CHKERR mField.getInterface<ProblemsManager>()->removeDofsOnEntities(
      simple->getProblemName(), "PHOTON_FLUENCE_RATE", boundary_ents);

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode PhotonDiffusion::assembleSystem() {
  MoFEMFunctionBegin;

  auto bc_mng = mField.getInterface<BcManager>();
  auto &bc_map = bc_mng->getBcMapByBlockName();

  auto add_domain_base_ops = [&](auto &pipeline) {
    auto jac_ptr = boost::make_shared<MatrixDouble>();
    auto inv_jac_ptr = boost::make_shared<MatrixDouble>();
    auto det_ptr = boost::make_shared<VectorDouble>();
    pipeline.push_back(new OpCalculateHOJacVolume(jac_ptr));
    pipeline.push_back(new OpInvertMatrix<3>(jac_ptr, det_ptr, inv_jac_ptr));
    pipeline.push_back(new OpSetHOInvJacToScalarBases(H1, inv_jac_ptr));
    pipeline.push_back(new OpSetHOWeights(det_ptr));
  };

  auto add_domain_lhs_ops = [&](auto &pipeline) {
    pipeline.push_back(new OpGetBlockData(commonDataPtr, blockSetPtr));

    pipeline.push_back(
        new OpDomainGradGrad("PHOTON_FLUENCE_RATE", "PHOTON_FLUENCE_RATE",
                             [&](double, double, double) -> double {
                               return commonDataPtr->coeffD;
                             }));
    auto get_mass_coefficient = [&](const double, const double, const double) {
      return inv_v * domainLhsFEPtr->ts_a + commonDataPtr->muA;
    };
    pipeline.push_back(new OpDomainMass(
        "PHOTON_FLUENCE_RATE", "PHOTON_FLUENCE_RATE", get_mass_coefficient));
  };

  auto add_domain_rhs_ops = [&](auto &pipeline) {
    auto grad_u_at_gauss_pts = boost::make_shared<MatrixDouble>();
    auto u_at_gauss_pts = boost::make_shared<VectorDouble>();
    auto dot_u_at_gauss_pts = boost::make_shared<VectorDouble>();

    pipeline.push_back(new OpCalculateScalarFieldGradient<SPACE_DIM>(
        "PHOTON_FLUENCE_RATE", grad_u_at_gauss_pts));
    pipeline.push_back(new OpCalculateScalarFieldValues("PHOTON_FLUENCE_RATE",
                                                        u_at_gauss_pts));
    pipeline.push_back(new OpCalculateScalarFieldValuesDot(
        "PHOTON_FLUENCE_RATE", dot_u_at_gauss_pts));

    pipeline.push_back(new OpGetBlockData(commonDataPtr, blockSetPtr));

    pipeline.push_back(
        new OpDomainGradTimesVec("PHOTON_FLUENCE_RATE", grad_u_at_gauss_pts,
                                 [&](double, double, double) -> double {
                                   return commonDataPtr->coeffD;
                                 }));
    pipeline.push_back(new OpDomainTimesScalarField(
        "PHOTON_FLUENCE_RATE", dot_u_at_gauss_pts,
        [&](const double, const double, const double) { return inv_v; }));
    pipeline.push_back(new OpDomainTimesScalarField(
        "PHOTON_FLUENCE_RATE", u_at_gauss_pts,
        [&](const double, const double, const double) {
          return commonDataPtr->muA;
        }));
  };

  auto add_boundary_base_ops = [&](auto &pipeline) {
    pipeline.push_back(new OpSetHOWeightsOnFace());
  };

  auto add_boundary_lhs_ops = [&](auto &pipeline) {
    for (auto b : bc_map) {
      if (std::regex_match(b.first, std::regex("(.*)EXT(.*)"))) {
        pipeline.push_back(new OpBoundaryMass(
            "PHOTON_FLUENCE_RATE", "PHOTON_FLUENCE_RATE",

            [&](const double, const double, const double) { return h; },

            b.second->getBcEntsPtr()));
      }
    }
  };

  auto add_boundary_rhs_ops = [&](auto &pipeline) {
    auto u_at_gauss_pts = boost::make_shared<VectorDouble>();
    pipeline.push_back(new OpCalculateScalarFieldValues("PHOTON_FLUENCE_RATE",
                                                        u_at_gauss_pts));
    for (auto b : bc_map) {
      if (std::regex_match(b.first, std::regex("(.*)EXT(.*)"))) {
        pipeline.push_back(new OpBoundaryTimeScalarField(
            "PHOTON_FLUENCE_RATE", u_at_gauss_pts,

            [&](const double, const double, const double) { return h; },

            b.second->getBcEntsPtr()));
      }
      if (std::regex_match(b.first, std::regex("(.*)LASER(.*)"))) {
        pipeline.push_back(new OpBoundarySource(
            "PHOTON_FLUENCE_RATE",

            [&](const double, const double, const double) {
              double time = boundaryRhsFEPtr->ts_t;
              return -h * beam_magnitude *
                     exp(-sqr(time - beam_max_time) / 2. / sqr(beam_std_dev));
            },

            b.second->getBcEntsPtr()));
      }
    }
  };

  auto pipeline_mng = mField.getInterface<PipelineManager>();
  add_domain_base_ops(pipeline_mng->getOpDomainLhsPipeline());
  add_domain_base_ops(pipeline_mng->getOpDomainRhsPipeline());
  add_domain_lhs_ops(pipeline_mng->getOpDomainLhsPipeline());
  add_domain_rhs_ops(pipeline_mng->getOpDomainRhsPipeline());

  add_boundary_base_ops(pipeline_mng->getOpBoundaryLhsPipeline());
  add_boundary_base_ops(pipeline_mng->getOpBoundaryRhsPipeline());
  add_boundary_lhs_ops(pipeline_mng->getOpBoundaryLhsPipeline());
  add_boundary_rhs_ops(pipeline_mng->getOpBoundaryRhsPipeline());

  domainLhsFEPtr = pipeline_mng->getDomainLhsFE();
  boundaryLhsFEPtr = pipeline_mng->getBoundaryLhsFE();
  boundaryRhsFEPtr = pipeline_mng->getBoundaryRhsFE();

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode PhotonDiffusion::solveSystem() {
  MoFEMFunctionBegin;

  auto *simple = mField.getInterface<Simple>();
  auto *pipeline_mng = mField.getInterface<PipelineManager>();

  auto create_post_process_element = [&]() {
    auto post_proc_fe = boost::make_shared<PostProcEle>(mField);
    post_proc_fe->generateReferenceElementMesh();
    post_proc_fe->addFieldValuesPostProc("PHOTON_FLUENCE_RATE");
    post_proc_fe->addFieldValuesGradientPostProc("PHOTON_FLUENCE_RATE");
    return post_proc_fe;
  };

  auto create_post_process_camera_element = [&]() {
    if (mField.check_finite_element("CAMERA_FE")) {
      auto post_proc_skin =
          boost::make_shared<PostProcFaceOnRefinedMesh>(mField);
      post_proc_skin->generateReferenceElementMesh();
      CHKERR post_proc_skin->addFieldValuesPostProc("PHOTON_FLUENCE_RATE");
      CHKERR post_proc_skin->addFieldValuesGradientPostProcOnSkin(
          "PHOTON_FLUENCE_RATE", simple->getDomainFEName());
      return post_proc_skin;
    } else {
      return boost::shared_ptr<PostProcFaceOnRefinedMesh>();
    }
  };

  auto create_post_process_integ_camera_element = [&]() {
    if (mField.check_finite_element("CAMERA_FE")) {
      boost::shared_ptr<VolSideFe> my_vol_side_fe_ptr =
          boost::make_shared<VolSideFe>(mField);
      auto jac_ptr = boost::make_shared<MatrixDouble>();
      auto inv_jac_ptr = boost::make_shared<MatrixDouble>();
      auto det_ptr = boost::make_shared<VectorDouble>();
      my_vol_side_fe_ptr->getOpPtrVector().push_back(
          new OpCalculateHOJacVolume(jac_ptr));
      my_vol_side_fe_ptr->getOpPtrVector().push_back(
          new OpInvertMatrix<3>(jac_ptr, det_ptr, inv_jac_ptr));
      my_vol_side_fe_ptr->getOpPtrVector().push_back(
          new OpSetHOInvJacToScalarBases(H1, inv_jac_ptr));
      my_vol_side_fe_ptr->getOpPtrVector().push_back(
          new OpSetHOWeights(det_ptr));
      my_vol_side_fe_ptr->getOpPtrVector().push_back(
          new OpCalculateScalarFieldGradient<3>("PHOTON_FLUENCE_RATE",
                                                commonDataPtr->approxValsGrad));
      my_vol_side_fe_ptr->getOpPtrVector().push_back(
          new OpGetBlockData(commonDataPtr, blockSetPtr));

      auto post_proc_integ_skin = boost::make_shared<BoundaryEle>(mField);
      post_proc_integ_skin->getOpPtrVector().push_back(
          new OpSetHOWeightsOnFace());
      post_proc_integ_skin->getOpPtrVector().push_back(
          new OpCalculateScalarFieldValues("PHOTON_FLUENCE_RATE",
                                           commonDataPtr->approxVals));
      post_proc_integ_skin->getOpPtrVector().push_back(
          new OpGetScalarFieldGradientValuesOnSkin(my_vol_side_fe_ptr));
      post_proc_integ_skin->getOpPtrVector().push_back(
          new OpCameraInteg(commonDataPtr));

      return post_proc_integ_skin;
    } else {
      return boost::shared_ptr<BoundaryEle>();
    }
  };

  auto set_time_monitor = [&](auto dm, auto solver) {
    MoFEMFunctionBegin;
    boost::shared_ptr<Monitor> monitor_ptr(new Monitor(
        dm, create_post_process_element(), create_post_process_camera_element(),
        create_post_process_integ_camera_element(), commonDataPtr));
    boost::shared_ptr<ForcesAndSourcesCore> null;
    CHKERR DMMoFEMTSSetMonitor(dm, solver, simple->getDomainFEName(),
                               monitor_ptr, null, null);
    MoFEMFunctionReturn(0);
  };

  auto dm = simple->getDM();
  auto X = smartCreateDMVector(dm);

  if (from_initial) {

    MOFEM_LOG("PHOTON", Sev::inform) << "reading vector in binary from file "
                                     << init_data_file_name << " ...";
    PetscViewer viewer;
    PetscViewerBinaryOpen(PETSC_COMM_WORLD, init_data_file_name, FILE_MODE_READ,
                          &viewer);
    VecLoad(X, viewer);

    CHKERR DMoFEMMeshToLocalVector(dm, X, INSERT_VALUES, SCATTER_REVERSE);
  }

  auto solver = pipeline_mng->createTS();

  CHKERR TSSetSolution(solver, X);
  CHKERR set_time_monitor(dm, solver);
  CHKERR TSSetSolution(solver, X);
  CHKERR TSSetFromOptions(solver);
  CHKERR TSSetUp(solver);
  CHKERR TSSolve(solver, NULL);

  CHKERR VecGhostUpdateBegin(X, INSERT_VALUES, SCATTER_FORWARD);
  CHKERR VecGhostUpdateEnd(X, INSERT_VALUES, SCATTER_FORWARD);
  CHKERR DMoFEMMeshToLocalVector(dm, X, INSERT_VALUES, SCATTER_REVERSE);

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode PhotonDiffusion::outputResults() {
  MoFEMFunctionBegin;

  // Processes to set output results are integrated in solveSystem()

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode PhotonDiffusion::runProgram() {
  MoFEMFunctionBegin;

  CHKERR readMesh();
  CHKERR createCommonData();
  CHKERR setupProblem();
  CHKERR setIntegrationRules();
  CHKERR initialCondition();
  CHKERR boundaryCondition();
  CHKERR assembleSystem();
  CHKERR solveSystem();
  CHKERR outputResults();

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode PhotonDiffusion::OpCameraInteg::doWork(int side, EntityType type,
                                                      EntData &data) {
  MoFEMFunctionBegin;
  const int nb_integration_pts = getGaussPts().size2();
  const double area = getMeasure();
  auto t_w = getFTensor0IntegrationWeight();
  auto t_val = getFTensor0FromVec(*(commonDataPtr->approxVals));
  auto t_val_grad = getFTensor1FromMat<3>(*(commonDataPtr->approxValsGrad));

  double values_integ = 0;
  double flux_norm_integ = 0;
  double total_area = 0;

  FTensor::Index<'i', 3> i;
  FTensor::Index<'j', 3> j;

  auto t_normal = getFTensor1Normal();
  t_normal(i) /= sqrt(t_normal(j) * t_normal(j));

  for (int gg = 0; gg != nb_integration_pts; ++gg) {
    const double alpha = t_w * area;

    values_integ += alpha * t_val;
    flux_norm_integ -=
        alpha * commonDataPtr->coeffD * t_val_grad(i) * t_normal(i);
    total_area += alpha;

    ++t_w;
    ++t_val;
    ++t_val_grad;
  }

  constexpr std::array<int, 3> indices = {CommonData::VALUES_INTEG,
                                          CommonData::FLUX_NORM_INTEG,
                                          CommonData::TOTAL_AREA};
  std::array<double, 3> values;
  values[0] = values_integ;
  values[1] = flux_norm_integ;
  values[2] = total_area;
  CHKERR VecSetValues(commonDataPtr->petscVec, 3, indices.data(), values.data(),
                      ADD_VALUES);
  MoFEMFunctionReturn(0);
}

int main(int argc, char *argv[]) {

  // Initialisation of MoFEM/PETSc and MOAB data structures
  const char param_file[] = "param_file.petsc";
  MoFEM::Core::Initialize(&argc, &argv, param_file, help);

  // Add logging channel for example
  auto core_log = logging::core::get();
  core_log->add_sink(
      LogManager::createSink(LogManager::getStrmWorld(), "PHOTON"));
  LogManager::setLog("PHOTON");
  MOFEM_LOG_TAG("PHOTON", "photon_diffusion")

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
    PhotonDiffusion heat_problem(m_field);
    CHKERR heat_problem.runProgram();
  }
  CATCH_ERRORS;

  // Finish work: cleaning memory, getting statistics, etc.
  MoFEM::Core::Finalize();

  return 0;
}