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

constexpr int SPACE_DIM =
    EXECUTABLE_DIMENSION; //< Space dimension of problem, mesh

using EntData = EntitiesFieldData::EntData;
using DomainEle = PipelineManager::ElementsAndOpsByDim<SPACE_DIM>::DomainEle;
using DomainEleOp = DomainEle::UserDataOperator;
using BoundaryEle =
    PipelineManager::ElementsAndOpsByDim<SPACE_DIM>::BoundaryEle;
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
    GAUSS>::OpMixDivTimesU<3, 1, SPACE_DIM>;

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
using DomainNaturalBCRhs =
    NaturalBC<DomainEleOp>::Assembly<PETSC>::LinearForm<GAUSS>;
using OpBodyForce =
    DomainNaturalBCRhs::OpFlux<NaturalMeshsetType<BLOCKSET>, 1, SPACE_DIM>;
using OpHeatSource =
    DomainNaturalBCRhs::OpFlux<NaturalMeshsetType<BLOCKSET>, 1, 1>;
using DomainNaturalBCLhs =
    NaturalBC<DomainEleOp>::Assembly<PETSC>::BiLinearForm<GAUSS>;
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

double default_young_modulus = 1;
double default_poisson_ratio = 0.25;
double ref_temp = 0.0;

double default_coeff_expansion = 1;
double default_heat_conductivity =
    1; // Force / (time temperature )  or Power /
       // (length temperature) // Time unit is hour. force unit kN
double default_heat_capacity = 1; // length^2/(time^2 temperature) // length is
                                  // millimeter time is hour

int order = 2;                    //< default approximation order

#include <ThermoElasticOps.hpp>   //< additional coupling opearyors
using namespace ThermoElasticOps; //< name space of coupling operators

using OpSetTemperatureRhs =
    DomainNaturalBCRhs::OpFlux<SetTargetTemperature, 1, 1>;
using OpSetTemperatureLhs =
    DomainNaturalBCLhs::OpFlux<SetTargetTemperature, 1, 1>;

auto save_range = [](moab::Interface &moab, const std::string name,
                     const Range r) {
  MoFEMFunctionBegin;
  auto out_meshset = get_temp_meshset_ptr(moab);
  CHKERR moab.add_entities(*out_meshset, r);
  CHKERR moab.write_file(name.c_str(), "VTK", "", out_meshset->get_ptr(), 1);
  MoFEMFunctionReturn(0);
};

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

  struct BlockedParameters
      : public boost::enable_shared_from_this<BlockedParameters> {
    MatrixDouble mD;
    double coeffExpansion;
    double heatConductivity;
    double heatCapacity;

    inline auto getDPtr() {
      return boost::shared_ptr<MatrixDouble>(shared_from_this(), &mD);
    }

    inline auto getCoeffExpansionPtr() {
      return boost::shared_ptr<double>(shared_from_this(), &coeffExpansion);
    }

    inline auto getHeatConductivityPtr() {
      return boost::shared_ptr<double>(shared_from_this(), &heatConductivity);
    }

    inline auto getHeatCapacityPtr() {
      return boost::shared_ptr<double>(shared_from_this(), &heatCapacity);
    }
  };

  MoFEMErrorCode addMatBlockOps(
      boost::ptr_deque<ForcesAndSourcesCore::UserDataOperator> &pipeline,
      std::string block_elastic_name, std::string block_thermal_name,
      boost::shared_ptr<BlockedParameters> blockedParamsPtr, Sev sev);
};

MoFEMErrorCode ThermoElasticProblem::addMatBlockOps(
    boost::ptr_deque<ForcesAndSourcesCore::UserDataOperator> &pipeline,
    std::string block_elastic_name, std::string block_thermal_name,
    boost::shared_ptr<BlockedParameters> blockedParamsPtr, Sev sev) {
  MoFEMFunctionBegin;

  struct OpMatElasticBlocks : public DomainEleOp {
    OpMatElasticBlocks(boost::shared_ptr<MatrixDouble> m, double bulk_modulus_K,
                       double shear_modulus_G, MoFEM::Interface &m_field,
                       Sev sev,
                       std::vector<const CubitMeshSets *> meshset_vec_ptr)
        : DomainEleOp(NOSPACE, DomainEleOp::OPSPACE), matDPtr(m),
          bulkModulusKDefault(bulk_modulus_K),
          shearModulusGDefault(shear_modulus_G) {
      CHK_THROW_MESSAGE(extractElasticBlockData(m_field, meshset_vec_ptr, sev),
                        "Can not get data from block");
    }

    MoFEMErrorCode doWork(int side, EntityType type,
                          EntitiesFieldData::EntData &data) {
      MoFEMFunctionBegin;

      for (auto &b : blockData) {

        if (b.blockEnts.find(getFEEntityHandle()) != b.blockEnts.end()) {
          CHKERR getMatDPtr(matDPtr, b.bulkModulusK, b.shearModulusG);
          MoFEMFunctionReturnHot(0);
        }
      }

      CHKERR getMatDPtr(matDPtr, bulkModulusKDefault, shearModulusGDefault);
      MoFEMFunctionReturn(0);
    }

  private:
    boost::shared_ptr<MatrixDouble> matDPtr;

    struct BlockData {
      double bulkModulusK;
      double shearModulusG;
      Range blockEnts;
    };

    double bulkModulusKDefault;
    double shearModulusGDefault;
    std::vector<BlockData> blockData;

    MoFEMErrorCode
    extractElasticBlockData(MoFEM::Interface &m_field,
                            std::vector<const CubitMeshSets *> meshset_vec_ptr,
                            Sev sev) {
      MoFEMFunctionBegin;

      for (auto m : meshset_vec_ptr) {
        MOFEM_TAG_AND_LOG("WORLD", sev, "Mat Elastic Block") << *m;
        std::vector<double> block_data;
        CHKERR m->getAttributes(block_data);
        if (block_data.size() < 2) {
          SETERRQ(PETSC_COMM_SELF, MOFEM_DATA_INCONSISTENCY,
                  "Expected that block has two attributes");
        }
        auto get_block_ents = [&]() {
          Range ents;
          CHKERR
          m_field.get_moab().get_entities_by_handle(m->meshset, ents, true);
          return ents;
        };

        double young_modulus = block_data[0];
        double poisson_ratio = block_data[1];
        double bulk_modulus_K = young_modulus / (3 * (1 - 2 * poisson_ratio));
        double shear_modulus_G = young_modulus / (2 * (1 + poisson_ratio));

        MOFEM_TAG_AND_LOG("WORLD", sev, "Mat Elastic Block")
            << m->getName() << ": E = " << young_modulus
            << " nu = " << poisson_ratio;

        blockData.push_back(
            {bulk_modulus_K, shear_modulus_G, get_block_ents()});
      }
      MOFEM_LOG_CHANNEL("WORLD");
      MoFEMFunctionReturn(0);
    }

    MoFEMErrorCode getMatDPtr(boost::shared_ptr<MatrixDouble> mat_D_ptr,
                              double bulk_modulus_K, double shear_modulus_G) {
      MoFEMFunctionBegin;
      //! [Calculate elasticity tensor]
      auto set_material_stiffness = [&]() {
        FTensor::Index<'i', SPACE_DIM> i;
        FTensor::Index<'j', SPACE_DIM> j;
        FTensor::Index<'k', SPACE_DIM> k;
        FTensor::Index<'l', SPACE_DIM> l;
        constexpr auto t_kd = FTensor::Kronecker_Delta_symmetric<int>();
        double A = (SPACE_DIM == 2)
                       ? 2 * shear_modulus_G /
                             (bulk_modulus_K + (4. / 3.) * shear_modulus_G)
                       : 1;
        auto t_D = getFTensor4DdgFromMat<SPACE_DIM, SPACE_DIM, 0>(*mat_D_ptr);
        t_D(i, j, k, l) =
            2 * shear_modulus_G * ((t_kd(i, k) ^ t_kd(j, l)) / 4.) +
            A * (bulk_modulus_K - (2. / 3.) * shear_modulus_G) * t_kd(i, j) *
                t_kd(k, l);
      };
      //! [Calculate elasticity tensor]
      constexpr auto size_symm = (SPACE_DIM * (SPACE_DIM + 1)) / 2;
      mat_D_ptr->resize(size_symm * size_symm, 1);
      set_material_stiffness();
      MoFEMFunctionReturn(0);
    }
  };

  double default_bulk_modulus_K =
      default_young_modulus / (3 * (1 - 2 * default_poisson_ratio));
  double default_shear_modulus_G =
      default_young_modulus / (2 * (1 + default_poisson_ratio));

  pipeline.push_back(new OpMatElasticBlocks(
      blockedParamsPtr->getDPtr(), default_bulk_modulus_K,
      default_bulk_modulus_K, mField, sev,

      // Get blockset using regular expression
      mField.getInterface<MeshsetsManager>()->getCubitMeshsetPtr(std::regex(

          (boost::format("%s(.*)") % block_elastic_name).str()

              ))

          ));

  struct OpMatThermalBlocks : public DomainEleOp {
    OpMatThermalBlocks(boost::shared_ptr<double> expansion_ptr,
                       boost::shared_ptr<double> conductivity_ptr,
                       boost::shared_ptr<double> capacity_ptr,
                       MoFEM::Interface &m_field, Sev sev,
                       std::vector<const CubitMeshSets *> meshset_vec_ptr)
        : DomainEleOp(NOSPACE, DomainEleOp::OPSPACE),
          expansionPtr(expansion_ptr), conductivityPtr(conductivity_ptr),
          capacityPtr(capacity_ptr) {
      CHK_THROW_MESSAGE(extractThermallockData(m_field, meshset_vec_ptr, sev),
                        "Can not get data from block");
    }

    MoFEMErrorCode doWork(int side, EntityType type,
                          EntitiesFieldData::EntData &data) {
      MoFEMFunctionBegin;

      for (auto &b : blockData) {

        if (b.blockEnts.find(getFEEntityHandle()) != b.blockEnts.end()) {
          *expansionPtr = b.extension;
          *conductivityPtr = b.conductivity;
          *capacityPtr = b.capacity;
          MoFEMFunctionReturnHot(0);
        }
      }

      *expansionPtr = default_coeff_expansion;
      *conductivityPtr = default_heat_conductivity;
      *capacityPtr = default_heat_capacity;

      MoFEMFunctionReturn(0);
    }

  private:
    struct BlockData {
      double extension;
      double conductivity;
      double capacity;
      Range blockEnts;
    };

    std::vector<BlockData> blockData;

    MoFEMErrorCode
    extractThermallockData(MoFEM::Interface &m_field,
                           std::vector<const CubitMeshSets *> meshset_vec_ptr,
                           Sev sev) {
      MoFEMFunctionBegin;

      for (auto m : meshset_vec_ptr) {
        MOFEM_TAG_AND_LOG("WORLD", sev, "Mat Thermal Block") << *m;
        std::vector<double> block_data;
        CHKERR m->getAttributes(block_data);
        if (block_data.size() < 3) {
          SETERRQ(PETSC_COMM_SELF, MOFEM_DATA_INCONSISTENCY,
                  "Expected that block has two attributes");
        }
        auto get_block_ents = [&]() {
          Range ents;
          CHKERR
          m_field.get_moab().get_entities_by_handle(m->meshset, ents, true);
          return ents;
        };

        MOFEM_TAG_AND_LOG("WORLD", sev, "Mat Thermal Block")
            << m->getName() << ": expansion = " << block_data[0]
            << " conductivity = " << block_data[1] << " capacity "
            << block_data[2];

        blockData.push_back(
            {block_data[0], block_data[1], block_data[2], get_block_ents()});
      }
      MOFEM_LOG_CHANNEL("WORLD");
      MoFEMFunctionReturn(0);
    }

    boost::shared_ptr<double> expansionPtr;
    boost::shared_ptr<double> conductivityPtr;
    boost::shared_ptr<double> capacityPtr;
  };

  pipeline.push_back(new OpMatThermalBlocks(
      blockedParamsPtr->getCoeffExpansionPtr(),
      blockedParamsPtr->getHeatConductivityPtr(),
      blockedParamsPtr->getHeatCapacityPtr(), mField, sev,

      // Get blockset using regular expression
      mField.getInterface<MeshsetsManager>()->getCubitMeshsetPtr(std::regex(

          (boost::format("%s(.*)") % block_thermal_name).str()

              ))

          ));

  MoFEMFunctionReturn(0);
}

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
  constexpr auto flux_space = (SPACE_DIM == 2) ? HCURL : HDIV;
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
                                 &default_young_modulus, PETSC_NULL);
    CHKERR PetscOptionsGetScalar(PETSC_NULL, "", "-poisson_ratio",
                                 &default_poisson_ratio, PETSC_NULL);
    CHKERR PetscOptionsGetScalar(PETSC_NULL, "", "-coeff_expansion",
                                 &default_coeff_expansion, PETSC_NULL);
    CHKERR PetscOptionsGetScalar(PETSC_NULL, "", "-ref_temp", &ref_temp,
                                 PETSC_NULL);

    CHKERR PetscOptionsGetScalar(PETSC_NULL, "", "-capacity",
                                 &default_heat_capacity, PETSC_NULL);
    CHKERR PetscOptionsGetScalar(PETSC_NULL, "", "-conductivity",
                                 &default_heat_conductivity, PETSC_NULL);

    MOFEM_LOG("ThermoElastic", Sev::inform)
        << "Young modulus " << default_young_modulus;
    MOFEM_LOG("ThermoElastic", Sev::inform)
        << "Poisson ratio " << default_poisson_ratio;
    MOFEM_LOG("ThermoElastic", Sev::inform)
        << "Coeff of expansion " << default_coeff_expansion;
    MOFEM_LOG("ThermoElastic", Sev::inform)
        << "Capacity " << default_heat_capacity;
    MOFEM_LOG("ThermoElastic", Sev::inform)
        << "Heat conductivity " << default_heat_conductivity;

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

  MOFEM_LOG("SYNC", Sev::noisy) << "bC";
  MOFEM_LOG_SEVERITY_SYNC(mField.get_comm(), Sev::noisy);

  auto simple = mField.getInterface<Simple>();
  auto bc_mng = mField.getInterface<BcManager>();

  CHKERR bc_mng->removeBlockDOFsOnEntities<DisplacementCubitBcData>(
      simple->getProblemName(), "U");
  CHKERR bc_mng->pushMarkDOFsOnEntities<HeatFluxCubitBcData>(
      simple->getProblemName(), "FLUX", false);

  auto get_skin = [&]() {
    Range body_ents;
    CHKERR mField.get_moab().get_entities_by_dimension(0, SPACE_DIM, body_ents);
    Skinner skin(&mField.get_moab());
    Range skin_ents;
    CHKERR skin.find_skin(0, body_ents, false, skin_ents);
    return skin_ents;
  };

  auto filter_flux_blocks = [&](auto skin) {
    auto remove_cubit_blocks = [&](auto c) {
      MoFEMFunctionBegin;
      for (auto m :

           mField.getInterface<MeshsetsManager>()->getCubitMeshsetPtr(c)

      ) {
        Range ents;
        CHKERR mField.get_moab().get_entities_by_dimension(
            m->getMeshset(), SPACE_DIM - 1, ents, true);
        skin = subtract(skin, ents);
      }
      MoFEMFunctionReturn(0);
    };

    auto remove_named_blocks = [&](auto n) {
      MoFEMFunctionBegin;
      for (auto m : mField.getInterface<MeshsetsManager>()->getCubitMeshsetPtr(
               std::regex(

                   (boost::format("%s(.*)") % n).str()

                       ))

      ) {
        Range ents;
        CHKERR mField.get_moab().get_entities_by_dimension(
            m->getMeshset(), SPACE_DIM - 1, ents, true);
        skin = subtract(skin, ents);
      }
      MoFEMFunctionReturn(0);
    };

    CHK_THROW_MESSAGE(remove_cubit_blocks(NODESET | TEMPERATURESET),
                      "remove_cubit_blocks");
    CHK_THROW_MESSAGE(remove_cubit_blocks(SIDESET | HEATFLUXSET),
                      "remove_cubit_blocks");
    CHK_THROW_MESSAGE(remove_named_blocks("TEMPERATURE"),
                      "remove_named_blocks");
    CHK_THROW_MESSAGE(remove_named_blocks("HEATFLUX"), "remove_named_blocks");

    return skin;
  };

  auto filter_true_skin = [&](auto skin) {
    Range boundary_ents;
    ParallelComm *pcomm =
        ParallelComm::get_pcomm(&mField.get_moab(), MYPCOMM_INDEX);
    CHKERR pcomm->filter_pstatus(skin, PSTATUS_SHARED | PSTATUS_MULTISHARED,
                                 PSTATUS_NOT, -1, &boundary_ents);
    return boundary_ents;
  };

  auto remove_flux_ents = filter_true_skin(filter_flux_blocks(get_skin()));

  CHKERR mField.getInterface<CommInterface>()->synchroniseEntities(
      remove_flux_ents);

  MOFEM_LOG("SYNC", Sev::noisy) << remove_flux_ents << endl;
  MOFEM_LOG_SEVERITY_SYNC(mField.get_comm(), Sev::noisy);

#ifdef NDEBUG

  CHKERR save_range(
      mField.get_moab(),
      (boost::format("flux_remove_%d.vtk") % mField.get_comm_rank()).str(),
      remove_flux_ents);

#endif

  CHKERR mField.getInterface<ProblemsManager>()->removeDofsOnEntities(
      simple->getProblemName(), "FLUX", remove_flux_ents);

  MoFEMFunctionReturn(0);
}
//! [Boundary condition]

//! [Push operators to pipeline]
MoFEMErrorCode ThermoElasticProblem::OPs() {
  MoFEMFunctionBegin;

  MOFEM_LOG("SYNC", Sev::noisy) << "OPs";
  MOFEM_LOG_SEVERITY_SYNC(mField.get_comm(), Sev::noisy);

  auto pipeline_mng = mField.getInterface<PipelineManager>();
  auto simple = mField.getInterface<Simple>();
  auto bc_mng = mField.getInterface<BcManager>();

  auto boundary_marker =
      bc_mng->getMergedBlocksMarker(vector<string>{"HEATFLUX"});

  auto integration_rule = [](int, int, int approx_order) {
    return 2 * approx_order;
  };
  CHKERR pipeline_mng->setDomainRhsIntegrationRule(integration_rule);
  CHKERR pipeline_mng->setDomainLhsIntegrationRule(integration_rule);
  CHKERR pipeline_mng->setBoundaryRhsIntegrationRule(integration_rule);
  CHKERR pipeline_mng->setBoundaryLhsIntegrationRule(integration_rule);

  auto block_params = boost::make_shared<BlockedParameters>();
  auto mDPtr = block_params->getDPtr();
  auto coeff_expansion_ptr = block_params->getCoeffExpansionPtr();
  auto heat_conductivity_ptr = block_params->getHeatConductivityPtr();
  auto heat_capacity_ptr = block_params->getHeatCapacityPtr();

  auto time_scale = boost::make_shared<TimeScale>();

  auto add_domain_rhs_ops = [&](auto &pipeline) {
    MoFEMFunctionBegin;
    CHKERR addMatBlockOps(pipeline, "MAT_ELASTIC", "MAT_THERMAL", block_params,
                          Sev::inform);
    CHKERR AddHOOps<SPACE_DIM, SPACE_DIM, SPACE_DIM>::add(pipeline, {H1, HDIV});

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
                                           mDPtr, coeff_expansion_ptr,
                                           mat_stress_ptr));

    pipeline.push_back(new OpSetBc("FLUX", true, boundary_marker));

    pipeline.push_back(new OpInternalForceCauchy(
        "U", mat_stress_ptr,
        [](double, double, double) constexpr { return 1; }));

    auto resistance = [heat_conductivity_ptr](const double, const double,
                                              const double) {
      return (1. / (*heat_conductivity_ptr));
    };
    // negative value is consequence of symmetric system
    auto capacity = [heat_capacity_ptr](const double, const double,
                                        const double) {
      return -(*heat_capacity_ptr);
    };
    auto unity = [](const double, const double, const double) constexpr {
      return -1.;
    };
    pipeline.push_back(new OpHdivFlux("FLUX", mat_flux_ptr, resistance));
    pipeline.push_back(new OpHDivTemp("FLUX", vec_temp_ptr, unity));
    pipeline.push_back(new OpBaseDivFlux("T", vec_temp_div_ptr, unity));
    pipeline.push_back(new OpBaseDotT("T", vec_temp_dot_ptr, capacity));

    pipeline.push_back(new OpUnSetBc("FLUX"));

    CHKERR DomainNaturalBCRhs::AddFluxToPipeline<OpHeatSource>::add(
        pipeline, mField, "T", {time_scale}, "HEAT_SOURCE", Sev::inform);
    CHKERR DomainNaturalBCRhs::AddFluxToPipeline<OpBodyForce>::add(
        pipeline, mField, "U", {time_scale}, "BODY_FORCE", Sev::inform);
    CHKERR DomainNaturalBCRhs::AddFluxToPipeline<OpSetTemperatureRhs>::add(
        pipeline, mField, "T", vec_temp_ptr, "SETTEMP", Sev::inform);

    MoFEMFunctionReturn(0);
  };

  auto add_domain_lhs_ops = [&](auto &pipeline) {
    MoFEMFunctionBegin;
    CHKERR addMatBlockOps(pipeline, "MAT_ELASTIC", "MAT_THERMAL", block_params,
                          Sev::verbose);
    CHKERR AddHOOps<SPACE_DIM, SPACE_DIM, SPACE_DIM>::add(pipeline, {H1, HDIV});

    pipeline.push_back(new OpSetBc("FLUX", true, boundary_marker));

    pipeline.push_back(new OpKCauchy("U", "U", mDPtr));
    pipeline.push_back(new ThermoElasticOps::OpKCauchyThermoElasticity(
        "U", "T", mDPtr, coeff_expansion_ptr));

    auto resistance = [heat_conductivity_ptr](const double, const double,
                                              const double) {
      return (1. / (*heat_conductivity_ptr));
    };
    auto capacity = [heat_capacity_ptr](const double, const double,
                                        const double) {
      return -(*heat_capacity_ptr);
    };
    pipeline.push_back(new OpHdivHdiv("FLUX", "FLUX", resistance));
    pipeline.push_back(new OpHdivT(
        "FLUX", "T", []() constexpr { return -1; }, true));

    auto op_capacity = new OpCapacity("T", "T", capacity);
    op_capacity->feScalingFun = [](const FEMethod *fe_ptr) {
      return fe_ptr->ts_a;
    };
    pipeline.push_back(op_capacity);

    pipeline.push_back(new OpUnSetBc("FLUX"));

    auto vec_temp_ptr = boost::make_shared<VectorDouble>();
    pipeline.push_back(new OpCalculateScalarFieldValues("T", vec_temp_ptr));
    CHKERR DomainNaturalBCLhs::AddFluxToPipeline<OpSetTemperatureLhs>::add(
        pipeline, mField, "T", vec_temp_ptr, "SETTEMP", Sev::verbose);

    MoFEMFunctionReturn(0);
  };

  auto add_boundary_rhs_ops = [&](auto &pipeline) {
    MoFEMFunctionBegin;

    CHKERR AddHOOps<SPACE_DIM - 1, SPACE_DIM, SPACE_DIM>::add(pipeline, {HDIV});

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

    CHKERR AddHOOps<SPACE_DIM - 1, SPACE_DIM, SPACE_DIM>::add(pipeline, {HDIV});

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

  CHKERR add_domain_rhs_ops(pipeline_mng->getOpDomainRhsPipeline());
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
  auto snes_ctx_ptr = getDMSnesCtx(dm);

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

    auto block_params = boost::make_shared<BlockedParameters>();
    auto mDPtr = block_params->getDPtr();
    auto coeff_expansion_ptr = block_params->getCoeffExpansionPtr();

    CHKERR addMatBlockOps(post_proc_fe->getOpPtrVector(), "MAT_ELASTIC",
                          "MAT_THERMAL", block_params, Sev::verbose);

    CHKERR AddHOOps<SPACE_DIM, SPACE_DIM, SPACE_DIM>::add(
        post_proc_fe->getOpPtrVector(), {H1, HDIV});

    auto mat_grad_ptr = boost::make_shared<MatrixDouble>();
    auto mat_strain_ptr = boost::make_shared<MatrixDouble>();
    auto mat_stress_ptr = boost::make_shared<MatrixDouble>();

    auto vec_temp_ptr = boost::make_shared<VectorDouble>();
    auto mat_flux_ptr = boost::make_shared<MatrixDouble>();

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
    post_proc_fe->getOpPtrVector().push_back(
        new OpStressThermal("U", mat_strain_ptr, vec_temp_ptr, mDPtr,
                            coeff_expansion_ptr, mat_stress_ptr));

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
      CHKERR DMoFEMLoopFiniteElements(dm, simple->getDomainFEName(),
                                      post_proc_fe,
                                      monitor_ptr->getCacheWeakPtr());
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
      auto is_mng = mField.getInterface<ISManager>();
      auto name_prb = simple->getProblemName();

      SmartPetscObj<IS> is_u;
      CHKERR is_mng->isCreateProblemFieldAndRank(name_prb, ROW, "U", 0,
                                                 SPACE_DIM, is_u);
      SmartPetscObj<IS> is_flux;
      CHKERR is_mng->isCreateProblemFieldAndRank(name_prb, ROW, "FLUX", 0, 0,
                                                 is_flux);
      SmartPetscObj<IS> is_T;
      CHKERR is_mng->isCreateProblemFieldAndRank(name_prb, ROW, "T", 0, 0,
                                                 is_T);
      IS is_tmp;
      CHKERR ISExpand(is_T, is_flux, &is_tmp);
      auto is_TFlux = SmartPetscObj<IS>(is_tmp);

      auto is_all_bc = bc_mng->getBlockIS(name_prb, "HEATFLUX", "FLUX", 0, 0);
      int is_all_bc_size;
      CHKERR ISGetSize(is_all_bc, &is_all_bc_size);
      MOFEM_LOG("ThermoElastic", Sev::inform)
          << "Field split block size " << is_all_bc_size;
      if (is_all_bc_size) {
        IS is_tmp2;
        CHKERR ISDifference(is_TFlux, is_all_bc, &is_tmp2);
        is_TFlux = SmartPetscObj<IS>(is_tmp2);
        CHKERR PCFieldSplitSetIS(pc, PETSC_NULL,
                                 is_all_bc); // boundary block
      }

      CHKERR ISSort(is_u);
      CHKERR ISSort(is_TFlux);
      CHKERR PCFieldSplitSetIS(pc, PETSC_NULL, is_TFlux);
      CHKERR PCFieldSplitSetIS(pc, PETSC_NULL, is_u);
    }

    MoFEMFunctionReturnHot(0);
  };

  auto D = createDMVector(dm);
  CHKERR TSSetSolution(solver, D);
  CHKERR TSSetFromOptions(solver);

  CHKERR set_section_monitor(solver);
  CHKERR set_fieldsplit_preconditioner(solver);
  CHKERR set_time_monitor(dm, solver);

  CHKERR TSSetUp(solver);
  CHKERR TSSolve(solver, NULL);

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
