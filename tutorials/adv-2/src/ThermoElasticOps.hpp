

/** \file ThermoElasticOps.hpp
 * \example ThermoElasticOps.hpp
 */

namespace ThermoElasticOps {

struct OpKCauchyThermoElasticity : public AssemblyDomainEleOp {
  OpKCauchyThermoElasticity(
    const std::string row_field_name, const std::string col_field_name,
    boost::shared_ptr<MatrixDouble> mDptr,
    boost::shared_ptr<double> coeff_expansion_ptr);

  MoFEMErrorCode iNtegrate(EntitiesFieldData::EntData &row_data,
                           EntitiesFieldData::EntData &col_data);

private:
  boost::shared_ptr<MatrixDouble> mDPtr;
  boost::shared_ptr<double> coeffExpansionPtr;
};

struct OpStressThermal : public DomainEleOp {
  OpStressThermal(const std::string field_name,
                  boost::shared_ptr<MatrixDouble> strain_ptr,
                  boost::shared_ptr<VectorDouble> temp_ptr,
                  boost::shared_ptr<MatrixDouble> m_D_ptr,
                  boost::shared_ptr<double> coeff_expansion_ptr,
                  boost::shared_ptr<MatrixDouble> stress_ptr);
  MoFEMErrorCode doWork(int side, EntityType type, EntData &data);

private:
  boost::shared_ptr<MatrixDouble> strainPtr;
  boost::shared_ptr<VectorDouble> tempPtr;
  boost::shared_ptr<MatrixDouble> mDPtr;
  boost::shared_ptr<double> coeffExpansionPtr;
  boost::shared_ptr<MatrixDouble> stressPtr;
};

OpKCauchyThermoElasticity::OpKCauchyThermoElasticity(
    const std::string row_field_name, const std::string col_field_name,
    boost::shared_ptr<MatrixDouble> mDptr,
    boost::shared_ptr<double> coeff_expansion_ptr)
    : AssemblyDomainEleOp(row_field_name, col_field_name,
                          DomainEleOp::OPROWCOL),
      mDPtr(mDptr), coeffExpansionPtr(coeff_expansion_ptr) {
  sYmm = false;
}

MoFEMErrorCode
OpKCauchyThermoElasticity::iNtegrate(EntitiesFieldData::EntData &row_data,
                                     EntitiesFieldData::EntData &col_data) {
  MoFEMFunctionBegin;

  auto &locMat = AssemblyDomainEleOp::locMat;

  const auto nb_integration_pts = row_data.getN().size1();
  const auto nb_row_base_functions = row_data.getN().size2();
  auto t_w = getFTensor0IntegrationWeight();

  constexpr auto t_kd = FTensor::Kronecker_Delta<int>();
  auto t_row_diff_base = row_data.getFTensor1DiffN<SPACE_DIM>();
  auto t_D = getFTensor4DdgFromMat<SPACE_DIM, SPACE_DIM, 0>(*mDPtr);

  FTensor::Index<'i', SPACE_DIM> i;
  FTensor::Index<'j', SPACE_DIM> j;
  FTensor::Index<'k', SPACE_DIM> k;
  FTensor::Index<'l', SPACE_DIM> l;
  FTensor::Tensor2_symmetric<double, SPACE_DIM> t_eigen_strain;
  t_eigen_strain(i, j) = (t_D(i, j, k, l) * t_kd(k, l)) * (*coeffExpansionPtr);

  for (auto gg = 0; gg != nb_integration_pts; ++gg) {

    double alpha = getMeasure() * t_w;
    auto rr = 0;
    for (; rr != AssemblyDomainEleOp::nbRows / SPACE_DIM; ++rr) {
      auto t_mat = getFTensor1FromMat<SPACE_DIM, 1>(locMat, rr * SPACE_DIM);
      auto t_col_base = col_data.getFTensor0N(gg, 0);
      for (auto cc = 0; cc != AssemblyDomainEleOp::nbCols; cc++) {

        t_mat(i) -=
            (t_row_diff_base(j) * t_eigen_strain(i, j)) * (t_col_base * alpha);

        ++t_mat;
        ++t_col_base;
      }

      ++t_row_diff_base;
    }
    for (; rr != nb_row_base_functions; ++rr)
      ++t_row_diff_base;

    ++t_w;
  }

  MoFEMFunctionReturn(0);
}

OpStressThermal::OpStressThermal(const std::string field_name,
                                 boost::shared_ptr<MatrixDouble> strain_ptr,
                                 boost::shared_ptr<VectorDouble> temp_ptr,
                                 boost::shared_ptr<MatrixDouble> m_D_ptr,
                                 boost::shared_ptr<double> coeff_expansion_ptr,
                                 boost::shared_ptr<MatrixDouble> stress_ptr)
    : DomainEleOp(field_name, DomainEleOp::OPROW), strainPtr(strain_ptr),
      tempPtr(temp_ptr), mDPtr(m_D_ptr), coeffExpansionPtr(coeff_expansion_ptr),
      stressPtr(stress_ptr) {
  // Operator is only executed for vertices
  std::fill(&doEntities[MBEDGE], &doEntities[MBMAXTYPE], false);
}

//! [Calculate stress]
MoFEMErrorCode OpStressThermal::doWork(int side, EntityType type,
                                              EntData &data) {
  MoFEMFunctionBegin;
  const auto nb_gauss_pts = getGaussPts().size2();
  stressPtr->resize((SPACE_DIM * (SPACE_DIM + 1)) / 2, nb_gauss_pts);

  constexpr auto t_kd = FTensor::Kronecker_Delta<int>();
  auto t_D = getFTensor4DdgFromMat<SPACE_DIM, SPACE_DIM, 0>(*mDPtr);
  auto t_strain = getFTensor2SymmetricFromMat<SPACE_DIM>(*strainPtr);
  auto t_stress = getFTensor2SymmetricFromMat<SPACE_DIM>(*stressPtr);
  auto t_temp = getFTensor0FromVec(*tempPtr);
  FTensor::Index<'i', SPACE_DIM> i;
  FTensor::Index<'j', SPACE_DIM> j;
  FTensor::Index<'k', SPACE_DIM> k;
  FTensor::Index<'l', SPACE_DIM> l;
  for (size_t gg = 0; gg != nb_gauss_pts; ++gg) {
    t_stress(i, j) =
        t_D(i, j, k, l) * (t_strain(k, l) - t_kd(k, l) * (t_temp - ref_temp) *
                                                (*coeffExpansionPtr));
    ++t_strain;
    ++t_stress;
    ++t_temp;
  }
  MoFEMFunctionReturn(0);
}
//! [Calculate stress]

struct SetTargetTemperature;

} // namespace ThermoElasticOps

//! [Target temperature]

template <AssemblyType A, IntegrationType I, typename OpBase>
struct MoFEM::OpFluxRhsImpl<ThermoElasticOps::SetTargetTemperature, 1, 1, A, I,
                            OpBase>;

template <AssemblyType A, IntegrationType I, typename OpBase>
struct MoFEM::OpFluxLhsImpl<ThermoElasticOps::SetTargetTemperature, 1, 1, A, I,
                            OpBase>;

template <AssemblyType A, IntegrationType I, typename OpBase>
struct MoFEM::AddFluxToRhsPipelineImpl<

    MoFEM::OpFluxRhsImpl<ThermoElasticOps::SetTargetTemperature, 1, 1, A, I,
                         OpBase>,
    A, I, OpBase

    > {

  AddFluxToRhsPipelineImpl() = delete;

  static MoFEMErrorCode add(

      boost::ptr_deque<ForcesAndSourcesCore::UserDataOperator> &pipeline,
      MoFEM::Interface &m_field, const std::string field_name,
      boost::shared_ptr<VectorDouble> temp_ptr, std::string block_name, Sev sev

  ) {
    MoFEMFunctionBegin;

    using OP_SOURCE = typename FormsIntegrators<OpBase>::template Assembly<
        A>::template LinearForm<I>::template OpSource<1, 1>;
    using OP_TEMP = typename FormsIntegrators<OpBase>::template Assembly<
        A>::template LinearForm<I>::template OpBaseTimesScalar<1>;

    auto add_op = [&](auto &&meshset_vec_ptr) {
      MoFEMFunctionBegin;
      for (auto m : meshset_vec_ptr) {
        std::vector<double> block_data;
        m->getAttributes(block_data);
        if (block_data.size() != 2) {
          SETERRQ(PETSC_COMM_WORLD, MOFEM_DATA_INCONSISTENCY,
                  "Expected two parameters");
        }
        double target_temperature = block_data[0];
        double beta =
            block_data[1]; // Set temperature parameter [ W/K * (1/m^3)]
        auto ents_ptr = boost::make_shared<Range>();
        CHKERR m_field.get_moab().get_entities_by_handle(m->meshset,
                                                         *(ents_ptr), true);

        MOFEM_TAG_AND_LOG("WORLD", sev, "SetTargetTemperature")
            << "Add " << *m << " target temperature " << target_temperature
            << " penalty " << beta;

        pipeline.push_back(new OP_SOURCE(
            field_name,
            [target_temperature, beta](double, double, double) {
              return target_temperature * beta;
            },
            ents_ptr));
        pipeline.push_back(new OP_TEMP(
            field_name, temp_ptr,
            [beta](double, double, double) { return -beta; }, ents_ptr));
      }
      MoFEMFunctionReturn(0);
    };

    CHKERR add_op(

        m_field.getInterface<MeshsetsManager>()->getCubitMeshsetPtr(std::regex(

            (boost::format("%s(.*)") % block_name).str()

                ))

    );

    MOFEM_LOG_CHANNEL("WORLD");

    MoFEMFunctionReturn(0);
  }
};

template <AssemblyType A, IntegrationType I, typename OpBase>
struct AddFluxToLhsPipelineImpl<

    OpFluxLhsImpl<ThermoElasticOps::SetTargetTemperature, 1, 1, A, I, OpBase>,
    A, I, OpBase

    > {

  AddFluxToLhsPipelineImpl() = delete;

  static MoFEMErrorCode add(

      boost::ptr_deque<ForcesAndSourcesCore::UserDataOperator> &pipeline,
      MoFEM::Interface &m_field, const std::string field_name,
      boost::shared_ptr<VectorDouble> temp_ptr, std::string block_name, Sev sev

  ) {
    MoFEMFunctionBegin;

    using OP_MASS = typename FormsIntegrators<OpBase>::template Assembly<
        A>::template BiLinearForm<I>::template OpMass<1, 1>;

    auto add_op = [&](auto &&meshset_vec_ptr) {
      MoFEMFunctionBegin;
      for (auto m : meshset_vec_ptr) {
        std::vector<double> block_data;
        m->getAttributes(block_data);
        if (block_data.size() != 2) {
          SETERRQ(PETSC_COMM_WORLD, MOFEM_DATA_INCONSISTENCY,
                  "Expected two parameters");
        }
        double beta =
            block_data[1]; // Set temperature parameter [ W/K * (1/m^3)]
        auto ents_ptr = boost::make_shared<Range>();
        CHKERR m_field.get_moab().get_entities_by_handle(m->meshset,
                                                         *(ents_ptr), true);

        MOFEM_TAG_AND_LOG("WORLD", sev, "SetTargetTemperature")
            << "Add " << *m << " penalty " << beta;

        pipeline.push_back(new OP_MASS(
            field_name, field_name,
            [beta](double, double, double) { return -beta; }, ents_ptr));
      }
      MoFEMFunctionReturn(0);
    };

    CHKERR add_op(

        m_field.getInterface<MeshsetsManager>()->getCubitMeshsetPtr(std::regex(

            (boost::format("%s(.*)") % block_name).str()

                ))

    );

    MOFEM_LOG_CHANNEL("WORLD");

    MoFEMFunctionReturn(0);
  }
};

//! [Target temperature]

