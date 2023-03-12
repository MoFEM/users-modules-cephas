/**
 * @file ElasticSpring.hpp
 * @author your name (you@domain.com)
 * @brief Implementation of elastic spring bc
 * @version 0.13.2
 * @date 2022-09-18
 *
 * @copyright Copyright (c) 2022
 *
 */

namespace ElasticExample {

template <CubitBC BC> struct SpringBcType {};

template <CubitBC> struct GetSpringStiffness {
  GetSpringStiffness() = delete;

  static MoFEMErrorCode getStiffness(double &normal_stiffness,
                                     double &tangent_stiffness,
                                     boost::shared_ptr<Range> &ents,
                                     MoFEM::Interface &m_field, int ms_id) {
    MoFEMFunctionBegin;

    auto cubit_meshset_ptr =
        m_field.getInterface<MeshsetsManager>()->getCubitMeshsetPtr(ms_id,
                                                                    BLOCKSET);
    std::vector<double> block_data;
    CHKERR cubit_meshset_ptr->getAttributes(block_data);
    if (block_data.size() != 2) {
      SETERRQ(PETSC_COMM_SELF, MOFEM_DATA_INCONSISTENCY,
              "Expected that block has two attribute");
    }
    normal_stiffness = block_data[0];
    tangent_stiffness = block_data[1];

    ents = boost::make_shared<Range>();
    CHKERR
    m_field.get_moab().get_entities_by_handle(cubit_meshset_ptr->meshset,
                                              *(ents), true);

    MoFEMFunctionReturn(0);
  }
};

} // namespace ElasticExample

template <int FIELD_DIM, AssemblyType A, typename EleOp>
struct OpFluxRhsImpl<ElasticExample::SpringBcType<BLOCKSET>, 1, FIELD_DIM, A,
                     GAUSS, EleOp> : OpBaseImpl<A, EleOp> {

  using OpBase = OpBaseImpl<A, EleOp>;

  OpFluxRhsImpl(MoFEM::Interface &m_field, int ms_id, std::string field_name,
                boost::shared_ptr<MatrixDouble> u_ptr, double scale);

protected:
  double normalStiffness;
  double tangentStiffness;
  double rhsScale;
  boost::shared_ptr<MatrixDouble> uPtr;
  MoFEMErrorCode iNtegrate(EntitiesFieldData::EntData &data);
};

template <int FIELD_DIM, AssemblyType A, typename EleOp>
struct OpFluxLhsImpl<ElasticExample::SpringBcType<BLOCKSET>, 1, FIELD_DIM, A,
                     GAUSS, EleOp> : OpBaseImpl<A, EleOp> {

  using OpBase = OpBaseImpl<A, EleOp>;

  OpFluxLhsImpl(MoFEM::Interface &m_field, int ms_id,
                std::string row_field_name, std::string col_field_name);

protected:
  double normalStiffness;
  double tangentStiffness;
  MoFEMErrorCode iNtegrate(EntitiesFieldData::EntData &row_data,
                           EntitiesFieldData::EntData &col_data);
};

template <int FIELD_DIM, AssemblyType A, typename EleOp>
OpFluxRhsImpl<ElasticExample::SpringBcType<BLOCKSET>, 1, FIELD_DIM, A, GAUSS,
              EleOp>::OpFluxRhsImpl(MoFEM::Interface &m_field, int ms_id,
                                    std::string field_name,
                                    boost::shared_ptr<MatrixDouble> u_ptr,
                                    double scale)
    : OpBase(field_name, field_name, OpBase::OPROW), uPtr(u_ptr), rhsScale(scale) {
  CHK_THROW_MESSAGE(ElasticExample::GetSpringStiffness<BLOCKSET>::getStiffness(
                        this->normalStiffness, this->tangentStiffness,
                        this->entsPtr, m_field, ms_id),
                    "Can not read spring stiffness data from blockset");
}

template <int FIELD_DIM, AssemblyType A, typename EleOp>
MoFEMErrorCode
MoFEM::OpFluxRhsImpl<ElasticExample::SpringBcType<BLOCKSET>, 1, FIELD_DIM, A,
                     GAUSS, EleOp>::iNtegrate(EntitiesFieldData::EntData
                                                  &row_data) {
  FTensor::Index<'i', FIELD_DIM> i;
  FTensor::Index<'j', FIELD_DIM> j;
  MoFEMFunctionBegin;
  // get element volume
  const double vol = OpBase::getMeasure();
  // get integration weights
  auto t_w = OpBase::getFTensor0IntegrationWeight();
  // get base function gradient on rows
  auto t_row_base = row_data.getFTensor0N();

  // get coordinate at integration points
  auto t_normal = OpBase::getFTensor1Normal();
  auto l2_norm = t_normal(i) * t_normal(i);
  // normal projection matrix
  FTensor::Tensor2<double, FIELD_DIM, FIELD_DIM> t_P;
  t_P(i, j) = t_normal(i) * t_normal(j) / l2_norm;
  constexpr auto t_kd = FTensor::Kronecker_Delta<double>();
  // tangential projection matrix
  FTensor::Tensor2<double, FIELD_DIM, FIELD_DIM> t_Q;
  t_Q(i, j) = t_kd(i, j) - t_P(i, j);
  // spring stiffness
  FTensor::Tensor2<double, FIELD_DIM, FIELD_DIM> t_D;
  t_D(i, j) = normalStiffness * t_P(i, j) + tangentStiffness * t_Q(i, j);

  // get displacements
  auto t_u = getFTensor1FromMat<FIELD_DIM>(*uPtr);
  // loop over integration points
  for (int gg = 0; gg != OpBase::nbIntegrationPts; gg++) {
    // take into account Jacobian
    const double alpha = t_w * vol * rhsScale;

    // calculate spring resistance
    FTensor::Tensor1<double, FIELD_DIM> t_reaction;
    t_reaction(i) = t_D(i, j) * t_u(j);

    // loop over rows base functions
    auto t_nf = getFTensor1FromArray<FIELD_DIM, FIELD_DIM>(OpBase::locF);
    int rr = 0;
    for (; rr != OpBase::nbRows / FIELD_DIM; ++rr) {
      t_nf(i) += (alpha * t_row_base) * t_reaction(i);
      ++t_row_base;
      ++t_nf;
    }

    for (; rr < OpBase::nbRowBaseFunctions; ++rr)
      ++t_row_base;
    ++t_w;
    ++t_u;
  }
  MoFEMFunctionReturn(0);
}

template <int FIELD_DIM, AssemblyType A, typename EleOp>
OpFluxLhsImpl<ElasticExample::SpringBcType<BLOCKSET>, 1, FIELD_DIM, A, GAUSS,
              EleOp>::OpFluxLhsImpl(MoFEM::Interface &m_field, int ms_id,
                                    std::string row_field_name,
                                    std::string col_field_name)
    : OpBase(row_field_name, col_field_name, OpBase::OPROWCOL) {
  CHK_THROW_MESSAGE(ElasticExample::GetSpringStiffness<BLOCKSET>::getStiffness(
                        this->normalStiffness, this->tangentStiffness,
                        this->entsPtr, m_field, ms_id),
                    "Can not read spring stiffness data from blockset");
}

template <int FIELD_DIM, AssemblyType A, typename EleOp>
MoFEMErrorCode
MoFEM::OpFluxLhsImpl<ElasticExample::SpringBcType<BLOCKSET>, 1, FIELD_DIM, A,
                     GAUSS,
                     EleOp>::iNtegrate(EntitiesFieldData::EntData &row_data,
                                       EntitiesFieldData::EntData &col_data) {

  FTensor::Index<'i', FIELD_DIM> i;
  FTensor::Index<'j', FIELD_DIM> j;

  MoFEMFunctionBegin;
  // get element volume
  const double vol = OpBase::getMeasure();
  // get integration weights
  auto t_w = OpBase::getFTensor0IntegrationWeight();
  // get base function gradient on rows
  auto t_row_base = row_data.getFTensor0N();

  // get coordinate at integration points
  auto t_normal = OpBase::getFTensor1Normal();
  auto l2_norm = t_normal(i) * t_normal(i);
  // normal projection matrix
  FTensor::Tensor2<double, FIELD_DIM, FIELD_DIM> t_P;
  t_P(i, j) = t_normal(i) * t_normal(j) / l2_norm;
  constexpr auto t_kd = FTensor::Kronecker_Delta<double>();
  // tangential projection matrix
  FTensor::Tensor2<double, FIELD_DIM, FIELD_DIM> t_Q;
  t_Q(i, j) = t_kd(i, j) - t_P(i, j);
  // spring stiffness
  FTensor::Tensor2<double, FIELD_DIM, FIELD_DIM> t_D;
  t_D(i, j) = normalStiffness * t_P(i, j) + tangentStiffness * t_Q(i, j);

  // loop over integration points
  for (int gg = 0; gg != OpBase::nbIntegrationPts; gg++) {
    // take into account Jacobian
    const double alpha = t_w * vol;
    // loop over rows base functions
    int rr = 0;
    for (; rr != OpBase::nbRows / FIELD_DIM; rr++) {
      // get column base functions gradient at gauss point gg
      auto t_col_base = col_data.getFTensor0N(gg, 0);
      // get sub matrix for the row
      auto t_m = OpBase::template getLocMat<FIELD_DIM>(FIELD_DIM * rr);
      // loop over columns
      for (int cc = 0; cc != OpBase::nbCols / FIELD_DIM; cc++) {
        // calculate element of local matrix
        t_m(i, j) += t_D(i, j) * (alpha * t_row_base * t_col_base);
        ++t_col_base;
        ++t_m;
      }
      ++t_row_base;
    }
    for (; rr < OpBase::nbRowBaseFunctions; ++rr)
      ++t_row_base;
    ++t_w; // move to another integration weight
  }
  MoFEMFunctionReturn(0);
}

template <CubitBC BCTYPE, int BASE_DIM, int FIELD_DIM, AssemblyType A,
          IntegrationType I, typename OpBase>
struct AddFluxToRhsPipelineImpl<

    OpFluxRhsImpl<ElasticExample::SpringBcType<BCTYPE>, BASE_DIM, FIELD_DIM, A,
                  I, OpBase>,
    A, I, OpBase

    > {

  AddFluxToRhsPipelineImpl() = delete;

  static MoFEMErrorCode add(

      boost::ptr_deque<ForcesAndSourcesCore::UserDataOperator> &pipeline,
      MoFEM::Interface &m_field, std::string field_name,
      boost::shared_ptr<MatrixDouble> u_ptr, double scale,
      std::string block_name, Sev sev

  ) {
    MoFEMFunctionBegin;

    using OP = OpFluxRhsImpl<ElasticExample::SpringBcType<BLOCKSET>, BASE_DIM,
                             FIELD_DIM, PETSC, GAUSS, OpBase>;

    auto add_op = [&](auto &&meshset_vec_ptr) {
      for (auto m : meshset_vec_ptr) {
        MOFEM_TAG_AND_LOG("WORLD", sev, "OpSpringRhs") << "Add " << *m;
        pipeline.push_back(
            new OP(m_field, m->getMeshsetId(), field_name, u_ptr, scale));
      }
      MOFEM_LOG_CHANNEL("WORLD");
    };

    switch (BCTYPE) {
    case BLOCKSET:
      add_op(

          m_field.getInterface<MeshsetsManager>()->getCubitMeshsetPtr(
              std::regex(

                  (boost::format("%s(.*)") % block_name).str()

                      ))

      );

      break;
    default:
      SETERRQ(PETSC_COMM_SELF, MOFEM_NOT_IMPLEMENTED,
              "Handling of bc type not implemented");
      break;
    }
    MoFEMFunctionReturn(0);
  }
};

template <CubitBC BCTYPE, int BASE_DIM, int FIELD_DIM, AssemblyType A,
          IntegrationType I, typename OpBase>
struct AddFluxToLhsPipelineImpl<

    OpFluxLhsImpl<ElasticExample::SpringBcType<BCTYPE>, BASE_DIM, FIELD_DIM, A,
                  I, OpBase>,
    A, I, OpBase

    > {

  AddFluxToLhsPipelineImpl() = delete;

  static MoFEMErrorCode add(

      boost::ptr_deque<ForcesAndSourcesCore::UserDataOperator> &pipeline,
      MoFEM::Interface &m_field, std::string row_field_name,
      std::string col_field_name, std::string block_name, Sev sev

  ) {
    MoFEMFunctionBegin;

    using OP = OpFluxLhsImpl<ElasticExample::SpringBcType<BLOCKSET>, BASE_DIM,
                             FIELD_DIM, PETSC, GAUSS, OpBase>;

    auto add_op = [&](auto &&meshset_vec_ptr) {
      for (auto m : meshset_vec_ptr) {
        MOFEM_TAG_AND_LOG("WORLD", sev, "OpSprngLhs") << "Add " << *m;
        pipeline.push_back(
            new OP(m_field, m->getMeshsetId(), row_field_name, col_field_name));
      }
      MOFEM_LOG_CHANNEL("WORLD");
    };

    switch (BCTYPE) {
    case BLOCKSET:
      add_op(

          m_field.getInterface<MeshsetsManager>()->getCubitMeshsetPtr(
              std::regex(

                  (boost::format("%s(.*)") % block_name).str()

                      ))

      );

      break;
    default:
      SETERRQ(PETSC_COMM_SELF, MOFEM_NOT_IMPLEMENTED,
              "Handling of bc type not implemented");
      break;
    }
    MoFEMFunctionReturn(0);
  }
};
