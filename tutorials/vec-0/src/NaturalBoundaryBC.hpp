/**
 * @file NaturalBoundaryBC.hpp
 * @brief Implementation of natural boundary conditions
 * @version 0.13.2
 * @date 2022-09-22
 * 
 * @copyright Copyright (c) 2022
 * 
 */

template <int BASE_DIM, int FIELD_DIM, AssemblyType A, IntegrationType I,
          typename OpBase>
struct AddFluxToRhsPipelineImpl<

    OpFluxRhsImpl<BoundaryBCs, BASE_DIM, FIELD_DIM, A, I, OpBase>, A, I, OpBase

    > {

  AddFluxToRhsPipelineImpl() = delete;

  using T =
      typename NaturalBC<OpBase>::template Assembly<A>::template LinearForm<I>;

  using OpForce =
      typename T::template OpFlux<NaturalForceMeshsets, 1, SPACE_DIM>;
  using OpSpringRhs =
      typename T::template OpFlux<ElasticExample::SpringBcType<BLOCKSET>, 1,
                                  SPACE_DIM>;

  static MoFEMErrorCode add(

      boost::ptr_deque<ForcesAndSourcesCore::UserDataOperator> &pipeline,
      MoFEM::Interface &m_field, std::string field_name, double scale, Sev sev

  ) {
    MoFEMFunctionBegin;
    CHKERR T::template AddFluxToPipeline<OpForce>::add(
        pipeline, m_field, field_name, {}, "FORCE", sev);
    auto u_ptr = boost::make_shared<MatrixDouble>();
    pipeline.push_back(
        new OpCalculateVectorFieldValues<SPACE_DIM>(field_name, u_ptr));
    CHKERR T::template AddFluxToPipeline<OpSpringRhs>::add(
        pipeline, m_field, field_name, u_ptr, scale, "SPRING", sev);
    MoFEMFunctionReturn(0);
  }
};

template <int BASE_DIM, int FIELD_DIM, AssemblyType A, IntegrationType I,
          typename OpBase>
struct AddFluxToLhsPipelineImpl<

    OpFluxLhsImpl<BoundaryBCs, BASE_DIM, FIELD_DIM, A, I, OpBase>, A, I, OpBase

    > {

  AddFluxToLhsPipelineImpl() = delete;

  using T = typename NaturalBC<OpBase>::template Assembly<
      A>::template BiLinearForm<I>;

  using OpSpringLhs =
      typename T::template OpFlux<ElasticExample::SpringBcType<BLOCKSET>,
                                  BASE_DIM, FIELD_DIM>;

  static MoFEMErrorCode add(

      boost::ptr_deque<ForcesAndSourcesCore::UserDataOperator> &pipeline,
      MoFEM::Interface &m_field, std::string field_name, Sev sev

  ) {
    MoFEMFunctionBegin;
    CHKERR T::template AddFluxToPipeline<OpSpringLhs>::add(
        pipeline, m_field, field_name, field_name, "SPRING", sev);
    MoFEMFunctionReturn(0);
  }
};