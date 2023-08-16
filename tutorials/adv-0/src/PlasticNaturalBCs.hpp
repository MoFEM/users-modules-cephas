/**
 * @file NaturalBCs.hpp
 * @brief Set natural conditions
 * @version 0.1
 * @date 2023-07-26
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#include <ElasticSpring.hpp>

namespace PlasticOps {

struct DomainBCs;
struct BoundaryBCs;

} // namespace PlasticOps


template <int BASE_DIM, int FIELD_DIM, AssemblyType A, IntegrationType I,
          typename OpBase>
struct AddFluxToRhsPipelineImpl<

    OpFluxRhsImpl<PlasticOps::DomainBCs, BASE_DIM, FIELD_DIM, A, I, OpBase>, A,
    I, OpBase

    > {

  AddFluxToRhsPipelineImpl() = delete;

  using T =
      typename NaturalBC<OpBase>::template Assembly<A>::template LinearForm<I>;
  using OpBodyForce = typename T::template OpFlux<NaturalMeshsetType<BLOCKSET>,
                                                  BASE_DIM, FIELD_DIM>;

  static MoFEMErrorCode add(

      boost::ptr_deque<ForcesAndSourcesCore::UserDataOperator> &pipeline,
      MoFEM::Interface &m_field, std::string field_name,
      std::vector<boost::shared_ptr<ScalingMethod>> smv, Sev sev

  ) {
    MoFEMFunctionBegin;
    CHKERR T::template AddFluxToPipeline<OpBodyForce>::add(
        pipeline, m_field, field_name, smv, "BODY_FORCE", sev);
    MoFEMFunctionReturn(0);
  }
};

template <int BASE_DIM, int FIELD_DIM, AssemblyType A, IntegrationType I,
          typename OpBase>
struct AddFluxToRhsPipelineImpl<

    OpFluxRhsImpl<PlasticOps::BoundaryBCs, BASE_DIM, FIELD_DIM, A, I, OpBase>,
    A, I, OpBase

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
      MoFEM::Interface &m_field, std::string field_name,
      std::vector<boost::shared_ptr<ScalingMethod>> smv, Sev sev

  ) {
    MoFEMFunctionBegin;
    CHKERR T::template AddFluxToPipeline<OpForce>::add(
        pipeline, m_field, field_name, smv, "FORCE", sev);
    auto u_ptr = boost::make_shared<MatrixDouble>();
    pipeline.push_back(
        new OpCalculateVectorFieldValues<SPACE_DIM>(field_name, u_ptr));
    CHKERR T::template AddFluxToPipeline<OpSpringRhs>::add(
        pipeline, m_field, field_name, u_ptr, 1, "SPRING", sev);
    MoFEMFunctionReturn(0);
  }
};

template <int BASE_DIM, int FIELD_DIM, AssemblyType A, IntegrationType I,
          typename OpBase>
struct AddFluxToLhsPipelineImpl<

    OpFluxLhsImpl<PlasticOps::BoundaryBCs, BASE_DIM, FIELD_DIM, A, I, OpBase>,
    A, I, OpBase

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