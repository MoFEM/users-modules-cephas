/**
 * @file NaturalDomainBC.hpp
 * @brief Boundary conditions in domain, i.e. body forces.
 * @version 0.13.2
 * @date 2022-09-22
 * 
 * @copyright Copyright (c) 2022
 * 
 */

template <int BASE_DIM, int FIELD_DIM, AssemblyType A, IntegrationType I,
          typename OpBase>
struct AddFluxToRhsPipelineImpl<

    OpFluxRhsImpl<DomainBCs, BASE_DIM, FIELD_DIM, A, I, OpBase>, A, I, OpBase

    > {

  AddFluxToRhsPipelineImpl() = delete;

  using T =
      typename NaturalBC<OpBase>::template Assembly<A>::template LinearForm<I>;
  using OpBodyForce = typename T::template OpFlux<NaturalMeshsetType<BLOCKSET>,
                                                  BASE_DIM, FIELD_DIM>;

  static MoFEMErrorCode add(

      boost::ptr_vector<ForcesAndSourcesCore::UserDataOperator> &pipeline,
      MoFEM::Interface &m_field, std::string field_name, Sev sev

  ) {
    MoFEMFunctionBegin;
    CHKERR T::template AddFluxToPipeline<OpBodyForce>::add(
        pipeline, m_field, field_name, {}, "BODY_FORCE", sev);
    MoFEMFunctionReturn(0);
  }
};