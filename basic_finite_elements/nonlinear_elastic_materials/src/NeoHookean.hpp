/** \file NeoHookean.hpp
 * \ingroup nonlinear_elastic_elem
 * \brief Implementation of Neo-Hookean elastic material
 * \example NeoHookean.hpp
 */



#ifndef __NEOHOOKEAN_HPP__
#define __NEOHOOKEAN_HPP__

/** \brief NeoHookan equation
 * \ingroup nonlinear_elastic_elem
 */
template <typename TYPE>
struct NeoHookean
    : public NonlinearElasticElement::FunctionsToCalculatePiolaKirchhoffI<
          TYPE> {

  using PiolaKirchoffI =
      NonlinearElasticElement::FunctionsToCalculatePiolaKirchhoffI<TYPE>;

  NeoHookean()
      : NonlinearElasticElement::FunctionsToCalculatePiolaKirchhoffI<TYPE>(),
        t_invC(PiolaKirchoffI::resizeAndSet(invC)) {}

  TYPE detC;
  TYPE logJ;
  MatrixBoundedArray<TYPE, 9> invC;
  FTensor::Tensor2<FTensor::PackPtr<TYPE *, 0>, 3, 3> t_invC;

  FTensor::Index<'i', 3> i;
  FTensor::Index<'j', 3> j;
  FTensor::Index<'k', 3> k;

  /** \brief calculate second Piola Kirchhoff
    *
    * \f$\mathbf{S} =
    \mu(\mathbf{I}-\mathbf{C}^{-1})+\lambda(\ln{J})\mathbf{C}^{-1}\f$

    For details look to: <br>
    NONLINEAR CONTINUUM MECHANICS FOR FINITE ELEMENT ANALYSIS, Javier Bonet,
    Richard D. Wood

    */
  MoFEMErrorCode NeoHooke_PiolaKirchhoffII() {
    MoFEMFunctionBeginHot;

    detC = determinantTensor3by3(this->C);
    CHKERR invertTensor3by3(this->C, detC, invC);
    this->J = determinantTensor3by3(this->F);

    logJ = log(sqrt(this->J * this->J));
    constexpr auto t_kd = FTensor::Kronecker_Delta<double>();

    this->t_S(i, j) = this->mu * (t_kd(i, j) - t_invC(i, j)) +
                      (this->lambda * logJ) * t_invC(i, j);

    MoFEMFunctionReturnHot(0);
  }

  virtual MoFEMErrorCode calculateP_PiolaKirchhoffI(
      const NonlinearElasticElement::BlockData block_data,
      boost::shared_ptr<const NumeredEntFiniteElement> fe_ptr) {
    MoFEMFunctionBegin;

    this->lambda = LAMBDA(block_data.E, block_data.PoissonRatio);
    this->mu = MU(block_data.E, block_data.PoissonRatio);
    CHKERR this->calculateC_CauchyDeformationTensor();
    CHKERR this->NeoHooke_PiolaKirchhoffII();

    this->t_P(i, j) = this->t_F(i, k) * this->t_S(k, j);

    MoFEMFunctionReturn(0);
  }

  /** \brief calculate elastic energy density
   *

   For details look to: <br>
   NONLINEAR CONTINUUM MECHANICS FOR FINITE ELEMENT ANALYSIS, Javier Bonet,
   Richard D. Wood

   */
  MoFEMErrorCode NeoHookean_ElasticEnergy() {
    MoFEMFunctionBegin;
    this->eNergy = this->t_C(i, i);
    this->eNergy = 0.5 * this->mu * (this->eNergy - 3);
    logJ = log(sqrt(this->J * this->J));
    this->eNergy += -this->mu * logJ + 0.5 * this->lambda * pow(logJ, 2);
    MoFEMFunctionReturn(0);
  }

  MoFEMErrorCode calculateElasticEnergy(
      const NonlinearElasticElement::BlockData block_data,
      boost::shared_ptr<const NumeredEntFiniteElement> fe_ptr) {
    MoFEMFunctionBegin;

    this->lambda = LAMBDA(block_data.E, block_data.PoissonRatio);
    this->mu = MU(block_data.E, block_data.PoissonRatio);
    CHKERR this->calculateC_CauchyDeformationTensor();
    this->J = determinantTensor3by3(this->F);
    CHKERR this->NeoHookean_ElasticEnergy();

    MoFEMFunctionReturn(0);
  }
};

#endif
