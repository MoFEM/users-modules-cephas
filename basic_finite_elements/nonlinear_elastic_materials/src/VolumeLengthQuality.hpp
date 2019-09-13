/** \file VolumeLengthQuality.hpp
 * \ingroup nonlinear_elastic_elem
 * \brief Implementation of Volume-Length-Quality measure with barrier
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

#ifndef __VOLUME_LENGTH_QUALITY_HPP__
#define __VOLUME_LENGTH_QUALITY_HPP__

#ifndef WITH_ADOL_C
#error "MoFEM need to be compiled with ADOL-C"
#endif

enum VolumeLengthQualityType {
  QUALITY,
  BARRIER_AND_QUALITY,
  BARRIER_AND_CHANGE_QUALITY,
  BARRIER_AND_CHANGE_QUALITY_SCALED_BY_VOLUME,
  LASTOP_VOLUMELENGTHQUALITYTYPE
};

static const char *VolumeLengthQualityTypeNames[] = {
    "QUALITY", "BARRIER_AND_QUALITY", "BARRIER_AND_CHANGE_QUALITY",
    "BARRIER_AND_CHANGE_QUALITY_SCALED_BY_VOLUME"};

/** \brief Volume Length Quality
  \ingroup nonlinear_elastic_elem

  */
template <typename TYPE>
struct VolumeLengthQuality
    : public NonlinearElasticElement::FunctionsToCalculatePiolaKirchhoffI<
          TYPE> {

  // VolumeLengthQualityType tYpe;
  int tYpe;
  double aLpha;
  double gAmma;

  VolumeLengthQuality(VolumeLengthQualityType type, double alpha, double gamma)
      : NonlinearElasticElement::FunctionsToCalculatePiolaKirchhoffI<TYPE>(),
        tYpe(type), aLpha(alpha), gAmma(gamma) {
    ierr = getMaterialOptions();
    CHKERRABORT(PETSC_COMM_WORLD, ierr);
  }

  MoFEMErrorCode getMaterialOptions() {
    MoFEMFunctionBeginHot;
    CHKERR PetscOptionsBegin(PETSC_COMM_WORLD, "",
                             "Get VolumeLengthQuality material options",
                             "none");
    CHKERR PetscOptionsEList(
        "-volume_length_type", "Volume length quality type", "",
        VolumeLengthQualityTypeNames, LASTOP_VOLUMELENGTHQUALITYTYPE,
        VolumeLengthQualityTypeNames[tYpe], &tYpe, PETSC_NULL);
    CHKERR PetscOptionsScalar("-volume_length_alpha",
                              "volume length alpha parameter", "", aLpha,
                              &aLpha, PETSC_NULL);
    CHKERR PetscOptionsScalar("-volume_length_gamma",
                              "volume length parameter (barrier)", "", gAmma,
                              &gAmma, PETSC_NULL);
    ierr = PetscOptionsEnd();
    MoFEMFunctionReturnHot(0);
  }

  VectorDouble coordsEdges;
  double lrmsSquared0;
  VectorDouble deltaChi;

  ublas::vector<TYPE> deltaX;
  ublas::matrix<TYPE> Q, dXdChiT;
  TYPE lrmsSquared, q, b, detF, currentVolume, tMp;

  /** Get coordinates of edges using cannonical element numeration
   */
  MoFEMErrorCode getEdgesFromElemCoords() {
    MoFEMFunctionBegin;
    if (coordsEdges.empty()) {
      coordsEdges.resize(6 * 2 * 3, false);
    }
    cblas_dcopy(3, &this->opPtr->getCoords()[0 * 3], 1,
                &coordsEdges[0 * 3 * 2 + 0], 1);
    cblas_dcopy(3, &this->opPtr->getCoords()[1 * 3], 1,
                &coordsEdges[0 * 3 * 2 + 3], 1);
    cblas_dcopy(3, &this->opPtr->getCoords()[0 * 3], 1,
                &coordsEdges[1 * 3 * 2 + 0], 1);
    cblas_dcopy(3, &this->opPtr->getCoords()[2 * 3], 1,
                &coordsEdges[1 * 3 * 2 + 3], 1);
    cblas_dcopy(3, &this->opPtr->getCoords()[0 * 3], 1,
                &coordsEdges[2 * 3 * 2 + 0], 1);
    cblas_dcopy(3, &this->opPtr->getCoords()[3 * 3], 1,
                &coordsEdges[2 * 3 * 2 + 3], 1);
    cblas_dcopy(3, &this->opPtr->getCoords()[1 * 3], 1,
                &coordsEdges[3 * 3 * 2 + 0], 1);
    cblas_dcopy(3, &this->opPtr->getCoords()[2 * 3], 1,
                &coordsEdges[3 * 3 * 2 + 3], 1);
    cblas_dcopy(3, &this->opPtr->getCoords()[1 * 3], 1,
                &coordsEdges[4 * 3 * 2 + 0], 1);
    cblas_dcopy(3, &this->opPtr->getCoords()[3 * 3], 1,
                &coordsEdges[4 * 3 * 2 + 3], 1);
    cblas_dcopy(3, &this->opPtr->getCoords()[2 * 3], 1,
                &coordsEdges[5 * 3 * 2 + 0], 1);
    cblas_dcopy(3, &this->opPtr->getCoords()[3 * 3], 1,
                &coordsEdges[5 * 3 * 2 + 3], 1);
    MoFEMFunctionReturn(0);
  }

  /** \brief Calculate mean element edge length

    \f[
    \Delta \boldsymbol\chi = \boldsymbol\chi^1 - \boldsymbol\chi^2
    \f]

    \f[
    \Delta X = \mathbf{F} \Delta \boldsymbol\chi
    \f]

    \f[
    l_\textrm{rms} = \sqrt{\frac{1}{6} \sum_{i=0}^6 l_i^2 } =
    L_\textrm{rms}dl_\textrm{rms} \f]

   */
  MoFEMErrorCode calculateLrms() {
    MoFEMFunctionBegin;
    if (deltaChi.size() != 3) {
      deltaChi.resize(3);
      deltaX.resize(3);
      dXdChiT.resize(3, 3);
    }
    lrmsSquared = 0;
    lrmsSquared0 = 0;
    dXdChiT.clear();
    for (int ee = 0; ee < 6; ee++) {
      for (int dd = 0; dd < 3; dd++) {
        deltaChi[dd] = coordsEdges[6 * ee + dd] - coordsEdges[6 * ee + 3 + dd];
      }
      noalias(deltaX) = prod(this->F, deltaChi);
      for (int dd = 0; dd < 3; dd++) {
        lrmsSquared += (1. / 6.) * deltaX[dd] * deltaX[dd];
        lrmsSquared0 += (1. / 6.) * deltaChi[dd] * deltaChi[dd];
      }
      noalias(dXdChiT) += outer_prod(deltaX, deltaChi);
    }
    MoFEMFunctionReturn(0);
  }

  /** \brief Calculate Q

  \f[
  \mathbf{Q} =
   \mathbf{F}^{-\mathsf{T}}
   -
   \frac{1}{2}
   \frac{1}{l^2_\textrm{rms}}
   \sum_i^6
     \Delta\mathbf{X}_i
     \Delta\boldsymbol\chi_i^\mathsf{T}
  \f]

  */
  MoFEMErrorCode calculateQ() {
    MoFEMFunctionBegin;
    if (Q.size1() == 0) {
      Q.resize(3, 3);
    }
    noalias(Q) = trans(this->invF) - 0.5 * dXdChiT / lrmsSquared;
    MoFEMFunctionReturn(0);
  }

  /** \brief Volume Length Quality

    Based on:
    Three‐dimensional brittle fracture: configurational‐force‐driven crack
    propagation International Journal for Numerical Methods in Engineering 97
    (7), 531-550

    \f[
    \mathcal{B}(a)=\frac{a}{2(1-\gamma)}-\ln{(a-\gamma)}
    \f]

    \f[
    q = q_0 b,
    \quad q_0 = 6\sqrt{2}\frac{V_0}{L^3_\textrm{rms}},
    \quad b = \frac{\textrm{det}(\mathbf{F})}{\textrm{d}l^3_\textrm{rms}}
    \f]

    \f[
    \mathbf{P} = \mathcal{B}(a)\mathbf{Q},
    \f]
    where \f$a\f$ depending on problem could be \f$q\f$ or \f$b\f$.

    */
  virtual MoFEMErrorCode calculateP_PiolaKirchhoffI(
      const NonlinearElasticElement::BlockData block_data,
      boost::shared_ptr<const NumeredEntFiniteElement> fe_ptr) {
    MoFEMFunctionBegin;

    CHKERR getEdgesFromElemCoords();
    CHKERR this->dEterminant(this->F, detF);
    if (this->invF.size1() != 3)
      this->invF.resize(3, 3);

    CHKERR this->iNvert(detF, this->F, this->invF);
    CHKERR calculateLrms();
    CHKERR calculateQ();

    double lrms03 = lrmsSquared0 * sqrt(lrmsSquared0);
    b = detF / (lrmsSquared * sqrt(lrmsSquared) / lrms03);

    currentVolume = detF * this->opPtr->getVolume();
    q = 6. * sqrt(2.) * currentVolume / (lrmsSquared * sqrt(lrmsSquared));

    if (this->P.size1() != 3)
      this->P.resize(3, 3);

    switch (tYpe) {
    case QUALITY:
      // Only use for testing, simple quality gradient
      noalias(this->P) = q * Q;
      break;
    case BARRIER_AND_QUALITY:
      // This is used form mesh smoothing
      tMp = q / (1.0 - gAmma) - 1.0 / (q - gAmma);
      noalias(this->P) = tMp * Q;
      break;
    case BARRIER_AND_CHANGE_QUALITY:
      // Works well with Arbitrary Lagrangian Formulation
      tMp = b / (1.0 - gAmma) - 1.0 / (b - gAmma);
      noalias(this->P) = tMp * Q;
      break;
    case BARRIER_AND_CHANGE_QUALITY_SCALED_BY_VOLUME:
      // When scaled by volume works well with ALE and face flipping.
      // Works well with smooth crack propagation
      tMp = currentVolume;
      tMp *= b / (1.0 - gAmma) - 1.0 / (b - gAmma);
      noalias(this->P) = tMp * Q;
      break;
    }

    // Divide by volume, to make units as they should be
    this->P *= aLpha / this->opPtr->getVolume();

    MoFEMFunctionReturn(0);
  }
};

#endif //__VOLUME_LENGTH_QUALITY_HPP__
