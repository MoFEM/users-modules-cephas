/** \file Hooke.hpp
 * \ingroup nonlinear_elastic_elem
 * \brief Implementation of linear elastic material
 */

/* MIT License
 *
 * Copyright (c) 2022
 * 
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 * 
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 * 
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#ifndef __HOOKE_HPP__
#define __HOOKE_HPP__

#ifndef WITH_ADOL_C
#error "MoFEM need to be compiled with ADOL-C"
#endif

/** \brief Hook equation
 * \ingroup nonlinear_elastic_elem
 */
template <typename TYPE>
struct Hooke
    : public NonlinearElasticElement::FunctionsToCalculatePiolaKirchhoffI<
          TYPE> {

  Hooke()
      : NonlinearElasticElement::FunctionsToCalculatePiolaKirchhoffI<TYPE>() {}

  ublas::matrix<TYPE> ePs;
  TYPE tR;

  /** \brief Hooke equation
   *
   * \f$\sigma = \lambda\textrm{tr}[\varepsilon]+2\mu\varepsilon\f$
   *
   */
  virtual MoFEMErrorCode calculateP_PiolaKirchhoffI(
      const NonlinearElasticElement::BlockData block_data,
      boost::shared_ptr<const NumeredEntFiniteElement> fe_ptr) {
    MoFEMFunctionBeginHot;
    this->lambda = LAMBDA(block_data.E, block_data.PoissonRatio);
    this->mu = MU(block_data.E, block_data.PoissonRatio);
    ePs.resize(3, 3, false);
    noalias(ePs) = this->F;
    for (int dd = 0; dd < 3; dd++) {
      ePs(dd, dd) -= 1;
    }
    ePs += trans(ePs);
    ePs *= 0.5;
    this->P.resize(3, 3, false);
    noalias(this->P) = 2 * this->mu * ePs;
    tR = 0;
    for (int dd = 0; dd < 3; dd++) {
      tR += ePs(dd, dd);
    }
    for (int dd = 0; dd < 3; dd++) {
      this->P(dd, dd) += this->lambda * tR;
    }
    MoFEMFunctionReturnHot(0);
  }

  /** \brief calculate density of strain energy
   *
   * \f$\Psi =
   * \frac{1}{2}\lambda(\textrm{tr}[\varepsilon])^2+\mu\varepsilon:\varepsilon\f$
   *
   */
  virtual MoFEMErrorCode calculateElasticEnergy(
      const NonlinearElasticElement::BlockData block_data,
      boost::shared_ptr<const NumeredEntFiniteElement> fe_ptr) {
    MoFEMFunctionBeginHot;
    this->lambda = LAMBDA(block_data.E, block_data.PoissonRatio);
    this->mu = MU(block_data.E, block_data.PoissonRatio);
    ePs.resize(3, 3);
    noalias(ePs) = this->F;
    for (int dd = 0; dd < 3; dd++) {
      ePs(dd, dd) -= 1;
    }
    ePs += trans(ePs);
    ePs *= 0.5;
    this->eNergy = 0;
    tR = 0;
    for (int dd = 0; dd < 3; dd++) {
      tR += ePs(dd, dd);
      for (int jj = 0; jj < 3; jj++) {
        this->eNergy += this->mu * ePs(dd, jj) * ePs(dd, jj);
      }
    }
    this->eNergy += 0.5 * this->lambda * tR * tR;
    MoFEMFunctionReturnHot(0);
  }
};

#endif //__HOOKE_HPP__
