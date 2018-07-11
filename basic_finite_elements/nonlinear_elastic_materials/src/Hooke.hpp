/** \file Hooke.hpp
 * \ingroup nonlinear_elastic_elem
 * \brief Implementation of linear elastic material
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

#ifndef __HOOKE_HPP__
#define __HOOKE_HPP__

#ifndef WITH_ADOL_C
  #error "MoFEM need to be compiled with ADOL-C"
#endif

/** \brief Hook equation
  * \ingroup nonlinear_elastic_elem
  */
template<typename TYPE>
struct Hooke: public NonlinearElasticElement::FunctionsToCalculatePiolaKirchhoffI<TYPE> {

    Hooke(): NonlinearElasticElement::FunctionsToCalculatePiolaKirchhoffI<TYPE>() {}

    ublas::matrix<TYPE> ePs;
    TYPE tR;

    /** \brief Hooke equation
      *
      * \f$\sigma = \lambda\textrm{tr}[\varepsilon]+2\mu\varepsilon\f$
      *
      */
    virtual MoFEMErrorCode calculateP_PiolaKirchhoffI(
      const NonlinearElasticElement::BlockData block_data,
      boost::shared_ptr<const NumeredEntFiniteElement> fe_ptr
    ) {
      MoFEMFunctionBeginHot;
      //
      this->lambda = LAMBDA(block_data.E,block_data.PoissonRatio);
      this->mu = MU(block_data.E,block_data.PoissonRatio);
      //std::cerr << block_data.E << " " << block_data.PoissonRatio << std::endl;
      ePs.resize(3,3,false);
      noalias(ePs) = this->F;
      for(int dd = 0;dd<3;dd++) {
        ePs(dd,dd) -= 1;
      }
      ePs += trans(ePs);
      ePs *= 0.5;
      this->P.resize(3,3,false);
      noalias(this->P) = 2*this->mu*ePs;
      tR = 0;
      for(int dd = 0;dd<3;dd++) {
        tR += ePs(dd,dd);
      }
      for(int dd =0;dd<3;dd++) {
        this->P(dd,dd) += this->lambda*tR;
      }
      //std::cerr << ePs << " : " << this->P << std::endl << std::endl;
      MoFEMFunctionReturnHot(0);
    }

    /** \brief calculate density of strain energy
      *
      * \f$\Psi = \frac{1}{2}\lambda(\textrm{tr}[\varepsilon])^2+\mu\varepsilon:\varepsilon\f$
      *
      */
    virtual MoFEMErrorCode calculateElasticEnergy(
      const NonlinearElasticElement::BlockData block_data,
      boost::shared_ptr<const NumeredEntFiniteElement> fe_ptr
    ) {
      MoFEMFunctionBeginHot;
      this->lambda = LAMBDA(block_data.E,block_data.PoissonRatio);
      this->mu = MU(block_data.E,block_data.PoissonRatio);
      ePs.resize(3,3);
      noalias(ePs) = this->F;
      for(int dd = 0;dd<3;dd++) {
        ePs(dd,dd) -= 1;
      }
      ePs += trans(ePs);
      ePs *= 0.5;
      // std::cerr << this->F << " : " << ePs << std::endl;
      this->eNergy = 0;
      tR = 0;
      for(int dd = 0;dd<3;dd++) {
        tR += ePs(dd,dd);
        for(int jj = 0;jj<3;jj++) {
          this->eNergy += this->mu*ePs(dd,jj)*ePs(dd,jj);
        }
      }
      this->eNergy += 0.5*this->lambda*tR*tR;
      MoFEMFunctionReturnHot(0);
    }

};

#endif //__HOOKE_HPP__
