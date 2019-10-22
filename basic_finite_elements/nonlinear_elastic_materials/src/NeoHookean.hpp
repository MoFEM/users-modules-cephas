/** \file NeoHookean.hpp
 * \ingroup nonlinear_elastic_elem
 * \brief Implementation of Neo-Hookean elastic material
 * \example NeoHookean.hpp
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

#ifndef __NEOHOOKEAN_HPP__
#define __NEOHOOKEAN_HPP__

/** \brief NeoHookan equation
  * \ingroup nonlinear_elastic_elem
  */
template<typename TYPE>
struct NeoHookean: public NonlinearElasticElement::FunctionsToCalculatePiolaKirchhoffI<TYPE> {

    NeoHookean(): NonlinearElasticElement::FunctionsToCalculatePiolaKirchhoffI<TYPE>() {}

    TYPE detC;
    TYPE logJ;
    ublas::matrix<TYPE,ublas::row_major,ublas::bounded_array<TYPE,9> > invC;

    /** \brief calculate second Piola Kirchhoff
      *
      * \f$\mathbf{S} = \mu(\mathbf{I}-\mathbf{C}^{-1})+\lambda(\ln{J})\mathbf{C}^{-1}\f$

      For details look to: <br>
      NONLINEAR CONTINUUM MECHANICS FOR FINITE ELEMENT ANALYSIS, Javier Bonet,
      Richard D. Wood

      */
    virtual MoFEMErrorCode NeoHooke_PiolaKirchhoffII() {
      MoFEMFunctionBeginHot;
      
      invC.resize(3,3);
      this->S.resize(3,3);
      ierr = this->dEterminant(this->C,detC); CHKERRG(ierr);
      ierr = this->iNvert(detC,this->C,invC); CHKERRG(ierr);
      ierr = this->dEterminant(this->F,this->J); CHKERRG(ierr);
      // if(this->J<=0) {
      //   cerr << this->J << endl;
      //   cerr << this->F << endl;
      // }
      logJ = log(sqrt(this->J*this->J));
      for(int i = 0;i<3;i++) {
        for(int j = 0;j<3;j++) {
          this->S(i,j) = this->mu*( ((i==j) ? 1 : 0) - invC(i,j) ) + this->lambda*logJ*invC(i,j);
        }
      }
      MoFEMFunctionReturnHot(0);
    }

    virtual MoFEMErrorCode calculateP_PiolaKirchhoffI(
      const NonlinearElasticElement::BlockData block_data,
      boost::shared_ptr<const NumeredEntFiniteElement> fe_ptr
    ) {
      MoFEMFunctionBeginHot;
      
      this->lambda = LAMBDA(block_data.E,block_data.PoissonRatio);
      this->mu = MU(block_data.E,block_data.PoissonRatio);
      ierr = this->calculateC_CauchyDeformationTensor(); CHKERRG(ierr);
      ierr = this->NeoHooke_PiolaKirchhoffII(); CHKERRG(ierr);
      this->P.resize(3,3);
      noalias(this->P) = prod(this->F,this->S);
      //std::cerr << "P: " << P << std::endl;
      MoFEMFunctionReturnHot(0);
    }

   /** \brief calculate elastic energy density
    *

    For details look to: <br>
    NONLINEAR CONTINUUM MECHANICS FOR FINITE ELEMENT ANALYSIS, Javier Bonet,
    Richard D. Wood

    */
    virtual MoFEMErrorCode NeoHookean_ElasticEnergy(){
        MoFEMFunctionBeginHot;
        this->eNergy = 0;
        for(int ii = 0;ii<3;ii++) {
            this->eNergy += this->C(ii,ii);
        }
        this->eNergy = 0.5*this->mu*(this->eNergy-3);
        logJ = log(sqrt(this->J*this->J));
        this->eNergy += -this->mu*logJ + 0.5*this->lambda*pow(logJ,2);
        MoFEMFunctionReturnHot(0);
    }

    MoFEMErrorCode calculateElasticEnergy(
      const NonlinearElasticElement::BlockData block_data,
      boost::shared_ptr<const NumeredEntFiniteElement> fe_ptr
    ) {
      MoFEMFunctionBeginHot;
      
      this->lambda = LAMBDA(block_data.E,block_data.PoissonRatio);
      this->mu = MU(block_data.E,block_data.PoissonRatio);
      ierr = this->calculateC_CauchyDeformationTensor(); CHKERRG(ierr);
      ierr = this->dEterminant(this->F,this->J); CHKERRG(ierr);
      ierr = this->NeoHookean_ElasticEnergy(); CHKERRG(ierr);
      MoFEMFunctionReturnHot(0);
    }

};



#endif
