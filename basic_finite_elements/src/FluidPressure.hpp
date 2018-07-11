/* \file FluidPressure.hpp
 *
 * \brief Implementation of fluid pressure element
 *
 * \todo Implement nonlinear case (consrvative force, i.e. normal follows surface normal)
 *
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


#ifndef __FLUID_PRESSURE_HPP
#define __FLUID_PRESSURE_HPP

/** \brief Fluid pressure forces

\todo Implementation for large displacements

*/
struct FluidPressure {

  MoFEM::Interface &mField;
  struct MyTriangleFE: public MoFEM::FaceElementForcesAndSourcesCore {

    MyTriangleFE(MoFEM::Interface &m_field):
    MoFEM::FaceElementForcesAndSourcesCore(m_field) {
    }
    int getRule(int order) { return order+1; };

    MoFEMErrorCode preProcess() {
      MoFEMFunctionBeginHot;
      MoFEMFunctionReturnHot(0);
    }

  };
  MyTriangleFE fe;
  MyTriangleFE& getLoopFe() { return fe; }

  FluidPressure(MoFEM::Interface &m_field): mField(m_field),fe(mField) {}

  typedef int MeshSetId;
  struct FluidData {
    double dEnsity; ///< fluid density [kg/m^2] or any consistent unit
    VectorDouble aCCeleration; ///< acceleration [m/s^2]
    VectorDouble zEroPressure; ///< fluid level of reference zero pressure.
    Range tRis; ///< range of surface elemennt to which fluid pressure is applied
    friend std::ostream& operator<<(std::ostream& os,const FluidPressure::FluidData &e);
  };
  std::map<MeshSetId,FluidData> setOfFluids;

  boost::ptr_vector<MethodForForceScaling> methodsOp;

  
  

  struct OpCalculatePressure: public MoFEM::FaceElementForcesAndSourcesCore::UserDataOperator {

    Vec F;
    FluidData &dAta;
    boost::ptr_vector<MethodForForceScaling> &methodsOp;
    bool allowNegativePressure; ///< allows for negative pressures
    bool hoGeometry;

    OpCalculatePressure(
      const std::string field_name,
      Vec _F,
      FluidData &data,
      boost::ptr_vector<MethodForForceScaling> &methods_op,
      bool allow_negative_pressure,
      bool ho_geometry
    ):
    MoFEM::FaceElementForcesAndSourcesCore::UserDataOperator(field_name,UserDataOperator::OPROW),
    F(_F),
    dAta(data),
    methodsOp(methods_op),
    allowNegativePressure(allow_negative_pressure),
    hoGeometry(ho_geometry) {
    }

    VectorDouble Nf;
    
    MoFEMErrorCode doWork(
      int side,EntityType type,DataForcesAndSourcesCore::EntData &data) {
      MoFEMFunctionBeginHot;
      if(data.getIndices().size()==0) MoFEMFunctionReturnHot(0);
      EntityHandle ent = getNumeredEntFiniteElementPtr()->getEnt();
      if(dAta.tRis.find(ent)==dAta.tRis.end()) MoFEMFunctionReturnHot(0);

      const FENumeredDofEntity *dof_ptr;
      ierr = getNumeredEntFiniteElementPtr()->getRowDofsByPetscGlobalDofIdx(data.getIndices()[0],&dof_ptr); CHKERRG(ierr);
      int rank = dof_ptr->getNbOfCoeffs();
      int nb_row_dofs = data.getIndices().size()/rank;

      Nf.resize(data.getIndices().size());
      Nf.clear();

      for(unsigned int gg = 0;gg<data.getN().size1();gg++) {

        VectorDouble dist;
        VectorDouble zero_pressure = dAta.zEroPressure;
        /*VectorDouble fluctuation;
        fluctuation.resize(3);
        fluctuation.clear();
        if(methodsOp.size()>0) {
          double acc_norm2 = norm_2(dAta.aCCeleration);
          if(acc_norm2>0) {
            fluctuation = dAta.aCCeleration/acc_norm2;
          }
          ierr = MethodForForceScaling::applyScale(getFEMethod(),methodsOp,fluctuation); CHKERRG(ierr);
        }
        noalias(zero_pressure) += fluctuation;*/
        dist = ublas::matrix_row<MatrixDouble >(getCoordsAtGaussPts(),gg);
        dist -= zero_pressure;
        double dot = cblas_ddot(3,&dist[0],1,&dAta.aCCeleration[0],1);
        // std::cerr << dot << " " << dAta.aCCeleration << " " << dist << std::endl;
        if(!allowNegativePressure) dot = fmax(0,dot);
        double pressure = dot*dAta.dEnsity;
        // std::cerr << dot << " " << dAta.dEnsity << " " << pressure << std::endl;

        for(int rr = 0;rr<rank;rr++) {
          double force;
          if(hoGeometry) {
            force = pressure*getNormalsAtGaussPt()(gg,rr);
          } else {
            force = pressure*getNormal()[rr];
          }
          cblas_daxpy(
            nb_row_dofs,getGaussPts()(2,gg)*force,&data.getN()(gg,0),1,&Nf[rr],rank
          );
        }

      }

      // std::cerr << Nf << std::endl;
      // std::cerr << std::endl;

      bool set = false;
      switch(getFEMethod()->ts_ctx) {
        case FEMethod::CTX_TSSETIFUNCTION:
        F = getFEMethod()->ts_F;
        set = true;
        break;
        default:
        break;
      }
      if(!set) {
        switch(getFEMethod()->snes_ctx) {
          case FEMethod::CTX_SNESSETFUNCTION:
          F = getFEMethod()->snes_f;
          set = true;
          default:
          break;
        }
      }

      if(F==PETSC_NULL) {
        SETERRQ(PETSC_COMM_SELF,MOFEM_IMPOSIBLE_CASE,"impossible case");
      }

      ierr = VecSetValues(
        F,
        data.getIndices().size(),
        &data.getIndices()[0],
        &Nf[0],
        ADD_VALUES
      ); CHKERRG(ierr);


      MoFEMFunctionReturnHot(0);
    }
  };

  MoFEMErrorCode addNeumannFluidPressureBCElements(
    const std::string field_name,const std::string mesh_nodals_positions = "MESH_NODE_POSITIONS"
  );

  MoFEMErrorCode setNeumannFluidPressureFiniteElementOperators(
    string field_name,Vec F,bool allow_negative_pressure = true,bool ho_geometry = false
  );
  
};

std::ostream& operator<<(std::ostream& os,const FluidPressure::FluidData &e);

#endif //__FLUID_PRESSSURE_HPP
