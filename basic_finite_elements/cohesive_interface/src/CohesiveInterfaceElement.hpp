/** \file CohesiveInterfaceElement.hpp
  \brief Implementation of linear interface element

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

namespace CohesiveElement {

/** \brief Cohesive element implementation

  \bug Interface element not working with HO geometry.
*/
struct CohesiveInterfaceElement {

  struct CommonData {
    MatrixDouble gapGlob;
    MatrixDouble gapLoc;
    ublas::vector<MatrixDouble > R;
  };
  CommonData commonData;

  struct MyPrism: public MoFEM::FlatPrismElementForcesAndSourcesCore {
    MyPrism(MoFEM::Interface &m_field): MoFEM::FlatPrismElementForcesAndSourcesCore(m_field) {}
    int getRule(int order) { return 2*order; };
  };
  MyPrism feRhs;
  MyPrism feLhs;
  MyPrism feHistory;

  CohesiveInterfaceElement(MoFEM::Interface &m_field):
    feRhs(m_field),feLhs(m_field),feHistory(m_field) {};

  virtual ~CohesiveInterfaceElement() {

  }

  MyPrism& getFeRhs() { return feRhs; }
  MyPrism& getFeLhs() { return feLhs; }
  MyPrism& getFeHistory() { return feHistory; }

  /** \brief Constitutive (physical) equation for interface

    This is linear degradation model. Material parameters are: strength
    \f$f_t\f$, interface fracture energy \f$G_f\f$, elastic material stiffness
    \f$E\f$. Parameter \f$\beta\f$ controls how interface opening is calculated.

    Model parameter is interface penalty thickness \f$h\f$.

    */
  struct PhysicalEquation {

    MoFEM::Interface &mField;
    bool isInitialised;
    PhysicalEquation(MoFEM::Interface &m_field):
      mField(m_field),isInitialised(false) {};

    virtual ~PhysicalEquation() {
    }

    double h,youngModulus,beta,ft,Gf;
    Range pRisms;
    Tag thKappa,thDamagedPrism;

    double E0,g0,kappa1;

    /** \brief Initialize history variable data

    Create tag on the prism/interface to store damage history variable

    */
    MoFEMErrorCode iNitailise(const FEMethod *fe_method) {
      MoFEMFunctionBeginHot;

      double def_damaged = 0;
      rval = mField.get_moab().tag_get_handle(
        "DAMAGED_PRISM",1,MB_TYPE_INTEGER,thDamagedPrism,MB_TAG_CREAT|MB_TAG_SPARSE,&def_damaged
      ); MOAB_THROW(rval);
      const int def_len = 0;
      rval = mField.get_moab().tag_get_handle("_KAPPA",def_len,MB_TYPE_DOUBLE,
      thKappa,MB_TAG_CREAT|MB_TAG_SPARSE|MB_TAG_VARLEN,NULL); CHKERRG(rval);
      E0 = youngModulus/h;
      g0 = ft/E0;
      kappa1 = 2*Gf/ft;
      MoFEMFunctionReturnHot(0);
    }

    /** \brief Calculate gap opening

    \f[
    g = \sqrt{ g_n^2 + \beta(g_{s1}^2 + g_{s2}^2)}
    \f]

    */
    double calcG(int gg,MatrixDouble gap_loc) {
      return sqrt(pow(gap_loc(gg,0),2)+beta*(pow(gap_loc(gg,1),2)+pow(gap_loc(gg,2),2)));
    }

    double *kappaPtr;
    int kappaSize;

    /** \brief Get pointer from the mesh to histoy variables \f$\kappa\f$
    */
    MoFEMErrorCode getKappa(int nb_gauss_pts,const FEMethod *fe_method) {
      MoFEMFunctionBeginHot;
      EntityHandle ent = fe_method->numeredEntFiniteElementPtr->getEnt();

      rval = mField.get_moab().tag_get_by_ptr(thKappa,&ent,1,(const void **)&kappaPtr,&kappaSize);
      if(rval != MB_SUCCESS || kappaSize != nb_gauss_pts) {
        VectorDouble kappa;
        kappa.resize(nb_gauss_pts);
        kappa.clear();
        int tag_size[1];
        tag_size[0] = nb_gauss_pts;
        void const* tag_data[] = { &kappa[0] };
        rval = mField.get_moab().tag_set_by_ptr(thKappa,&ent,1,tag_data,tag_size);  CHKERRG(rval);
        rval = mField.get_moab().tag_get_by_ptr(thKappa,&ent,1,(const void **)&kappaPtr,&kappaSize);  CHKERRG(rval);
      }
      MoFEMFunctionReturnHot(0);
    }

    MatrixDouble Dglob,Dloc;

    /** \brief Calculate stiffness material matrix

    \f[
    \mathbf{D}_\textrm{loc} = (1-\Omega) \mathbf{I} E_0
    \f]
    where \f$E_0\f$ is initial interface penalty stiffness

    \f[
    \mathbf{D}_\textrm{glob} = \mathbf{R}^\textrm{T} \mathbf{D}_\textrm{loc}\mathbf{R}
    \f]

    */
    MoFEMErrorCode calcDglob(const double omega,MatrixDouble &R) {
      MoFEMFunctionBeginHot;
      Dglob.resize(3,3);
      Dloc.resize(3,3);
      Dloc.clear();
      double E = (1-omega)*E0;
      Dloc(0,0) = E;
      Dloc(1,1) = E;
      Dloc(2,2) = E;
      Dglob = prod( Dloc, R );
      Dglob = prod( trans(R), Dglob );
      MoFEMFunctionReturnHot(0);
    }

    /** \brief Calculate damage

    \f[
    \Omega = \frac{1}{2} \frac{(2 G_f E_0+f_t^2)\kappa}{(ft+E_0 \kappa)G_f}
    \f]

    */
    MoFEMErrorCode calcOmega(const double kappa,double& omega) {
      MoFEMFunctionBeginHot;
      omega = 0;
      if(kappa>=kappa1) {
        omega = 1;
        MoFEMFunctionReturnHot(0);
      } else if(kappa>0) {
        double a = (2.0*Gf*E0+ft*ft)*kappa;
        double b = (ft+E0*kappa)*Gf;
        omega = 0.5*a/b;
      }
      MoFEMFunctionReturnHot(0);
    }

    /** \brief Calculate tangent material stiffness
    */
    MoFEMErrorCode calcTangetDglob(const double omega,double g,const VectorDouble& gap_loc,MatrixDouble &R) {
      MoFEMFunctionBeginHot;
      Dglob.resize(3,3);
      Dloc.resize(3,3);
      double domega = 0.5*(2*Gf*E0+ft*ft)/((ft+(g-ft/E0)*E0)*Gf) - 0.5*((g-ft/E0)*(2*Gf*E0+ft*ft)*E0)/(pow(ft+(g-ft/E0)*E0,2)*Gf);
      Dloc.resize(3,3);
      //r0
      Dloc(0,0) = (1-omega)*E0 - domega*E0*gap_loc[0]*gap_loc[0]/g;
      Dloc(0,1) = -domega*E0*gap_loc[0]*beta*gap_loc[1]/g;
      Dloc(0,2) = -domega*E0*gap_loc[0]*beta*gap_loc[2]/g;
      //r1
      Dloc(1,0) = -domega*E0*gap_loc[1]*gap_loc[0]/g;
      Dloc(1,1) = (1-omega)*E0 - domega*E0*gap_loc[1]*beta*gap_loc[1]/g;
      Dloc(1,2) = -domega*E0*gap_loc[1]*beta*gap_loc[2]/g;
      //r2
      Dloc(2,0) = -domega*E0*gap_loc[2]*gap_loc[0]/g;
      Dloc(2,1) = -domega*E0*gap_loc[2]*beta*gap_loc[1]/g;
      Dloc(2,2) = (1-omega)*E0 - domega*E0*gap_loc[2]*beta*gap_loc[2]/g;
      Dglob = prod(Dloc,R);
      Dglob = prod(trans(R),Dglob);
      MoFEMFunctionReturnHot(0);
    }

    /** \brief Calculate tractions

    \f[
    \mathbf{t} = \mathbf{D}_\textrm{glob}\mathbf{g}
    \f]

    */
    virtual MoFEMErrorCode calculateTraction(
      VectorDouble &traction,
      int gg,CommonData &common_data,
      const FEMethod *fe_method
    ) {
      MoFEMFunctionBeginHot;

      if(!isInitialised) {
        ierr = iNitailise(fe_method); CHKERRG(ierr);
        isInitialised = true;
      }
      if(gg==0) {
        ierr = getKappa(common_data.gapGlob.size1(),fe_method); CHKERRG(ierr);
      }
      double g = calcG(gg,common_data.gapLoc);
      double kappa = fmax(g-g0,kappaPtr[gg]);
      double omega = 0;
      ierr = calcOmega(kappa,omega); CHKERRG(ierr);
      //std::cerr << gg << " " << omega << std::endl;
      ierr = calcDglob(omega,common_data.R[gg]); CHKERRG(ierr);
      traction.resize(3);
      ublas::matrix_row<MatrixDouble > gap_glob(common_data.gapGlob,gg);
      noalias(traction) = prod(Dglob,gap_glob);
      MoFEMFunctionReturnHot(0);
    }

    /** \brief Calculate tangent stiffness
    */
    virtual MoFEMErrorCode calculateTangentStiffeness(
      MatrixDouble &tangent_matrix,
      int gg,CommonData &common_data,
      const FEMethod *fe_method
    ) {
      MoFEMFunctionBeginHot;

      try {
        if(!isInitialised) {
          ierr = iNitailise(fe_method); CHKERRG(ierr);
          isInitialised = true;
        }
        if(gg==0) {
          ierr = getKappa(common_data.gapGlob.size1(),fe_method); CHKERRG(ierr);
        }
        double g = calcG(gg,common_data.gapLoc);
        double kappa = fmax(g-g0,kappaPtr[gg]);
        double omega = 0;
        ierr = calcOmega(kappa,omega); CHKERRG(ierr);
        //std::cerr << gg << " " << omega << std::endl;
        int iter;
        ierr = SNESGetIterationNumber(fe_method->snes,&iter); CHKERRG(ierr);
        if((kappa <= kappaPtr[gg])||(kappa>=kappa1)||(iter <= 1)) {
          ierr = calcDglob(omega,common_data.R[gg]); CHKERRG(ierr);
        } else {
          ublas::matrix_row<MatrixDouble > g_loc(common_data.gapLoc,gg);
          ierr = calcTangetDglob(omega,g,g_loc,common_data.R[gg]);
        }
        tangent_matrix.resize(3,3);
        noalias(tangent_matrix) = Dglob;
        //std::cerr << "t " << tangent_matrix << std::endl;
      } catch (const std::exception& ex) {
        std::ostringstream ss;
        ss << "throw in method: " << ex.what() << std::endl;
        SETERRQ(PETSC_COMM_SELF,1,ss.str().c_str());
      }
      MoFEMFunctionReturnHot(0);
    }

    /** \brief Update history variables when converged
    */
    virtual MoFEMErrorCode updateHistory(
      CommonData &common_data,const FEMethod *fe_method
    ) {
      MoFEMFunctionBeginHot;


      if(!isInitialised) {
        ierr = iNitailise(fe_method); CHKERRG(ierr);
        isInitialised = true;
      }
      ierr = getKappa(common_data.gapGlob.size1(),fe_method); CHKERRG(ierr);
      bool all_gauss_pts_damaged = true;
      for(unsigned int gg = 0;gg<common_data.gapGlob.size1();gg++) {
        double omega = 0;
        double g = calcG(gg,common_data.gapLoc);
        double kappa = fmax(g-g0,kappaPtr[gg]);
        kappaPtr[gg] = kappa;
        ierr = calcOmega(kappa,omega); CHKERRG(ierr);
        //if(omega < 1.) {
        all_gauss_pts_damaged = false;
        //}
      }
      if(all_gauss_pts_damaged) {
        EntityHandle ent = fe_method->numeredEntFiniteElementPtr->getEnt();
        int set_prism_as_demaged = 1;
        rval = mField.get_moab().tag_set_data(thDamagedPrism,&ent,1,&set_prism_as_demaged); CHKERRG(rval);
      }
      MoFEMFunctionReturnHot(0);
    }

  };


  /** \brief Set negative sign to shape functions on face 4
    */
  struct OpSetSignToShapeFunctions: public FlatPrismElementForcesAndSourcesCore::UserDataOperator {

    OpSetSignToShapeFunctions(const std::string field_name):
    FlatPrismElementForcesAndSourcesCore::UserDataOperator(field_name,ForcesAndSourcesCore::UserDataOperator::OPROW) {}

    MoFEMErrorCode doWork(int side,EntityType type,DataForcesAndSourcesCore::EntData &data) {
      MoFEMFunctionBeginHot;
      if(data.getN().size1()==0)  MoFEMFunctionReturnHot(0);
      if(data.getN().size2()==0)  MoFEMFunctionReturnHot(0);
      switch(type) {
        case MBVERTEX:
        for(unsigned int gg = 0;gg<data.getN().size1();gg++) {
          for(int nn = 3;nn<6;nn++) {
            data.getN()(gg,nn) *= -1;
          }
        }
        break;
        case MBEDGE:
        if(side < 3) MoFEMFunctionReturnHot(0);
        data.getN() *= -1;
        break;
        case MBTRI:
        if(side == 3) MoFEMFunctionReturnHot(0);
        data.getN() *= -1;
        break;
        default:
        SETERRQ(PETSC_COMM_SELF,1,"data inconsitency");
      }
      MoFEMFunctionReturnHot(0);
    }

  };

  /** \brief Operator calculate gap, normal vector and rotation matrix
  */
  struct OpCalculateGapGlobal: public FlatPrismElementForcesAndSourcesCore::UserDataOperator {

    CommonData &commonData;
    OpCalculateGapGlobal(const std::string field_name,CommonData &common_data):
      FlatPrismElementForcesAndSourcesCore::UserDataOperator(field_name,ForcesAndSourcesCore::UserDataOperator::OPROW),
      commonData(common_data) {}

    MoFEMErrorCode doWork(
      int side,EntityType type,DataForcesAndSourcesCore::EntData &data) {
      MoFEMFunctionBeginHot;
      try {
        int nb_dofs = data.getIndices().size();
        if(nb_dofs == 0) MoFEMFunctionReturnHot(0);
        int nb_gauss_pts = data.getN().size1();
        if(type == MBVERTEX) {
          commonData.R.resize(nb_gauss_pts);
          for(int gg = 0;gg<nb_gauss_pts;gg++) {
            commonData.R[gg].resize(3,3);
            double nrm2_normal = 0;
            double nrm2_tangent1 = 0;
            double nrm2_tangent2 = 0;
            for(int dd = 0;dd<3;dd++) {
              nrm2_normal += pow(getNormalsAtGaussPtsF3()(gg,dd),2);
              nrm2_tangent1 += pow(getTangent1AtGaussPtF3()(gg,dd),2);
              nrm2_tangent2 += pow(getTangent2AtGaussPtF3()(gg,dd),2);
            }
            nrm2_normal = sqrt(nrm2_normal);
            nrm2_tangent1 = sqrt(nrm2_tangent1);
            nrm2_tangent2 = sqrt(nrm2_tangent2);
            for(int dd = 0;dd<3;dd++) {
              commonData.R[gg](0,dd) = getNormalsAtGaussPtsF3()(gg,dd)/nrm2_normal;
              commonData.R[gg](1,dd) = getTangent1AtGaussPtF3()(gg,dd)/nrm2_tangent1;
              commonData.R[gg](2,dd) = getTangent2AtGaussPtF3()(gg,dd)/nrm2_tangent2;
            }
          }
        }
        if(type == MBVERTEX) {
          commonData.gapGlob.resize(nb_gauss_pts,3);
          commonData.gapGlob.clear();
        }
        for(int gg = 0;gg<nb_gauss_pts;gg++) {
          for(int dd = 0;dd<3;dd++) {
            commonData.gapGlob(gg,dd) += cblas_ddot(
              nb_dofs/3,&data.getN(gg)[0],1,&data.getFieldData()[dd],3);
            }
          }
        } catch (const std::exception& ex) {
          std::ostringstream ss;
          ss << "throw in method: " << ex.what() << std::endl;
          SETERRQ(PETSC_COMM_SELF,1,ss.str().c_str());
        }
      MoFEMFunctionReturnHot(0);
    }

  };

  /** \brief Operator calculate gap in local coordinate system
  */
  struct OpCalculateGapLocal: public FlatPrismElementForcesAndSourcesCore::UserDataOperator {

    CommonData &commonData;
    OpCalculateGapLocal(const std::string field_name,CommonData &common_data):
      FlatPrismElementForcesAndSourcesCore::UserDataOperator(field_name,ForcesAndSourcesCore::UserDataOperator::OPROW),
      commonData(common_data) {}

    MoFEMErrorCode doWork(int side,EntityType type,DataForcesAndSourcesCore::EntData &data) {
      MoFEMFunctionBeginHot;
      try {
        if(type == MBVERTEX) {
          int nb_gauss_pts = data.getN().size1();
          commonData.gapLoc.resize(nb_gauss_pts,3);
          for(int gg = 0;gg<nb_gauss_pts;gg++) {
            ublas::matrix_row<MatrixDouble > gap_glob(commonData.gapGlob,gg);
            ublas::matrix_row<MatrixDouble > gap_loc(commonData.gapLoc,gg);
            gap_loc = prod(commonData.R[gg],gap_glob);
          }
        }
      } catch (const std::exception& ex) {
        std::ostringstream ss;
        ss << "throw in method: " << ex.what() << std::endl;
        SETERRQ(PETSC_COMM_SELF,1,ss.str().c_str());
      }
      MoFEMFunctionReturnHot(0);
    }

  };

  /** \brief Operator calculate right hand side vector
  */
  struct OpRhs: public FlatPrismElementForcesAndSourcesCore::UserDataOperator {

    CommonData &commonData;
    PhysicalEquation &physicalEqations;
    OpRhs(const std::string field_name,CommonData &common_data,PhysicalEquation &physical_eqations):
      FlatPrismElementForcesAndSourcesCore::UserDataOperator(field_name,ForcesAndSourcesCore::UserDataOperator::OPROW),
      commonData(common_data),physicalEqations(physical_eqations) {}

    VectorDouble traction,Nf;
    MoFEMErrorCode doWork(int side,EntityType type,DataForcesAndSourcesCore::EntData &data) {
      MoFEMFunctionBeginHot;

      try {
        int nb_dofs = data.getIndices().size();
        if(nb_dofs == 0) MoFEMFunctionReturnHot(0);
        if(physicalEqations.pRisms.find(getNumeredEntFiniteElementPtr()->getEnt()) == physicalEqations.pRisms.end()) {
          MoFEMFunctionReturnHot(0);
        }
        Nf.resize(nb_dofs);
        Nf.clear();
        int nb_gauss_pts = data.getN().size1();
        for(int gg = 0;gg<nb_gauss_pts;gg++) {
          ierr = physicalEqations.calculateTraction(traction,gg,commonData,getFEMethod()); CHKERRG(ierr);
          double w = getGaussPts()(2,gg)*cblas_dnrm2(3,&getNormalsAtGaussPtsF3()(gg,0),1)*0.5;
          for(int nn = 0;nn<nb_dofs/3;nn++) {
            for(int dd = 0;dd<3;dd++) {
              Nf[3*nn+dd] += w*data.getN(gg)[nn]*traction[dd];
            }
          }
        }
        ierr = VecSetValues(getFEMethod()->snes_f,
        data.getIndices().size(),&data.getIndices()[0],&Nf[0],ADD_VALUES); CHKERRG(ierr);
      } catch (const std::exception& ex) {
        std::ostringstream ss;
        ss << "throw in method: " << ex.what() << std::endl;
        SETERRQ(PETSC_COMM_SELF,1,ss.str().c_str());
      }
      MoFEMFunctionReturnHot(0);
    }

  };

  /** \brief Operator calculate element stiffens matrix
  */
  struct OpLhs: public FlatPrismElementForcesAndSourcesCore::UserDataOperator {

    CommonData &commonData;
    PhysicalEquation &physicalEqations;
    OpLhs(const std::string field_name,CommonData &common_data,PhysicalEquation &physical_eqations):
    FlatPrismElementForcesAndSourcesCore::UserDataOperator(field_name,ForcesAndSourcesCore::UserDataOperator::OPROWCOL),
    commonData(common_data),physicalEqations(physical_eqations) { sYmm = false; }

    MatrixDouble K,D,ND;
    MoFEMErrorCode doWork(
      int row_side,int col_side,
      EntityType row_type,EntityType col_type,
      DataForcesAndSourcesCore::EntData &row_data,
      DataForcesAndSourcesCore::EntData &col_data
    ) {
      MoFEMFunctionBeginHot;

      try {
        int nb_row = row_data.getIndices().size();
        if(nb_row == 0) MoFEMFunctionReturnHot(0);
        int nb_col = col_data.getIndices().size();
        if(nb_col == 0) MoFEMFunctionReturnHot(0);
        if(physicalEqations.pRisms.find(getNumeredEntFiniteElementPtr()->getEnt())
        == physicalEqations.pRisms.end()) {
          MoFEMFunctionReturnHot(0);
        }
        //std::cerr << row_side << " " << row_type << " " << row_data.getN() << std::endl;
        //std::cerr << col_side << " " << col_type << " " << col_data.getN() << std::endl;
        ND.resize(nb_row,3);
        K.resize(nb_row,nb_col);
        K.clear();
        int nb_gauss_pts = row_data.getN().size1();
        for(int gg = 0;gg<nb_gauss_pts;gg++) {
          ierr = physicalEqations.calculateTangentStiffeness(D,gg,commonData,getFEMethod()); CHKERRG(ierr);
          double w = getGaussPts()(2,gg)*cblas_dnrm2(3,&getNormalsAtGaussPtsF3()(gg,0),1)*0.5;
          ND.clear();
          for(int nn = 0; nn<nb_row/3;nn++) {
            for(int dd = 0;dd<3;dd++) {
              for(int DD = 0;DD<3;DD++) {
                ND(3*nn+dd,DD) += row_data.getN(gg)[nn]*D(dd,DD);
              }
            }
          }
          for(int nn = 0; nn<nb_row/3; nn++) {
            for(int dd = 0;dd<3;dd++) {
              for(int NN = 0; NN<nb_col/3; NN++) {
                for(int DD = 0; DD<3;DD++) {
                  K(3*nn+dd,3*NN+DD) += w*ND(3*nn+dd,DD)*col_data.getN(gg)[NN];
                }
              }
            }
          }
        }
        ierr = MatSetValues(getFEMethod()->snes_B,
          nb_row,&row_data.getIndices()[0],
          nb_col,&col_data.getIndices()[0],
          &K(0,0),ADD_VALUES
        ); CHKERRG(ierr);
      } catch (const std::exception& ex) {
        std::ostringstream ss;
        ss << "throw in method: " << ex.what() << std::endl;
        SETERRQ(PETSC_COMM_SELF,1,ss.str().c_str());
      }
      MoFEMFunctionReturnHot(0);
    }

  };

  /** \brief Operator update history variables
  */
  struct OpHistory: public FlatPrismElementForcesAndSourcesCore::UserDataOperator {

    CommonData &commonData;
    PhysicalEquation &physicalEqations;
    OpHistory(const std::string field_name,CommonData &common_data,PhysicalEquation &physical_eqations):
      FlatPrismElementForcesAndSourcesCore::UserDataOperator(field_name,ForcesAndSourcesCore::UserDataOperator::OPROW),
      commonData(common_data),physicalEqations(physical_eqations) {}

      MoFEMErrorCode doWork(int side,EntityType type,DataForcesAndSourcesCore::EntData &data) {
        MoFEMFunctionBeginHot;

        if(type != MBVERTEX) MoFEMFunctionReturnHot(0);
        if(physicalEqations.pRisms.find(getNumeredEntFiniteElementPtr()->getEnt()) == physicalEqations.pRisms.end()) {
          MoFEMFunctionReturnHot(0);
        }
        ierr = physicalEqations.updateHistory(commonData,getFEMethod()); CHKERRG(ierr);
        MoFEMFunctionReturnHot(0);
      }

  };

  /** \brief Driver function settting all operators needed for interface element
  */
  MoFEMErrorCode addOps(const std::string field_name,boost::ptr_vector<CohesiveInterfaceElement::PhysicalEquation> &interfaces) {
    MoFEMFunctionBeginHot;

    //Rhs
    feRhs.getOpPtrVector().push_back(new OpSetSignToShapeFunctions(field_name));
    feRhs.getOpPtrVector().push_back(new OpCalculateGapGlobal(field_name,commonData));
    feRhs.getOpPtrVector().push_back(new OpCalculateGapLocal(field_name,commonData));
    //Lhs
    feLhs.getOpPtrVector().push_back(new OpSetSignToShapeFunctions(field_name));
    feLhs.getOpPtrVector().push_back(new OpCalculateGapGlobal(field_name,commonData));
    feLhs.getOpPtrVector().push_back(new OpCalculateGapLocal(field_name,commonData));
    //History
    feHistory.getOpPtrVector().push_back(new OpSetSignToShapeFunctions(field_name));
    feHistory.getOpPtrVector().push_back(new OpCalculateGapGlobal(field_name,commonData));
    feHistory.getOpPtrVector().push_back(new OpCalculateGapLocal(field_name,commonData));

    //add equations/data for physical interfaces
    boost::ptr_vector<CohesiveInterfaceElement::PhysicalEquation>::iterator pit;
    for(pit = interfaces.begin();pit!=interfaces.end();pit++) {
      feRhs.getOpPtrVector().push_back(new OpRhs(field_name,commonData,*pit));
      feLhs.getOpPtrVector().push_back(new OpLhs(field_name,commonData,*pit));
      feHistory.getOpPtrVector().push_back(new OpHistory(field_name,commonData,*pit));
    }

    MoFEMFunctionReturnHot(0);
  }

};

}
