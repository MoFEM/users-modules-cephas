/** \file SmallStrainTranverslyIsotropic.hpp
 * \ingroup nonlinear_elastic_elem
 * \brief Implementation of linear transverse isotropic material.
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

#ifndef __SMALLSTRAINTRANVERSLYISOTROPIC_HPP__
#define __SMALLSTRAINTRANVERSLYISOTROPIC_HPP__

#ifndef WITH_ADOL_C
  #error "MoFEM need to be compiled with ADOL-C"
#endif

/** \brief Hook equation
  * \ingroup nonlinear_elastic_elem
  */
template<typename TYPE>
struct SmallStrainTranverslyIsotropic: public NonlinearElasticElement::FunctionsToCalculatePiolaKirchhoffI<TYPE> {

  SmallStrainTranverslyIsotropic(): NonlinearElasticElement::FunctionsToCalculatePiolaKirchhoffI<TYPE>() {}

  

  ublas::matrix<TYPE> sTrain;
  ublas::vector<TYPE> voightStrain;
  TYPE tR;

  MoFEMErrorCode calculateStrain() {
    MoFEMFunctionBeginHot;
    sTrain.resize(3,3,false);
    noalias(sTrain) = this->F;
    for(int dd = 0;dd<3;dd++) {
      sTrain(dd,dd) -= 1;
    }
    sTrain += trans(sTrain);
    sTrain *= 0.5;
    voightStrain.resize(6,false);
    voightStrain[0] = sTrain(0,0);
    voightStrain[1] = sTrain(1,1);
    voightStrain[2] = sTrain(2,2);
    voightStrain[3] = 2*sTrain(0,1);
    voightStrain[4] = 2*sTrain(1,2);
    voightStrain[5] = 2*sTrain(2,0);
    MoFEMFunctionReturnHot(0);
  }

  double nu_p, nu_pz, E_p, E_z, G_zp;
  ublas::symmetric_matrix<TYPE,ublas::upper> localStiffnessMatrix;
  MoFEMErrorCode calculateLocalStiffnesMatrix() {
    MoFEMFunctionBeginHot;
    double nu_zp=(nu_pz*E_z)/E_p;
    double delta=((1+nu_p)*(1-nu_p-(2*nu_pz*nu_zp)))/(E_p*E_p*E_z);

    localStiffnessMatrix.resize(6);
    localStiffnessMatrix.clear();
    localStiffnessMatrix(0,0)=localStiffnessMatrix(1,1)=(1-nu_pz*nu_zp)/(E_p*E_z*delta);
    localStiffnessMatrix(2,2)=(1-nu_p*nu_p)/(E_p*E_p*delta);

    localStiffnessMatrix(0,1)=localStiffnessMatrix(1,0)=(nu_p+nu_zp*nu_pz)/(E_p*E_z*delta);
    localStiffnessMatrix(0,2)=localStiffnessMatrix(2,0)=localStiffnessMatrix(1,2)=localStiffnessMatrix(2,1)=(nu_zp+nu_p*nu_zp)/(E_p*E_z*delta);

    localStiffnessMatrix(3,3)=E_p/(2*(1+nu_p));
    localStiffnessMatrix(4,4)=localStiffnessMatrix(5,5)=G_zp;

    MoFEMFunctionReturnHot(0);
  }

  ublas::matrix<TYPE> aARotMat;
  ublas::vector<TYPE> axVector;
  TYPE axAngle;

  /**
  * \brief Function to Calculate the Rotation Matrix at a given axis and angle of rotation

  * This function computes the rotational matrix for a given axis of rotation
  * and angle of rotation about that angle <br>

  *\param axVector A vector representing the axis of rotation
  *\param axAngle Angle of rotation along the axis (in radians)
  */
  MoFEMErrorCode calculateAxisAngleRotationalMatrix() {
    MoFEMFunctionBeginHot;

    aARotMat.resize(3,3,false);
    aARotMat.clear();

    TYPE norm = sqrt(pow(axVector[0],2) + pow(axVector[1],2) + pow(axVector[2],2));

    aARotMat(0,0) = 1-((1-cos(axAngle))*(pow(axVector[1],2)+pow(axVector[2],2))/pow(norm,2));
    aARotMat(1,1) = 1-((1-cos(axAngle))*(pow(axVector[0],2)+pow(axVector[2],2))/pow(norm,2));
    aARotMat(2,2) = 1-((1-cos(axAngle))*(pow(axVector[0],2)+pow(axVector[1],2))/pow(norm,2));

    aARotMat(0,1) = ((1-cos(axAngle))*axVector[0]*axVector[1]-norm*axVector[2]*sin(axAngle))/pow(norm,2);
    aARotMat(1,0) = ((1-cos(axAngle))*axVector[0]*axVector[1]+norm*axVector[2]*sin(axAngle))/pow(norm,2);

    aARotMat(0,2) = ((1-cos(axAngle))*axVector[0]*axVector[2]+norm*axVector[1]*sin(axAngle))/pow(norm,2);
    aARotMat(2,0) = ((1-cos(axAngle))*axVector[0]*axVector[2]-norm*axVector[1]*sin(axAngle))/pow(norm,2);

    aARotMat(1,2) = ((1-cos(axAngle))*axVector[1]*axVector[2]-norm*axVector[0]*sin(axAngle))/pow(norm,2);
    aARotMat(2,1) = ((1-cos(axAngle))*axVector[1]*axVector[2]+norm*axVector[0]*sin(axAngle))/pow(norm,2);

    MoFEMFunctionReturnHot(0);
  }

  ublas::matrix<TYPE> stressRotMat;

  /**
  * \brief Function to Calculate Stress Transformation Matrix
  * This function computes the stress transformation Matrix at a give axis and angle of rotation <br>
  * One can also output the axis/angle rotational Matrix
  */
  MoFEMErrorCode stressTransformation() {
    MoFEMFunctionBeginHot;

    stressRotMat.resize(6,6,false);
    stressRotMat.clear();

    stressRotMat(0, 0) =       aARotMat(0,0) * aARotMat(0,0);
    stressRotMat(0, 1) =       aARotMat(1,0) * aARotMat(1,0);
    stressRotMat(0, 2) =       aARotMat(2,0) * aARotMat(2,0);
    stressRotMat(0, 3) = 2.0 * aARotMat(1,0) * aARotMat(0,0);
    stressRotMat(0, 4) = 2.0 * aARotMat(2,0) * aARotMat(1,0);
    stressRotMat(0, 5) = 2.0 * aARotMat(0,0) * aARotMat(2,0);

    stressRotMat(1, 0) =       aARotMat(0,1) * aARotMat(0,1);
    stressRotMat(1, 1) =       aARotMat(1,1) * aARotMat(1,1);
    stressRotMat(1, 2) =       aARotMat(2,1) * aARotMat(2,1);
    stressRotMat(1, 3) = 2.0 * aARotMat(1,1) * aARotMat(0,1);
    stressRotMat(1, 4) = 2.0 * aARotMat(2,1) * aARotMat(1,1);
    stressRotMat(1, 5) = 2.0 * aARotMat(0,1) * aARotMat(2,1);

    stressRotMat(2, 0) =       aARotMat(0,2) * aARotMat(0,2);
    stressRotMat(2, 1) =       aARotMat(1,2) * aARotMat(1,2);
    stressRotMat(2, 2) =       aARotMat(2,2) * aARotMat(2,2);
    stressRotMat(2, 3) = 2.0 * aARotMat(1,2) * aARotMat(0,2);
    stressRotMat(2, 4) = 2.0 * aARotMat(2,2) * aARotMat(1,2);
    stressRotMat(2, 5) = 2.0 * aARotMat(0,2) * aARotMat(2,2);

    stressRotMat(3, 0) =   aARotMat(0,1) * aARotMat(0,0);
    stressRotMat(3, 1) =   aARotMat(1,1) * aARotMat(1,0);
    stressRotMat(3, 2) =   aARotMat(2,1) * aARotMat(2,0);
    stressRotMat(3, 3) = ( aARotMat(1,1) * aARotMat(0,0) + aARotMat(0,1) * aARotMat(1,0) );
    stressRotMat(3, 4) = ( aARotMat(2,1) * aARotMat(1,0) + aARotMat(1,1) * aARotMat(2,0) );
    stressRotMat(3, 5) = ( aARotMat(0,1) * aARotMat(2,0) + aARotMat(2,1) * aARotMat(0,0) );

    stressRotMat(4, 0) =   aARotMat(0,2) * aARotMat(0,1);
    stressRotMat(4, 1) =   aARotMat(1,2) * aARotMat(1,1);
    stressRotMat(4, 2) =   aARotMat(2,2) * aARotMat(2,1);
    stressRotMat(4, 3) = ( aARotMat(1,2) * aARotMat(0,1) + aARotMat(0,2) * aARotMat(1,1) );
    stressRotMat(4, 4) = ( aARotMat(2,2) * aARotMat(1,1) + aARotMat(1,2) * aARotMat(2,1) );
    stressRotMat(4, 5) = ( aARotMat(0,2) * aARotMat(2,1) + aARotMat(2,2) * aARotMat(0,1) );

    stressRotMat(5, 0) =   aARotMat(0,0) * aARotMat(0,2);
    stressRotMat(5, 1) =   aARotMat(1,0) * aARotMat(1,2);
    stressRotMat(5, 2) =   aARotMat(2,0) * aARotMat(2,2);
    stressRotMat(5, 3) = ( aARotMat(1,0) * aARotMat(0,2) + aARotMat(0,0) * aARotMat(1,2) );
    stressRotMat(5, 4) = ( aARotMat(2,0) * aARotMat(1,2) + aARotMat(1,0) * aARotMat(2,2) );
    stressRotMat(5, 5) = ( aARotMat(0,0) * aARotMat(2,2) + aARotMat(2,0) * aARotMat(0,2) );

    MoFEMFunctionReturnHot(0);
  }

  ublas::matrix<TYPE> strainRotMat;

  /**
  * \brief Function to Calculate Strain Transformation Matrix<br>
  * This function computes the strain transformation Matrix at a give axis and angle of rotation <br>
  * One can also output the axis/angle rotational Matrix
  */
  MoFEMErrorCode strainTransformation() {
    MoFEMFunctionBeginHot;

    strainRotMat.resize(6,6,false);
    strainRotMat.clear();

    strainRotMat(0, 0) = aARotMat(0,0) * aARotMat(0,0);
    strainRotMat(0, 1) = aARotMat(1,0) * aARotMat(1,0);
    strainRotMat(0, 2) = aARotMat(2,0) * aARotMat(2,0);
    strainRotMat(0, 3) = aARotMat(1,0) * aARotMat(0,0);
    strainRotMat(0, 4) = aARotMat(2,0) * aARotMat(1,0);
    strainRotMat(0, 5) = aARotMat(0,0) * aARotMat(2,0);

    strainRotMat(1, 0) = aARotMat(0,1) * aARotMat(0,1);
    strainRotMat(1, 1) = aARotMat(1,1) * aARotMat(1,1);
    strainRotMat(1, 2) = aARotMat(2,1) * aARotMat(2,1);
    strainRotMat(1, 3) = aARotMat(1,1) * aARotMat(0,1);
    strainRotMat(1, 4) = aARotMat(2,1) * aARotMat(1,1);
    strainRotMat(1, 5) = aARotMat(0,1) * aARotMat(2,1);

    strainRotMat(2, 0) = aARotMat(0,2) * aARotMat(0,2);
    strainRotMat(2, 1) = aARotMat(1,2) * aARotMat(1,2);
    strainRotMat(2, 2) = aARotMat(2,2) * aARotMat(2,2);
    strainRotMat(2, 3) = aARotMat(1,2) * aARotMat(0,2);
    strainRotMat(2, 4) = aARotMat(2,2) * aARotMat(1,2);
    strainRotMat(2, 5) = aARotMat(0,2) * aARotMat(2,2);

    strainRotMat(3, 0) = 2.0 * aARotMat(0,1) * aARotMat(0,0);
    strainRotMat(3, 1) = 2.0 * aARotMat(1,1) * aARotMat(1,0);
    strainRotMat(3, 2) = 2.0 * aARotMat(2,1) * aARotMat(2,0);
    strainRotMat(3, 3) =     ( aARotMat(1,1) * aARotMat(0,0) + aARotMat(0,1) * aARotMat(1,0) );
    strainRotMat(3, 4) =     ( aARotMat(2,1) * aARotMat(1,0) + aARotMat(1,1) * aARotMat(2,0) );
    strainRotMat(3, 5) =     ( aARotMat(0,1) * aARotMat(2,0) + aARotMat(2,1) * aARotMat(0,0) );

    strainRotMat(4, 0) = 2.0 * aARotMat(0,2) * aARotMat(0,1);
    strainRotMat(4, 1) = 2.0 * aARotMat(1,2) * aARotMat(1,1);
    strainRotMat(4, 2) = 2.0 * aARotMat(2,2) * aARotMat(2,1);
    strainRotMat(4, 3) =     ( aARotMat(1,2) * aARotMat(0,1) + aARotMat(0,2) * aARotMat(1,1) );
    strainRotMat(4, 4) =     ( aARotMat(2,2) * aARotMat(1,1) + aARotMat(1,2) * aARotMat(2,1) );
    strainRotMat(4, 5) =     ( aARotMat(0,2) * aARotMat(2,1) + aARotMat(2,2) * aARotMat(0,1) );

    strainRotMat(5, 0) = 2.0 * aARotMat(0,0) * aARotMat(0,2);
    strainRotMat(5, 1) = 2.0 * aARotMat(1,0) * aARotMat(1,2);
    strainRotMat(5, 2) = 2.0 * aARotMat(2,0) * aARotMat(2,2);
    strainRotMat(5, 3) =     ( aARotMat(1,0) * aARotMat(0,2) + aARotMat(0,0) * aARotMat(1,2) );
    strainRotMat(5, 4) =     ( aARotMat(2,0) * aARotMat(1,2) + aARotMat(1,0) * aARotMat(2,2) );
    strainRotMat(5, 5) =     ( aARotMat(0,0) * aARotMat(2,2) + aARotMat(2,0) * aARotMat(0,2) );

    MoFEMFunctionReturnHot(0);
  }

  ublas::matrix<TYPE> dR;
  ublas::matrix<TYPE> globalStiffnessMatrix;

  MoFEMErrorCode calculateGlobalStiffnesMatrix() {
    MoFEMFunctionBeginHot;

    dR.resize(6,6,false);
    noalias(dR) = prod(localStiffnessMatrix,strainRotMat);
    globalStiffnessMatrix.resize(6,6,false);
    noalias(globalStiffnessMatrix) = prod(stressRotMat,dR);

    MoFEMFunctionReturnHot(0);
  }

  virtual MoFEMErrorCode calculateAngles() {
    MoFEMFunctionBeginHot;
    MoFEMFunctionReturnHot(0);
  }

  ublas::vector<TYPE> voigtStress;

  /** \brief Calculate global stress

  This is small strain approach, i.e. Piola stress
  is like a Cauchy stress, since configurations are notation
  distinguished.

  */
  virtual MoFEMErrorCode calculateP_PiolaKirchhoffI(
    const NonlinearElasticElement::BlockData block_data,
    boost::shared_ptr<const NumeredEntFiniteElement> fe_ptr
  ) {
    MoFEMFunctionBeginHot;
    ierr = calculateAngles(); CHKERRG(ierr);
    ierr = calculateStrain(); CHKERRG(ierr);
    ierr = calculateLocalStiffnesMatrix(); CHKERRG(ierr);
    ierr = calculateAxisAngleRotationalMatrix(); CHKERRG(ierr);
    ierr = stressTransformation(); CHKERRG(ierr);
    axAngle = -axAngle;
    ierr = calculateAxisAngleRotationalMatrix(); CHKERRG(ierr);
    ierr = strainTransformation(); CHKERRG(ierr);
    ierr = calculateGlobalStiffnesMatrix(); CHKERRG(ierr);

    voigtStress.resize(6,false);
    noalias(voigtStress) = prod(globalStiffnessMatrix,voightStrain);
    this->P.resize(3,3,false);
    this->P(0,0) = voigtStress[0];
    this->P(1,1) = voigtStress[1];
    this->P(2,2) = voigtStress[2];
    this->P(0,1) = voigtStress[3];
    this->P(1,2) = voigtStress[4];
    this->P(0,2) = voigtStress[5];
    this->P(1,0) = this->P(0,1);
    this->P(2,1) = this->P(1,2);
    this->P(2,0) = this->P(0,2);

    MoFEMFunctionReturnHot(0);
  }

  /** \brief calculate density of strain energy
  *
  */
  virtual MoFEMErrorCode calculateElasticEnergy(
  const NonlinearElasticElement::BlockData block_data,
  boost::shared_ptr<const NumeredEntFiniteElement> fe_ptr
) {
    MoFEMFunctionBeginHot;

    ierr = calculateAngles(); CHKERRG(ierr);
    ierr = calculateStrain(); CHKERRG(ierr);
    ierr = calculateLocalStiffnesMatrix(); CHKERRG(ierr);
    ierr = calculateAxisAngleRotationalMatrix(); CHKERRG(ierr);
    ierr = stressTransformation(); CHKERRG(ierr);
    axAngle = -axAngle;
    ierr = calculateAxisAngleRotationalMatrix(); CHKERRG(ierr);
    ierr = strainTransformation(); CHKERRG(ierr);
    ierr = calculateGlobalStiffnesMatrix(); CHKERRG(ierr);

    voigtStress.resize(6,false);
    noalias(voigtStress) = prod(globalStiffnessMatrix,voightStrain);
    this->eNergy = 0.5*inner_prod(voigtStress,voightStrain);
    MoFEMFunctionReturnHot(0);
  }

  VectorDouble normalizedPhi;
  VectorDouble axVectorDouble;
  double axAngleDouble;

  MoFEMErrorCode calculateFibreAngles() {
    MoFEMFunctionBeginHot;

    try {

      int gg = this->gG; // number of integration point
      MatrixDouble &phi = (this->commonDataPtr->gradAtGaussPts["POTENTIAL_FIELD"][gg]);
      normalizedPhi.resize(3,false);
      double nrm2_phi = sqrt(pow(phi(0,0),2)+pow(phi(0,1),2)+pow(phi(0,2),2));
      for(int ii = 0;ii<3;ii++) {
        normalizedPhi[ii] = -phi(0,ii)/nrm2_phi;
      }

      axVectorDouble.resize(3,false);
      const double zVec[3]={ 0.0,0.0,1.0 };
      axVectorDouble[0] = normalizedPhi[1]*zVec[2]-normalizedPhi[2]*zVec[1];
      axVectorDouble[1] = normalizedPhi[2]*zVec[0]-normalizedPhi[0]*zVec[2];
      axVectorDouble[2] = normalizedPhi[0]*zVec[1]-normalizedPhi[1]*zVec[0];
      double nrm2_ax_vector = norm_2(axVectorDouble);
      const double eps = 1e-12;
      if(nrm2_ax_vector<eps) {
        axVectorDouble[0] = 0;
        axVectorDouble[1] = 0;
        axVectorDouble[2] = 1;
        nrm2_ax_vector = 1;
      }
      axAngleDouble = asin(nrm2_ax_vector);

    } catch (const std::exception& ex) {
      std::ostringstream ss;
      ss << "throw in method: " << ex.what() << std::endl;
      SETERRQ(PETSC_COMM_SELF,1,ss.str().c_str());
    }

    MoFEMFunctionReturnHot(0);
  }

};


struct SmallStrainTranverslyIsotropicDouble: public SmallStrainTranverslyIsotropic<double> {

  SmallStrainTranverslyIsotropicDouble(): SmallStrainTranverslyIsotropic<double>() {}

  virtual MoFEMErrorCode calculateAngles() {
    MoFEMFunctionBeginHot;

    try {

      ierr = calculateFibreAngles(); CHKERRG(ierr);
      axVector.resize(3,false);
      axVector[0] = axVectorDouble[0];
      axVector[1] = axVectorDouble[1];
      axVector[2] = axVectorDouble[2];
      axAngle = axAngleDouble;

    } catch (const std::exception& ex) {
      std::ostringstream ss;
      ss << "throw in method: " << ex.what() << std::endl;
      SETERRQ(PETSC_COMM_SELF,1,ss.str().c_str());
    }
    MoFEMFunctionReturnHot(0);
  }

  virtual MoFEMErrorCode getDataOnPostProcessor(
    std::map<std::string,std::vector<VectorDouble > > &field_map,
    std::map<std::string,std::vector<MatrixDouble > > &grad_map
  ) {
    MoFEMFunctionBeginHot;
    int nb_gauss_pts = grad_map["POTENTIAL_FIELD"].size();
    this->commonDataPtr->gradAtGaussPts["POTENTIAL_FIELD"].resize(nb_gauss_pts);
    for(int gg = 0;gg<nb_gauss_pts;gg++) {
      this->commonDataPtr->gradAtGaussPts["POTENTIAL_FIELD"][gg].resize(1,3,false);
      for(int ii = 0;ii<3;ii++) {
        this->commonDataPtr->gradAtGaussPts["POTENTIAL_FIELD"][gg](0,ii) =
        ((grad_map["POTENTIAL_FIELD"])[gg])(0,ii);
      }
    }
    MoFEMFunctionReturnHot(0);
  }

};

struct SmallStrainTranverslyIsotropicADouble: public SmallStrainTranverslyIsotropic<adouble> {

  SmallStrainTranverslyIsotropicADouble(): SmallStrainTranverslyIsotropic<adouble>() {}

  int nbActiveVariables0;

  virtual MoFEMErrorCode setUserActiveVariables(
    int &nb_active_variables
  ) {
    MoFEMFunctionBeginHot;

    try {

      ierr = calculateFibreAngles(); CHKERRG(ierr);
      axVector.resize(3,false);
      axVector[0] <<= axVectorDouble[0];
      axVector[1] <<= axVectorDouble[1];
      axVector[2] <<= axVectorDouble[2];
      axAngle <<= axAngleDouble;
      nbActiveVariables0 = nb_active_variables;
      nb_active_variables += 4;

    } catch (const std::exception& ex) {
      std::ostringstream ss;
      ss << "throw in method: " << ex.what() << std::endl;
      SETERRQ(PETSC_COMM_SELF,1,ss.str().c_str());
    }

    MoFEMFunctionReturnHot(0);
  }

  virtual MoFEMErrorCode setUserActiveVariables(
    VectorDouble &active_variables) {
    MoFEMFunctionBeginHot;

    try {

      int shift = nbActiveVariables0; // is a number of elements in F
      ierr = calculateFibreAngles(); CHKERRG(ierr);
      active_variables[shift+0] = axVectorDouble[0];
      active_variables[shift+1] = axVectorDouble[1];
      active_variables[shift+2] = axVectorDouble[2];
      active_variables[shift+3] = axAngleDouble;

    } catch (const std::exception& ex) {
      std::ostringstream ss;
      ss << "throw in method: " << ex.what() << std::endl;
      SETERRQ(PETSC_COMM_SELF,1,ss.str().c_str());
    }

    MoFEMFunctionReturnHot(0);
  }

};

#endif //__SMALLSTRAINTRANVERSLYISOTROPIC_HPP__
