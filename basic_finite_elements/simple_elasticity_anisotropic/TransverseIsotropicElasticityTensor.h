// #include <cmath>
// #include <vector>
// #include <Eigen/Dense>
// #include <FTensor.hpp>

class TransverseIsotropicElasticityTensor{

    std::vector<double> eulerAngle;           // euler angle, in degree, material coordinate
    double elasticModulusLongitudinal;        // EL
    double elasticModulusTransverse;          // ET
    double shearModulusOutPlane;              // Gxz and Gyz
    double poissonRatioInPlane;               // Vxy
    double poissonRatioOutPlane;              // Vzx and Vzy

    FTensor::Tensor4<double, 3, 3, 3, 3> elasticTensor_Global;

public:
    TransverseIsotropicElasticityTensor(std::vector<double> eulerAngle_dft    = {0.0,0.0,0.0},
                                        double elasticModulusLongitudinal_dft = 1.0e9,
                                        double elasticModulusTransverse_dft   = 1.0e9,
                                        double shearModulusOutPlane_dft       = 1.0e9,
                                        double poissonRatioInPlane_dft        = 0.0,
                                        double poissonRatioOutPlane_dft       = 0.0);
    
    // TransverseIsotropicElasticityTensor();
    virtual ~TransverseIsotropicElasticityTensor(){}

    /**
     *  Angle respectively between x-axis and N-axis, z-axis and Z-axis, N-axis and X-axis
     */
    void setEulerAngle(const double a1,
                       const double a2,
                       const double a3);

    /**
     * set material constants
     */
    void setElasticModulusLongitudinal(const double elasticModulusLongitudinal_In);
    void setElasticModulusTransverse(const double elasticModulusTransverse_In);
    void setShearModulusInPlane(const double shearModulusInPlane_In);
    void setShearModulusOutPlane(const double shearModulusOutPlane_In);
    void setPoissonRatioInPlane(const double poissonRatioInPlane_In);
    void setPoissonRatioOutPlane(const double poissonRatioOutPlane_In);

    /**
     * print the material constants
     */

    void printMaterialConstants();

    /**
     * calculate elasticity tensor
     */
    void calculateElasticityTensor();

    /**
     * output the elasticityTensor
     */
    FTensor::Tensor4<double, 3, 3, 3, 3> getElasticityTensor();

};


// TransverseIsotropicElasticityTensor
// ::TransverseIsotropicElasticityTensor(){
//     eulerAngle.assign (3,0.0);
// }
TransverseIsotropicElasticityTensor
::TransverseIsotropicElasticityTensor(std::vector<double> eulerAngle_dft,
                                      double elasticModulusLongitudinal_dft,
                                      double elasticModulusTransverse_dft,
                                      double shearModulusOutPlane_dft,
                                      double poissonRatioInPlane_dft,
                                      double poissonRatioOutPlane_dft){
    eulerAngle = eulerAngle_dft;
    elasticModulusLongitudinal = elasticModulusLongitudinal_dft;
    elasticModulusTransverse   = elasticModulusTransverse_dft;
    shearModulusOutPlane       = shearModulusOutPlane_dft;
    poissonRatioInPlane        = poissonRatioInPlane_dft;
    poissonRatioOutPlane       = poissonRatioOutPlane_dft;

}

void TransverseIsotropicElasticityTensor
::setEulerAngle(const double a1_In,
                const double a2_In,
                const double a3_In){
    eulerAngle[0] = a1_In;
    eulerAngle[1] = a2_In;
    eulerAngle[2] = a3_In;
}

void TransverseIsotropicElasticityTensor
::setElasticModulusLongitudinal(const double elasticModulusLongitudinal_In){
    elasticModulusLongitudinal = elasticModulusLongitudinal_In;
}

void TransverseIsotropicElasticityTensor
::setElasticModulusTransverse(const double elasticModulusTransverse_In){
    elasticModulusTransverse = elasticModulusTransverse_In;
}

void TransverseIsotropicElasticityTensor
::setShearModulusOutPlane(const double shearModulusOutPlane_In){
    shearModulusOutPlane = shearModulusOutPlane_In;
}

void TransverseIsotropicElasticityTensor
::setPoissonRatioInPlane(const double poissonRatioInPlane_In){
    poissonRatioInPlane = poissonRatioInPlane_In;
}

void TransverseIsotropicElasticityTensor
::setPoissonRatioOutPlane(const double poissonRatioOutPlane_In){
    poissonRatioOutPlane = poissonRatioOutPlane_In;
}



void TransverseIsotropicElasticityTensor::printMaterialConstants(){
    std::cout << "The angles between material and global coordinates are: " << std::endl;
    std::cout << "eulerAngle[0] = " << eulerAngle[0] << std::endl;
    std::cout << "eulerAngle[1] = " << eulerAngle[1] << std::endl;
    std::cout << "eulerAngle[2] = " << eulerAngle[2] << std::endl;

    std::cout << "The longitudinal elastic modulus is: " << std::endl;
    std::cout << "EL = " << elasticModulusLongitudinal << std::endl;

    std::cout << "The transverse elastic modulus is: " << std::endl;
    std::cout << "ET = " << elasticModulusTransverse << std::endl;

    std::cout << "The shear modulus out plane is: " << std::endl;
    std::cout << "Gxz or Gyz = " << shearModulusOutPlane << std::endl;

    std::cout << "The in-plane Poisson's ratio is: " << std::endl;
    std::cout << "Vxy = " << poissonRatioInPlane << std::endl;

    std::cout << "The out-plane Poisson's ratio is: " << std::endl;
    std::cout << "Vzx or Vzy = " << poissonRatioOutPlane << std::endl;
}

void TransverseIsotropicElasticityTensor::calculateElasticityTensor(){

    Eigen::MatrixXd complianceMatrix(6,6), elasticMatrix(6,6);

    FTensor::Tensor2<double, 3, 3> rotationalTensor; // local to global
    FTensor::Tensor4<double, 3, 3, 3, 3> elasticTensor_Local;
    FTensor::Tensor4<double, 3, 3, 3, 3> elasticTensor_temp;

    FTensor::Index<'i', 3> i;
    FTensor::Index<'j', 3> j;
    FTensor::Index<'k', 3> k;
    FTensor::Index<'l', 3> l;
    FTensor::Index<'p', 3> p;
    FTensor::Index<'q', 3> q;
    FTensor::Index<'r', 3> r;
    FTensor::Index<'s', 3> s;

    rotationalTensor(i,j)         = 0.0;
    elasticTensor_Local(i,j,k,l)  = rotationalTensor(i,j)
                                  * rotationalTensor(k,l);   // initialize to 0
    elasticTensor_Global(i,j,k,l) = rotationalTensor(i,j)
                                  * rotationalTensor(k,l);   // initialize to 0


    /**
     * initialize and calculate the compliance matrix
     */
    for(int ii=0; ii<6; ++ii){
        for(int jj=0; jj<6; ++jj){
            complianceMatrix(ii,jj) = 0.0;
        }
    }

    complianceMatrix(0,0) = 1.0 / elasticModulusTransverse;
    complianceMatrix(1,1) = 1.0 / elasticModulusTransverse;
    complianceMatrix(2,2) = 1.0 / elasticModulusLongitudinal;
    complianceMatrix(3,3) = 0.5 / shearModulusOutPlane;
    complianceMatrix(4,4) = 0.5 / shearModulusOutPlane;
    complianceMatrix(5,5) = 1.0 * (1+poissonRatioInPlane) / elasticModulusTransverse;
    complianceMatrix(0,1) = - poissonRatioInPlane  / elasticModulusTransverse;
    complianceMatrix(1,0) = complianceMatrix(0,1);
    complianceMatrix(0,2) = - poissonRatioOutPlane / elasticModulusLongitudinal;
    complianceMatrix(2,0) = complianceMatrix(0,2);
    complianceMatrix(1,2) = - poissonRatioOutPlane / elasticModulusLongitudinal;
    complianceMatrix(2,1) = complianceMatrix(1,2);

    elasticMatrix = complianceMatrix.inverse();

    // std::cout << complianceMatrix << std::endl;

    elasticTensor_Local(0,0,0,0) = elasticMatrix(0,0);   // 11 -> 1111
    elasticTensor_Local(1,1,1,1) = elasticMatrix(1,1);   // 22 -> 2222
    elasticTensor_Local(2,2,2,2) = elasticMatrix(2,2);   // 33 -> 3333
    elasticTensor_Local(1,2,1,2) = elasticMatrix(3,3);   // 44 -> 2323
    elasticTensor_Local(0,2,0,2) = elasticMatrix(4,4);   // 55 -> 1313
    elasticTensor_Local(0,1,0,1) = elasticMatrix(5,5);   // 66 -> 1212
    elasticTensor_Local(0,0,1,1) = elasticMatrix(0,1);   // 12 -> 1122
    elasticTensor_Local(1,1,0,0) = elasticMatrix(1,0);   // 21 -> 2211
    elasticTensor_Local(0,0,2,2) = elasticMatrix(0,2);   // 13 -> 1133
    elasticTensor_Local(2,2,0,0) = elasticMatrix(2,0);   // 31 -> 3311
    elasticTensor_Local(1,1,2,2) = elasticMatrix(1,2);   // 23 -> 2233
    elasticTensor_Local(2,2,1,1) = elasticMatrix(2,1);   // 32 -> 3322

    elasticTensor_Local(2,1,2,1) = elasticTensor_Local(1,2,1,2);
    elasticTensor_Local(1,0,1,0) = elasticTensor_Local(0,1,0,1);
    elasticTensor_Local(2,0,2,0) = elasticTensor_Local(0,2,0,2);
    /**
     * Global to local:
     *                         R = z_alpha * x_beta * z_gamma * r
     *         Refer: https://en.wikipedia.org/wiki/Euler_angles 
     * 
     *         inverse or transverse of the matrix in reference
     */
    double alpha = eulerAngle[0] * (M_PI / 180.0);  // Angle between x-axis and N-axis
    double beta  = eulerAngle[1] * (M_PI / 180.0);  // Angle between z-axis and X-axis
    double gamma = eulerAngle[2] * (M_PI / 180.0);  // Angle between N-axis and X-axis

    double c1 = cos(alpha);
    double c2 = cos(beta );
    double c3 = cos(gamma);
    double s1 = sin(alpha);
    double s2 = sin(beta );
    double s3 = sin(gamma);

    rotationalTensor(0,0) =  c1*c3 - c2*s1*s3;
    rotationalTensor(0,1) =  c3*s1 + c1*c2*s3;
    rotationalTensor(0,2) =  s2*s3;           
    rotationalTensor(1,0) = -c1*s3 - c2*c3*s1;
    rotationalTensor(1,1) =  c1*c2*c3 - s1*s3;
    rotationalTensor(1,2) =  c3*s2;           
    rotationalTensor(2,0) =  s1*s2;           
    rotationalTensor(2,1) = -c1*s2;           
    rotationalTensor(2,2) =  c2;

    /** 
     *        D_ijkl = A_ip * A_jq * A_kr * A_ls * D_pqrs
     */
    elasticTensor_Global(p,q,r,l) = rotationalTensor(l,s)*elasticTensor_Local(p,q,r,s);
    elasticTensor_temp = elasticTensor_Global;
    elasticTensor_Global(p,q,k,l) = rotationalTensor(k,r)*elasticTensor_temp(p,q,r,l);
    elasticTensor_temp = elasticTensor_Global;
    elasticTensor_Global(p,j,k,l) = rotationalTensor(j,q)*elasticTensor_temp(p,q,k,l);
    elasticTensor_temp = elasticTensor_Global;
    elasticTensor_Global(i,j,k,l) = rotationalTensor(i,p)*elasticTensor_temp(p,j,k,l);

    // std::cout << elasticTensor_Global << std::endl;
}

FTensor::Tensor4<double, 3, 3, 3, 3>
TransverseIsotropicElasticityTensor::getElasticityTensor(){

    return elasticTensor_Global;
    
}