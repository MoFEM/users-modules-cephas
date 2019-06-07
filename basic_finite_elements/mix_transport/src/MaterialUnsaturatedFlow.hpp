/** \file MaterialUnsaturatedFlow.hpp
 * \brief Mix implementation of transport element
 *
 * \ingroup mofem_mix_transport_elem
 *
 */

#ifndef __MATERIALUNSATURATEDFLOW_HPP__
#define __MATERIALUNSATURATEDFLOW_HPP__

namespace MixTransport {

struct CommonMaterialData : public GenericMaterial {

  CommonMaterialData() {
    blockId = 0;
    matName = "SimpleDarcy";
    sCale = 1;
    thetaS = 0.38;
    thetaR = 0.068;
    thetaM = 0.068;
    alpha = 0.8;
    n = 1.09;
    Ks = 0.048;
    hS = 0;
    Ah = 0.;
    AhZ = 0.;
    AhZZ = 0;
  }

  int blockId;         ///< Block Id
  std::string matName; ///< material name

  double Ks;     ///< Saturated hydraulic conductivity [m/day]
  double hS;     ///< minimum capillary height [m]
  double thetaS; ///< saturated water content
  double thetaR; ///< residual water contents
  double thetaM; ///< model parameter
  double alpha;  ///< model parameter
  double n;      ///< model parameter

  // Initial hydraulic head
  //
  double Ah;   ///< Initial hydraulic head coefficient
  double AhZ;  ///< Initial hydraulic head coefficient
  double AhZZ; ///< Initial hydraulic head coefficient

  double initalPcEval() const { return Ah + AhZ * z + AhZZ * z * z; }

  void addOptions(po::options_description &o, const std::string &prefix) {
    o.add_options()((prefix + ".material_name").c_str(),
                    po::value<std::string>(&matName)->default_value(matName))(
        (prefix + ".ePsilon0").c_str(),
        po::value<double>(&ePsilon0)->default_value(ePsilon0))(
        (prefix + ".ePsilon1").c_str(),
        po::value<double>(&ePsilon1)->default_value(ePsilon1))(
        (prefix + ".sCale").c_str(),
        po::value<double>(&sCale)->default_value(sCale))(
        (prefix + ".thetaS").c_str(),
        po::value<double>(&thetaS)->default_value(thetaS))(
        (prefix + ".thetaR").c_str(),
        po::value<double>(&thetaR)->default_value(thetaR))(
        (prefix + ".thetaM").c_str(),
        po::value<double>(&thetaM)->default_value(thetaM))(
        (prefix + ".alpha").c_str(),
        po::value<double>(&alpha)->default_value(alpha))(
        (prefix + ".n").c_str(), po::value<double>(&n)->default_value(n))(
        (prefix + ".hS").c_str(), po::value<double>(&hS)->default_value(hS))(
        (prefix + ".Ks").c_str(), po::value<double>(&Ks)->default_value(Ks))(
        (prefix + ".Ah").c_str(), po::value<double>(&Ah)->default_value(Ah))(
        (prefix + ".AhZ").c_str(), po::value<double>(&AhZ)->default_value(AhZ))(
        (prefix + ".AhZZ").c_str(),
        po::value<double>(&AhZZ)->default_value(AhZZ));
  }

  void printMatParameters(const int id, const std::string &prefix) const {
    PetscPrintf(PETSC_COMM_WORLD, "Mat name %s-%s block id %d\n",
                prefix.c_str(), matName.c_str(), id);
    PetscPrintf(PETSC_COMM_WORLD, "Material name: %s\n", matName.c_str());
    PetscPrintf(PETSC_COMM_WORLD, "thetaS=%6.4g\n", thetaS);
    PetscPrintf(PETSC_COMM_WORLD, "thetaR=%6.4g\n", thetaR);
    PetscPrintf(PETSC_COMM_WORLD, "thetaM=%6.4g\n", thetaM);
    PetscPrintf(PETSC_COMM_WORLD, "alpha=%6.4g\n", alpha);
    PetscPrintf(PETSC_COMM_WORLD, "n=%6.4g\n", n);
    PetscPrintf(PETSC_COMM_WORLD, "hS=%6.4g\n", hS);
    PetscPrintf(PETSC_COMM_WORLD, "Ks=%6.4g\n", Ks);
    PetscPrintf(PETSC_COMM_WORLD, "Ah=%6.4g\n", Ah);
    PetscPrintf(PETSC_COMM_WORLD, "AhZ=%6.4g\n", AhZ);
    PetscPrintf(PETSC_COMM_WORLD, "AhZZ=%6.4g\n", AhZZ);
    PetscPrintf(PETSC_COMM_WORLD, "ePsilon0=%6.4g\n", ePsilon0);
    PetscPrintf(PETSC_COMM_WORLD, "ePsilon1=%6.4g\n", ePsilon1);
    PetscPrintf(PETSC_COMM_WORLD, "sCale=%6.4g\n", sCale);
  }

  void evaluateThetaM() {
    // thetaM =  thetaR + (thetaS - thetaR) *
    //                               pow(1 + pow(-alpha * hS, n), (1. - 1. / n));
}

  typedef boost::function<boost::shared_ptr<CommonMaterialData>(
      const CommonMaterialData &data)>
      RegisterHook;
};

struct MaterialDarcy : public CommonMaterialData {

  static boost::shared_ptr<CommonMaterialData>
  createMatPtr(const CommonMaterialData &data) {
    return boost::shared_ptr<CommonMaterialData>(new MaterialDarcy(data));
  }

  MaterialDarcy(const CommonMaterialData &data) : CommonMaterialData(data) {}

  MoFEMErrorCode calK() {
    MoFEMFunctionBeginHot;
    K = Ks;
    MoFEMFunctionReturnHot(0);
  };

  MoFEMErrorCode calDiffK() {
    MoFEMFunctionBeginHot;
    diffK = 0;
    MoFEMFunctionReturnHot(0);
  };

  MoFEMErrorCode calC() {
    MoFEMFunctionBeginHot;
    C = ePsilon1;
    MoFEMFunctionReturnHot(0);
  }

  MoFEMErrorCode calDiffC() {
    MoFEMFunctionBeginHot;
    diffC = 0;
    MoFEMFunctionReturnHot(0);
  }

  MoFEMErrorCode calTheta() {
    MoFEMFunctionBeginHot;
    tHeta = thetaS;
    MoFEMFunctionReturnHot(0);
  }

  virtual MoFEMErrorCode calSe() {
    MoFEMFunctionBeginHot;
    Se = 1;
    MoFEMFunctionReturnHot(0);
  }

  MoFEMErrorCode calDiffDiffK() {
    MoFEMFunctionBeginHot;
    diffDiffK = 0;
    MoFEMFunctionReturnHot(0);
  };
};

struct MaterialWithAutomaticDifferentiation : public CommonMaterialData {
  MaterialWithAutomaticDifferentiation(const CommonMaterialData &data)
      : CommonMaterialData(data) {}

  template <typename TYPE> inline TYPE funSe(TYPE &theta) {
    return (theta - thetaR) / (thetaS - thetaR);
  }

  virtual void recordTheta() = 0;

  virtual void recordKr() = 0;

  double Kr;
  MoFEMErrorCode calK() {
    MoFEMFunctionBeginHot;
    if (h < hS) {
      int r = ::function(2 * blockId + 1, 1, 1, &h, &Kr);
      if (r < 0) {
        SETERRQ(PETSC_COMM_SELF, MOFEM_OPERATION_UNSUCCESSFUL,
                "ADOL-C function evaluation with error");
      }
      K = Ks * Kr;
    } else {
      K = Ks;
    }
    K += Ks * ePsilon0;
    MoFEMFunctionReturnHot(0);
  };

  double diffKr;
  MoFEMErrorCode calDiffK() {
    MoFEMFunctionBeginHot;
    if (h < hS) {
      diffK = 0;
      int r = ::gradient(2 * blockId + 1, 1, &h, &diffKr);
      if (r < 0) {
        SETERRQ(PETSC_COMM_SELF, MOFEM_OPERATION_UNSUCCESSFUL,
                "ADOL-C function evaluation with error");
      }
      diffK = Ks * diffKr;
    } else {
      diffK = 0;
    }
    MoFEMFunctionReturnHot(0);
  };

  MoFEMErrorCode calC() {
    MoFEMFunctionBeginHot;
    if (h < hS) {
      int r = ::gradient(2 * blockId + 0, 1, &h, &C);
      if (r < 0) {
        SETERRQ(PETSC_COMM_SELF, MOFEM_OPERATION_UNSUCCESSFUL,
                "ADOL-C function evaluation with error");
      }
      } else {
      C = 0;
    }
    C += ePsilon1;
    MoFEMFunctionReturnHot(0);
  }

  MoFEMErrorCode calDiffC() {
    MoFEMFunctionBeginHot;
    if (h < hS) {
      double v = 1;
      int r = ::hess_vec(2 * blockId + 0, 1, &h, &v, &diffC);
      if (r < 0) {
        SETERRQ(PETSC_COMM_SELF, MOFEM_OPERATION_UNSUCCESSFUL,
                "ADOL-C function evaluation with error");
      }
    } else {
      diffC = 0;
    }
    MoFEMFunctionReturnHot(0);
  }

  MoFEMErrorCode calTheta() {
    MoFEMFunctionBeginHot;
    if (h < hS) {
      int r = ::function(2 * blockId + 0, 1, 1, &h, &tHeta);
      if (r < 0) {
        SETERRQ(PETSC_COMM_SELF, MOFEM_OPERATION_UNSUCCESSFUL,
                "ADOL-C function evaluation with error");
      }
    } else {
      tHeta = thetaS;
    }
    MoFEMFunctionReturnHot(0);
  }

  MoFEMErrorCode calSe() {
    MoFEMFunctionBeginHot;
    if (h < hS) {
      int r = ::function(2 * blockId + 0, 1, 1, &h, &tHeta);
      Se = funSe(tHeta);
      if (r < 0) {
        SETERRQ(PETSC_COMM_SELF, MOFEM_OPERATION_UNSUCCESSFUL,
                "ADOL-C function evaluation with error");
      }
    } else {
      tHeta = thetaS;
      Se = 1.;
    }
    MoFEMFunctionReturnHot(0);
  }
};

struct MaterialVanGenuchten : public MaterialWithAutomaticDifferentiation {

  static boost::shared_ptr<CommonMaterialData>
  createMatPtr(const CommonMaterialData &data) {
    return boost::shared_ptr<CommonMaterialData>(
        new MaterialVanGenuchten(data));
  }

  MaterialVanGenuchten(const CommonMaterialData &data)
      : MaterialWithAutomaticDifferentiation(data) {
    recordTheta();
    recordKr();
  }

  adouble ah;
  adouble aTheta;
  adouble aKr;
  adouble aSe;
  adouble aSeStar;

  template <typename TYPE> inline TYPE funTheta(TYPE &h, const double m) {
    return thetaR + (thetaM - thetaR) / pow(1 + pow(-alpha * h, n), m);
  }

  template <typename TYPE>
  inline TYPE funFunSeStar(TYPE &SeStar, const double m) {
    return pow(1 - pow(SeStar, 1. / m), m);
  }

  inline adouble funKr(adouble &ah) {
    const double m = 1 - 1 / n;
    aTheta = funTheta(ah, m);
    aSe = funSe(aTheta);
    aSeStar = aSe * (thetaS - thetaR) / (thetaM - thetaR);
    double one = 1;
    const double c = funFunSeStar<double>(one, m);
    return sqrt(aSe) *
           pow((1 - funFunSeStar<adouble>(aSeStar, m)) / (1 - c), 2);
  }

  virtual void recordTheta() {
    h = -1-hS;
    trace_on(2 * blockId + 0, true);
    ah <<= h;
    const double m = 1 - 1 / n;
    aTheta = funTheta(ah, m);
    double r_theta;
    aTheta >>= r_theta;
    trace_off();
  }

  virtual void recordKr() {
    h = -1-hS;
    trace_on(2 * blockId + 1, true);
    ah <<= h;
    aKr = funKr(ah);
    double r_Kr;
    aKr >>= r_Kr;
    trace_off();
  }

  void printTheta(const double b, const double e, double s,
                  const std::string &prefix) {
    const double m = 1 - 1 / n;
    h = b;
    for (; h >= e; h += s) {
      s = -pow(-s, 0.9);
      double theta = funTheta(h, m);
      double Se = (theta - thetaR) / (thetaS - thetaR);
      PetscPrintf(PETSC_COMM_SELF, "%s %6.4e %6.4e %6.4e\n", prefix.c_str(), h,
                  theta, Se);
    }
  }

  void printKappa(const double b, const double e, double s,
                  const std::string &prefix) {
    h = b;
    for (; h >= e; h += s) {
      s = -pow(-s, 0.9);
      calK();
      PetscPrintf(PETSC_COMM_SELF, "%s %6.4e %6.4e %6.4e\n", prefix.c_str(), h,
                  Kr, K);
    }
  }

  void printC(const double b, const double e, double s,
              const std::string &prefix) {
    h = b;
    for (; h >= e; h += s) {
      s = -pow(-s, 0.9);
      calC();
      PetscPrintf(PETSC_COMM_SELF, "%s %6.4e %6.4e\n", prefix.c_str(), h, C);
    }
  }
};

struct MaterialModifiedVanG : public MaterialWithAutomaticDifferentiation {

  static boost::shared_ptr<CommonMaterialData>
  createMatPtr(const CommonMaterialData &data) {
    return boost::shared_ptr<CommonMaterialData>(
        new MaterialModifiedVanG(data));
  }

  MaterialModifiedVanG(const CommonMaterialData &data)
      : MaterialWithAutomaticDifferentiation(data) {
    thetaM = thetaR +
             (thetaS - thetaR) * pow(1 + pow(-alpha * hS, n), (1. - 1. / n));
    recordTheta();
    recordKr();
  }

  adouble ah;
  adouble aTheta;
  adouble aKr;
  adouble aSe;
  adouble aSeStar;

  double Kr;
  MoFEMErrorCode calK() {
    MoFEMFunctionBeginHot;
    if (h < hS) {
      double m = 1. - 1. / n;
      double p = 1. / (pow(-alpha * h, n) + 1) ;
      double ratio = pow((thetaS - thetaR) / (thetaM - thetaR), 1. / m);
      double num1 = 1 - pow( 1 - ratio * p , m);
      double num2 = 1 - pow(1 - ratio, m);
      Kr = pow(p, 0.5 * m) * pow(num1 / num2, 2.);
      K = Ks * Kr;
    } else {
      K = Ks;
    }
    //K += Ks * ePsilon0;
    MoFEMFunctionReturnHot(0);
  };

  double diffKr;
  MoFEMErrorCode calDiffK() {
    MoFEMFunctionBeginHot;
    if (h < hS) {

      double m = 1. - 1. / n;
      double p = pow(-alpha * h, n);
      double ratio = pow( (thetaS - thetaR) / (thetaM - thetaR), 1./m );
      double denom = 2. * h * (p - ratio + 1) * pow( pow(1 - ratio, m) -1. , 2.);
      double mult1 = - m*n*p*pow(p+1., -0.5*m - 1. );
      double mult2 = pow(1 - ratio / (p+1), m) - 1.;
      double mult3_1 = (p - 5.*ratio + 1) * pow( 1 - ratio / (p +1.), m );
      double mult3_2 = - p + ratio - 1.;
      double mult3 = mult3_1 + mult3_2;
      diffKr = mult1 * mult2 * mult3 / denom;
      
        // if (Kr < 1.e-6) {
        //   diffKr = 1.e-6;
        // }
      //printf("See???  h is: %e   and diffK is: %e\n", h, diffK);
      diffK = Ks * diffKr;
    } else {
      diffK = 0.;
    }
    MoFEMFunctionReturnHot(0);
  };

  //double diffDiffK;
  MoFEMErrorCode calDiffDiffK() {
    
    double m = 1. - 1. / n;
    double f1 = (thetaS - thetaR) / (thetaM - thetaR);

    const double diff_diff =
        (pow(-(alpha * h), n) * m * n *
         (-2 * (1 + pow(-(alpha * h), n)) *
              pow(pow(1 + pow(-(alpha * h), n), -m), 0.5) *
              pow(f1 / pow(1 + pow(-(alpha * h), n), m), 1 / m) *
              pow(1 - pow(f1 / pow(1 + pow(-(alpha * h), n), m), 1 / m),
                    -1 + m) *
              (1 - pow(1 - pow(f1 / pow(1 + pow(-(alpha * h), n), m),
                                   1 / m),
                         m)) *
              (-1 + n) -
          0.5 * pow(pow(1 + pow(-(alpha * h), n), -m), 0.5) *
              (1. + pow(-(alpha * h), n)) *
              pow(-1. + pow(1 - pow(f1 / pow(1 + pow(-(alpha * h), n),
                                                     m),
                                          1 / m),
                                m),
                    2) *
              (-1. + n) -
          2 * pow(-(alpha * h), n) *
              pow(pow(1 + pow(-(alpha * h), n), -m), 0.5) *
              pow(f1 / pow(1 + pow(-(alpha * h), n), m), 1 / m) *
              pow(1 - pow(f1 / pow(1 + pow(-(alpha * h), n), m), 1 / m),
                    -1 + m) *
              (1 - pow(1 - pow(f1 / pow(1 + pow(-(alpha * h), n), m),
                                   1 / m),
                         m)) *
              (-1 - m) * n +
          2 * pow(-(alpha * h), n) *
              pow(pow(1 + pow(-(alpha * h), n), -m), 0.5) *
              pow(f1 / pow(1 + pow(-(alpha * h), n), m), 1 / m) *
              pow(1 - pow(f1 / pow(1 + pow(-(alpha * h), n), m), 1 / m),
                    -1 + m) *
              (1 - pow(1 - pow(f1 / pow(1 + pow(-(alpha * h), n), m),
                                   1 / m),
                         m)) *
              (1 - m) * n -
          2 * pow(-(alpha * h), n) *
              pow(pow(1 + pow(-(alpha * h), n), -m), 0.5) *
              pow(f1 / pow(1 + pow(-(alpha * h), n), m), 2 / m) *
              pow(1 - pow(f1 / pow(1 + pow(-(alpha * h), n), m), 1 / m),
                    -2 + m) *
              (1 - pow(1 - pow(f1 / pow(1 + pow(-(alpha * h), n), m),
                                   1 / m),
                         m)) *
              (-1 + m) * n +
          2 * pow(-(alpha * h), n) *
              pow(pow(1 + pow(-(alpha * h), n), -m), 0.5) *
              pow(f1 / pow(1 + pow(-(alpha * h), n), m), 2 / m) *
              pow(1 - pow(f1 / pow(1 + pow(-(alpha * h), n), m), 1 / m),
                    2 * (-1 + m)) *
              m * n +
          2. * pow(-(alpha * h), n) *
              pow(pow(1 + pow(-(alpha * h), n), -m), 0.5) *
              pow(f1 / pow(1 + pow(-(alpha * h), n), m), 1 / m) *
              pow(1 - pow(f1 / pow(1 + pow(-(alpha * h), n), m), 1 / m),
                    -1 + m) *
              (1 - pow(1 - pow(f1 / pow(1 + pow(-(alpha * h), n), m),
                                   1 / m),
                         m)) *
              m * n -
          (0.25 * pow(-(alpha * h), n) *
           pow(-1. +
                     pow(1 - pow(f1 / pow(1 + pow(-(alpha * h), n), m),
                                     1 / m),
                           m),
                 2) *
           m * n) /
              (pow(1 + pow(-(alpha * h), n), 2 * m) *
               pow(pow(1 + pow(-(alpha * h), n), -m), 1.5)) +
          0.5 * pow(-(alpha * h), n) *
              pow(pow(1 + pow(-(alpha * h), n), -m), 0.5) *
              pow(-1. + pow(1 - pow(f1 / pow(1 + pow(-(alpha * h), n),
                                                     m),
                                          1 / m),
                                m),
                    2) *
              (1. + m) * n)) /
        (pow(-1 + pow(1 - pow(f1, 1 / m), m), 2) * pow(h, 2) *
         pow(1 + pow(-(alpha * h), n), 2));
        
        diffDiffK = Ks * diff_diff;
    // } else {
    //   diffDiffK = 0.;
    // }

    MoFEMFunctionReturnHot(0);
    }

  MoFEMErrorCode calC() {
    MoFEMFunctionBeginHot;
    if (h < hS) {
      double m = 1. - 1. / n;
      double p = pow(-alpha * h, n);
      C = -m * n * (thetaS - thetaR) * p * pow(p + 1., -m - 1.) / h;
      //printf(">>>>>>>>>>>>>>>>calC<<<<<<<<<<---------  %e\n", C);
      // if (r < 0) {
      //   SETERRQ(PETSC_COMM_SELF, MOFEM_OPERATION_UNSUCCESSFUL,
      //           "ADOL-C function evaluation with error");
      // }
    } else {
      C = 0;
    }
    //C += ePsilon1;
    MoFEMFunctionReturnHot(0);
  }

  MoFEMErrorCode calDiffC() {
    MoFEMFunctionBeginHot;
    if (h < hS) {
      double m = 1. - 1. / n;
      double p = pow(-alpha * h, n);
      diffC = m * n * (thetaS - thetaR) * p * pow(p + 1., -m - 2.) *
              ((m * n + 1) * p - n + 1) / pow(h, 2.);
      //printf(">>>>>>>>>>>>>>>>calDiffC<<<<<<<<<<--------- %e\n", diffC);
      // if (r < 0) {
      //   SETERRQ(PETSC_COMM_SELF, MOFEM_OPERATION_UNSUCCESSFUL,
      //           "ADOL-C function evaluation with error");
      // }
    } else {
      diffC = 0;
    }
    MoFEMFunctionReturnHot(0);
  }

  template <typename TYPE> inline TYPE funTheta(TYPE &h, const double m) {
    std::cout << "thetaM now is:" << thetaM << "\n";
      return thetaR + (thetaM - thetaR) / pow(1 + pow(-alpha * h, n), m);
  }

  template <typename TYPE>
  inline TYPE funFunSeStar(TYPE &SeStar, const double m) {
    return pow(1 - pow(SeStar, 1. / m), m);
  }

  inline adouble funKr(adouble &ah) {
    const double m = 1 - 1 / n;
    std::cout << "thetaM now is 2:" << thetaM << "\n";
    aTheta = funTheta(ah, m);
    aSe = funSe(aTheta);
    aSeStar = aSe * (thetaS - thetaR) / (thetaM - thetaR);
    double one = 1;
    const double c = funFunSeStar<double>(one, m);
    return sqrt(aSe) *
           pow((1 - funFunSeStar<adouble>(aSeStar, m)) / (1. - c), 2);
  }

  virtual void recordTheta() {
    
    h = -1-hS;
    trace_on(2 * blockId + 0, true);
    ah <<= h;
    const double m = 1 - 1 / n;
    aTheta = funTheta(ah, m);
    double r_theta;
    aTheta >>= r_theta;
    trace_off();
  }

  virtual void recordKr() {
    h = -1-hS;
    trace_on(2 * blockId + 1, true);
    ah <<= h;
    aKr = funKr(ah);
    double r_Kr;
    aKr >>= r_Kr;
    trace_off();
  }

  void printTheta(const double b, const double e, double s,
                  const std::string &prefix) {
    const double m = 1 - 1 / n;
    h = b;
    for (; h >= e; h += s) {
      s = -pow(-s, 0.9);
      double theta = funTheta(h, m);
      double Se = (theta - thetaR) / (thetaS - thetaR);
      PetscPrintf(PETSC_COMM_SELF, "%s %6.4e %6.4e %6.4e\n", prefix.c_str(), h,
                  theta, Se);
    }
  }

  void printKappa(const double b, const double e, double s,
                  const std::string &prefix) {
    h = b;
    for (; h >= e; h += s) {
      s = -pow(-s, 0.9);
      calK();
      PetscPrintf(PETSC_COMM_SELF, "%s %6.4e %6.4e %6.4e\n", prefix.c_str(), h,
                  Kr, K);
    }
  }

  void printC(const double b, const double e, double s,
              const std::string &prefix) {
    h = b;
    for (; h >= e; h += s) {
      s = -pow(-s, 0.9);
      calC();
      PetscPrintf(PETSC_COMM_SELF, "%s %6.4e %6.4e\n", prefix.c_str(), h, C);
    }
  }
};

struct RegisterMaterials {
  static map<std::string, CommonMaterialData::RegisterHook>
      mapOfRegistredMaterials;
  MoFEMErrorCode operator()() const {
    MoFEMFunctionBeginHot;
    mapOfRegistredMaterials["SimpleDarcy"] = MaterialDarcy::createMatPtr;
    mapOfRegistredMaterials["VanGenuchten"] =
        MaterialVanGenuchten::createMatPtr;
    mapOfRegistredMaterials["ModifiedVanG"] =
        MaterialModifiedVanG::createMatPtr;
    MoFEMFunctionReturnHot(0);
  }
};

} // namespace MixTransport

#endif //__MATERIALUNSATURATEDFLOW_HPP__
