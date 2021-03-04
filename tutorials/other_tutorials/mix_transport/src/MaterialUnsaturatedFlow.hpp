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
    thetaM = 0.38;
    thetaR = 0.068;
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
  double Ah;   ///< Initial hydraulic head coefficient
  double AhZ;  ///< Initial hydraulic head coefficient
  double AhZZ; ///< Initial hydraulic head coefficient

  double initialPcEval() const { return Ah + AhZ * z + AhZZ * z * z; }

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
        po::value<double>(&AhZZ)->default_value(AhZZ))(
        (prefix + ".scaleZ").c_str(),
        po::value<double>(&scaleZ)->default_value(scaleZ));
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
    PetscPrintf(PETSC_COMM_WORLD, "scaleZ=%6.4g\n", scaleZ);
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
    return pow(1 - pow(SeStar, 1 / m), m);
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
    h = -1 - hS;
    trace_on(2 * blockId + 0, true);
    ah <<= h;
    const double m = 1 - 1 / n;
    aTheta = funTheta(ah, m);
    double r_theta;
    aTheta >>= r_theta;
    trace_off();
  }

  virtual void recordKr() {
    h = -1 - hS;
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
    MoFEMFunctionReturnHot(0);
  }
};

} // namespace MixTransport

#endif //__MATERIALUNSATURATEDFLOW_HPP__
