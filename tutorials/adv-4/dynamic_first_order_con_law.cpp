/**
 * \file dynamic_first_order_con_law.cpp
 * \example dynamic_first_order_con_law.cpp
 *
 * Explicit first order conservation laws for solid dynamics
 *
 */

#include <MoFEM.hpp>
#include <MatrixFunction.hpp>
using namespace MoFEM;

// template <int DIM> struct ElementsAndOps {};

// template <> struct ElementsAndOps<2> {
//   using DomainEle = FaceElementForcesAndSourcesCore;
// };

// template <> struct ElementsAndOps<3> {
//   using DomainEle = VolumeElementForcesAndSourcesCore;
// };


template <typename T> 
inline double trace(FTensor::Tensor2<T, 2, 2> &t_stress) {
  constexpr double third = boost::math::constants::third<double>();
  return (t_stress(0, 0) + t_stress(1, 1));
};

template <typename T>
inline double trace(FTensor::Tensor2<T, 3, 3> &t_stress) {
  constexpr double third = boost::math::constants::third<double>();
  return (t_stress(0, 0) + t_stress(1, 1) + t_stress(2, 2));
};

constexpr int SPACE_DIM =
    EXECUTABLE_DIMENSION; //< Space dimension of problem, mesh

using EntData = EntitiesFieldData::EntData;
using DomainEle = PipelineManager::ElementsAndOpsByDim<SPACE_DIM>::DomainEle;
using DomainEleOp = DomainEle::UserDataOperator;
using PostProcEle = PostProcBrokenMeshInMoab<DomainEle>;
using BoundaryEle =
    PipelineManager::ElementsAndOpsByDim<SPACE_DIM>::BoundaryEle;
  using BoundaryEleOp = BoundaryEle::UserDataOperator;
using SetPtsData = FieldEvaluatorInterface::SetPtsData;

using OpMassV = FormsIntegrators<DomainEleOp>::Assembly<PETSC>::BiLinearForm<
    GAUSS>::OpMass<1, SPACE_DIM>;
using OpMassF = FormsIntegrators<DomainEleOp>::Assembly<PETSC>::BiLinearForm<
    GAUSS>::OpMass<1, SPACE_DIM * SPACE_DIM>;

using OpInertiaForce = FormsIntegrators<DomainEleOp>::Assembly<
    PETSC>::LinearForm<GAUSS>::OpBaseTimesVector<1, SPACE_DIM, 1>;

using OpBodyForce = FormsIntegrators<DomainEleOp>::Assembly<PETSC>::LinearForm<
    GAUSS>::OpBaseTimesVector<1, SPACE_DIM, 0>;

using DomainNaturalBC =
    NaturalBC<DomainEleOp>::Assembly<PETSC>::LinearForm<GAUSS>;
using OpBodyForceVector =
    DomainNaturalBC::OpFlux<NaturalMeshsetTypeVectorScaling<BLOCKSET>, 1,
                            SPACE_DIM>;
                            
using OpGradTimesTensor2 = FormsIntegrators<DomainEleOp>::Assembly<
          AssemblyType::PETSC>::LinearForm<IntegrationType::GAUSS>::OpGradTimesTensor<1, SPACE_DIM, SPACE_DIM>;

using OpRhsTestPiola = FormsIntegrators<DomainEleOp>::Assembly<
    AssemblyType::PETSC>::LinearForm<IntegrationType::GAUSS>::OpBaseTimesVector<1, SPACE_DIM * SPACE_DIM, 1>;


using OpGradTimesPiola = FormsIntegrators<DomainEleOp>::Assembly<
          AssemblyType::PETSC>::LinearForm<IntegrationType::GAUSS>::OpGradTimesTensor<1, SPACE_DIM, SPACE_DIM>;


using BoundaryNaturalBC = NaturalBC<BoundaryEleOp>::Assembly<PETSC>::LinearForm<GAUSS>;
using OpForce = BoundaryNaturalBC::OpFlux<NaturalForceMeshsets, 1, SPACE_DIM>;



/** \brief Save field DOFS on vertices/tags
 */
template <int FIELD_DIM> 
struct InitialiseDeformationGradient : public MoFEM::DofMethod {

  MoFEM::Interface &mField;
  std::string tagName;
  InitialiseDeformationGradient(MoFEM::Interface &m_field, std::string tag_name)
      : mField(m_field), tagName(tag_name) {}

  Tag tH;

  MoFEMErrorCode preProcess() {
    MoFEMFunctionBegin;
    if (!fieldPtr) {
      SETERRQ(mField.get_comm(), MOFEM_DATA_INCONSISTENCY,
              "Null pointer, probably field not found");
    }
    // if (fieldPtr->getSpace() != H1) {
    //   SETERRQ(mField.get_comm(), MOFEM_DATA_INCONSISTENCY,
    //           "Field must be in H1 space");
    // }
    std::vector<double> def_vals(fieldPtr->getNbOfCoeffs(), 0);
    rval = mField.get_moab().tag_get_handle(tagName.c_str(), tH);
    if (rval != MB_SUCCESS) {
      CHKERR mField.get_moab().tag_get_handle(
          tagName.c_str(), fieldPtr->getNbOfCoeffs(), MB_TYPE_DOUBLE, tH,
          MB_TAG_CREAT | MB_TAG_SPARSE, &def_vals[0]);
    }

    MoFEMFunctionReturn(0);
  }

  MoFEMErrorCode postProcess() {
    MoFEMFunctionBegin;
    MoFEMFunctionReturn(0);
  }

  MoFEMErrorCode operator()() {
    MoFEMFunctionBegin;
    FTensor::Index<'i', SPACE_DIM> i;
    FTensor::Index<'j', SPACE_DIM> j;
    constexpr auto t_kd = FTensor::Kronecker_Delta<double>();
    // if (dofPtr->getEntType() != MBVERTEX)
    //   MoFEMFunctionReturnHot(0);
    EntityHandle ent = dofPtr->getEnt();
    int rank = dofPtr->getNbOfCoeffs();

    // double tag_val[rank];
    // CHKERR mField.get_moab().tag_get_data(tH, &ent, 1, tag_val);
    // auto t_F = getFTensor2FromPtr<FIELD_DIM,FIELD_DIM>(tag_val);
    // t_F(i, j) = t_kd(i, j);
    // CHKERR mField.get_moab().tag_set_data(tH, &ent, 1, &tag_val);
    
    

    Range nodes;
    CHKERR mField.get_moab().get_entities_by_type(0, dofPtr->getEntType(), nodes);
    int length;
    CHKERR mField.get_moab().tag_get_length(tH, length);

    std::vector<double> check(/*nodes.size() */ length);
    CHKERR mField.get_moab().tag_get_data(tH, &ent, 1, &*check.begin());
    cerr << "check " << check << "\n";
    cerr << "check[]" << check[0] << " "<< check[1] << " "<< check[2] << " "<< check[3] << " "<< check[4] << " "<< check[5] << " "<< check[6] << " "<< check[7] << " "<< check[8] << " " << "\n";

    std::vector<double> data(/*nodes.size() */ length);
    CHKERR mField.get_moab().tag_get_data(tH, &ent, 1, &*data.begin());
    auto t_F = getFTensor2FromPtr<FIELD_DIM, FIELD_DIM>(&*data.begin());
    t_F(i, j) = t_kd(i, j);
    cerr << "data " << data << " t_F " << t_F << "\n";
    
    CHKERR mField.get_moab().tag_set_data(tH, &ent, 1, &*data.begin());

    
    
    MoFEMFunctionReturn(0);
  }
};

template <int FIELD_DIM> struct OpCalculateExplicitMass : public DomainEleOp {
  OpCalculateExplicitMass(const std::string row_field_name,
                          const std::string col_field_name,
                          SmartPetscObj<Mat> explicit_mat,
                          ScalarFun beta = scalar_fun_one)
      : DomainEleOp(row_field_name, col_field_name, DomainEleOp::OPROWCOL),
        M(explicit_mat), betaCoeff(beta) {
          // sYmm = true;
          sYmm = false;
        }

  MoFEMErrorCode doWork(int row_side, int col_side, EntityType row_type,
                        EntityType col_type, EntData &row_data,
                        EntData &col_data) {

    MoFEMFunctionBegin;

    const int nb_row_dofs = row_data.getIndices().size();
    const int nb_col_dofs = col_data.getIndices().size();
    if (nb_row_dofs && nb_col_dofs) {
      const int nb_integration_pts = getGaussPts().size2();
      mat.resize(nb_row_dofs, nb_col_dofs, false);
      mat.clear();
      matLumped.resize(nb_row_dofs, nb_col_dofs, false);
      matLumped.clear();

      const double nb_row_base_functions = row_data.getN().size2();

      // get element volume
      const double vol = getMeasure();
      // get integration weights
      auto t_w = getFTensor0IntegrationWeight();
      // get base function gradient on rows
      auto t_row_base = row_data.getFTensor0N();
      // get coordinate at integration points
      auto t_coords = getFTensor1CoordsAtGaussPts();

      FTensor::Index<'i', FIELD_DIM> i;
      auto get_t_vec = [&](const int rr) {
        std::array<double *, FIELD_DIM> ptrs;
        for (auto i = 0; i != FIELD_DIM; ++i)
          ptrs[i] = &mat(rr + i, i);
        return FTensor::Tensor1<FTensor::PackPtr<double *, FIELD_DIM>,
                                FIELD_DIM>(ptrs);
      };

      // loop over integration points
      for (int gg = 0; gg != nb_integration_pts; gg++) {
        const double beta =
            vol * betaCoeff(t_coords(0), t_coords(1), t_coords(2));
        // take into account Jacobian
        const double alpha = t_w * beta;
        // loop over rows base functions
        int rr = 0;
        for (; rr != nb_row_dofs / FIELD_DIM; rr++) {
          // get column base functions gradient at gauss point gg
          auto t_col_base = col_data.getFTensor0N(gg, 0);
          // get mat vec
          auto t_vec = get_t_vec(FIELD_DIM * rr);
          // loop over columns
          for (int cc = 0; cc != nb_col_dofs / FIELD_DIM; cc++) {
            // calculate element of local matrix
            t_vec(i) += alpha * (t_row_base * t_col_base);
            ++t_col_base;
            ++t_vec;
          }
          ++t_row_base;
        }
        for (; rr < nb_row_base_functions; ++rr)
          ++t_row_base;
        ++t_coords;
        ++t_w; // move to another integration weight
      }

      // for (MatrixDouble::iterator1 it1 = mat.begin1(); it1 != mat.end1(); ++it1) {
      //   for (MatrixDouble::iterator2 it2 = it1.begin(); it2 != it1.end(); ++it2) {
      //     std::cout << "(" << it2.index1() << "," << it2.index2()
      //               << ") = " << *it2 << endl;
      //   }
      //   cout << endl;
      // }
            CHKERR MatSetValues(M, row_data, col_data, &mat(0,
            0),
                                ADD_VALUES);
            // if (row_side != col_side || row_type != col_type) {
            //   transMat.resize(nb_col_dofs, nb_row_dofs, false);
            //   noalias(transMat) = trans(mat);
            //   CHKERR MatSetValues(M, col_data, row_data,
            //   &transMat(0, 0),
            //                       ADD_VALUES);
            // }
    }
  MoFEMFunctionReturn(0);
};
private:
MatrixDouble mat, transMat;
MatrixDouble matLumped;
ScalarFun betaCoeff;
SmartPetscObj<Mat> M;
};

constexpr double rho = 1;
constexpr double omega = 1.;
constexpr double young_modulus = 1.;
constexpr double poisson_ratio = 0.;
double bulk_modulus_K = young_modulus / (3. * (1. - 2. * poisson_ratio));
double shear_modulus_G = young_modulus / (2. * (1. + poisson_ratio));
double mu = young_modulus/(2. * (1. + poisson_ratio));
double lamme_lambda = young_modulus * poisson_ratio /( (1. + poisson_ratio) * (1. - 2. * poisson_ratio) );

// #include <HenckyOps.hpp>
// using namespace HenckyOps;

static double *ts_time_ptr;
static double *ts_aa_ptr;

// Operator to Calculate F
template <int DIM_0, int DIM_1>
struct OpCalculateFStab : public ForcesAndSourcesCore::UserDataOperator {
  OpCalculateFStab(boost::shared_ptr<MatrixDouble> def_grad_ptr,
                   boost::shared_ptr<MatrixDouble> def_grad_stab_ptr,
                   boost::shared_ptr<MatrixDouble> def_grad_dot_ptr,
                   double tau_F_ptr,
                   boost::shared_ptr<MatrixDouble> grad_x_ptr,
                   boost::shared_ptr<MatrixDouble> grad_vel_ptr)
      : ForcesAndSourcesCore::UserDataOperator(NOSPACE, OPLAST),
        defGradPtr(def_grad_ptr), defGradStabPtr(def_grad_stab_ptr),
        defGradDotPtr(def_grad_dot_ptr), tauFPtr(tau_F_ptr),
        gradxPtr(grad_x_ptr), gradVelPtr(grad_vel_ptr) {}

  MoFEMErrorCode doWork(int side, EntityType type,
                        DataForcesAndSourcesCore::EntData &data) {
    MoFEMFunctionBegin;
    // Define Indicies
    FTensor::Index<'i', SPACE_DIM> i;
    FTensor::Index<'j', SPACE_DIM> j;

    // Number of Gauss points
    const size_t nb_gauss_pts = getGaussPts().size2();
   
    defGradStabPtr->resize(DIM_0 * DIM_1, nb_gauss_pts, false);
    defGradStabPtr->clear();

    constexpr auto t_kd = FTensor::Kronecker_Delta<double>();

    // Extract matrix from data matrix
    auto t_F = getFTensor2FromMat<SPACE_DIM, SPACE_DIM>(*defGradPtr);
    auto t_Fstab = getFTensor2FromMat<SPACE_DIM, SPACE_DIM>(*defGradStabPtr);
    auto t_F_dot = getFTensor2FromMat<SPACE_DIM, SPACE_DIM>(*defGradDotPtr);
    
    // tau_F = alpha deltaT
    auto tau_F = tauFPtr;
    double xi_F = 0.1;
    auto t_gradx = getFTensor2FromMat<SPACE_DIM, SPACE_DIM>(*gradxPtr);
    auto t_gradVel = getFTensor2FromMat<SPACE_DIM, SPACE_DIM>(*gradVelPtr);

    for (auto gg = 0; gg != nb_gauss_pts; ++gg) {
      // Stabilised Deformation Gradient
            t_Fstab(i, j) = t_F(i, j) +
                            tau_F * (t_gradVel(i, j) - t_F_dot(i, j)) +
                            xi_F * (t_gradx(i, j) - t_F(i, j) - t_kd(i, j));

            // if(sqrt(t_gradx(i,j)*t_gradx(i,j)) > 1.e-28)
            //   cerr << t_gradx <<"\n";

            ++t_F;
            ++t_Fstab;
            ++t_gradVel;
            ++t_F_dot;
            
            ++t_gradx;
    }

    MoFEMFunctionReturn(0);
  }
  private:
    double tauFPtr;
    boost::shared_ptr<MatrixDouble> defGradPtr;
    boost::shared_ptr<MatrixDouble> defGradStabPtr;
    boost::shared_ptr<MatrixDouble> defGradDotPtr;
    boost::shared_ptr<MatrixDouble> gradxPtr;
    boost::shared_ptr<MatrixDouble> gradVelPtr;
};


// Operator to Calculate P
  template <int DIM_0, int DIM_1>
  struct OpCalculatePiola : public ForcesAndSourcesCore::UserDataOperator {
    OpCalculatePiola(double shear_modulus, double bulk_modulus, double m_u, double lambda_lamme,
                     boost::shared_ptr<MatrixDouble> first_piola_ptr,
                     boost::shared_ptr<MatrixDouble> def_grad_ptr)
        : ForcesAndSourcesCore::UserDataOperator(NOSPACE, OPLAST),
          shearModulus(shear_modulus),
          bulkModulus(bulk_modulus), mU(m_u), lammeLambda(lambda_lamme), firstPiolaPtr(first_piola_ptr),
          defGradPtr(def_grad_ptr) {}

    MoFEMErrorCode doWork(int side, EntityType type,
                          DataForcesAndSourcesCore::EntData &data) {
      MoFEMFunctionBegin;
      // Define Indicies
      FTensor::Index<'i', SPACE_DIM> i;
      FTensor::Index<'j', SPACE_DIM> j;
      FTensor::Index<'k', SPACE_DIM> k;

      // Define Kronecker Delta
      constexpr auto t_kd = FTensor::Kronecker_Delta<double>();

      // Number of Gauss points
      const size_t nb_gauss_pts = getGaussPts().size2();

      // Resize Piola
      firstPiolaPtr->resize(DIM_0 * DIM_1, nb_gauss_pts, false);//ignatios check
      firstPiolaPtr->clear();

      // Extract matrix from data matrix
      auto t_P = getFTensor2FromMat<SPACE_DIM, SPACE_DIM>(*firstPiolaPtr);
      auto t_F = getFTensor2FromMat<SPACE_DIM, SPACE_DIM>(*defGradPtr);
      const double two_o_three = 2. / 3.;
      for (auto gg = 0; gg != nb_gauss_pts; ++gg) {

      // Calculate P- assuming neo-hookean
      //  t_P(i, j) =
      //     shearModulus * (t_F(i, j) + t_F(j, i)) +
      //     lammeLambda * ( trace(t_F) ) * t_kd(i, j);

      t_P(i, j) = 
       shearModulus * (t_F(i, j) + t_F(j, i) -  two_o_three * trace(t_F) * t_kd(i, j)) +
          bulkModulus * trace(t_F) * t_kd(i, j);
      
      // t_P(i, j) =
      //     mU * (t_F(i, j) + t_F(j, i) - two_o_three * trace(t_F) * t_kd(i, j)) +
      //     bulkModulus * (trace(t_F) - 3.) * t_kd(i, j);
      ++t_F;
      ++t_P;
      }

      MoFEMFunctionReturn(0);
    }

  private:
    double shearModulus;
    double bulkModulus;
    double mU;
    double lammeLambda;
    boost::shared_ptr<MatrixDouble> firstPiolaPtr;
    boost::shared_ptr<MatrixDouble> defGradPtr;
  };

template <int DIM_0, int DIM_1>
  struct OpCalculateTranspose : public ForcesAndSourcesCore::UserDataOperator {
    OpCalculateTranspose(boost::shared_ptr<MatrixDouble> in_ptr,
                     boost::shared_ptr<MatrixDouble> out_ptr)
        : ForcesAndSourcesCore::UserDataOperator(NOSPACE, OPLAST),
          inMat(in_ptr),
          outMat(out_ptr) {}

    MoFEMErrorCode doWork(int side, EntityType type,
                          DataForcesAndSourcesCore::EntData &data) {
      MoFEMFunctionBegin;
      // Define Indicies
      FTensor::Index<'i', SPACE_DIM> i;
      FTensor::Index<'j', SPACE_DIM> j;

      constexpr auto t_kd = FTensor::Kronecker_Delta<double>();

      // Number of Gauss points
      const size_t nb_gauss_pts = getGaussPts().size2();

      // Resize Piola
      outMat->resize(DIM_0 * DIM_1, nb_gauss_pts, false);
      outMat->clear();

      // Extract matrix from data matrix
      auto t_out = getFTensor2FromMat<SPACE_DIM, SPACE_DIM>(*outMat);
      auto t_in = getFTensor2FromMat<SPACE_DIM, SPACE_DIM>(*inMat);
      const double two_o_three = 2. / 3.;
      for (auto gg = 0; gg != nb_gauss_pts; ++gg) {

      t_out(j, i) = t_in(i, j);
      // if((t_out( i, j) * t_out( i, j)) > 1.e-12 )
      //   cerr << "t_in " << t_in << "\n";
      ++t_out;
      ++t_in;
      }

      MoFEMFunctionReturn(0);
    }

  private:
    boost::shared_ptr<MatrixDouble> inMat;
    boost::shared_ptr<MatrixDouble> outMat;
  };

template <int DIM_0, int DIM_1>
  struct OpCalculateDeformationGradient : public ForcesAndSourcesCore::UserDataOperator {
    OpCalculateDeformationGradient(boost::shared_ptr<MatrixDouble> def_grad_ptr,
                     boost::shared_ptr<MatrixDouble> grad_tensor_ptr)
        : ForcesAndSourcesCore::UserDataOperator(NOSPACE, OPLAST),
          defGradPtr(def_grad_ptr),
          gradTensorPtr(grad_tensor_ptr) {}

    MoFEMErrorCode doWork(int side, EntityType type,
                          DataForcesAndSourcesCore::EntData &data) {
      MoFEMFunctionBegin;
      // Define Indicies
      FTensor::Index<'i', SPACE_DIM> i;
      FTensor::Index<'j', SPACE_DIM> j;

      // Define Kronecker Delta
      constexpr auto t_kd = FTensor::Kronecker_Delta<double>();

      // Number of Gauss points
      const size_t nb_gauss_pts = getGaussPts().size2();

      // Resize Piola
      defGradPtr->resize(DIM_0 * DIM_1, nb_gauss_pts, false);
      defGradPtr->clear();

      // Extract matrix from data matrix
      auto t_F = getFTensor2FromMat<SPACE_DIM, SPACE_DIM>(*defGradPtr);
      auto t_H = getFTensor2FromMat<SPACE_DIM, SPACE_DIM>(*gradTensorPtr);
      for (auto gg = 0; gg != nb_gauss_pts; ++gg) {

      t_F(i, j) = t_H(i, j) + t_kd(i, j);

      ++t_F;
      ++t_H;
      }

      MoFEMFunctionReturn(0);
    }

  private:
    boost::shared_ptr<MatrixDouble> gradTensorPtr;
    boost::shared_ptr<MatrixDouble> defGradPtr;
  };

struct Example;
struct TSPrePostProc {

  TSPrePostProc() = default;
  virtual ~TSPrePostProc() = default;

  /**
   * @brief Used to setup TS solver
   *
   * @param ts
   * @return MoFEMErrorCode
   */
  MoFEMErrorCode tsSetUp(TS ts);

  // SmartPetscObj<VecScatter> getScatter(Vec x, Vec y, enum FR fr);
  SmartPetscObj<Vec> getSubVector();
  SmartPetscObj<Vec> T;

  SmartPetscObj<DM> solverSubDM;
  SmartPetscObj<Vec> globSol;
  Example *fsRawPtr;
  static MoFEMErrorCode tsPostStage(TS ts, PetscReal stagetime, PetscInt stageindex, Vec* Y);
  static MoFEMErrorCode tsPostStep(TS ts);
  static MoFEMErrorCode tsPreStep(TS ts);

private:
  /**
   * @brief Pre process time step
   *
   * Refine mesh and update fields
   *
   * @param ts
   * @return MoFEMErrorCode
   */
  // static MoFEMErrorCode tsPreProc(TS ts);

  /**
   * @brief Post process time step.
   *
   * Currently that function do not make anything major
   *
   * @param ts
   * @return MoFEMErrorCode
   */
  // static MoFEMErrorCode tsPostProc(TS ts);

  

  // static MoFEMErrorCode tsSetIFunction(TS ts, PetscReal t, Vec u, Vec u_t,
  //                                      Vec f,
  //                                      void *ctx); //< Wrapper for SNES Rhs
  // static MoFEMErrorCode tsSetIJacobian(TS ts, PetscReal t, Vec u, Vec u_t,
  //                                      PetscReal a, Mat A, Mat B,
  //                                      void *ctx); ///< Wrapper for SNES Lhs
  // static MoFEMErrorCode tsMonitor(TS ts, PetscInt step, PetscReal t, Vec u,
  //                                 void *ctx);      ///< Wrapper for TS monitor
  // static MoFEMErrorCode pcSetup(PC pc);
  // static MoFEMErrorCode pcApply(PC pc, Vec pc_f, Vec pc_x);

  SmartPetscObj<Vec> globRes; //< global residual
  SmartPetscObj<Mat> subB;    //< sub problem tangent matrix
  SmartPetscObj<KSP> subKSP;  //< sub problem KSP solver

  boost::shared_ptr<SnesCtx>
      snesCtxPtr; //< infernal data (context) for MoFEM SNES fuctions
  boost::shared_ptr<TsCtx>
      tsCtxPtr;   //<  internal data (context) for MoFEM TS functions.
};

static boost::weak_ptr<TSPrePostProc> tsPrePostProc;

struct LinMomTimeScale : public MoFEM::TimeScale {
    using MoFEM::TimeScale::TimeScale;
    double getScale(const double time) {
      return sin(2. * M_PI  * MoFEM::TimeScale::getScale(time));
    };
  };

struct CommonData {
  SmartPetscObj<Mat> M;   ///< Mass matrix
  SmartPetscObj<KSP> ksp; ///< Linear solver
};


struct Example {

  Example(MoFEM::Interface &m_field) : mField(m_field) {}

  MoFEMErrorCode runProblem();

private:
  MoFEM::Interface &mField;

  MoFEMErrorCode readMesh();
  MoFEMErrorCode setupProblem();
  MoFEMErrorCode boundaryCondition();
  MoFEMErrorCode assembleSystem();
  MoFEMErrorCode solveSystem();
  MoFEMErrorCode outputResults();
  MoFEMErrorCode checkResults();
  friend struct TSPrePostProc;

  struct DynamicFirstOrderConsTimeScale : public MoFEM::TimeScale {
    using MoFEM::TimeScale::TimeScale;
    double getScale(const double time) {
      // return 0.001 * sin( 0.1 * time);
      return 0.001;
    };
  };
};

//! [Run problem]
MoFEMErrorCode Example::runProblem() {
  MoFEMFunctionBegin;
  CHKERR readMesh();
  CHKERR setupProblem();
  CHKERR boundaryCondition();
  CHKERR assembleSystem();
  CHKERR solveSystem();
  CHKERR outputResults();
  CHKERR checkResults();
  MoFEMFunctionReturn(0);
}
//! [Run problem]

//! [Read mesh]
MoFEMErrorCode Example::readMesh() {
  MoFEMFunctionBegin;
  auto simple = mField.getInterface<Simple>();
  CHKERR simple->getOptions();
  CHKERR simple->loadFile();
  MoFEMFunctionReturn(0);
}
//! [Read mesh]

//! [Set up problem]
MoFEMErrorCode Example::setupProblem() {
  MoFEMFunctionBegin;
  Simple *simple = mField.getInterface<Simple>();
  enum bases { AINSWORTH, DEMKOWICZ, LASBASETOPT };
  const char *list_bases[LASBASETOPT] = {"ainsworth", "demkowicz"};
  PetscInt choice_base_value = AINSWORTH;
  CHKERR PetscOptionsGetEList(PETSC_NULL, NULL, "-base", list_bases,
                              LASBASETOPT, &choice_base_value, PETSC_NULL);

  FieldApproximationBase base;
  switch (choice_base_value) {
  case AINSWORTH:
    base = AINSWORTH_LEGENDRE_BASE;
    MOFEM_LOG("WORLD", Sev::inform)
        << "Set AINSWORTH_LEGENDRE_BASE for displacements";
    break;
  case DEMKOWICZ:
    base = DEMKOWICZ_JACOBI_BASE;
    MOFEM_LOG("WORLD", Sev::inform)
        << "Set DEMKOWICZ_JACOBI_BASE for displacements";
    break;
  default:
    base = LASTBASE;
    break;
  }
  // Add field
  CHKERR simple->addDomainField("V", H1, base, SPACE_DIM);
  CHKERR simple->addBoundaryField("V", H1, base, SPACE_DIM);
  CHKERR simple->addDomainField("F", H1, base,
                                SPACE_DIM * SPACE_DIM);
  CHKERR simple->addDataField("x_1", H1, base,
                              SPACE_DIM);
  CHKERR simple->addDataField("x_2", H1, base,
                              SPACE_DIM);
  CHKERR simple->addDataField("F_0", H1, base,
                                SPACE_DIM * SPACE_DIM);
  CHKERR simple->addDataField("F_dot", H1, base,
                                SPACE_DIM * SPACE_DIM);
                              
  CHKERR simple->addDataField("GEOMETRY", H1, base,
                              SPACE_DIM);
  int order = 2;
  CHKERR PetscOptionsGetInt(PETSC_NULL, "", "-order", &order, PETSC_NULL);
  CHKERR simple->setFieldOrder("V", order);
  CHKERR simple->setFieldOrder("F", order);
  CHKERR simple->setFieldOrder("F_0", order);
  CHKERR simple->setFieldOrder("F_dot", order);
  CHKERR simple->setFieldOrder("x_1", order); 
  CHKERR simple->setFieldOrder("x_2", order); 
  CHKERR simple->setFieldOrder("GEOMETRY", order);
  CHKERR simple->setUp();

  auto project_ho_geometry = [&]() {
    
    Projection10NodeCoordsOnField ent_method_x(mField, "x_1");
    CHKERR mField.loop_dofs("x_1", ent_method_x);
    Projection10NodeCoordsOnField ent_method_x_2(mField, "x_2");
    CHKERR mField.loop_dofs("x_2", ent_method_x_2);

    Projection10NodeCoordsOnField ent_method(mField, "GEOMETRY");
    return mField.loop_dofs("GEOMETRY", ent_method);

    
  };
  CHKERR project_ho_geometry();
  // InitialiseDeformationGradient<SPACE_DIM> ent_method_F(mField, "F");
  // mField.loop_dofs("F", ent_method_F);
  
  MoFEMFunctionReturn(0);
}
//! [Set up problem]

//! [Boundary condition]
MoFEMErrorCode Example::boundaryCondition() {
  MoFEMFunctionBegin;

  auto simple = mField.getInterface<Simple>();
  auto bc_mng = mField.getInterface<BcManager>();
  auto *pipeline_mng = mField.getInterface<PipelineManager>();
  auto time_scale = boost::make_shared<TimeScale>();

  pipeline_mng->getBoundaryExplicitRhsFE().reset();
  CHKERR AddHOOps<SPACE_DIM - 1, SPACE_DIM, SPACE_DIM>::add(
        pipeline_mng->getOpBoundaryExplicitRhsPipeline(), {NOSPACE}, "GEOMETRY");

  CHKERR BoundaryNaturalBC::AddFluxToPipeline<OpForce>::add(
      pipeline_mng->getOpBoundaryExplicitRhsPipeline(), mField, "V",
      {boost::make_shared<DynamicFirstOrderConsTimeScale>()}, "FORCE",
      Sev::inform);

  auto integration_rule = [](int, int, int approx_order) {
    return 8 * (approx_order);
  };

  CHKERR pipeline_mng->setBoundaryExplicitRhsIntegrationRule(integration_rule);
  CHKERR pipeline_mng->setDomainExplicitRhsIntegrationRule(integration_rule);


  CHKERR bc_mng->removeBlockDOFsOnEntities<DisplacementCubitBcData>(
      simple->getProblemName(), "V");

  auto get_pre_proc_hook = [&]() {
    return EssentialPreProc<DisplacementCubitBcData>(
        mField, pipeline_mng->getDomainExplicitRhsFE(),
        {boost::make_shared<DynamicFirstOrderConsTimeScale>()});
  };
  pipeline_mng->getDomainExplicitRhsFE()->preProcessHook = get_pre_proc_hook();

  MoFEMFunctionReturn(0);
}
//! [Boundary condition]

MoFEMErrorCode TSPrePostProc::tsPostStage(TS ts, PetscReal stagetime, PetscInt stageindex, Vec* Y) {
  MoFEMFunctionBegin;
// cerr << "tsPostStage " <<"\n";
if (auto ptr = tsPrePostProc.lock()) {

    int size;
    // CHKERR VecGetSize(ptr->T, &size);
    // CHKERR VecGhostUpdateBegin(ptr->T, INSERT_VALUES, SCATTER_FORWARD);
    // CHKERR VecGhostUpdateEnd(ptr->T, INSERT_VALUES, SCATTER_FORWARD);
    // CHKERR DMoFEMMeshToLocalVector(ptr->solverSubDM, ptr->T, INSERT_VALUES, SCATTER_REVERSE);
    
    // CHKERR VecView(ptr->T, PETSC_VIEWER_STDOUT_WORLD);
    

    auto &m_field = ptr->fsRawPtr->mField;
    auto fb = m_field.getInterface<FieldBlas>();
    double dt;
    CHKERR TSGetTimeStep(ts, &dt);
    double time;
    CHKERR TSGetTime(ts, &time);
    PetscPrintf(PETSC_COMM_WORLD, "Timestep %e time %e\n", dt, time);
    // double pseudo_time_step;
    // CHKERR TSPseudoComputeTimeStep(ts, &pseudo_time_step);
    
    // PetscPrintf(PETSC_COMM_WORLD, "Timestep %e time %e pseudo-time-step %e\n", dt, time, pseudo_time_step);
    //v = (x_t+1 - x_t) / Δt
    //x_t+1 = Δt * v + x_t 
    // cerr << "dt " << dt <<"\n";
    CHKERR fb->fieldCopy(1., "x_1", "x_2");
    CHKERR fb->fieldAxpy(dt, "V", "x_2");
    CHKERR fb->fieldCopy(1., "x_2", "x_1");
    
    CHKERR fb->fieldCopy(-1./dt, "F_0", "F_dot");
    CHKERR fb->fieldAxpy(1./dt, "F", "F_dot");
    CHKERR fb->fieldCopy(1., "F", "F_0");
}
  MoFEMFunctionReturn(0);
}

MoFEMErrorCode TSPrePostProc::tsPostStep(TS ts) {
  MoFEMFunctionBegin;

if (auto ptr = tsPrePostProc.lock()) {
    auto &m_field = ptr->fsRawPtr->mField;
    auto fb = m_field.getInterface<FieldBlas>();
    //find trajectory V and F
    //
    //x_t+1 = Δt * v + x_t 
    // CHKERR fb->fieldCopy(1., "x_2", "x_1");
}
  MoFEMFunctionReturn(0);
}

MoFEMErrorCode TSPrePostProc::tsPreStep(TS ts) {
  MoFEMFunctionBegin;

  if (auto ptr = tsPrePostProc.lock()) {
    auto &m_field = ptr->fsRawPtr->mField;
    auto *simple = m_field.getInterface<Simple>();
    auto *pipeline_mng = m_field.getInterface<PipelineManager>();
    ;

    double dt;
    CHKERR TSGetTimeStep(ts, &dt);
    double time;
    CHKERR TSGetTime(ts, &time);
    int step_num;
    CHKERR TSGetStepNumber(ts, &step_num);
    // TSTrajectory tj;
    // CHKERR TSGetTrajectory(ts, &tj);
    
    // auto T = createDMVector(simple->getDM());
    // Vec TT = smartVectorDuplicate(T);
    // CHKERR TSGetSolution(ts, &TT);
    
    // CHKERR VecGhostUpdateBegin(TT,
    //                            ADD_VALUES, SCATTER_REVERSE);
    // CHKERR VecGhostUpdateEnd(TT,
    //                          ADD_VALUES, SCATTER_REVERSE);
    // CHKERR VecAssemblyBegin(TT);
    // CHKERR VecAssemblyEnd(TT);
    
    // CHKERR VecCopy(TT, T);

    // CHKERR TSTrajectoryGetVecs(tj, ts, step_num, &time, T,
    //                            pipeline_mng->getDomainExplicitRhsFE()->ts_u_t);

    // // if (*(pipeline_mng->getDomainExplicitRhsFE()
    // //           ->vecAssembleSwitch)) {

    // CHKERR VecGhostUpdateBegin(pipeline_mng->getDomainExplicitRhsFE()->ts_u_t,
    //                            ADD_VALUES, SCATTER_REVERSE);
    // CHKERR VecGhostUpdateEnd(pipeline_mng->getDomainExplicitRhsFE()->ts_u_t,
    //                          ADD_VALUES, SCATTER_REVERSE);
    // CHKERR VecAssemblyBegin(pipeline_mng->getDomainExplicitRhsFE()->ts_u_t);
    // CHKERR VecAssemblyEnd(pipeline_mng->getDomainExplicitRhsFE()->ts_u_t);
    
  }
  MoFEMFunctionReturn(0);
}

//! [Push operators to pipeline]
MoFEMErrorCode Example::assembleSystem() {
  MoFEMFunctionBegin;
  auto *simple = mField.getInterface<Simple>();
  auto *pipeline_mng = mField.getInterface<PipelineManager>();
  

  auto get_body_force = [this](const double, const double, const double) {
    FTensor::Index<'i', SPACE_DIM> i;
    FTensor::Tensor1<double, SPACE_DIM> t_source;
    t_source(i) = 0.;
    t_source(0) = 0.1;
    t_source(1) = 1.;
    return t_source;
  };

  auto rho_ptr = boost::make_shared<double>(rho);

  auto add_rho_block = [this, rho_ptr](auto &pip, auto block_name, Sev sev) {
    MoFEMFunctionBegin;

    struct OpMatRhoBlocks : public DomainEleOp {
      OpMatRhoBlocks(boost::shared_ptr<double> rho_ptr,
                     MoFEM::Interface &m_field, Sev sev,
                     std::vector<const CubitMeshSets *> meshset_vec_ptr)
          : DomainEleOp(NOSPACE, DomainEleOp::OPSPACE), rhoPtr(rho_ptr) {
        CHK_THROW_MESSAGE(extractRhoData(m_field, meshset_vec_ptr, sev),
                          "Can not get data from block");
      }

      MoFEMErrorCode doWork(int side, EntityType type,
                            EntitiesFieldData::EntData &data) {

        MoFEMFunctionBegin;

        for (auto &b : blockData) {
          if (b.blockEnts.find(getFEEntityHandle()) != b.blockEnts.end()) {
            *rhoPtr = b.rho;
            MoFEMFunctionReturnHot(0);
          }
        }

        *rhoPtr = rho;

        MoFEMFunctionReturn(0);
      }

    private:
      struct BlockData {
        double rho;
        Range blockEnts;
      };

      std::vector<BlockData> blockData;

      MoFEMErrorCode
      extractRhoData(MoFEM::Interface &m_field,
                     std::vector<const CubitMeshSets *> meshset_vec_ptr,
                     Sev sev) {
        MoFEMFunctionBegin;

        for (auto m : meshset_vec_ptr) {
          MOFEM_TAG_AND_LOG("WORLD", sev, "Mat Rho Block") << *m;
          std::vector<double> block_data;
          CHKERR m->getAttributes(block_data);
          if (block_data.size() < 1) {
            SETERRQ(PETSC_COMM_SELF, MOFEM_DATA_INCONSISTENCY,
                    "Expected that block has two attributes");
          }
          auto get_block_ents = [&]() {
            Range ents;
            CHKERR
            m_field.get_moab().get_entities_by_handle(m->meshset, ents, true);
            return ents;
          };

          MOFEM_TAG_AND_LOG("WORLD", sev, "Mat Rho Block")
              << m->getName() << ": ro = " << block_data[0];

          blockData.push_back({block_data[0], get_block_ents()});
        }
        MOFEM_LOG_CHANNEL("WORLD");
        MOFEM_LOG_CHANNEL("WORLD");
        MoFEMFunctionReturn(0);
      }

      boost::shared_ptr<double> rhoPtr;
    };

  //   pip.push_back(new OpMatRhoBlocks(
  //       rho_ptr, mField, sev,

  //       // Get blockset using regular expression
  //       mField.getInterface<MeshsetsManager>()->getCubitMeshsetPtr(std::regex(

  //           (boost::format("%s(.*)") % block_name).str()

  //               ))

  //           ));
    MoFEMFunctionReturn(0);
  };

  // Get pointer to U_tt shift in domain element
  auto get_rho = [rho_ptr](const double, const double, const double) {
    return *rho_ptr;
  };

  // specific time scaling
  auto get_time_scale = [this](const double time) {
    return sin(time * omega * M_PI);
  };

  auto apply_rhs = [&](auto &pip, SmartPetscObj<Vec> &v_f_dot) {
    MoFEMFunctionBegin;

    CHKERR AddHOOps<SPACE_DIM, SPACE_DIM, SPACE_DIM>::add(pip, {H1},
                                                          "GEOMETRY");
    
    // Calculate unknown F
    auto mat_F_tensor_ptr = boost::make_shared<MatrixDouble>();
    pip.push_back(new OpCalculateTensor2FieldValues<SPACE_DIM, SPACE_DIM>(
        "F", mat_F_tensor_ptr));

    // auto mat_dot_F_tensor_ptr = boost::make_shared<MatrixDouble>();
    // pip.push_back(new OpCalculateTensor2FieldValuesDot<SPACE_DIM, SPACE_DIM>(
    //     "F", mat_dot_F_tensor_ptr));
    
    //  pip.push_back(new OpCalculateVectorFieldValuesDot<SPACE_DIM*SPACE_DIM>("F", mat_dot_F_tensor_ptr));

    // Calculate rate of F
    // auto mat_dot_F_tensor_ptr = boost::make_shared<MatrixDouble>();
    // pip.push_back(new OpCalculateTensor2FieldValuesDot<SPACE_DIM, SPACE_DIM>(
    //     "F", mat_dot_F_tensor_ptr));

    // Calculate Gradient of velocity
    auto mat_v_grad_ptr = boost::make_shared<MatrixDouble>();
    pip.push_back(new OpCalculateVectorFieldGradient<SPACE_DIM, SPACE_DIM>(
        "V", mat_v_grad_ptr));

  auto mat_v_grad_trans_ptr = boost::make_shared<MatrixDouble>();
  pip.push_back(new OpCalculateTranspose<SPACE_DIM, SPACE_DIM>(
      mat_v_grad_ptr, mat_v_grad_trans_ptr));

  
  auto gravity_vector_ptr = boost::make_shared<MatrixDouble>();
  gravity_vector_ptr->resize(SPACE_DIM, 1);
  auto set_body_force = [&]() {
    FTensor::Index<'i', SPACE_DIM> i;
    MoFEMFunctionBegin;
    auto t_force = getFTensor1FromMat<SPACE_DIM, 0>(*gravity_vector_ptr);
    double unit_weight = 1.;
    CHKERR PetscOptionsGetReal(PETSC_NULL, "", "-unit_weight", &unit_weight,
                               PETSC_NULL);
    t_force(i) = 0;
    if (SPACE_DIM == 2) {
      t_force(1) = -unit_weight;
    } else if (SPACE_DIM == 3) {
      t_force(2) = unit_weight;
    }
    MoFEMFunctionReturn(0);
  };

  // CHKERR set_body_force();
  // pip.push_back(new OpBodyForce(
  //   "V", gravity_vector_ptr, [](double, double, double) { return 1.; }));

  // some operator that calculates F^st
  //  pip.push_back(new OpCalculateVectorFieldGradient<SPACE_DIM, SPACE_DIM>(
  //      "x_2", mat_x_grad_ptr));
    
  //  // Calculate P
  //   auto mat_P_ptr = boost::make_shared<MatrixDouble>();
  //   pip.push_back(new OpCalculatePiola<SPACE_DIM, SPACE_DIM>(shear_modulus_G, bulk_modulus_K, mu, lamme_lambda, mat_P_ptr, mat_F_tensor_ptr));

  //  // Calculate F
   double tau = 0.2;
  
  // Calculate P stab
  
  auto one = [&](const double, const double, const double) {
    return 3. * bulk_modulus_K;
  };
  auto minus_one = [](const double, const double, const double) { return -1.; };

  // OpCalculateTensor2FieldValues_General(
  //   const std::string field_name,
  //   boost::shared_ptr<
  //       ublas::matrix<double, ublas::row_major, DoubleAllocator>>
  //       data_ptr,
  //   SmartPetscObj<Vec> data_vec, const EntityType zero_type = MBVERTEX)
  //   : ForcesAndSourcesCore::UserDataOperator(
  //         field_name, ForcesAndSourcesCore::UserDataOperator::OPROW),
  //     dataPtr(data_ptr), zeroType(zero_type), dataVec(data_vec) {
  auto mat_dot_F_tensor_ptr = boost::make_shared<MatrixDouble>();
    pip.push_back(new OpCalculateTensor2FieldValues<SPACE_DIM, SPACE_DIM>(
        "F_dot", mat_dot_F_tensor_ptr));

  // pip.push_back(new OpCalculateTensor2FieldValues<SPACE_DIM, SPACE_DIM>("F", mat_dot_F_tensor_ptr, v_f_dot));

  // Calculate Gradient of Spatial Positions
  auto mat_x_grad_ptr = boost::make_shared<MatrixDouble>();
  pip.push_back(
    new OpCalculateVectorFieldGradient<SPACE_DIM, SPACE_DIM>("x_2",
                                                             mat_x_grad_ptr));

  auto mat_F_stab_ptr = boost::make_shared<MatrixDouble>();
  pip.push_back(new OpCalculateFStab<SPACE_DIM, SPACE_DIM>(
      mat_F_tensor_ptr, mat_F_stab_ptr, mat_dot_F_tensor_ptr, tau,
      mat_x_grad_ptr, mat_v_grad_ptr));

auto mat_P_stab_ptr = boost::make_shared<MatrixDouble>();
  pip.push_back(new OpCalculatePiola<SPACE_DIM, SPACE_DIM>(
      shear_modulus_G, bulk_modulus_K, mu, lamme_lambda, mat_P_stab_ptr,
      mat_F_stab_ptr));

  pip.push_back(new OpGradTimesTensor2("V", mat_P_stab_ptr, minus_one));
  // pip.push_back(new OpGradTimesPiola("V", mat_P_stab_ptr, one));

  pip.push_back(new OpRhsTestPiola("F", mat_v_grad_ptr, one));

  // CHKERR add_rho_block(pip, "MAT_RHO", Sev::inform);

  // auto mat_acceleration_ptr = boost::make_shared<MatrixDouble>();
  // // Apply inertia
  // pip.push_back(new OpCalculateVectorFieldValuesDotDot<SPACE_DIM>(
  //     "V", mat_acceleration_ptr));
  // pip.push_back(new OpInertiaForce("V", mat_acceleration_ptr, get_rho));

  // CHKERR DomainNaturalBC::AddFluxToPipeline<OpBodyForceVector>::add(
  //     pip, mField, "V", {},
  //     {boost::make_shared<TimeScaleVector<SPACE_DIM>>("-time_vector_file",
  //                                                     true)},
  //     "BODY_FORCE", Sev::inform);

  MoFEMFunctionReturn(0);
  };

  
  
  // CHKERR apply_rhs(pipeline_mng->getCastExplicitDomainRhsFE());
  
  // pipeline_mng->getDomainExplicitRhsFE().reset();
  auto u_t = createDMVector(simple->getDM());
  pipeline_mng->getDomainExplicitRhsFE()->ts_ctx = TSMethod::CTX_TSSETIFUNCTION;
  pipeline_mng->getDomainExplicitRhsFE()->ts_u_t = u_t;
  pipeline_mng->getDomainExplicitRhsFE()->data_ctx = PetscData::CtxSetX_T;
  // auto u_t = smartVectorDuplicate(pipeline_mng->getDomainExplicitRhsFE()->ts_F);
  // Add hook to the element to calculate g.
  // pipeline_mng->getDomainExplicitRhsFE()->preProcessHook = [&]() {
  //   MoFEMFunctionBegin;
  //   // pipeline_mng->getDomainExplicitRhsFE()->ts_ctx = TSMethod::CTX_TSSETIFUNCTION;
    
    
  //   MoFEMFunctionReturn(0);
  // };

  CHKERR apply_rhs(pipeline_mng->getOpDomainExplicitRhsPipeline(), u_t);

  auto integration_rule = [](int, int, int approx_order) {
    return 8 * approx_order;
  };
  CHKERR pipeline_mng->setDomainExplicitRhsIntegrationRule(integration_rule);
  // CHKERR pipeline_mng->setDomainLhsIntegrationRule(integration_rule);

  // auto get_bc_hook = [&]() {
  //   EssentialPreProc<DisplacementCubitBcData> hook(
  //       mField, pipeline_mng->getDomainExplicitRhsFE(),
  //       {boost::make_shared<TimeScale>()});
  //   return hook;
  // };

  // pipeline_mng->getDomainExplicitRhsFE()->preProcessHook = get_bc_hook();

  MoFEMFunctionReturn(0);
}
//! [Push operators to pipeline]

/**
 * @brief Monitor solution
 *
 * This functions is called by TS solver at the end of each step. It is used
 * to output results to the hard drive.
 */
struct Monitor : public FEMethod {
  MoFEM::Interface &mField;
  Monitor(SmartPetscObj<DM> dm, MoFEM::Interface &m_field,
          boost::shared_ptr<PostProcEle> post_proc,
          boost::shared_ptr<MatrixDouble> pass_field_ptr,
          std::array<double, 3> pass_field_eval_coords,
          boost::shared_ptr<SetPtsData> pass_field_eval_data)
      : dM(dm), mField(m_field), postProc(post_proc), fieldPtr(pass_field_ptr),
        fieldEvalCoords(pass_field_eval_coords),
        fieldEvalData(pass_field_eval_data){};
  MoFEMErrorCode postProcess() {
    MoFEMFunctionBegin;

    // cerr << "wagawaga\n";
    auto *simple = mField.getInterface<Simple>();

    if (SPACE_DIM == 3) {
      CHKERR mField.getInterface<FieldEvaluatorInterface>()->evalFEAtThePoint3D(
          fieldEvalCoords.data(), 1e-12, simple->getProblemName(),
          simple->getDomainFEName(), fieldEvalData, mField.get_comm_rank(),
          mField.get_comm_rank(), nullptr, MF_EXIST, QUIET);
    } else {
      CHKERR mField.getInterface<FieldEvaluatorInterface>()->evalFEAtThePoint2D(
          fieldEvalCoords.data(), 1e-12, simple->getProblemName(),
          simple->getDomainFEName(), fieldEvalData, mField.get_comm_rank(),
          mField.get_comm_rank(), nullptr, MF_EXIST, QUIET);
    }

    if (fieldPtr->size1()) {
      // cerr << "ASDDASDASFSAGASGAS\n";
      auto t_p = getFTensor1FromMat<SPACE_DIM>(*fieldPtr);
      // PetscPrintf(PETSC_COMM_WORLD, "Velocities x: %e y: %e z: %e\n", t_p(0), t_p(1), t_p(2));
      // cerr << "Velocities x: " << t_p(0) << " y: " << t_p(1) << " z: " << t_p(2) <<"\n";
      // MOFEM_LOG("EXAMPLE", Sev::noisy)
      //     << "Velocities x: " << t_p(0) << " y: " << t_p(1) << " z: " << t_p(2);
    }


  for (auto m :
       mField.getInterface<MeshsetsManager>()->getCubitMeshsetPtr(std::regex(
           (boost::format("%s(.*)") % "Data_Vertex").str()
               ))
  ) {
    Range ents;
    mField.get_moab().get_entities_by_dimension(
        m->getMeshset(), 0, ents, true);
    auto print_vets = [](boost::shared_ptr<FieldEntity> ent_ptr) {
      MoFEMFunctionBegin;
      if(!(ent_ptr->getPStatus() & PSTATUS_NOT_OWNED)) {
        MOFEM_LOG("SYNC", Sev::inform) << "Velocities: " << ent_ptr->getEntFieldData()[0] << " " << ent_ptr->getEntFieldData()[1] << " " << ent_ptr->getEntFieldData()[2] << "\n";
      }
      MoFEMFunctionReturn(0);
    };
    CHKERR mField.getInterface<FieldBlas>()->fieldLambdaOnEntities(
        print_vets, "V", &ents);
    MOFEM_LOG_SEVERITY_SYNC(mField.get_comm(), Sev::inform);
  }

    constexpr int save_every_nth_step = 1;
    if (ts_step % save_every_nth_step == 0) {
      // CHKERR DMoFEMLoopFiniteElements(dM, "dFE", postProc);
      // CHKERR postProc->writeFile(
      //     "out_step_" + boost::lexical_cast<std::string>(ts_step) + ".h5m");
    }
    MoFEMFunctionReturn(0);
  }

private:
  SmartPetscObj<DM> dM;
  boost::shared_ptr<PostProcEle> postProc;
  boost::shared_ptr<MatrixDouble> fieldPtr;
  std::array<double, 3> fieldEvalCoords;
  boost::shared_ptr<SetPtsData> fieldEvalData;
};

//! [Solve]
MoFEMErrorCode Example::solveSystem() {
  MoFEMFunctionBegin;
  auto *simple = mField.getInterface<Simple>();
  auto *pipeline_mng = mField.getInterface<PipelineManager>();

  auto dm = simple->getDM();

  // Setup postprocessing
  auto post_proc_fe = boost::make_shared<PostProcEle>(mField);
  CHKERR AddHOOps<SPACE_DIM, SPACE_DIM, SPACE_DIM>::add(
      post_proc_fe->getOpPtrVector(), {H1});
  auto u_ptr = boost::make_shared<MatrixDouble>();
  post_proc_fe->getOpPtrVector().push_back(
      new OpCalculateVectorFieldValues<SPACE_DIM>("V", u_ptr));
  auto X_ptr = boost::make_shared<MatrixDouble>();
  post_proc_fe->getOpPtrVector().push_back(
      new OpCalculateVectorFieldValues<SPACE_DIM>("GEOMETRY", X_ptr));

  auto x_ptr = boost::make_shared<MatrixDouble>();
  post_proc_fe->getOpPtrVector().push_back(
      new OpCalculateVectorFieldValues<SPACE_DIM>("x_1", x_ptr));

  // Calculate unknown F
  auto mat_H_tensor_ptr = boost::make_shared<MatrixDouble>();
  post_proc_fe->getOpPtrVector().push_back(
      new OpCalculateTensor2FieldValues<SPACE_DIM, SPACE_DIM>(
          "F", mat_H_tensor_ptr));

  // Calculate P
  auto mat_P_ptr = boost::make_shared<MatrixDouble>();
  post_proc_fe->getOpPtrVector().push_back(
      new OpCalculatePiola<SPACE_DIM, SPACE_DIM>(
          shear_modulus_G, bulk_modulus_K, mu, lamme_lambda, mat_P_ptr, mat_H_tensor_ptr));

  auto mat_F_ptr = boost::make_shared<MatrixDouble>();
  post_proc_fe->getOpPtrVector().push_back(
      new OpCalculateDeformationGradient<SPACE_DIM, SPACE_DIM>(
          mat_F_ptr, mat_H_tensor_ptr));

  auto mat_v_grad_ptr = boost::make_shared<MatrixDouble>();
  post_proc_fe->getOpPtrVector().push_back(
      new OpCalculateVectorFieldGradient<SPACE_DIM, SPACE_DIM>("V",
                                                               mat_v_grad_ptr));

  using OpPPMap = OpPostProcMapInMoab<SPACE_DIM, SPACE_DIM>;

  post_proc_fe->getOpPtrVector().push_back(

      new OpPPMap(

          post_proc_fe->getPostProcMesh(), post_proc_fe->getMapGaussPts(),

          {},

          {{"V", u_ptr}, {"GEOMETRY", X_ptr}, {"x", x_ptr}},

          {{"FIRST_PIOLA", mat_P_ptr}, {"F", mat_F_ptr}, {"V_grad", mat_v_grad_ptr}},

          {}

          )

  );

  // Add monitor to time solver
  
  auto rho_ptr = boost::make_shared<double>(rho);
  auto get_rho = [rho_ptr](const double, const double, const double) {
    return *rho_ptr;
  };

  auto add_rho_block = [this, rho_ptr](auto &pip, auto block_name, Sev sev) {
    MoFEMFunctionBegin;

    struct OpMatRhoBlocks : public DomainEleOp {
      OpMatRhoBlocks(boost::shared_ptr<double> rho_ptr,
                     MoFEM::Interface &m_field, Sev sev,
                     std::vector<const CubitMeshSets *> meshset_vec_ptr)
          : DomainEleOp(NOSPACE, DomainEleOp::OPSPACE), rhoPtr(rho_ptr) {
        CHK_THROW_MESSAGE(extractRhoData(m_field, meshset_vec_ptr, sev),
                          "Can not get data from block");
      }

      MoFEMErrorCode doWork(int side, EntityType type,
                            EntitiesFieldData::EntData &data) {

        MoFEMFunctionBegin;

        for (auto &b : blockData) {
          if (b.blockEnts.find(getFEEntityHandle()) != b.blockEnts.end()) {
            *rhoPtr = b.rho;
            MoFEMFunctionReturnHot(0);
          }
        }

        *rhoPtr = rho;

        MoFEMFunctionReturn(0);
      }

    private:
      struct BlockData {
        double rho;
        Range blockEnts;
      };

      std::vector<BlockData> blockData;

      MoFEMErrorCode
      extractRhoData(MoFEM::Interface &m_field,
                     std::vector<const CubitMeshSets *> meshset_vec_ptr,
                     Sev sev) {
        MoFEMFunctionBegin;

        for (auto m : meshset_vec_ptr) {
          MOFEM_TAG_AND_LOG("WORLD", sev, "Mat Rho Block") << *m;
          std::vector<double> block_data;
          CHKERR m->getAttributes(block_data);
          if (block_data.size() < 1) {
            SETERRQ(PETSC_COMM_SELF, MOFEM_DATA_INCONSISTENCY,
                    "Expected that block has two attributes");
          }
          auto get_block_ents = [&]() {
            Range ents;
            CHKERR
            m_field.get_moab().get_entities_by_handle(m->meshset, ents, true);
            return ents;
          };

          MOFEM_TAG_AND_LOG("WORLD", sev, "Mat Rho Block")
              << m->getName() << ": ro = " << block_data[0];

          blockData.push_back({block_data[0], get_block_ents()});
        }
        MOFEM_LOG_CHANNEL("WORLD");
        MOFEM_LOG_CHANNEL("WORLD");
        MoFEMFunctionReturn(0);
      }

      boost::shared_ptr<double> rhoPtr;
    };

    // pip.push_back(new OpMatRhoBlocks(
    //     rho_ptr, mField, sev,

    //     // Get blockset using regular expression
    //     mField.getInterface<MeshsetsManager>()->getCubitMeshsetPtr(std::regex(

    //         (boost::format("%s(.*)") % block_name).str()

    //             ))

    //         ));
    MoFEMFunctionReturn(0);
  };

  
    SmartPetscObj<Mat> M;   ///< Mass matrix
    SmartPetscObj<KSP> ksp; ///< Linear solver
    // SmartPetscObj<Vector> lumpVec;
    
    // boost::shared_ptr<CommonData> data(new CommonData());

    auto ts_pre_post_proc = boost::make_shared<TSPrePostProc>();
    tsPrePostProc = ts_pre_post_proc;

    CHKERR DMCreateMatrix_MoFEM(dm, M);
    CHKERR MatZeroEntries(M);
    
    boost::shared_ptr<DomainEle> vol_mass_ele(new DomainEle(mField));
    
    vol_mass_ele->B = M;

    auto integration_rule = [](int, int, int approx_order) {
      return 8 * approx_order;
    };
    
    vol_mass_ele->getRuleHook = integration_rule;
    
    // CHKERR AddHOOps<SPACE_DIM, SPACE_DIM, SPACE_DIM>::add(
    //                   vol_mass_ele->getOpPtrVector(), {H1}, "GEOMETRY");
    // CHKERR AddHOOps<SPACE_DIM, SPACE_DIM, SPACE_DIM>::add(
    //     pipeline_mng->getOpDomainExplicitRhsPipeline(), {H1}, "GEOMETRY");
    auto energy_consistency = [&](const double, const double, const double) { return 3.*bulk_modulus_K; };
    vol_mass_ele->getOpPtrVector().push_back(new OpMassV("V", "V"));
    vol_mass_ele->getOpPtrVector().push_back(new OpMassF("F", "F", energy_consistency));


    // vol_mass_ele->getOpPtrVector().push_back(new OpCalculateExplicitMass<SPACE_DIM>("V", "V", M, get_rho));
    // vol_mass_ele->getOpPtrVector().push_back(new OpCalculateExplicitMass<SPACE_DIM*SPACE_DIM>("F", "F", M));
        
    CHKERR DMoFEMLoopFiniteElements(dm, simple->getDomainFEName(),
                                    vol_mass_ele);
    CHKERR MatAssemblyBegin(M, MAT_FINAL_ASSEMBLY);
    CHKERR MatAssemblyEnd(M, MAT_FINAL_ASSEMBLY);
    
    auto lumpVec = createDMVector(simple->getDM());
    CHKERR MatGetRowSum(M, lumpVec);

    CHKERR MatZeroEntries(M);
    CHKERR MatDiagonalSet(M, lumpVec, INSERT_VALUES);


    //CHKERR VecView(lumpVec, PETSC_VIEWER_STDOUT_WORLD);

    // auto lumpVecCheck = createDMVector(simple->getDM());
    // CHKERR MatGetRowSum(M, lumpVecCheck);
    // CHKERR VecView(lumpVecCheck, PETSC_VIEWER_STDOUT_WORLD);

    // MatView(M,PETSC_VIEWER_STDOUT_WORLD);
    // Create and septup KSP (linear solver), we need this to calculate g(t,u) =
    // M^-1G(t,u)
    ksp = createKSP(mField.get_comm());
    CHKERR KSPSetOperators(ksp, M, M);
    CHKERR KSPSetFromOptions(ksp);
    CHKERR KSPSetUp(ksp);

  auto solve_boundary_for_g = [&]() {
      MoFEMFunctionBegin;
      if (*(pipeline_mng->getBoundaryExplicitRhsFE()->vecAssembleSwitch)) {

        CHKERR VecGhostUpdateBegin(pipeline_mng->getBoundaryExplicitRhsFE()->ts_F, ADD_VALUES,
                                   SCATTER_REVERSE);
        CHKERR VecGhostUpdateEnd(pipeline_mng->getBoundaryExplicitRhsFE()->ts_F, ADD_VALUES,
                                 SCATTER_REVERSE);
        CHKERR VecAssemblyBegin(pipeline_mng->getBoundaryExplicitRhsFE()->ts_F);
        CHKERR VecAssemblyEnd(pipeline_mng->getBoundaryExplicitRhsFE()->ts_F);
        *(pipeline_mng->getBoundaryExplicitRhsFE()->vecAssembleSwitch) = false;
        
  
        auto D = smartVectorDuplicate(pipeline_mng->getBoundaryExplicitRhsFE()->ts_F);
        // cerr << "ksp.use_count() " <<  ksp.use_count() <<"\n";
      // auto ptr_ksp = ksp();
      CHKERR KSPSolve(ksp, pipeline_mng->getBoundaryExplicitRhsFE()->ts_F, D);
      CHKERR VecGhostUpdateBegin(D, INSERT_VALUES, SCATTER_FORWARD);
      CHKERR VecGhostUpdateEnd(D, INSERT_VALUES, SCATTER_FORWARD);
      CHKERR VecCopy(D, pipeline_mng->getBoundaryExplicitRhsFE()->ts_F);
      }
      
      MoFEMFunctionReturn(0);
    };

  pipeline_mng->getBoundaryExplicitRhsFE()->postProcessHook =
      solve_boundary_for_g;

  MoFEM::SmartPetscObj<TS> ts;
  ts = pipeline_mng->createTSEX(dm);

  //Field eval
  PetscBool field_eval_flag = PETSC_TRUE;
  boost::shared_ptr<MatrixDouble> field_ptr;
  boost::shared_ptr<SetPtsData> field_eval_data;

  std::array<double, 3> field_eval_coords = {0.5, 0.5, 0.};
  int dim = 3;
    // CHKERR PetscOptionsGetRealArray(NULL, NULL, "-field_eval_coords",
    //                                 field_eval_coords.data(), &dim,
    //                                 &field_eval_flag);

    if (field_eval_flag) {
      field_eval_data =
          mField.getInterface<FieldEvaluatorInterface>()->getData<DomainEle>();
      if(SPACE_DIM == 3){
      CHKERR mField.getInterface<FieldEvaluatorInterface>()->buildTree3D(
          field_eval_data, simple->getDomainFEName());
      } else {
        CHKERR mField.getInterface<FieldEvaluatorInterface>()->buildTree2D(
          field_eval_data, simple->getDomainFEName());
      }

      field_eval_data->setEvalPoints(field_eval_coords.data(), 1);

      auto no_rule = [](int, int, int) { return -1; };

      auto fe_ptr = field_eval_data->feMethodPtr.lock();
      fe_ptr->getRuleHook = no_rule;

      field_ptr = boost::make_shared<MatrixDouble>();
      fe_ptr->getOpPtrVector().push_back(
          new OpCalculateVectorFieldValues<3>("GEOMETRY", field_ptr));
    }


  boost::shared_ptr<FEMethod> null_fe;
  auto monitor_ptr = boost::make_shared<Monitor>(dm, mField, post_proc_fe, field_ptr, field_eval_coords, field_eval_data);
  
    
  CHKERR DMMoFEMTSSetMonitor(dm, ts, simple->getDomainFEName(), null_fe,
                             null_fe, monitor_ptr);

  double ftime = 1;
  // CHKERR TSSetMaxTime(ts, ftime);
  CHKERR TSSetExactFinalTime(ts, TS_EXACTFINALTIME_MATCHSTEP);

  auto T = createDMVector(simple->getDM());
  CHKERR DMoFEMMeshToLocalVector(simple->getDM(), T, INSERT_VALUES,
                                 SCATTER_FORWARD);
  CHKERR TSSetSolution(ts, T);
  CHKERR TSSetFromOptions(ts);

  auto fb = mField.getInterface<FieldBlas>();

  
  CHKERR TSSetPostStage(ts, TSPrePostProc::tsPostStage);
  CHKERR TSSetPostStep(ts, TSPrePostProc::tsPostStep);
  CHKERR TSSetPreStep(ts, TSPrePostProc::tsPreStep);
  
  boost::shared_ptr<ForcesAndSourcesCore> null;
  
if (auto ptr = tsPrePostProc.lock()) {
  // if (true) {

    ptr->fsRawPtr = this;
  
  //   ptr->T = createDMVector(dm);
  // ptr->T = T;
  // ptr->solverSubDM = simple->getDM();
  CHKERR TSSetUp(ts);
  // CHKERR TSSetSaveTrajectory(ts);
  CHKERR TSSolve(ts, NULL);

  CHKERR TSGetTime(ts, &ftime);

//   PetscInt steps, snesfails, rejects, nonlimits, limits;
// #if PETSC_VERSION_GE(3, 8, 0)
//   CHKERR TSGetStepNumber(ts, &steps);
// #else
//   CHKERR TSGetTimeStepNumber(ts, &steps);
// #endif
//   // CHKERR TSGetSNESFailures(ts, &snesfails);
//   // CHKERR TSGetStepRejections(ts, &rejects);
//   // CHKERR TSGetSNESIterations(ts, &nonlimits);
//   CHKERR TSGetKSPIterations(ts, &limits);
//   MOFEM_LOG_C("EXAMPLE", Sev::inform,
//               "steps %d (%d rejected, %d SNES fails), ftime %g, nonlimits "
//               "%d, limits %d\n",
//               steps, rejects, snesfails, ftime, nonlimits, limits);
  } 

  MoFEMFunctionReturn(0);
}
//! [Solve]

//! [Postprocess results]
MoFEMErrorCode Example::outputResults() {
  MoFEMFunctionBegin;
  PetscBool test_flg = PETSC_FALSE;
  CHKERR PetscOptionsGetBool(PETSC_NULL, "", "-test", &test_flg, PETSC_NULL);
  if (test_flg) {
    auto *simple = mField.getInterface<Simple>();
    auto T = createDMVector(simple->getDM());
    CHKERR DMoFEMMeshToLocalVector(simple->getDM(), T, INSERT_VALUES,
                                   SCATTER_FORWARD);
    double nrm2;
    CHKERR VecNorm(T, NORM_2, &nrm2);
    MOFEM_LOG("EXAMPLE", Sev::inform) << "Regression norm " << nrm2;
    constexpr double regression_value = 0.0194561;
    if (fabs(nrm2 - regression_value) > 1e-2)
      SETERRQ(PETSC_COMM_WORLD, MOFEM_ATOM_TEST_INVALID,
              "Regression test failed; wrong norm value.");
  }
  MoFEMFunctionReturn(0);
}
//! [Postprocess results]

//! [Check]
MoFEMErrorCode Example::checkResults() {
  MoFEMFunctionBegin;
  MoFEMFunctionReturn(0);
}
//! [Check]

static char help[] = "...\n\n";

int main(int argc, char *argv[]) {

  // Initialisation of MoFEM/PETSc and MOAB data structures
  const char param_file[] = "param_file.petsc";
  MoFEM::Core::Initialize(&argc, &argv, param_file, help);

  // Add logging channel for example
  auto core_log = logging::core::get();
  core_log->add_sink(
      LogManager::createSink(LogManager::getStrmWorld(), "EXAMPLE"));
  LogManager::setLog("EXAMPLE");
  MOFEM_LOG_TAG("EXAMPLE", "example");

  try {

    //! [Register MoFEM discrete manager in PETSc]
    DMType dm_name = "DMMOFEM";
    CHKERR DMRegister_MoFEM(dm_name);
    //! [Register MoFEM discrete manager in PETSc

    //! [Create MoAB]
    moab::Core mb_instance;              ///< mesh database
    moab::Interface &moab = mb_instance; ///< mesh database interface
    //! [Create MoAB]

    //! [Create MoFEM]
    MoFEM::Core core(moab);           ///< finite element database
    MoFEM::Interface &m_field = core; ///< finite element database interface
    //! [Create MoFEM]

    //! [Example]
    Example ex(m_field);
    CHKERR ex.runProblem();
    //! [Example]
  }
  CATCH_ERRORS;

  CHKERR MoFEM::Core::Finalize();
}
