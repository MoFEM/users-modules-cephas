#ifndef __UFOPERATORS2D_HPP__
#define __UFOPERATORS2D_HPP__

#include <stdlib.h>
#include <BasicFiniteElements.hpp>

namespace UFOperators2D {
  

  using VolEle = MoFEM::FaceElementForcesAndSourcesCore;
  using FaceEle = MoFEM::EdgeElementForcesAndSourcesCore;

  using OpVolEle = VolEle::UserDataOperator;
  using OpFaceEle = FaceEle::UserDataOperator;

  using EntData = DataForcesAndSourcesCore::EntData;

  using PostProc = PostProcFaceOnRefinedMesh;


  const int save_every_nth_step = 4;
  const double natural_bc_values = -1;
  const double essential_bc_values = 0.0;


  // const int 2 = 3;
  FTensor::Index<'i', 3> i;

  struct BlockData {
    int block_id;
    double K_s; // K_s
    double ll;          // l
    double theta_r;       // theta_r
    double theta_s;      // theta_s
    double theta_m;       // theta_m
    double nn;       // n
    double alpha;   // alpha
    double h_s;     // h_s

    Range block_ents;

    BlockData()
    : K_s(4.8000)
    , ll(0.5000)
    , theta_r(0.01000000)
    , theta_s(0.45000000)
    , theta_m(0.45008000)
    , nn(1.17200000)
    , alpha(0.00170000)
    , h_s(-2.0000)
    {}

    double get_waterContent(double head){
      double ret_val;
      double m = 1 - 1.0 / nn;
      if(head < h_s){
        ret_val = theta_r + (theta_m-theta_r) / 
                                  pow(1 + pow(-alpha * head, nn), m); 
      } else {
        ret_val = theta_s; 
      }
      return ret_val;
    }

    double get_conductivity(double head){
      double ret_val;
      if(head < h_s){
        ret_val = K_s * get_relativeConductivity(head);
      }else {
        ret_val = K_s;
      }
      return ret_val;
    }
    double get_relativeConductivity(double head){
      double S_e = get_effSaturation(head);
      double F_e = get_Fe(S_e);
      double F_1 = get_Fe(1.0);

      return pow(S_e, ll) * pow(( (1 - F_e) / (1 - F_1) ), 2);
    }
    double get_Fe(double eff_satu){
      double m = 1 - 1.0 / nn;
      double S_eStar = get_eff_satuStar(eff_satu);
      return pow(1 - pow(S_eStar, 1.0 / m), m);
    }

    double get_eff_satuStar(double eff_satu){
      return (theta_s - theta_r) / (theta_m - theta_r) * eff_satu;
             
    }

    double get_effSaturation(double head){
      double theta = get_waterContent(head);
      return (theta - theta_r) / (theta_s - theta_r);
    }

    double get_capacity(double head){
      double m = 1 - 1.0 / nn;
      double ret_val;
      if(head < h_s){
        ret_val = alpha * m * nn * (theta_m - theta_r) * pow(-alpha * head, nn-1) /
                  pow(1 + pow(-alpha * head, nn), m+1);
      } else {
      ret_val = 0;
    }
     return ret_val;
    }

    double get_diffCapacity(double head){
      double m = 1.0 - 1.0 / nn;
      double ret_val;
      if(head < h_s){
      double denom = 1 + pow(-alpha * head, nn);
        ret_val = pow(alpha, 2) * (theta_m - theta_r) * m * nn * pow(-alpha * head, nn-2) / pow(denom, m+1) *
              ( (m + 1) * nn * pow(-alpha * head, nn) / denom + (nn-1) );
      }else{
        ret_val = 0;
      }
      
      return ret_val;
    }

    double get_diffConductivity(double head){
      double DK_r = get_diffRelativeConductivity(head);
      double ret_val;
      if(head < h_s){
        ret_val = K_s * DK_r;
      }else{
        ret_val = 0;
      }
      return ret_val;
    }

    double get_diffRelativeConductivity(double head){
      double S_e = get_effSaturation(head);
      double DS_e = get_diffEffSaturation(head);
      double F_e = get_Fe(S_e);
      double F_1 = get_Fe(1.0);
      double DF_e = get_diffFe(S_e);
      return pow(S_e, ll-1) * DS_e * ( (1 - F_e) / (1 - F_1) ) * 
             (ll * ( (1 - F_e) / (1 - F_1) ) - 2.0 * S_e * DF_e / (1 - F_1));
    }
    double get_diffFe(double s_e){
      double m = 1 - 1.0 / nn;
      double S_estar = get_eff_satuStar(s_e);
      double DS_estar = get_diffEffSatuStar(s_e);
      return -DS_estar * pow(1-pow(S_estar, 1.0/m), m-1) * pow(S_estar, 1.0/m - 1);
    }

    double get_diffEffSatuStar(double s_e){
      return (theta_s - theta_r) / (theta_m - theta_r);
    }

    double get_diffEffSaturation(double head){
      double Dtheta = get_capacity(head);
      return Dtheta / (theta_s - theta_r);
    }

  };


  struct PreviousData {
    MatrixDouble grads;    ///< Gradients of field "u" at integration points
    VectorDouble values;     ///< Values of field "u" at integration points
    VectorDouble dot_values; ///< Rate of values of field "u" at integration points
    VectorDouble slow_values;

    MatrixDouble invJac; ///< Inverse of element jacobian

    PreviousData() {}
  };


  struct OpAssembleStiffRhs : OpVolEle {
      
    OpAssembleStiffRhs(std::string field, 
                       boost::shared_ptr<PreviousData> &data,
                       std::map<int, BlockData> &block_map)
        : OpVolEle(field, OpVolEle::OPROW), commonData(data)
        , setOfBlock(block_map) {}

    MoFEMErrorCode doWork(int side, EntityType type, EntData &data) {
      MoFEMFunctionBegin;
      const int nb_dofs = data.getIndices().size();
      if (nb_dofs) {
        auto find_block_data = [&]() {
          EntityHandle fe_ent = getFEEntityHandle();
          BlockData *block_raw_ptr = nullptr;
          for (auto &m : setOfBlock) {
            if (m.second.block_ents.find(fe_ent) != m.second.block_ents.end()) {
              block_raw_ptr = &m.second;
              break;
            }
          }
          return block_raw_ptr;
        };

        auto block_data_ptr = find_block_data();
        if (!block_data_ptr)
          SETERRQ(PETSC_COMM_SELF, MOFEM_DATA_INCONSISTENCY, "Block not found");

        auto &block_data = *block_data_ptr;

        vecF.resize(nb_dofs, false);
        vecF.clear();
        const int nb_integration_pts = getGaussPts().size2();
        
        auto t_dot_val = getFTensor0FromVec(commonData->dot_values);
        auto t_val = getFTensor0FromVec(commonData->values);
        auto t_grad = getFTensor1FromMat<2>(commonData->grads);

        auto t_base = data.getFTensor0N();
        auto t_diff_base = data.getFTensor1DiffN<2>();
        auto t_w = getFTensor0IntegrationWeight();
      
        FTensor::Index<'i', 2> i;


        const double vol = getMeasure();
        
        for (int gg = 0; gg != nb_integration_pts; ++gg) {
          const double a = vol * t_w;

          const double K_h = block_data.get_conductivity(t_val);
          const double C_h = block_data.get_capacity(t_val); 
          for (int rr = 0; rr != nb_dofs; ++rr) {
            vecF[rr] += a * (t_base * C_h * t_dot_val + K_h *
            t_diff_base(i) * t_grad(i));
            ++t_diff_base;
            ++t_base;
          }
          // cout << "vecF : " << vecF << endl;
          ++t_dot_val;
          ++t_grad;
          ++t_val;
          ++t_w;
        }
        CHKERR VecSetOption(getFEMethod()->ts_F, VEC_IGNORE_NEGATIVE_INDICES,
                            PETSC_TRUE);
        CHKERR VecSetValues(getFEMethod()->ts_F, data, &*vecF.begin(),
                            ADD_VALUES);
      }
      MoFEMFunctionReturn(0);
    }

  private:
    boost::shared_ptr<PreviousData> commonData;
    std::map<int, BlockData> setOfBlock;
    VectorDouble vecF;
    std::string field;
  };

  struct OpAssembleStiffLhs : OpVolEle {

    OpAssembleStiffLhs(std::string fieldu,
                       boost::shared_ptr<PreviousData> &data,
                       std::map<int, BlockData> &block_map)
        : OpVolEle(fieldu, fieldu, OpVolEle::OPROWCOL), commonData(data), setOfBlock(block_map) {
      sYmm = true;
    }
    MoFEMErrorCode doWork(int row_side, int col_side, EntityType row_type,
                          EntityType col_type, EntData &row_data,
                          EntData &col_data) {
      MoFEMFunctionBegin;

      const int nb_row_dofs = row_data.getIndices().size();
      const int nb_col_dofs = col_data.getIndices().size();
      // cerr << "In doWork() : (row, col) = (" << nb_row_dofs << ", " <<
      // nb_col_dofs << ")" << endl;
      if (nb_row_dofs && nb_col_dofs) {
        auto find_block_data = [&]() {
          EntityHandle fe_ent = getFEEntityHandle();
          BlockData *block_raw_ptr = nullptr;
          for (auto &m : setOfBlock) {
            if (m.second.block_ents.find(fe_ent) != m.second.block_ents.end()) {
              block_raw_ptr = &m.second;
              break;
            }
          }
          return block_raw_ptr;
        };

        auto block_data_ptr = find_block_data();
        if (!block_data_ptr)
          SETERRQ(PETSC_COMM_SELF, MOFEM_DATA_INCONSISTENCY, "Block not found");

        auto &block_data = *block_data_ptr;

        mat.resize(nb_row_dofs, nb_col_dofs, false);
        mat.clear();
        const int nb_integration_pts = getGaussPts().size2();
        auto t_row_base = row_data.getFTensor0N();


        auto t_val = getFTensor0FromVec(commonData->values);
        auto t_dot_val = getFTensor0FromVec(commonData->dot_values);
        auto t_grad = getFTensor1FromMat<2>(commonData->grads);

        auto t_row_diff_base = row_data.getFTensor1DiffN<2>();
        auto t_w = getFTensor0IntegrationWeight();
        const double ts_a = getFEMethod()->ts_a;
        const double vol = getMeasure();

        FTensor::Index<'i', 2> i;

        // cout << "B0 : " << block_data.B0 << endl;

        for (int gg = 0; gg != nb_integration_pts; ++gg) {
          const double a = vol * t_w;
          const double K_h = block_data.get_conductivity(t_val);
          const double C_h = block_data.get_capacity(t_val);
          const double DK_h = block_data.get_diffConductivity(t_val);
          const double DC_h = block_data.get_diffCapacity(t_val);

          // cout << "C_h : " << C_h << endl; 
          for (int rr = 0; rr != nb_row_dofs; ++rr) {
            auto t_col_base = col_data.getFTensor0N(gg, 0);
            auto t_col_diff_base = col_data.getFTensor1DiffN<2>(gg, 0);
            for (int cc = 0; cc != nb_col_dofs; ++cc) {

              mat(rr, cc) +=
                  a * (t_row_base * (DC_h * t_dot_val + C_h * ts_a) * t_col_base +
                  DK_h * t_grad(i) * t_row_diff_base(i) * t_col_base + K_h * t_row_diff_base(i) * t_col_diff_base(i));

              ++t_col_base;
              ++t_col_diff_base;
            }
            // cout << "mat : " << mat << endl;
            ++t_row_base;
            ++t_row_diff_base;
          }
          ++t_w;
          ++t_val;
          ++t_grad;
          ++t_dot_val;
        }
        CHKERR MatSetValues(getFEMethod()->ts_B, row_data, col_data, &mat(0, 0),
                            ADD_VALUES);
        if (row_side != col_side || row_type != col_type) {
          transMat.resize(nb_col_dofs, nb_row_dofs, false);
          noalias(transMat) = trans(mat);
          CHKERR MatSetValues(getFEMethod()->ts_B, col_data, row_data,
                              &transMat(0, 0), ADD_VALUES);
        }
      }
      MoFEMFunctionReturn(0);
    }

  private:
    boost::shared_ptr<PreviousData> commonData;
    std::map<int, BlockData> setOfBlock;
    MatrixDouble mat, transMat;
  };

  struct OpAssembleNaturalBCRhs : OpFaceEle // R_tau_2
  {
    OpAssembleNaturalBCRhs(std::string mass_field, Range &natural_bd_ents)
        : OpFaceEle(mass_field, OpFaceEle::OPROW),
          natural_bd_ents(natural_bd_ents) {
    }

    MoFEMErrorCode doWork(int side, EntityType type, EntData &data) {
      MoFEMFunctionBegin;
      const int nb_dofs = data.getIndices().size();

      if (nb_dofs) {
        EntityHandle row_side_ent = getFEEntityHandle();
        bool is_natural =
            (natural_bd_ents.find(row_side_ent) != natural_bd_ents.end());
        if (is_natural) {
          // cerr << "In NaturalBCRhsTau..." << endl;
          vecF.resize(nb_dofs, false);
          vecF.clear();
          const int nb_integration_pts = getGaussPts().size2();
          auto t_row_base = data.getFTensor0N();

        
          auto t_w = getFTensor0IntegrationWeight();
          const double vol = getMeasure();

          for (int gg = 0; gg != nb_integration_pts; ++gg) {
            const double a = vol * t_w;
         
            double h = natural_bc_values;
            for (int rr = 0; rr != nb_dofs; ++rr) {
              vecF[rr] += t_row_base * h * a;
              ++t_row_base;
            }
            // cout << "vecF : " << vecF << endl;
            ++t_w;
          }
          CHKERR VecSetOption(getFEMethod()->ts_F, VEC_IGNORE_NEGATIVE_INDICES,
                              PETSC_TRUE);
          CHKERR VecSetValues(getFEMethod()->ts_F, data, &*vecF.begin(),
                              ADD_VALUES);
        }
      }
      MoFEMFunctionReturn(0);
    }

  private:
    VectorDouble vecF;
    Range natural_bd_ents;
  };



  struct Monitor : public FEMethod {
    double &eRror;
    Monitor(MPI_Comm &comm, const int &rank, SmartPetscObj<DM> &dm,
            boost::shared_ptr<PostProc> &post_proc, double &err)
        : cOmm(comm)
        , rAnk(rank) 
        , dM(dm)
        , postProc(post_proc)
        , eRror(err){};
    MoFEMErrorCode preProcess() { return 0; }
    MoFEMErrorCode operator()() { return 0; }
    MoFEMErrorCode postProcess() {
      MoFEMFunctionBegin;
      if (ts_step % save_every_nth_step == 0) {
        CHKERR DMoFEMLoopFiniteElements(dM, "dFE", postProc);
        CHKERR postProc->writeFile(
            "out_level_" + boost::lexical_cast<std::string>(ts_step) + ".h5m");
      }
     
      MoFEMFunctionReturn(0);
    }

  private:
    SmartPetscObj<DM> dM;
    boost::shared_ptr<PostProc> postProc;
    MPI_Comm cOmm;
    const int rAnk;
  };

}; // end UFOperators2D namespace

#endif