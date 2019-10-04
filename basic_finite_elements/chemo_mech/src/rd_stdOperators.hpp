#ifndef __STDOPERATORS_HPP__
#define __STDOPERATORS_HPP__

#include <stdlib.h>
#include <BasicFiniteElements.hpp>

namespace StdRDOperators {

  using Ele = FaceElementForcesAndSourcesCore;
  using OpEle = FaceElementForcesAndSourcesCore::UserDataOperator;

  using BoundaryEle = MoFEM::EdgeElementForcesAndSourcesCore;
  using OpBoundaryEle = BoundaryEle::UserDataOperator;

  using EntData = DataForcesAndSourcesCore::EntData;

  const double B = 0;
  const int save_every_nth_step = 1;
  const double natural_bc_values = 0.0;
  const double essential_bc_values = 0.0;
  const int order = 2;
  // const int dim = 3;
  FTensor::Index<'i', 3> i;

  struct BlockData {
    int block_id;
    MatrixDouble coef;
    VectorDouble rate;

    Range block_ents;

    double B0; // species mobility

    BlockData()
      : B0(1e0) {
      coef.resize(3, 3, false);
      rate.resize(3, false);
      coef.clear();
      rate.clear();
      for (int i = 0; i < 3; ++i) {
        coef(i, i) = 1;
        rate[i] = 1;
        }
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

  struct OpAssembleMass : OpEle {
    OpAssembleMass(std::string fieldu, SmartPetscObj<Mat> m)
        : OpEle(fieldu, fieldu, OpEle::OPROWCOL), M(m)

    {
      sYmm = true;
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
        auto t_row_base = row_data.getFTensor0N();
        auto t_w = getFTensor0IntegrationWeight();
        const double vol = getMeasure();
        for (int gg = 0; gg != nb_integration_pts; ++gg) {
          const double a = t_w * vol;
          for (int rr = 0; rr != nb_row_dofs; ++rr) {
            auto t_col_base = col_data.getFTensor0N(gg, 0);
            for (int cc = 0; cc != nb_col_dofs; ++cc) {
              mat(rr, cc) += a * t_row_base * t_col_base;
              ++t_col_base;
            }
            ++t_row_base;
          }
          ++t_w;
        }
        CHKERR MatSetValues(M, row_data, col_data, &mat(0, 0), ADD_VALUES);
        if (row_side != col_side || row_type != col_type) {
          transMat.resize(nb_col_dofs, nb_row_dofs, false);
          noalias(transMat) = trans(mat);
          CHKERR MatSetValues(M, col_data, row_data, &transMat(0, 0),
                              ADD_VALUES);
        }
      }
      MoFEMFunctionReturn(0);
    }

  private:
    MatrixDouble mat, transMat;
    SmartPetscObj<Mat> M;
  };

  struct OpComputeSlowValue : public OpEle {
    OpComputeSlowValue(std::string mass_field,
                       boost::shared_ptr<PreviousData> &data1,
                       boost::shared_ptr<PreviousData> &data2,
                       boost::shared_ptr<PreviousData> &data3,
                       std::map<int, BlockData> &block_map)
        : OpEle(mass_field, OpEle::OPROW), commonData1(data1),
          commonData2(data2), commonData3(data3), massField(mass_field),
          setOfBlock(block_map) {}
    MoFEMErrorCode doWork(int side, EntityType type, EntData &data) {
      MoFEMFunctionBegin;
      boost::shared_ptr<VectorDouble> slow_value_ptr1(
          commonData1, &commonData1->slow_values);
      boost::shared_ptr<VectorDouble> slow_value_ptr2(
          commonData2, &commonData2->slow_values);
      boost::shared_ptr<VectorDouble> slow_value_ptr3(
          commonData3, &commonData3->slow_values);

      VectorDouble &vec1 = *slow_value_ptr1;
      VectorDouble &vec2 = *slow_value_ptr2;
      VectorDouble &vec3 = *slow_value_ptr3;
      const int nb_integration_pts = getGaussPts().size2();
      if (type == MBVERTEX) {
        vec1.resize(nb_integration_pts, false);
        vec2.resize(nb_integration_pts, false);
        vec3.resize(nb_integration_pts, false);
        vec1.clear();
        vec2.clear();
        vec3.clear();
      }
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

        const int nb_integration_pts = getGaussPts().size2();

        auto t_slow_values1 = getFTensor0FromVec(vec1);
        auto t_slow_values2 = getFTensor0FromVec(vec2);
        auto t_slow_values3 = getFTensor0FromVec(vec3);

        auto t_mass_values1 = getFTensor0FromVec(commonData1->values);
        auto t_mass_values2 = getFTensor0FromVec(commonData2->values);
        auto t_mass_values3 = getFTensor0FromVec(commonData3->values);

        for (int gg = 0; gg != nb_integration_pts; ++gg) {
          t_slow_values1 = block_data.rate[0] * t_mass_values1 *
                           (1.0 - block_data.coef(0, 0) * t_mass_values1 -
                                  block_data.coef(0, 1) * t_mass_values2 -
                                  block_data.coef(0, 2) * t_mass_values3);
          t_slow_values2 = block_data.rate[1] * t_mass_values2 *
                           (1.0 - block_data.coef(1, 0) * t_mass_values1 -
                                  block_data.coef(1, 1) * t_mass_values2 -
                                  block_data.coef(1, 2) * t_mass_values3);

          t_slow_values3 = block_data.rate[2] * t_mass_values3 *
                           (1.0 - block_data.coef(2, 0) * t_mass_values1 -
                                  block_data.coef(2, 1) * t_mass_values2 -
                                  block_data.coef(2, 2) * t_mass_values3);
          ++t_slow_values1;
          ++t_slow_values2;
          ++t_slow_values3;

          ++t_mass_values1;
          ++t_mass_values2;
          ++t_mass_values3;
        }
      }
      MoFEMFunctionReturn(0);
    }

  private:
    std::string massField;
    boost::shared_ptr<PreviousData> commonData1;
    boost::shared_ptr<PreviousData> commonData2;
    boost::shared_ptr<PreviousData> commonData3;
    std::map<int, BlockData> setOfBlock;
  };

  struct OpAssembleSlowRhs : OpEle {
    typedef boost::function<double(const double, const double, const double)>
        FVal;
    typedef boost::function<FTensor::Tensor1<double, 3>(
        const double, const double, const double)>
        FGrad;
    OpAssembleSlowRhs(std::string field, 
                      boost::shared_ptr<PreviousData> &data,
                      FVal exact_value, 
                      FVal exact_dot, 
                      FVal exact_lap)
        : OpEle(field, OpEle::OPROW)
        , commonData(data)
        , exactValue(exact_value)
        , exactDot(exact_dot)
        , exactLap(exact_lap) {}

    MoFEMErrorCode doWork(int side, EntityType type, EntData &data) {
      MoFEMFunctionBegin;
   
      const int nb_dofs = data.getIndices().size();
      if (nb_dofs) {
        vecF.resize(nb_dofs, false);
        vecF.clear();
        const int nb_integration_pts = getGaussPts().size2();

        auto t_slow_value = getFTensor0FromVec(commonData->slow_values);

        auto t_base = data.getFTensor0N();
        auto t_w = getFTensor0IntegrationWeight();
        const double vol = getMeasure();
        const double ct = getFEMethod()->ts_t;
        auto t_coords = getFTensor1CoordsAtGaussPts();
        for (int gg = 0; gg != nb_integration_pts; ++gg) {
          const double a = vol * t_w;

          double u_dot = exactDot(t_coords(NX), t_coords(NY), ct);
          double u_lap = exactLap(t_coords(NX), t_coords(NY), ct);

          double f = u_dot - u_lap;

          // const double f = a * t_slow_value;  

          for (int rr = 0; rr != nb_dofs; ++rr) {
            const double b = a * f * t_base;
            vecF[rr] += b;
            ++t_base;
          }
          ++t_slow_value;
          ++t_w;
          ++t_coords;
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
    VectorDouble vecF;

    FVal exactValue;
    FVal exactDot;
    FVal exactLap;

    FTensor::Number<0> NX;
    FTensor::Number<1> NY;
    FTensor::Number<2> NZ;
  };

  template <int DIM> struct OpAssembleStiffRhs : OpEle {
    typedef boost::function<double(const double, const double, const double)>
        FVal;
    typedef boost::function<FTensor::Tensor1<double, 3>(
        const double, const double, const double)>
        FGrad;
      
    OpAssembleStiffRhs(std::string field, 
                       boost::shared_ptr<PreviousData> &data,
                       std::map<int, BlockData> &block_map
                      //  FVal exact_value, 
                      //  FVal exact_dot, 
                      //  FVal exact_lap
                       )
        : OpEle(field, OpEle::OPROW), commonData(data), setOfBlock(block_map) {}

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
        auto t_grad = getFTensor1FromMat<DIM>(commonData->grads);

        auto t_base = data.getFTensor0N();
        auto t_diff_base = data.getFTensor1DiffN<DIM>();
        auto t_w = getFTensor0IntegrationWeight();
      
        FTensor::Index<'i', DIM> i;

        // cout << "B0 : " << block_data.B0 << endl;

        const double vol = getMeasure();
        for (int gg = 0; gg != nb_integration_pts; ++gg) {
          const double a = vol * t_w;
          for (int rr = 0; rr != nb_dofs; ++rr) {
            vecF[rr] += a * (t_base * t_dot_val + (block_data.B0 + B * t_val) *
            t_diff_base(i) * t_grad(i));
            ++t_diff_base;
            ++t_base;
          }
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

  template <int DIM> struct OpAssembleStiffLhs : OpEle {

    OpAssembleStiffLhs(std::string fieldu,
                       boost::shared_ptr<PreviousData> &data,
                       std::map<int, BlockData> &block_map)
        : OpEle(fieldu, fieldu, OpEle::OPROWCOL), commonData(data), setOfBlock(block_map) {
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
        auto t_grad = getFTensor1FromMat<DIM>(commonData->grads);

        auto t_row_diff_base = row_data.getFTensor1DiffN<DIM>();
        auto t_w = getFTensor0IntegrationWeight();
        const double ts_a = getFEMethod()->ts_a;
        const double vol = getMeasure();

        FTensor::Index<'i', DIM> i;

        // cout << "B0 : " << block_data.B0 << endl;

        for (int gg = 0; gg != nb_integration_pts; ++gg) {
          const double a = vol * t_w;
          
          for (int rr = 0; rr != nb_row_dofs; ++rr) {
            auto t_col_base = col_data.getFTensor0N(gg, 0);
            auto t_col_diff_base = col_data.getFTensor1DiffN<DIM>(gg, 0);
            for (int cc = 0; cc != nb_col_dofs; ++cc) {

              mat(rr, cc) +=
                  a * (t_row_base * t_col_base * ts_a + (block_data.B0 + B * t_val) *
                  t_row_diff_base(i) * t_col_diff_base(i)
                        + B * t_col_base * t_grad(i) * t_row_diff_base(i));

              ++t_col_base;
              ++t_col_diff_base;
            }
            ++t_row_base;
            ++t_row_diff_base;
          }
          ++t_w;
          ++t_val;
          ++t_grad;
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

  struct OpAssembleNaturalBCRhs : OpBoundaryEle // R_tau_2
  {
    OpAssembleNaturalBCRhs(std::string mass_field, Range &natural_bd_ents)
        : OpBoundaryEle(mass_field, OpBoundaryEle::OPROW),
          natural_bd_ents(natural_bd_ents) {
      cerr << "OpAssembleNaturalBCRhsTau()" << endl;
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

          auto dir = getDirection();
          FTensor::Tensor1<double, 3> t_normal(-dir[1], dir[0], dir[2]);
          FTensor::Index<'i', 3> i;
          auto t_w = getFTensor0IntegrationWeight();
          const double pi = 3.141592653589793;
          const double ct = getFEMethod()->ts_t;
          for (int gg = 0; gg != nb_integration_pts; ++gg) {
            const double a = t_w;
            double x = getCoordsAtGaussPts()(gg, 0);
            double y = getCoordsAtGaussPts()(gg, 1);
            
            double mm = - 10 * 8 * pi * cos(2 * pi * x) * sin(2 * pi * y) * sin(2 * pi * ct);
            double nn = - 10 * 8 * pi * sin(2 * pi * x) * cos(2 * pi * y) * sin(2 * pi * ct);
            
            FTensor::Tensor1<double, 3> t_bd_val(mm, nn, 0.0);
            double h = t_bd_val(i) * t_normal(i);
            for (int rr = 0; rr != nb_dofs; ++rr) {
              vecF[rr] += t_row_base * h * a;
              ++t_row_base;
            }
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
    Monitor(SmartPetscObj<DM> &dm,
            boost::shared_ptr<PostProcFaceOnRefinedMesh> &post_proc)
        : dM(dm), postProc(post_proc){};
    MoFEMErrorCode preProcess() { return 0; }
    MoFEMErrorCode operator()() { return 0; }
    MoFEMErrorCode postProcess() {
      MoFEMFunctionBegin;
      if (ts_step % save_every_nth_step == 0) {
        CHKERR DMoFEMLoopFiniteElements(dM, "dFE", postProc);
        CHKERR postProc->writeFile(
            "out_level_" + boost::lexical_cast<std::string>(ts_step) + ".h5m");
      }
      // cout << "ct : " << ts_t << endl;
      MoFEMFunctionReturn(0);
    }

  private:
    SmartPetscObj<DM> dM;
    boost::shared_ptr<PostProcFaceOnRefinedMesh> postProc;
  };

}; // end StdRDOperators namespace

#endif