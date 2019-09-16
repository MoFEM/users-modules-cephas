#ifndef __STDOPERATORS_HPP__
#define __STDOPERATORS_HPP__

#include <stdlib.h>
#include <BasicFiniteElements.hpp>

namespace StdRDOperators {

  using Ele = FaceElementForcesAndSourcesCore;
  using OpEle = FaceElementForcesAndSourcesCore::UserDataOperator;
  using EntData = DataForcesAndSourcesCore::EntData;

  const double B = 0;
  const int save_every_nth_step = 4;
  const double natural_bc_values = 0.0;
  const double essential_bc_values = 0.0;
  const int order = 2;
  // const int dim = 3;
  FTensor::Index<'i', 3> i;

  struct BlockData {
    int block_id;
    double a11, a12, a13, a21, a22, a23, a31, a32, a33;

        double r1, r2, r3;

    Range block_ents;

    double B0; // species mobility

    BlockData()
        : a11(1), a12(0), a13(0), a21(0), a22(1), a23(0), a31(0), a32(0),
          a33(1), r1(1), r2(1), r3(1), B0(1e-3) {}
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
          commonData2(data2), commonData3(data3),
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
          t_slow_values1 = block_data.r1 * t_mass_values1 *
                           (1.0 - block_data.a11 * t_mass_values1 -
                            block_data.a12 * t_mass_values2 -
                            block_data.a13 * t_mass_values3);
          t_slow_values2 = block_data.r2 * t_mass_values2 *
                           (1.0 - block_data.a21 * t_mass_values1 -
                            block_data.a22 * t_mass_values2 -
                            block_data.a23 * t_mass_values3);

          t_slow_values3 = block_data.r3 * t_mass_values3 *
                           (1.0 - block_data.a31 * t_mass_values1 -
                            block_data.a32 * t_mass_values2 -
                            block_data.a33 * t_mass_values3);
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
    boost::shared_ptr<PreviousData> commonData1;
    boost::shared_ptr<PreviousData> commonData2;
    boost::shared_ptr<PreviousData> commonData3;
    std::map<int, BlockData> setOfBlock;
  };

  struct OpAssembleSlowRhs : OpEle {
    OpAssembleSlowRhs(std::string field, boost::shared_ptr<PreviousData> &data)
        : OpEle(field, OpEle::OPROW), commonData(data) {}

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
        for (int gg = 0; gg != nb_integration_pts; ++gg) {
          const double a = vol * t_w;
          const double f = a * t_slow_value;
          for (int rr = 0; rr != nb_dofs; ++rr) {
            const double b = f * t_base;
            vecF[rr] += b;
            ++t_base;
          }
          ++t_slow_value;
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
    VectorDouble vecF;
  };

  template <int DIM> struct OpAssembleStiffRhs : OpEle {
    OpAssembleStiffRhs(std::string field, boost::shared_ptr<PreviousData> &data,
                       std::map<int, BlockData> &block_map)
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
      MoFEMFunctionReturn(0);
    }

  private:
    SmartPetscObj<DM> dM;
    boost::shared_ptr<PostProcFaceOnRefinedMesh> postProc;
  };

}; // end StdRDOperators namespace

#endif