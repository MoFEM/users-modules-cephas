

/**
 * @brief Dual base
 *
 */
namespace DualBaseOps {

struct OpSolveDualBase : public DomainEleOp {

  OpSolveDualBase() : DomainEleOp(L2) {}

protected:
  boost::shared_ptr<MatrixDouble> baseCoeffs;
  boost::shared_ptr<MatrixDouble> primarBase;
  boost::shared_ptr<MatrixDouble> dualBase;
};

struct OpSetDualBase : public DomainEleOp {
  OpSetDualBase(const std::string row_name, const std::string col_name,
                boost::shared_ptr<PlasticOps::CommonData> common_data_ptr)
      : DomainEleOp(row_name, col_name, DomainEleOp::OPROWCOL),
        commonDataPtr(common_data_ptr) {
    sYmm = false;
  }

  MoFEMErrorCode doWork(int row_side, int col_side, EntityType row_type,
                        EntityType col_type, EntData &row_data,
                        EntData &col_data) {
    MoFEMFunctionBegin;

    const size_t nb_row_dofs = row_data.getIndices().size();
    const size_t nb_col_dofs = col_data.getIndices().size();
    if (nb_row_dofs && nb_col_dofs) {

      const int rank = row_data.getFieldDofs()[0]->getNbOfCoeffs();

      const size_t nb_integration_pts = row_data.getN().size1();
      const size_t nb_row = row_data.getN().size2();
      const size_t nb_col = col_data.getN().size2();
      auto t_w = getFTensor0IntegrationWeight();

      auto &dual_base = commonDataPtr->dualBaseMat;
      dual_base.resize(nb_integration_pts, nb_row, false);
      // dual_base = row_data.getN();
      dual_base.clear();

      MatrixDouble A(nb_row, nb_col, false);
      A.clear();

      auto t_row_base = row_data.getFTensor0N();
      for (size_t gg = 0; gg != nb_integration_pts; ++gg) {
        const double alpha = getMeasure() * t_w;

        noalias(A) += t_w * outer_prod(row_data.getN(gg, nb_row),
                                       col_data.getN(gg, nb_col));
        ++t_w;
      }

      auto invA = A;
      CHKERR computeMatrixInverse(invA);

      for (size_t gg = 0; gg != nb_integration_pts; ++gg) {
        auto shape_function =
            getVectorAdaptor(&dual_base.data()[gg * nb_row], nb_row);
        noalias(shape_function) = prod(invA, row_data.getN(gg, nb_row));
      }
      row_data.getN().swap(dual_base);
    }

    MoFEMFunctionReturn(0);
  }

protected:
  boost::shared_ptr<PlasticOps::CommonData> commonDataPtr;
};

struct OpUnsetDualBase : public OpSetDualBase {
  using OpSetDualBase::OpSetDualBase;
  MoFEMErrorCode doWork(int row_side, int col_side, EntityType row_type,
                        EntityType col_type, EntData &row_data,
                        EntData &col_data) {
    MoFEMFunctionBegin;

    const size_t nb_row_dofs = row_data.getIndices().size();
    const size_t nb_col_dofs = col_data.getIndices().size();
    if (nb_row_dofs && nb_col_dofs) {
      auto &dual_base = commonDataPtr->dualBaseMat;
      dual_base.swap(row_data.getN());
    }

    MoFEMFunctionReturn(0);
  }
};

}; // namespace DualBaseOps