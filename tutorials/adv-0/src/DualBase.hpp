

/**
 * @brief Dual base
 *
 */
namespace DualBaseOps {

struct OpCalculateDualBase : public DomainEleOp {
  OpCalculateDualBase(const std::string row_name,
                      boost::shared_ptr<PlasticOps::CommonData> common_data_ptr)
      : DomainEleOp(row_name, DomainEleOp::OPROW),
        commonDataPtr(common_data_ptr) {
    sYmm = false;
  }

  MoFEMErrorCode doWork(int side, EntityType type, EntData &data) {
    MoFEMFunctionBegin;

    const size_t nb_row_dofs = data.getIndices().size();
    if (nb_row_dofs) {

      // auto &dual_base = commonDataPtr->dualBaseMat;
      // dual_base = data.getN()/2;

      const int rank = data.getFieldDofs()[0]->getNbOfCoeffs();

      const size_t nb_integration_pts = data.getN().size1();
      const size_t nb_row = data.getN().size2();
      auto t_w = getFTensor0IntegrationWeight();

      auto &dual_base = commonDataPtr->dualBaseMat;
      dual_base.resize(nb_integration_pts, nb_row, false);
      dual_base.clear();

      MatrixDouble A(nb_row, nb_row, false);
      A.clear();

      auto t_row_base = data.getFTensor0N();
      for (size_t gg = 0; gg != nb_integration_pts; ++gg) {
        const double alpha = getMeasure() * t_w;

        noalias(A) += t_w * outer_prod(data.getN(gg, nb_row),
                                       data.getN(gg, nb_row));
        ++t_w;
      }

      auto invA = A;
      CHKERR computeMatrixInverse(invA);

      for (size_t gg = 0; gg != nb_integration_pts; ++gg) {
        auto shape_function =
            getVectorAdaptor(&dual_base.data()[gg * nb_row], nb_row);
        noalias(shape_function) = prod(invA, data.getN(gg, nb_row));
      }
    }

    MoFEMFunctionReturn(0);
  }

protected:
  boost::shared_ptr<PlasticOps::CommonData> commonDataPtr;
};

struct OpDualSwap : public OpCalculateDualBase {
  using OpCalculateDualBase::OpCalculateDualBase;
  MoFEMErrorCode doWork(int side, EntityType type, EntData &data) {
    MoFEMFunctionBegin;

    const size_t nb_row_dofs = data.getIndices().size();
    if (nb_row_dofs) {
      auto &dual_base = commonDataPtr->dualBaseMat;
      dual_base.swap(data.getN());
    }

    MoFEMFunctionReturn(0);
  }
};

}; // namespace DualBaseOps