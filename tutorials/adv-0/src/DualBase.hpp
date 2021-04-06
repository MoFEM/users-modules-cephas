

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

};