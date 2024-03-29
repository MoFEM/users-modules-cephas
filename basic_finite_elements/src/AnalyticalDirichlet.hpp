/** \file AnalyticalDirichlet.hpp

  Enforce Dirichlet boundary condition for given analytical function,

*/



#ifndef __ANALYTICALDIRICHLETBC_HPP__
#define __ANALYTICALDIRICHLETBC_HPP__

using namespace boost::numeric;
using namespace MoFEM;

/** \brief Analytical Dirichlet boundary conditions
  \ingroup user_modules
  */
struct AnalyticalDirichletBC {

  /** \brief finite element to approximate analytical solution on surface
   */
  struct ApproxField {

    struct MyTriFE : public MoFEM::FaceElementForcesAndSourcesCore {

      int addToRule; ///< this is add to integration rule if 2nd order geometry
                     ///< approximation
      MyTriFE(MoFEM::Interface &m_field)
          : MoFEM::FaceElementForcesAndSourcesCore(m_field), addToRule(4) {}
      int getRule(int order) { return 2 * order + addToRule; };
    };

    ApproxField(MoFEM::Interface &m_field) : feApprox(m_field) {}
    virtual ~ApproxField() = default;

    MyTriFE feApprox;
    MyTriFE &getLoopFeApprox() { return feApprox; }

    /** \brief Lhs operator used to build matrix
     */
    struct OpLhs
        : public MoFEM::FaceElementForcesAndSourcesCore::UserDataOperator {

      OpLhs(const std::string field_name);

      MatrixDouble NN, transNN;
      MoFEMErrorCode doWork(int row_side, int col_side, EntityType row_type,
                            EntityType col_type,
                            EntitiesFieldData::EntData &row_data,
                            EntitiesFieldData::EntData &col_data);
    };

    /** \brief Rhs operator used to build matrix
     */
    template <typename FUNEVAL>
    struct OpRhs
        : public MoFEM::FaceElementForcesAndSourcesCore::UserDataOperator {

      // Range tRis;
      boost::shared_ptr<FUNEVAL> functionEvaluator;
      int fieldNumber;

      OpRhs(const std::string field_name,
            boost::shared_ptr<FUNEVAL> function_evaluator, int field_number)
          : MoFEM::FaceElementForcesAndSourcesCore::UserDataOperator(
                field_name, ForcesAndSourcesCore::UserDataOperator::OPROW),
            functionEvaluator(function_evaluator), fieldNumber(field_number) {}

      VectorDouble NTf;
      VectorInt iNdices;

      MoFEMErrorCode doWork(int side, EntityType type,
                            EntitiesFieldData::EntData &data) {
        MoFEMFunctionBegin;

        unsigned int nb_row = data.getIndices().size();
        if (nb_row == 0)
          MoFEMFunctionReturnHot(0);

        const auto &dof_ptr = data.getFieldDofs()[0];
        unsigned int rank = dof_ptr->getNbOfCoeffs();

        const auto &gauss_pts = getGaussPts();
        const auto &coords_at_gauss_pts = getCoordsAtGaussPts();

        NTf.resize(nb_row / rank);
        iNdices.resize(nb_row / rank);

        for (unsigned int gg = 0; gg < data.getN().size1(); gg++) {

          const double area = norm_2(getNormalsAtGaussPts(gg)) * 0.5;
          const double val = gauss_pts(2, gg) * area;
          const double x = coords_at_gauss_pts(gg, 0);
          const double y = coords_at_gauss_pts(gg, 1);
          const double z = coords_at_gauss_pts(gg, 2);

          VectorDouble a = (*functionEvaluator)(x, y, z)[fieldNumber];

          if (a.size() != rank) {
            SETERRQ(PETSC_COMM_SELF, MOFEM_DATA_INCONSISTENCY,
                    "data inconsistency");
          }

          for (unsigned int rr = 0; rr < rank; rr++) {

            ublas::noalias(iNdices) = ublas::vector_slice<VectorInt>(
                data.getIndices(),
                ublas::slice(rr, rank, data.getIndices().size() / rank));

            noalias(NTf) = data.getN(gg, nb_row / rank) * a[rr] * val;
            CHKERR VecSetValues(getFEMethod()->snes_f, iNdices.size(),
                                &iNdices[0], &*NTf.data().begin(), ADD_VALUES);
          }
        }

        MoFEMFunctionReturn(0);
      }
    };
  };

  /**
   * \brief Structure used to enforce analytical boundary conditions
   */
  struct DirichletBC : public DirichletDisplacementBc {

    DirichletBC(MoFEM::Interface &m_field, const std::string &field, Mat A,
                Vec X, Vec F);

    DirichletBC(MoFEM::Interface &m_field, const std::string &field);

    boost::shared_ptr<Range> trisPtr;

    MoFEMErrorCode iNitialize();
    MoFEMErrorCode iNitialize(Range &tris);
  };

  ApproxField approxField;
  AnalyticalDirichletBC(MoFEM::Interface &m_field);

  /**
   * \brief Set operators used to calculate the rhs vector and the lhs matrix

   * To enforce analytical function on boundary, first function has to be
   approximated
   * by finite element base functions. This is done by solving system of linear
   * equations, Following function set finite element operators to calculate
   * the left hand side matrix and the left hand side matrix.

   * @param  m_field            interface
   * @param  field_name         field name
   * @param  function_evaluator analytical function to evaluate
   * @param  field_number       field index
   * @param  nodals_positions   name of the field for ho-geometry description
   * @return                    error code
   */
  template <typename FUNEVAL>
  MoFEMErrorCode
  setApproxOps(MoFEM::Interface &m_field, const std::string field_name,
               boost::shared_ptr<FUNEVAL> function_evaluator,
               const int field_number = 0,
               const string nodals_positions = "MESH_NODE_POSITIONS") {
    MoFEMFunctionBeginHot;
    if (approxField.getLoopFeApprox().getOpPtrVector().empty()) {
      if (m_field.check_field(nodals_positions))
        CHKERR addHOOpsFace3D(nodals_positions, approxField.getLoopFeApprox(),
                              false, false);
      approxField.getLoopFeApprox().getOpPtrVector().push_back(
          new ApproxField::OpLhs(field_name));
    }
    approxField.getLoopFeApprox().getOpPtrVector().push_back(
        new ApproxField::OpRhs<FUNEVAL>(field_name, function_evaluator,
                                        field_number));
    MoFEMFunctionReturnHot(0);
  }

  // /**
  //  * \deprecated no need to use function with argument of triangle range
  //  */
  // template<typename FUNEVAL> DEPRECATED MoFEMErrorCode setApproxOps(
  //   MoFEM::Interface &m_field,
  //   string field_name,
  //   Range& tris,
  //   boost::shared_ptr<FUNEVAL> function_evaluator,
  //   int field_number = 0,
  //   string nodals_positions = "MESH_NODE_POSITIONS"
  // ) {
  //   return setApproxOps(
  //     m_field,field_name,tris,function_evaluator,field_number,nodals_positions
  //   );
  // }

  /**
   * \brief set finite element
   * @param  m_field          mofem interface
   * @param  fe               finite element name
   * @param  field            field name
   * @param  tris             faces where analytical boundary is given
   * @param  nodals_positions field having higher order geometry description
   * @return                  error code
   */
  MoFEMErrorCode
  setFiniteElement(MoFEM::Interface &m_field, string fe, string field,
                   Range &tris,
                   string nodals_positions = "MESH_NODE_POSITIONS");

  // /**
  // \deprecated use setFiniteElement instead
  // */
  // DEPRECATED MoFEMErrorCode initializeProblem(
  //   MoFEM::Interface &m_field,
  //   string fe,
  //   string field,
  //   Range& tris,
  //   string nodals_positions = "MESH_NODE_POSITIONS"
  // ) {
  //   return setFiniteElement(m_field,fe,field,tris,nodals_positions);
  // }

  Mat A;
  Vec D, F;
  KSP kspSolver;

  /**
   * \brief set problem solver and create matrices and vectors
   * @param  m_field mofem interface
   * @param  problem problem name
   * @return         error code
   */
  MoFEMErrorCode setUpProblem(MoFEM::Interface &m_field, string problem);

  // /**
  //  * \deprecated use setUpProblem instead
  //  */
  // DEPRECATED MoFEMErrorCode setProblem(
  //   MoFEM::Interface &m_field,string problem
  // ) {
  //   return setUpProblem(m_field,problem);
  // }

  /**
   * \brief solve boundary problem
   *

   * This functions solve for DOFs on boundary where analytical solution is
   * given. i.e. finding values of DOFs which approximate analytical solution.

   * @param  m_field mofem interface
   * @param  problem problem name
   * @param  fe      finite element name
   * @param  bc      Driblet boundary structure used to apply boundary
   conditions
   * @param  tris    triangles on boundary
   * @return         error code
   */
  MoFEMErrorCode solveProblem(MoFEM::Interface &m_field, string problem,
                              string fe, DirichletBC &bc, Range &tris);

  /**
   * \brief solve boundary problem
   *

   * This functions solve for DOFs on boundary where analytical solution is
   * given.

   * @param  m_field mofem interface
   * @param  problem problem name
   * @param  fe      finite element name
   * @param  bc      Driblet boundary structure used to apply boundary
   conditions
   * @return         [description]
   */
  MoFEMErrorCode solveProblem(MoFEM::Interface &m_field, string problem,
                              string fe, DirichletBC &bc);

  /**
   * \brief Destroy problem
   *
   * Destroy matrices and vectors used to solve boundary problem, i.e. finding
   * values of DOFs which approximate analytical solution.
   *
   * @return error code
   */
  MoFEMErrorCode destroyProblem();
};

#endif //__ANALYTICALDIRICHLETBC_HPP__
