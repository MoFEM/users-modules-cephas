/** \file AnalyticalDirichlet.hpp

  Enforce Dirichlet boundary condition for given analytical function,

*/

/* This file is part of MoFEM.
 * MoFEM is free software: you can redistribute it and/or modify it under
 * the terms of the GNU Lesser General Public License as published by the
 * Free Software Foundation, either version 3 of the License, or (at your
 * option) any later version.
 *
 * MoFEM is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public
 * License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with MoFEM. If not, see <http://www.gnu.org/licenses/>. */

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
    virtual ~ApproxField() {}

    MyTriFE feApprox;
    MyTriFE &getLoopFeApprox() { return feApprox; }

    MatrixDouble hoCoords;
    struct OpHoCoord
        : public MoFEM::FaceElementForcesAndSourcesCore::UserDataOperator {

      MatrixDouble &hoCoords;
      OpHoCoord(const std::string field_name, MatrixDouble &ho_coords);

      MoFEMErrorCode doWork(int side, EntityType type,
                            DataForcesAndSourcesCore::EntData &data);
    };

    /** \brief Lhs operator used to build matrix
     */
    struct OpLhs
        : public MoFEM::FaceElementForcesAndSourcesCore::UserDataOperator {

      MatrixDouble &hoCoords;
      OpLhs(const std::string field_name, MatrixDouble &ho_coords);

      MatrixDouble NN, transNN;
      MoFEMErrorCode doWork(int row_side, int col_side, EntityType row_type,
                            EntityType col_type,
                            DataForcesAndSourcesCore::EntData &row_data,
                            DataForcesAndSourcesCore::EntData &col_data);
    };

    /** \brief Rhs operator used to build matrix
     */
    template <typename FUNEVAL>
    struct OpRhs
        : public MoFEM::FaceElementForcesAndSourcesCore::UserDataOperator {

      // Range tRis;
      MatrixDouble &hoCoords;
      boost::shared_ptr<FUNEVAL> functionEvaluator;
      int fieldNumber;

      OpRhs(const std::string field_name,
            // Range tris,
            MatrixDouble &ho_coords,
            boost::shared_ptr<FUNEVAL> function_evaluator, int field_number)
          : MoFEM::FaceElementForcesAndSourcesCore::UserDataOperator(
                field_name, ForcesAndSourcesCore::UserDataOperator::OPROW),
            hoCoords(ho_coords), functionEvaluator(function_evaluator),
            fieldNumber(field_number) {}

      VectorDouble NTf;
      VectorInt iNdices;

      MoFEMErrorCode doWork(int side, EntityType type,
                            DataForcesAndSourcesCore::EntData &data) {
        MoFEMFunctionBegin;

        unsigned int nb_row = data.getIndices().size();
        if (nb_row == 0)
          MoFEMFunctionReturnHot(0);

        const auto &dof_ptr = data.getFieldDofs()[0];
        unsigned int rank = dof_ptr->getNbOfCoeffs();

        NTf.resize(nb_row / rank);
        iNdices.resize(nb_row / rank);

        for (unsigned int gg = 0; gg < data.getN().size1(); gg++) {

          double x, y, z;
          double val = getGaussPts()(2, gg);
          if (hoCoords.size1() == data.getN().size1()) {
            double area = norm_2(getNormalsAtGaussPts(gg)) * 0.5;
            val *= area;
            x = hoCoords(gg, 0);
            y = hoCoords(gg, 1);
            z = hoCoords(gg, 2);
          } else {
            val *= getArea();
            x = getCoordsAtGaussPts()(gg, 0);
            y = getCoordsAtGaussPts()(gg, 1);
            z = getCoordsAtGaussPts()(gg, 2);
          }

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
            ierr = VecSetValues(getFEMethod()->snes_f, iNdices.size(),
                                &iNdices[0], &*NTf.data().begin(), ADD_VALUES);
            CHKERRG(ierr);
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

    MoFEMErrorCode iNitalize();
    MoFEMErrorCode iNitalize(Range &tris);
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
      if (m_field.check_field(nodals_positions)) {
        approxField.getLoopFeApprox().getOpPtrVector().push_back(
            new ApproxField::OpHoCoord(nodals_positions, approxField.hoCoords));
      }
      approxField.getLoopFeApprox().getOpPtrVector().push_back(
          new ApproxField::OpLhs(field_name, approxField.hoCoords));
    }
    approxField.getLoopFeApprox().getOpPtrVector().push_back(
        new ApproxField::OpRhs<FUNEVAL>(field_name, approxField.hoCoords,
                                        function_evaluator, field_number));
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
