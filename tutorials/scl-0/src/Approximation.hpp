/**
 * \file Approximation.cpp
 * \example Approximation.cpp
 *
 * Setting opetators for Least square problem, or to calculate mass matrix for
 * scalar problem.
 *
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

typedef boost::function<double(const double, const double, const double)>
    ScalarFun;

template <typename EleOp> struct OpTools {

  struct OpBase : public EleOp {

    OpBase(const std::string row_field_name, const std::string col_field_name,
           const typename EleOp::OpType type,
           boost::shared_ptr<std::vector<bool>> boundary_marker = nullptr)
        : EleOp(row_field_name, col_field_name, type, false) {}

    /**
     * \brief Do calculations for the left hand side
     * @param  row_side row side number (local number) of entity on element
     * @param  col_side column side number (local number) of entity on element
     * @param  row_type type of row entity MBVERTEX, MBEDGE, MBTRI or MBTET
     * @param  col_type type of column entity MBVERTEX, MBEDGE, MBTRI or MBTET
     * @param  row_data data for row
     * @param  col_data data for column
     * @return          error code
     */
    MoFEMErrorCode doWork(int row_side, int col_side, EntityType row_type,
                          EntityType col_type, EntData &row_data,
                          EntData &col_data);

    /**
     * @brief Do calculations for the right hand side
     *
     * @param row_side
     * @param row_type
     * @param row_data
     * @return MoFEMErrorCode
     */
    MoFEMErrorCode doWork(int row_side, EntityType row_type, EntData &row_data);

  protected:

    int nbRows;           ///< number of dofs on rows
    int nbCols;           ///< number if dof on column
    int nbIntegrationPts; ///< number of integration points

    MatrixDouble locMat; ///< local entity block matrix
    VectorDouble locF;   ///< local entity vector

    /**
     * \brief Integrate grad-grad operator
     * @param  row_data row data (consist base functions on row entity)
     * @param  col_data column data (consist base functions on column entity)
     * @return          error code
     */
    virtual MoFEMErrorCode iNtegrate(EntData &row_data, EntData &col_data) {
      MoFEMFunctionBegin;
      SETERRQ(PETSC_COMM_SELF, MOFEM_NOT_IMPLEMENTED, "Should not be used");
      MoFEMFunctionReturn(0);
    }

    virtual MoFEMErrorCode aSsemble(EntData &row_data, EntData &col_data);

    /**
     * \brief Class dedicated to integrate operator
     * @param  data entity data on element row
     * @return      error code
     */
    virtual MoFEMErrorCode iNtegrate(EntData &data) {
      MoFEMFunctionBegin;
      SETERRQ(PETSC_COMM_SELF, MOFEM_NOT_IMPLEMENTED, "Should not be used");
      MoFEMFunctionReturn(0);
    }

    /**
     * \brief Class dedicated to assemble operator to global system vector
     * @param  data entity data (indices, base functions, etc. ) on element row
     * @return      error code
     */
    virtual MoFEMErrorCode aSsemble(EntData &data);
  };

  //! [Source operator]
  template <int DIM> struct OpSource : public OpBase {

    OpSource(const std::string field_name, ScalarFun source_fun,
             boost::shared_ptr<std::vector<bool>> boundary_marker = nullptr)
        : OpBase(field_name, field_name, OpBase::OPROW, boundary_marker),
          sourceFun(source_fun) {}

  protected:
    ScalarFun sourceFun;
    FTensor::Index<'i', DIM> i; ///< summit Index

    MoFEMErrorCode iNtegrate(EntData &row_data) {
      MoFEMFunctionBegin;
      // get element volume
      const double vol = OpBase::getMeasure();
      // get integration weights
      auto t_w = OpBase::getFTensor0IntegrationWeight();
      // get base function gradient on rows
      auto t_row_base = row_data.getFTensor0N();
      // get coordinate at integration points
      auto t_coords = OpBase::getFTensor1CoordsAtGaussPts();
      // loop over integration points
      for (int gg = 0; gg != OpBase::nbIntegrationPts; gg++) {
        // take into account Jacobean
        const double alpha = t_w * vol;
        // loop over rows base functions
        for (int rr = 0; rr != OpBase::nbRows; ++rr) {
          OpBase::locF[rr] += alpha * t_row_base *
                              sourceFun(t_coords(0), t_coords(1), t_coords(2));
          ++t_row_base;
        }
        ++t_coords;
        ++t_w; // move to another integration weight
      }
      MoFEMFunctionReturn(0);
    }
  };
  //! [Source operator]

  //! [Mass operator]
  struct OpMass : public OpBase {

    OpMass(const std::string row_field_name, const std::string col_field_name,
           ScalarFun beta,
           boost::shared_ptr<std::vector<bool>> boundary_marker = nullptr)
        : OpBase(row_field_name, col_field_name, OpBase::OPROWCOL,
                 boundary_marker),
          betaCoeff(beta) {}

  protected:
    ScalarFun betaCoeff;

    MoFEMErrorCode iNtegrate(EntData &row_data, EntData &col_data) {
      MoFEMFunctionBegin;
      // get element volume
      const double vol = OpBase::getMeasure();
      // get integration weights
      auto t_w = OpBase::getFTensor0IntegrationWeight();
      // get base function gradient on rows
      auto t_row_base = row_data.getFTensor0N();
      // get coordinate at integration points
      auto t_coords = OpBase::getFTensor1CoordsAtGaussPts();

      // loop over integration points
      for (int gg = 0; gg != OpBase::nbIntegrationPts; gg++) {
        const double beta =
            vol * betaCoeff(t_coords(0), t_coords(1), t_coords(2));
        // take into account Jacobean
        const double alpha = t_w * beta;
        // loop over rows base functions
        for (int rr = 0; rr != OpBase::nbRows; rr++) {
          // get column base functions gradient at gauss point gg
          auto t_col_base = col_data.getFTensor0N(gg, 0);
          // loop over columns
          for (int cc = 0; cc != OpBase::nbCols; cc++) {
            // calculate element of local matrix
            OpBase::locMat(rr, cc) += alpha * (t_row_base * t_col_base);
            ++t_col_base;
          }
          ++t_row_base;
        }
        ++t_coords;
        ++t_w; // move to another integration weight
      }
      MoFEMFunctionReturn(0);
    }
  };
  //! [Mass operator]
};

template <typename EleOp>
MoFEMErrorCode
OpTools<EleOp>::OpBase::doWork(int row_side, int col_side, EntityType row_type,
                               EntityType col_type, EntData &row_data,
                               EntData &col_data) {
  MoFEMFunctionBegin;
  // get number of dofs on row
  OpBase::nbRows = row_data.getIndices().size();
  // if no dofs on row, exit that work, nothing to do here
  if (!OpBase::nbRows)
    MoFEMFunctionReturnHot(0);
  // get number of dofs on column
  OpBase::nbCols = col_data.getIndices().size();
  // if no dofs on Columbia, exit nothing to do here
  if (!OpBase::nbCols)
    MoFEMFunctionReturnHot(0);
  // get number of integration points
  OpBase::nbIntegrationPts = OpBase::getGaussPts().size2();
  // set size of local entity bock
  OpBase::locMat.resize(OpBase::nbRows, OpBase::nbCols, false);
  // clear matrix
  OpBase::locMat.clear();
  // integrate local matrix for entity block
  CHKERR this->iNtegrate(row_data, col_data);
  // assemble local matrix
  CHKERR this->aSsemble(row_data, col_data);
  MoFEMFunctionReturn(0);
}

/**
 * \brief Assemble local entity block matrix
 * @param  row_data row data (consist base functions on row entity)
 * @param  col_data column data (consist base functions on column entity)
 * @return          error code
 */
template <typename EleOp>
MoFEMErrorCode OpTools<EleOp>::OpBase::aSsemble(EntData &row_data,
                                                EntData &col_data) {
  MoFEMFunctionBegin;
  // assemble local matrix
  CHKERR MatSetValues(this->getKSPB(), row_data, col_data,
                      &*locMat.data().begin(), ADD_VALUES);
  MoFEMFunctionReturn(0);
}

/**
 * \brief This function is called by finite element
 *
 * Do work is composed from two operations, integrate and assembly. Also,
 * it set values nbRows, and nbIntegrationPts.
 *
 */
template <typename EleOp>
MoFEMErrorCode OpTools<EleOp>::OpBase::doWork(int row_side, EntityType row_type,
                                              EntData &row_data) {
  MoFEMFunctionBegin;
  // get number of dofs on row
  OpBase::nbRows = row_data.getIndices().size();
  if (!OpBase::nbRows)
    MoFEMFunctionReturnHot(0);
  // get number of integration points
  OpBase::nbIntegrationPts = OpBase::getGaussPts().size2();
  // resize and clear the right hand side vector
  OpBase::locF.resize(nbRows);
  OpBase::locF.clear();
  // integrate local vector
  CHKERR this->iNtegrate(row_data);
  // assemble local vector
  CHKERR this->aSsemble(row_data);
  MoFEMFunctionReturn(0);
}

/**
 * \brief assemble local entity vector to the global right hand side
 * @param  data entity data, i.e. global indices of local vector
 * @return      error code
 */
template <typename EleOp>
MoFEMErrorCode
OpTools<EleOp>::OpBase::aSsemble(DataForcesAndSourcesCore::EntData &data) {
  MoFEMFunctionBegin;
  // get values from local vector
  const double *vals = &*locF.data().begin();
  // assemble vector
  CHKERR VecSetOption(this->getKSPf(), VEC_IGNORE_NEGATIVE_INDICES, PETSC_TRUE);
  CHKERR VecSetValues(this->getKSPf(), data, vals, ADD_VALUES);
  MoFEMFunctionReturn(0);
}
