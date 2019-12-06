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

struct OpBase : public EleOp {

  OpBase(const EleOp::OpType type) : EleOp("U", "U", type, true) {}

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
  bool isDiag;          ///< true if this block is on diagonal

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

MoFEMErrorCode OpBase::doWork(int row_side, int col_side, EntityType row_type,
                              EntityType col_type, EntData &row_data,
                              EntData &col_data) {
  MoFEMFunctionBegin;
  // get number of dofs on row
  nbRows = row_data.getIndices().size();
  // if no dofs on row, exit that work, nothing to do here
  if (!nbRows)
    MoFEMFunctionReturnHot(0);
  // get number of dofs on column
  nbCols = col_data.getIndices().size();
  // if no dofs on Columbia, exit nothing to do here
  if (!nbCols)
    MoFEMFunctionReturnHot(0);
  // get number of integration points
  nbIntegrationPts = getGaussPts().size2();
  // set size of local entity bock
  locMat.resize(nbRows, nbCols, false);
  // clear matrix
  locMat.clear();
  // check if entity block is on matrix diagonal
  if (row_side == col_side && row_type == col_type) {
    isDiag = true; // yes, it is
  } else {
    isDiag = false;
  }
  // integrate local matrix for entity block
  CHKERR iNtegrate(row_data, col_data);
  // assemble local matrix
  CHKERR aSsemble(row_data, col_data);
  MoFEMFunctionReturn(0);
}

/**
 * \brief Assemble local entity block matrix
 * @param  row_data row data (consist base functions on row entity)
 * @param  col_data column data (consist base functions on column entity)
 * @return          error code
 */
MoFEMErrorCode OpBase::aSsemble(EntData &row_data, EntData &col_data) {
  MoFEMFunctionBegin;
  // assemble local matrix
  CHKERR MatSetValues(getFEMethod()->ksp_B, row_data, col_data,
                      &*locMat.data().begin(), ADD_VALUES);

  if (!isDiag && sYmm) {
    // if not diagonal term and since global matrix is symmetric assemble
    // transpose term.
    locMat = trans(locMat);
    CHKERR MatSetValues(getFEMethod()->ksp_B, col_data, row_data,
                        &*locMat.data().begin(), ADD_VALUES);
  }
  MoFEMFunctionReturn(0);
}

/**
 * \brief This function is called by finite element
 *
 * Do work is composed from two operations, integrate and assembly. Also,
 * it set values nbRows, and nbIntegrationPts.
 *
 */
MoFEMErrorCode OpBase::doWork(int row_side, EntityType row_type,
                              EntData &row_data) {
  MoFEMFunctionBegin;
  // get number of dofs on row
  nbRows = row_data.getIndices().size();
  if (!nbRows)
    MoFEMFunctionReturnHot(0);
  // get number of integration points
  nbIntegrationPts = getGaussPts().size2();
  // resize and clear the right hand side vector
  locF.resize(nbRows);
  locF.clear();
  // integrate local vector
  CHKERR iNtegrate(row_data);
  // assemble local vector
  CHKERR aSsemble(row_data);
  MoFEMFunctionReturn(0);
}

/**
 * \brief assemble local entity vector to the global right hand side
 * @param  data entity data, i.e. global indices of local vector
 * @return      error code
 */
MoFEMErrorCode OpBase::aSsemble(DataForcesAndSourcesCore::EntData &data) {
  MoFEMFunctionBegin;
  // get values from local vector
  const double *vals = &*locF.data().begin();
  // assemble vector
  CHKERR VecSetOption(getFEMethod()->ksp_f, VEC_IGNORE_NEGATIVE_INDICES,
                      PETSC_TRUE);
  CHKERR VecSetValues(getFEMethod()->ksp_f, data, vals, ADD_VALUES);
  MoFEMFunctionReturn(0);
}

typedef boost::function<double(const double, const double, const double)>
    ScalarFun;

//! [Source operator]
template <int DIM> struct OpSource : public OpBase {

  ScalarFun sourceFun;

  OpSource(ScalarFun source_fun) : OpBase(OPROW), sourceFun(source_fun) {}

  FTensor::Index<'i', DIM> i; ///< summit Index

  MoFEMErrorCode iNtegrate(EntData &row_data) {
    MoFEMFunctionBegin;
    // get element volume
    const double vol = getMeasure();
    // get integration weights
    auto t_w = getFTensor0IntegrationWeight();
    // get base function gradient on rows
    auto t_row_base = row_data.getFTensor0N();
    // get coordinate at integration points
    auto t_coords = getFTensor1CoordsAtGaussPts();
    // loop over integration points
    for (int gg = 0; gg != nbIntegrationPts; gg++) {
      // take into account Jacobean
      const double alpha = t_w * vol;
      // loop over rows base functions
      for (int rr = 0; rr != nbRows; ++rr) {
        locF[rr] += alpha * t_row_base *
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

//! [Grad operator]
template <int DIM> struct OpGradGrad : public OpBase {

  const double betaCoeff;

  OpGradGrad(const double beta) : OpBase(OPROWCOL), betaCoeff(beta) {}

  FTensor::Index<'i', DIM> i; ///< summit Index

  MoFEMErrorCode iNtegrate(EntData &row_data, EntData &col_data) {
    MoFEMFunctionBegin;
    // get element volume
    const double vol = getMeasure();
    // get integration weights
    auto t_w = getFTensor0IntegrationWeight();
    // get base function gradient on rows
    auto t_row_base = row_data.getFTensor0N();
    auto t_row_grad = row_data.getFTensor1DiffN<DIM>();
    double beta = vol * betaCoeff;
    // loop over integration points
    for (int gg = 0; gg != nbIntegrationPts; gg++) {
      // take into account Jacobean
      const double alpha = t_w * beta;
      // fist element to local matrix
      FTensor::Tensor0<double *> a(&*locMat.data().begin());
      // loop over rows base functions
      for (int rr = 0; rr != nbRows; rr++) {
        // get column base functions gradient at gauss point gg
        auto t_col_base = col_data.getFTensor0N(gg, 0);
        auto t_col_grad = col_data.getFTensor1DiffN<DIM>(gg, 0);
        // loop over columns
        for (int cc = 0; cc != nbCols; cc++) {
          // calculate element of local matrix
          a += alpha * (t_row_grad(i) * t_col_grad(i));
          ++t_col_base;
          ++t_col_grad; // move to another gradient of base function on column
          ++a;          // move to another element of local matrix in column
        }
        ++t_row_base;
        ++t_row_grad; // move to another element of gradient of base function on
                      // row
      }
      ++t_w; // move to another integration weight
    }
    MoFEMFunctionReturn(0);
  }

};
//! [Grad operator]


//! [Mass operator]
template <int DIM> struct OpMass : public OpBase {

  const double betaCoeff;

  OpMass(const double beta) : OpBase(OPROWCOL), betaCoeff(beta) {}

  FTensor::Index<'i', DIM> i; ///< summit Index

  MoFEMErrorCode iNtegrate(EntData &row_data, EntData &col_data) {
    MoFEMFunctionBegin;
    // get element volume
    const double vol = getMeasure();
    // get integration weights
    auto t_w = getFTensor0IntegrationWeight();
    // get base function gradient on rows
    auto t_row_base = row_data.getFTensor0N();
    auto t_row_grad = row_data.getFTensor1DiffN<DIM>();
    double beta = vol * betaCoeff;
    // loop over integration points
    for (int gg = 0; gg != nbIntegrationPts; gg++) {
      // take into account Jacobean
      const double alpha = t_w * beta;
      // fist element to local matrix
      FTensor::Tensor0<double *> a(&*locMat.data().begin());
      // loop over rows base functions
      for (int rr = 0; rr != nbRows; rr++) {
        // get column base functions gradient at gauss point gg
        auto t_col_base = col_data.getFTensor0N(gg, 0);
        auto t_col_grad = col_data.getFTensor1DiffN<DIM>(gg, 0);
        // loop over columns
        for (int cc = 0; cc != nbCols; cc++) {
          // calculate element of local matrix
          a += alpha * (t_row_base * t_col_base);
          ++t_col_base;
          ++t_col_grad; // move to another gradient of base function on column
          ++a;          // move to another element of local matrix in column
        }
        ++t_row_base;
        ++t_row_grad; // move to another element of gradient of base function on
                      // row
      }
      ++t_w; // move to another integration weight
    }
    MoFEMFunctionReturn(0);
  }

};
//! [Mass operator]