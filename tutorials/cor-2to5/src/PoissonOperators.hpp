/**
 * \file PoissonOperators.hpp
 * \example PoissonOperators.hpp
 *
 */

/* MIT License
 *
 * Copyright (c) 2022
 * 
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 * 
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 * 
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#ifndef __POISSONOPERATORS_HPP__
#define __POISSONOPERATORS_HPP__

namespace PoissonExample {

/**
 * \brief Calculate the grad-grad operator and assemble matrix
 *
 * Calculate
 * \f[
 * \mathbf{K}=\int_\Omega \nabla \boldsymbol\phi \cdot \nabla \boldsymbol\phi
 * \textrm{d}\Omega \f] and assemble to global matrix.
 *
 * This operator is executed on element for each unique combination of entities.
 *
 */
struct OpK : public VolumeElementForcesAndSourcesCore::UserDataOperator {

  OpK(bool symm = true)
      : VolumeElementForcesAndSourcesCore::UserDataOperator("U", "U", OPROWCOL,
                                                            symm) {}

  /**
   * \brief Do calculations for give operator
   * @param  row_side row side number (local number) of entity on element
   * @param  col_side column side number (local number) of entity on element
   * @param  row_type type of row entity MBVERTEX, MBEDGE, MBTRI or MBTET
   * @param  col_type type of column entity MBVERTEX, MBEDGE, MBTRI or MBTET
   * @param  row_data data for row
   * @param  col_data data for column
   * @return          error code
   */
  MoFEMErrorCode doWork(int row_side, int col_side, EntityType row_type,
                        EntityType col_type,
                        EntitiesFieldData::EntData &row_data,
                        EntitiesFieldData::EntData &col_data) {
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

protected:
  ///< error code

  int nbRows;           ///< number of dofs on rows
  int nbCols;           ///< number if dof on column
  int nbIntegrationPts; ///< number of integration points
  bool isDiag;          ///< true if this block is on diagonal

  FTensor::Index<'i', 3> i; ///< summit Index
  MatrixDouble locMat;      ///< local entity block matrix

  /**
   * \brief Integrate grad-grad operator
   * @param  row_data row data (consist base functions on row entity)
   * @param  col_data column data (consist base functions on column entity)
   * @return          error code
   */
  virtual MoFEMErrorCode
  iNtegrate(EntitiesFieldData::EntData &row_data,
            EntitiesFieldData::EntData &col_data) {
    MoFEMFunctionBegin;
    // set size of local entity bock
    locMat.resize(nbRows, nbCols, false);
    // clear matrix
    locMat.clear();
    // get element volume
    double vol = getVolume();
    // get integration weights
    auto t_w = getFTensor0IntegrationWeight();
    // get base function gradient on rows
    auto t_row_grad = row_data.getFTensor1DiffN<3>();
    // loop over integration points
    for (int gg = 0; gg != nbIntegrationPts; gg++) {
      // take into account Jacobean
      const double alpha = t_w * vol;
      // noalias(locMat) +=
      // alpha*prod(row_data.getDiffN(gg),trans(col_data.getDiffN(gg))); take
      // fist element to local matrix
      FTensor::Tensor0<double *> a(&*locMat.data().begin());
      // loop over rows base functions
      for (int rr = 0; rr != nbRows; rr++) {
        // get column base functions gradient at gauss point gg
        auto t_col_grad = col_data.getFTensor1DiffN<3>(gg, 0);
        // loop over columns
        for (int cc = 0; cc != nbCols; cc++) {
          // calculate element of local matrix
          a += alpha * (t_row_grad(i) * t_col_grad(i));
          ++t_col_grad; // move to another gradient of base function on column
          ++a;          // move to another element of local matrix in column
        }
        ++t_row_grad; // move to another element of gradient of base function on
                      // row
      }
      ++t_w; // move to another integration weight
    }
    MoFEMFunctionReturn(0);
  }

  /**
   * \brief Assemble local entity block matrix
   * @param  row_data row data (consist base functions on row entity)
   * @param  col_data column data (consist base functions on column entity)
   * @return          error code
   */
  virtual MoFEMErrorCode aSsemble(EntitiesFieldData::EntData &row_data,
                                  EntitiesFieldData::EntData &col_data) {
    MoFEMFunctionBegin;
    // get pointer to first global index on row
    const int *row_indices = &*row_data.getIndices().data().begin();
    // get pointer to first global index on column
    const int *col_indices = &*col_data.getIndices().data().begin();
    Mat B = getFEMethod()->ksp_B != PETSC_NULL ? getFEMethod()->ksp_B
                                               : getFEMethod()->snes_B;
    // assemble local matrix
    CHKERR MatSetValues(B, nbRows, row_indices, nbCols, col_indices,
                        &*locMat.data().begin(), ADD_VALUES);

    if (!isDiag && sYmm) {
      // if not diagonal term and since global matrix is symmetric assemble
      // transpose term.
      locMat = trans(locMat);
      CHKERR MatSetValues(B, nbCols, col_indices, nbRows, row_indices,
                          &*locMat.data().begin(), ADD_VALUES);
    }
    MoFEMFunctionReturn(0);
  }
};

/**
 * \brief template class for integration oh the right hand side
 */
template <typename OPBASE> struct OpBaseRhs : public OPBASE {

  OpBaseRhs(const std::string field_name) : OPBASE(field_name, OPBASE::OPROW) {}

  /**
   * \brief This function is called by finite element
   *
   * Do work is composed from two operations, integrate and assembly. Also,
   * it set values nbRows, and nbIntegrationPts.
   *
   */
  MoFEMErrorCode doWork(int row_side, EntityType row_type,
                        EntitiesFieldData::EntData &row_data) {
    MoFEMFunctionBegin;
    // get number of dofs on row
    nbRows = row_data.getIndices().size();
    if (!nbRows)
      MoFEMFunctionReturnHot(0);
    // get number of integration points
    nbIntegrationPts = OPBASE::getGaussPts().size2();
    // integrate local vector
    CHKERR iNtegrate(row_data);
    // assemble local vector
    CHKERR aSsemble(row_data);
    MoFEMFunctionReturn(0);
  }

  /**
   * \brief Class dedicated to integrate operator
   * @param  data entity data on element row
   * @return      error code
   */
  virtual MoFEMErrorCode iNtegrate(EntitiesFieldData::EntData &data) = 0;

  /**
   * \brief Class dedicated to assemble operator to global system vector
   * @param  data entity data (indices, base functions, etc. ) on element row
   * @return      error code
   */
  virtual MoFEMErrorCode aSsemble(EntitiesFieldData::EntData &data) = 0;

protected:
  ///< error code
  int nbRows;           ///< number of dofs on row
  int nbIntegrationPts; ///< number of integration points
};

/**
 * \brief Operator calculate source term,
 *
 * \f[
 * \mathbf{F} = \int_\Omega \boldsymbol\phi f \textrm{d}\Omega
 * \f]
 *
 */
struct OpF
    : public OpBaseRhs<VolumeElementForcesAndSourcesCore::UserDataOperator> {

  typedef boost::function<double(const double, const double, const double)>
      FSource;

  OpF(FSource f_source)
      : OpBaseRhs<VolumeElementForcesAndSourcesCore::UserDataOperator>("U"),
        fSource(f_source) {}

protected:
  FTensor::Number<0> NX;
  FTensor::Number<1> NY;
  FTensor::Number<2> NZ;
  FSource fSource;

  VectorDouble locVec;

  /**
   * \brief Integrate local entity vector
   * @param  data entity data on element row
   * @return      error code
   */
  MoFEMErrorCode iNtegrate(EntitiesFieldData::EntData &data) {
    MoFEMFunctionBegin;
    // set size of local vector
    locVec.resize(nbRows, false);
    // clear local entity vector
    locVec.clear();
    // get finite element volume
    double vol = getVolume();
    // get integration weights
    auto t_w = getFTensor0IntegrationWeight();
    // get base functions on entity
    auto t_v = data.getFTensor0N();
    // get coordinates at integration points
    auto t_coords = getFTensor1CoordsAtGaussPts();
    // loop over all integration points
    for (int gg = 0; gg != nbIntegrationPts; gg++) {
      // evaluate constant term
      const double alpha =
          vol * t_w * fSource(t_coords(NX), t_coords(NY), t_coords(NZ));
      // get element of local vector
      FTensor::Tensor0<FTensor::PackPtr<double *, 1>> t_a(
          &*locVec.data().begin());
      // loop over base functions
      for (int rr = 0; rr != nbRows; rr++) {
        // add to local vector source term
        t_a -= alpha * t_v;
        ++t_a; // move to next element of local vector
        ++t_v; // move to next base function
      }
      ++t_w;      // move to next integration weight
      ++t_coords; // move to next physical coordinates at integration point
    }
    MoFEMFunctionReturn(0);
  }

  /**
   * \brief assemble local entity vector to the global right hand side
   * @param  data entity data, i.e. global indices of local vector
   * @return      error code
   */
  MoFEMErrorCode aSsemble(EntitiesFieldData::EntData &data) {
    MoFEMFunctionBegin;
    // get global indices of local vector
    const int *indices = &*data.getIndices().data().begin();
    // get values from local vector
    const double *vals = &*locVec.data().begin();
    Vec f = getFEMethod()->ksp_f != PETSC_NULL ? getFEMethod()->ksp_f
                                               : getFEMethod()->snes_f;
    // assemble vector
    CHKERR VecSetValues(f, nbRows, indices, vals, ADD_VALUES);
    MoFEMFunctionReturn(0);
  }
};

/**
 * \brief Calculate constrains matrix
 *
 * \f[
 * \mathbf{C} = \int_{\partial\Omega} \boldsymbol\psi \boldsymbol\phi
 * \textrm{d}\partial\Omega \f] where \f$\lambda \f$ is base function on
 * boundary
 *
 */
struct OpC : public FaceElementForcesAndSourcesCore::UserDataOperator {

  OpC(const bool assemble_transpose)
      : FaceElementForcesAndSourcesCore::UserDataOperator("L", "U", OPROWCOL,
                                                          false),
        assembleTranspose(assemble_transpose) {}

  MoFEMErrorCode doWork(int row_side, int col_side, EntityType row_type,
                        EntityType col_type,
                        EntitiesFieldData::EntData &row_data,
                        EntitiesFieldData::EntData &col_data) {
    MoFEMFunctionBegin;
    // get number of dofs on row
    nbRows = row_data.getIndices().size();
    // exit here if no dofs on row, nothing to do
    if (!nbRows)
      MoFEMFunctionReturnHot(0);
    // get number of dofs on column,
    nbCols = col_data.getIndices().size();
    // exit here if no dofs on roe, nothing to do
    if (!nbCols)
      MoFEMFunctionReturnHot(0);
    // get number of integration points
    nbIntegrationPts = getGaussPts().size2();
    // integrate local constrains matrix
    CHKERR iNtegrate(row_data, col_data);
    // assemble local constrains matrix
    CHKERR aSsemble(row_data, col_data);
    MoFEMFunctionReturn(0);
  }

private:
  ///< error code

  int nbRows;                   ///< number of dofs on row
  int nbCols;                   ///< number of dofs on column
  int nbIntegrationPts;         ///< number of integration points
  const bool assembleTranspose; ///< assemble transpose, i.e. CT if set to true

  MatrixDouble locMat; ///< local constrains matrix

  /** \brief Integrate local constrains matrix
   */
  inline MoFEMErrorCode iNtegrate(EntitiesFieldData::EntData &row_data,
                                  EntitiesFieldData::EntData &col_data) {
    MoFEMFunctionBegin;
    // set size of local constrains matrix
    locMat.resize(nbRows, nbCols, false);
    // clear matrix
    locMat.clear();
    // get area of element
    const double area = getArea();
    // get integration weights
    auto t_w = getFTensor0IntegrationWeight();
    // get base functions on entity
    auto t_row = row_data.getFTensor0N();
    // run over integration points
    for (int gg = 0; gg != nbIntegrationPts; gg++) {
      const double alpha = area * t_w;
      // get element of local matrix
      FTensor::Tensor0<FTensor::PackPtr<double *, 1>> c(
          &*locMat.data().begin());
      // run over base functions on rows
      for (int rr = 0; rr != nbRows; rr++) {
        // get first base functions on column for integration point gg
        auto t_col = col_data.getFTensor0N(gg, 0);
        // run over base function on column
        for (int cc = 0; cc != nbCols; cc++) {
          // integrate element of constrains matrix
          c += alpha * t_row * t_col;
          ++t_col; // move to next base function on column
          ++c;     // move to next element of local matrix
        }
        ++t_row; // move to next base function on row
      }
      ++t_w; // move to next integrate weight
    }
    MoFEMFunctionReturn(0);
  }

  /**
   * \brief integrate local constrains matrix
   */
  inline MoFEMErrorCode aSsemble(EntitiesFieldData::EntData &row_data,
                                 EntitiesFieldData::EntData &col_data) {
    MoFEMFunctionBegin;
    // get indices on row
    const int *row_indices = &*row_data.getIndices().data().begin();
    // get indices on column
    const int *col_indices = &*col_data.getIndices().data().begin();
    Mat B = getFEMethod()->ksp_B != PETSC_NULL ? getFEMethod()->ksp_B
                                               : getFEMethod()->snes_B;
    // assemble local matrix
    CHKERR MatSetValues(B, nbRows, row_indices, nbCols, col_indices,
                        &*locMat.data().begin(), ADD_VALUES);
    // cerr << locMat << endl;
    if (assembleTranspose) {
      // assemble transpose of local matrix
      locMat = trans(locMat);
      CHKERR MatSetValues(B, nbCols, col_indices, nbRows, row_indices,
                          &*locMat.data().begin(), ADD_VALUES);
    }
    MoFEMFunctionReturn(0);
  }
};

/**
 * \brief Assemble constrains vector
 *
 * \f[
 * \mathbf{g} = \int_{\partial\Omega} \boldsymbol\psi \overline{u}
 * \textrm{d}\partial\Omega \f]
 *
 */
struct Op_g
    : public OpBaseRhs<FaceElementForcesAndSourcesCore::UserDataOperator> {

  typedef boost::function<double(const double, const double, const double)>
      FVal;

  Op_g(FVal f_value, const string field_name = "L", const double beta = 1)
      : OpBaseRhs<FaceElementForcesAndSourcesCore::UserDataOperator>(
            field_name),
        fValue(f_value), bEta(beta) {}

protected:
  FTensor::Number<0> NX; ///< x-direction index
  FTensor::Number<1> NY; ///< y-direction index
  FTensor::Number<2> NZ; ///< z-direction index
  FVal fValue; ///< Function pointer evaluating values of "U" at the boundary

  VectorDouble locVec;
  const double bEta;

  /**
   * \brief Integrate local constrains vector
   */
  MoFEMErrorCode iNtegrate(EntitiesFieldData::EntData &data) {
    MoFEMFunctionBegin;
    // set size to local vector
    locVec.resize(nbRows, false);
    // clear local vector
    locVec.clear();
    // get face area
    const double area = getArea() * bEta;
    // get integration weight
    auto t_w = getFTensor0IntegrationWeight();
    // get base function
    auto t_l = data.getFTensor0N();
    // get coordinates at integration point
    auto t_coords = getFTensor1CoordsAtGaussPts();
    // make loop over integration points
    for (int gg = 0; gg != nbIntegrationPts; gg++) {
      // evaluate function on boundary and scale it by area and integration
      // weight
      double alpha =
          area * t_w * fValue(t_coords(NX), t_coords(NY), t_coords(NZ));
      // get element of vector
      FTensor::Tensor0<FTensor::PackPtr<double *, 1>> t_a(
          &*locVec.data().begin());
      //
      for (int rr = 0; rr != nbRows; rr++) {
        t_a += alpha * t_l;
        ++t_a;
        ++t_l;
      }
      ++t_w;
      ++t_coords;
    }
    MoFEMFunctionReturn(0);
  }

  /**
   * \brief assemble constrains vectors
   */
  MoFEMErrorCode aSsemble(EntitiesFieldData::EntData &data) {
    MoFEMFunctionBegin;
    const int *indices = &*data.getIndices().data().begin();
    const double *vals = &*locVec.data().begin();
    Vec f = getFEMethod()->ksp_f != PETSC_NULL ? getFEMethod()->ksp_f
                                               : getFEMethod()->snes_f;
    CHKERR VecSetValues(f, nbRows, indices, &*vals, ADD_VALUES);
    MoFEMFunctionReturn(0);
  }
};

/**
 * \brief Evaluate error
 */
struct OpError
    : public OpBaseRhs<VolumeElementForcesAndSourcesCore::UserDataOperator> {

  typedef boost::function<double(const double, const double, const double)>
      UVal;
  typedef boost::function<FTensor::Tensor1<double, 3>(
      const double, const double, const double)>
      GVal;

  OpError(UVal u_value, GVal g_value,
          boost::shared_ptr<VectorDouble> &field_vals,
          boost::shared_ptr<MatrixDouble> &grad_vals, Vec global_error)
      : OpBaseRhs<VolumeElementForcesAndSourcesCore::UserDataOperator>("ERROR"),
        globalError(global_error), uValue(u_value), gValue(g_value),
        fieldVals(field_vals), gradVals(grad_vals) {}

  MoFEMErrorCode doWork(int row_side, EntityType row_type,
                        EntitiesFieldData::EntData &row_data) {
    MoFEMFunctionBegin;
    nbRows = row_data.getFieldData().size();
    if (!nbRows)
      MoFEMFunctionReturnHot(0);
    nbIntegrationPts = getGaussPts().size2();
    CHKERR iNtegrate(row_data);
    CHKERR aSsemble(row_data);
    MoFEMFunctionReturn(0);
  }

private:
  Vec globalError; ///< ghost vector with global (integrated over volume) error

  FTensor::Number<0> NX;
  FTensor::Number<1> NY;
  FTensor::Number<2> NZ;
  FTensor::Index<'i', 3> i;
  UVal uValue; ///< function with exact solution
  GVal gValue; ///< function with exact solution for gradient

  boost::shared_ptr<VectorDouble> fieldVals;
  boost::shared_ptr<MatrixDouble> gradVals;

  /**
   * \brief Integrate error
   */
  MoFEMErrorCode iNtegrate(EntitiesFieldData::EntData &data) {
    MoFEMFunctionBegin;
    // clear field dofs
    data.getFieldData().clear();
    // get volume of element
    const double vol = getVolume();
    // get integration weight
    auto t_w = getFTensor0IntegrationWeight();
    // get solution at integration point
    auto t_u = getFTensor0FromVec(*fieldVals);
    // get solution at integration point
    auto t_grad = getFTensor1FromMat<3>(*gradVals);
    // get coordinates at integration point
    auto t_coords = getFTensor1CoordsAtGaussPts();
    // keep exact gradient and error or gradient
    FTensor::Tensor1<double, 3> t_exact_grad, t_error_grad;
    // integrate over
    for (int gg = 0; gg != nbIntegrationPts; gg++) {
      double alpha = vol * t_w;
      // evaluate exact value
      double exact_u = uValue(t_coords(NX), t_coords(NY), t_coords(NZ));
      // evaluate exact gradient
      t_exact_grad = gValue(t_coords(NX), t_coords(NY), t_coords(NZ));
      // calculate gradient errro
      t_error_grad(i) = t_grad(i) - t_exact_grad(i);
      // error
      double error = pow(t_u - exact_u, 2) + t_error_grad(i) * t_error_grad(i);
      // iterate over base functions
      data.getFieldData()[0] += alpha * error;
      ++t_w;      // move to next integration point
      ++t_u;      // next value of function at integration point
      ++t_grad;   // next gradient at integration point
      ++t_coords; // next coordinate at integration point
    }
    MoFEMFunctionReturn(0);
  }

  /**
   * \brief Assemble error
   */
  MoFEMErrorCode aSsemble(EntitiesFieldData::EntData &data) {
    MoFEMFunctionBegin;
    // set error on mesh
    data.getFieldDofs()[0]->getFieldData() = sqrt(data.getFieldData()[0]);
    // assemble vector to global error
    CHKERR VecSetValue(globalError, 0, data.getFieldData()[0], ADD_VALUES);
    MoFEMFunctionReturn(0);
  }
};

struct OpKt : public OpK {

  OpKt(boost::function<double(const double)> a,
       boost::function<double(const double)> diff_a,
       boost::shared_ptr<VectorDouble> &field_vals,
       boost::shared_ptr<MatrixDouble> &grad_vals)
      : OpK(false), A(a), diffA(diff_a), fieldVals(field_vals),
        gradVals(grad_vals) {}

protected:
  /**
   * \brief Integrate grad-grad operator
   * @param  row_data row data (consist base functions on row entity)
   * @param  col_data column data (consist base functions on column entity)
   * @return          error code
   */
  inline MoFEMErrorCode iNtegrate(EntitiesFieldData::EntData &row_data,
                                  EntitiesFieldData::EntData &col_data) {
    MoFEMFunctionBegin;
    // set size of local entity bock
    locMat.resize(nbRows, nbCols, false);
    // clear matrix
    locMat.clear();
    // get element volume
    double vol = getVolume();
    // get integration weights
    auto t_w = getFTensor0IntegrationWeight();
    // get solution at integration point
    auto t_u = getFTensor0FromVec(*fieldVals);
    // get solution at integration point
    auto t_grad = getFTensor1FromMat<3>(*gradVals);
    // get base function gradient on rows
    auto t_row_grad = row_data.getFTensor1DiffN<3>();
    // loop over integration points
    for (int gg = 0; gg != nbIntegrationPts; gg++) {
      // take into account Jacobian
      const double alpha = t_w * vol;
      const double beta = alpha * A(t_u);
      FTensor::Tensor1<double, 3> t_gamma;
      t_gamma(i) = (alpha * diffA(t_u)) * t_grad(i);
      // take fist element to local matrix
      FTensor::Tensor0<FTensor::PackPtr<double *, 1>> a(
          &*locMat.data().begin());
      // loop over rows base functions
      for (int rr = 0; rr != nbRows; rr++) {
        // get column base function
        auto t_col = col_data.getFTensor0N(gg, 0);
        // get column base functions gradient at gauss point gg
        auto t_col_grad = col_data.getFTensor1DiffN<3>(gg, 0);
        // loop over columns
        for (int cc = 0; cc != nbCols; cc++) {
          // calculate element of local matrix
          a += (t_row_grad(i) * beta) * t_col_grad(i) +
               t_row_grad(i) * (t_gamma(i) * t_col);
          ++t_col;      // move to next base function
          ++t_col_grad; // move to another gradient of base function on column
          ++a;          // move to another element of local matrix in column
        }
        ++t_row_grad; // move to another element of gradient of base function on
                      // row
      }
      ++t_w;    // move to another integration weight
      ++t_u;    // move to next value at integration point
      ++t_grad; // move to next gradient value
    }
    MoFEMFunctionReturn(0);
  }

  boost::function<double(const double)> A;
  boost::function<double(const double)> diffA;
  boost::shared_ptr<VectorDouble> fieldVals;
  boost::shared_ptr<MatrixDouble> gradVals;
};

struct OpResF_Domain : public OpF {

  OpResF_Domain(FSource f_source, boost::function<double(const double)> a,
                boost::shared_ptr<VectorDouble> &field_vals,
                boost::shared_ptr<MatrixDouble> &grad_vals)
      : OpF(f_source), A(a), fieldVals(field_vals), gradVals(grad_vals) {}

protected:
  FTensor::Index<'i', 3> i;

  /**
   * \brief Integrate local entity vector
   * @param  data entity data on element row
   * @return      error code
   */
  MoFEMErrorCode iNtegrate(EntitiesFieldData::EntData &data) {
    MoFEMFunctionBegin;
    // set size of local vector
    locVec.resize(nbRows, false);
    // clear local entity vector
    locVec.clear();
    // get finite element volume
    double vol = getVolume();
    // get integration weights
    auto t_w = getFTensor0IntegrationWeight();
    // get solution at integration point
    auto t_u = getFTensor0FromVec(*fieldVals);
    // get solution at integration point
    auto t_grad = getFTensor1FromMat<3>(*gradVals);
    // get base functions on entity
    auto t_v = data.getFTensor0N();
    // get base function gradient on rows
    auto t_v_grad = data.getFTensor1DiffN<3>();
    // get coordinates at integration points
    auto t_coords = getFTensor1CoordsAtGaussPts();
    // loop over all integration points
    for (int gg = 0; gg != nbIntegrationPts; gg++) {
      // evaluate constant term
      const double alpha = vol * t_w;
      const double source_term =
          alpha * fSource(t_coords(NX), t_coords(NY), t_coords(NZ));
      FTensor::Tensor1<double, 3> grad_term;
      grad_term(i) = (alpha * A(t_u)) * t_grad(i);
      // get element of local vector
      FTensor::Tensor0<FTensor::PackPtr<double *, 1>> t_a(
          &*locVec.data().begin());
      // loop over base functions
      for (int rr = 0; rr != nbRows; rr++) {
        // add to local vector source term
        t_a += t_v_grad(i) * grad_term(i) + t_v * source_term;
        ++t_a;      // move to next element of local vector
        ++t_v;      // move to next base function
        ++t_v_grad; // move to next gradient of base function
      }
      ++t_w;      // move to next integration weights
      ++t_u;      // move to next value
      ++t_grad;   // move to next gradient value
      ++t_coords; // move to next physical coordinates at integration point
    }
    MoFEMFunctionReturn(0);
  }

  boost::function<double(const double)> A;
  boost::shared_ptr<VectorDouble> fieldVals;
  boost::shared_ptr<MatrixDouble> gradVals;
};

struct OpRes_g : public Op_g {

  OpRes_g(FVal f_value, boost::shared_ptr<VectorDouble> &field_vals)
      : Op_g(f_value, "L", 1), fieldVals(field_vals) {}

protected:
  boost::shared_ptr<VectorDouble> fieldVals;

  /**
   * \brief Integrate local constrains vector
   */
  MoFEMErrorCode iNtegrate(EntitiesFieldData::EntData &data) {
    MoFEMFunctionBegin;
    // set size to local vector
    locVec.resize(nbRows, false);
    // clear local vector
    locVec.clear();
    // get face area
    const double area = getArea() * bEta;
    // get integration weight
    auto t_w = getFTensor0IntegrationWeight();
    // get base function
    auto t_l = data.getFTensor0N();
    // get solution at integration point
    auto t_u = getFTensor0FromVec(*fieldVals);
    // get coordinates at integration point
    auto t_coords = getFTensor1CoordsAtGaussPts();
    // make loop over integration points
    for (int gg = 0; gg != nbIntegrationPts; gg++) {
      // evaluate function on boundary and scale it by area and integration
      // weight
      double alpha = area * t_w;
      // get element of vector
      FTensor::Tensor0<FTensor::PackPtr<double *, 1>> t_a(
          &*locVec.data().begin());
      for (int rr = 0; rr != nbRows; rr++) {
        t_a += alpha * t_l *
               (t_u - fValue(t_coords(NX), t_coords(NY), t_coords(NZ)));
        ++t_a;
        ++t_l;
      }
      ++t_w;
      ++t_u;
      ++t_coords;
    }
    MoFEMFunctionReturn(0);
  }
};

struct OpResF_Boundary : public Op_g {

  OpResF_Boundary(boost::shared_ptr<VectorDouble> &lambda_vals)
      : Op_g(FVal(), "U", 1), lambdaVals(lambda_vals) {}

protected:
  boost::shared_ptr<VectorDouble> lambdaVals;

  /**
   * \brief Integrate local constrains vector
   */
  MoFEMErrorCode iNtegrate(EntitiesFieldData::EntData &data) {
    MoFEMFunctionBegin;
    // set size to local vector
    locVec.resize(nbRows, false);
    // clear local vector
    locVec.clear();
    // get face area
    const double area = getArea() * bEta;
    // get integration weight
    auto t_w = getFTensor0IntegrationWeight();
    // get base function
    auto t_u = data.getFTensor0N();
    // get solution at integration point
    auto t_lambda = getFTensor0FromVec(*lambdaVals);
    // make loop over integration points
    for (int gg = 0; gg != nbIntegrationPts; gg++) {
      // evaluate function on boundary and scale it by area and integration
      // weight
      double alpha = area * t_w;
      // get element of vector
      FTensor::Tensor0<FTensor::PackPtr<double *, 1>> t_a(
          &*locVec.data().begin());
      for (int rr = 0; rr != nbRows; rr++) {
        t_a += alpha * t_u * t_lambda;
        ++t_a;
        ++t_u;
      }
      ++t_w;
      ++t_lambda;
    }
    MoFEMFunctionReturn(0);
  }
};

/**
 * \brief Set integration rule to volume elements
 *
 * This rule is used to integrate \f$\nabla v \cdot \nabla u\f$, thus
 * if approximation field and testing field are polynomial order "p",
 * then rule for exact integration is 2*(p-1).
 *
 * Integration rule is order of polynomial which is calculated exactly. Finite
 * element selects integration method based on return of this function.
 *
 */
struct VolRule {
  int operator()(int, int, int p) const { return 2 * (p - 1); }
};

/**
 * \brief Set integration rule to boundary elements
 *
 * This is uses to integrate values on the face. Is used to integrate
 * \f$(\mathbf{n} \cdot \lambda) u\f$, where Lagrange multiplayer
 * is order "p_row" and approximate function is order "p_col".
 *
 * Integration rule is order of polynomial which is calculated exactly. Finite
 * element selects integration method based on return of this function.
 *
 */
struct FaceRule {
  int operator()(int p_row, int p_col, int p_data) const {
    return 2 * p_data + 1;
  }
};

/**
 * \brief Create finite elements instances
 *
 * Create finite element instances and add operators to finite elements.
 *
 */
struct CreateFiniteElements {

  CreateFiniteElements(MoFEM::Interface &m_field) : mField(m_field) {}

  /**
   * \brief Create finite element to calculate matrix and vectors
   */
  MoFEMErrorCode createFEToAssembleMatrixAndVector(
      boost::function<double(const double, const double, const double)> f_u,
      boost::function<double(const double, const double, const double)>
          f_source,
      boost::shared_ptr<ForcesAndSourcesCore> &domain_lhs_fe,
      boost::shared_ptr<ForcesAndSourcesCore> &boundary_lhs_fe,
      boost::shared_ptr<ForcesAndSourcesCore> &domain_rhs_fe,
      boost::shared_ptr<ForcesAndSourcesCore> &boundary_rhs_fe,
      bool trans = true) const {
    MoFEMFunctionBegin;

    // Create elements element instances
    domain_lhs_fe = boost::shared_ptr<ForcesAndSourcesCore>(
        new VolumeElementForcesAndSourcesCore(mField));
    boundary_lhs_fe = boost::shared_ptr<ForcesAndSourcesCore>(
        new FaceElementForcesAndSourcesCore(mField));
    domain_rhs_fe = boost::shared_ptr<ForcesAndSourcesCore>(
        new VolumeElementForcesAndSourcesCore(mField));
    boundary_rhs_fe = boost::shared_ptr<ForcesAndSourcesCore>(
        new FaceElementForcesAndSourcesCore(mField));

    // Set integration rule to elements instances
    domain_lhs_fe->getRuleHook = VolRule();
    domain_rhs_fe->getRuleHook = VolRule();
    boundary_lhs_fe->getRuleHook = FaceRule();
    boundary_rhs_fe->getRuleHook = FaceRule();

    // Add operators to element instances
    // Add operator grad-grad for calculate matrix
    domain_lhs_fe->getOpPtrVector().push_back(new OpK());
    // Add operator to calculate source terms
    domain_rhs_fe->getOpPtrVector().push_back(new OpF(f_source));
    // Add operator calculating constrains matrix
    boundary_lhs_fe->getOpPtrVector().push_back(new OpC(trans));
    // Add operator calculating constrains vector
    boundary_rhs_fe->getOpPtrVector().push_back(new Op_g(f_u));

    MoFEMFunctionReturn(0);
  }

  /**
   * \brief Create finite element to calculate error
   */
  MoFEMErrorCode createFEToEvaluateError(
      boost::function<double(const double, const double, const double)> f_u,
      boost::function<FTensor::Tensor1<double, 3>(const double, const double,
                                                  const double)>
          g_u,
      Vec global_error,
      boost::shared_ptr<ForcesAndSourcesCore> &domain_error) const {
    MoFEMFunctionBegin;
    // Create finite element instance to calculate error
    domain_error = boost::shared_ptr<ForcesAndSourcesCore>(
        new VolumeElementForcesAndSourcesCore(mField));
    domain_error->getRuleHook = VolRule();
    // Set integration rule
    // Crate shared vector storing values of field "u" on integration points on
    // element. element is local and is used to exchange data between operators.
    boost::shared_ptr<VectorDouble> values_at_integration_ptr =
        boost::make_shared<VectorDouble>();
    // Storing gradients of field
    boost::shared_ptr<MatrixDouble> grad_at_integration_ptr =
        boost::make_shared<MatrixDouble>();
    // Add default operator to calculate field values at integration points
    domain_error->getOpPtrVector().push_back(
        new OpCalculateScalarFieldValues("U", values_at_integration_ptr));
    // Add default operator to calculate field gradient at integration points
    domain_error->getOpPtrVector().push_back(
        new OpCalculateScalarFieldGradient<3>("U", grad_at_integration_ptr));
    // Add operator to integrate error element by element.
    domain_error->getOpPtrVector().push_back(
        new OpError(f_u, g_u, values_at_integration_ptr,
                    grad_at_integration_ptr, global_error));
    MoFEMFunctionReturn(0);
  }

  /**
   * \brief Create finite element to post-process results
   */
  MoFEMErrorCode creatFEToPostProcessResults(
      boost::shared_ptr<ForcesAndSourcesCore> &post_proc_volume) const {

    MoFEMFunctionBegin;

    // Note that user can stack together arbitrary number of operators to
    // compose complex PDEs.

    // Post-process results. This is standard element, with functionality
    // enabling refining mesh for post-processing. In addition in
    // PostProcOnRefMesh.hpp are implanted set of  users operators to
    // post-processing fields. Here using simplified mechanism for
    // post-processing finite element, we add operators to save data from field
    // on mesh tags for ParaView visualization.
    post_proc_volume = boost::shared_ptr<ForcesAndSourcesCore>(
        new PostProcVolumeOnRefinedMesh(mField));
    // Add operators to the elements, starting with some generic
    CHKERR boost::static_pointer_cast<PostProcVolumeOnRefinedMesh>(
        post_proc_volume)
        ->generateReferenceElementMesh();
    CHKERR boost::static_pointer_cast<PostProcVolumeOnRefinedMesh>(
        post_proc_volume)
        ->addFieldValuesPostProc("U");
    CHKERR boost::static_pointer_cast<PostProcVolumeOnRefinedMesh>(
        post_proc_volume)
        ->addFieldValuesPostProc("ERROR");
    CHKERR boost::static_pointer_cast<PostProcVolumeOnRefinedMesh>(
        post_proc_volume)
        ->addFieldValuesGradientPostProc("U");

    MoFEMFunctionReturn(0);
  }

  /**
   * \brief Create finite element to calculate matrix and vectors
   */
  MoFEMErrorCode createFEToAssembleMatrixAndVectorForNonlinearProblem(
      boost::function<double(const double, const double, const double)> f_u,
      boost::function<double(const double, const double, const double)>
          f_source,
      boost::function<double(const double)> a,
      boost::function<double(const double)> diff_a,
      boost::shared_ptr<ForcesAndSourcesCore> &domain_lhs_fe,
      boost::shared_ptr<ForcesAndSourcesCore> &boundary_lhs_fe,
      boost::shared_ptr<ForcesAndSourcesCore> &domain_rhs_fe,
      boost::shared_ptr<ForcesAndSourcesCore> &boundary_rhs_fe,
      ForcesAndSourcesCore::RuleHookFun vol_rule,
      ForcesAndSourcesCore::RuleHookFun face_rule = FaceRule(),
      bool trans = true) const {
    MoFEMFunctionBegin;

    // Create elements element instances
    domain_lhs_fe = boost::shared_ptr<ForcesAndSourcesCore>(
        new VolumeElementForcesAndSourcesCore(mField));
    boundary_lhs_fe = boost::shared_ptr<ForcesAndSourcesCore>(
        new FaceElementForcesAndSourcesCore(mField));
    domain_rhs_fe = boost::shared_ptr<ForcesAndSourcesCore>(
        new VolumeElementForcesAndSourcesCore(mField));
    boundary_rhs_fe = boost::shared_ptr<ForcesAndSourcesCore>(
        new FaceElementForcesAndSourcesCore(mField));

    // Set integration rule to elements instances
    domain_lhs_fe->getRuleHook = vol_rule;
    domain_rhs_fe->getRuleHook = vol_rule;
    boundary_lhs_fe->getRuleHook = face_rule;
    boundary_rhs_fe->getRuleHook = face_rule;

    // Set integration rule
    // Crate shared vector storing values of field "u" on integration points on
    // element. element is local and is used to exchange data between operators.
    boost::shared_ptr<VectorDouble> values_at_integration_ptr =
        boost::make_shared<VectorDouble>();
    // Storing gradients of field
    boost::shared_ptr<MatrixDouble> grad_at_integration_ptr =
        boost::make_shared<MatrixDouble>();
    // multipliers values
    boost::shared_ptr<VectorDouble> multiplier_at_integration_ptr =
        boost::make_shared<VectorDouble>();

    // Add operators to element instances
    // Add default operator to calculate field values at integration points
    domain_lhs_fe->getOpPtrVector().push_back(
        new OpCalculateScalarFieldValues("U", values_at_integration_ptr));
    // Add default operator to calculate field gradient at integration points
    domain_lhs_fe->getOpPtrVector().push_back(
        new OpCalculateScalarFieldGradient<3>("U", grad_at_integration_ptr));
    // Add operator grad-(1+u^2)grad for calculate matrix
    domain_lhs_fe->getOpPtrVector().push_back(new OpKt(
        a, diff_a, values_at_integration_ptr, grad_at_integration_ptr));

    // Add default operator to calculate field values at integration points
    domain_rhs_fe->getOpPtrVector().push_back(
        new OpCalculateScalarFieldValues("U", values_at_integration_ptr));
    // Add default operator to calculate field gradient at integration points
    domain_rhs_fe->getOpPtrVector().push_back(
        new OpCalculateScalarFieldGradient<3>("U", grad_at_integration_ptr));
    // Add operator to calculate source terms
    domain_rhs_fe->getOpPtrVector().push_back(new OpResF_Domain(
        f_source, a, values_at_integration_ptr, grad_at_integration_ptr));

    // Add operator calculating constrains matrix
    boundary_lhs_fe->getOpPtrVector().push_back(new OpC(trans));

    // Add default operator to calculate field values at integration points
    boundary_rhs_fe->getOpPtrVector().push_back(
        new OpCalculateScalarFieldValues("U", values_at_integration_ptr));
    // Add default operator to calculate values of Lagrange multipliers
    boundary_rhs_fe->getOpPtrVector().push_back(
        new OpCalculateScalarFieldValues("L", multiplier_at_integration_ptr));
    // Add operator calculating constrains vector
    boundary_rhs_fe->getOpPtrVector().push_back(
        new OpRes_g(f_u, values_at_integration_ptr));
    boundary_rhs_fe->getOpPtrVector().push_back(
        new OpResF_Boundary(multiplier_at_integration_ptr));

    MoFEMFunctionReturn(0);
  }

private:
  MoFEM::Interface &mField;
};

} // namespace PoissonExample

#endif //__POISSONOPERATORS_HPP__
