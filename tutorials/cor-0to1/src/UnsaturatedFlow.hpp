/** \file UnsaturatedFlow.hpp
 * \brief Mix implementation of transport element
 * \example UnsaturatedFlow.hpp
 *
 * \ingroup mofem_mix_transport_elem
 *
 */

/*
 * This file is part of MoFEM.
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

#ifndef __UNSATURATD_FLOW_HPP__
#define __UNSATURATD_FLOW_HPP__

namespace MixTransport {

/**
 * \brief Generic material model for unsaturated water transport
 *
 * \note This is abstact class, no physical material model is implemented
 * here.
 */
struct GenericMaterial {

  virtual ~GenericMaterial() {}

  static double ePsilon0; ///< Regularization parameter
  static double ePsilon1; ///< Regularization parameter
  static double scaleZ; ///< Scale z direction
  
  double sCale;           ///< Scale time dependent eq.

  double h;   ///< hydraulic head
  double h_t; ///< rate of hydraulic head
  // double h_flux_residual;  ///< residual at point
  double K;     ///< Hydraulic conductivity [L/s]
  double diffK; ///< Derivative of hydraulic conductivity [L/s * L^2/F]
  double C;     ///< Capacity [S^2/L^2]
  double diffC; ///< Derivative of capacity [S^2/L^2 * L^2/F ]
  double tHeta; ///< Water content
  double Se;    ///< Effective saturation

  Range tEts; ///< Elements with this material

  double x, y, z; ///< in meters (L)

  /**
   * \brief Initialize head
   * @return value of head
   */
  virtual double initialPcEval() const = 0;
  virtual void printMatParameters(const int id,
                                  const std::string &prefix) const = 0;

  virtual MoFEMErrorCode calK() {
    MoFEMFunctionBegin;
    SETERRQ(PETSC_COMM_SELF, MOFEM_NOT_IMPLEMENTED,
            "Not implemented how to calculate hydraulic conductivity");
    MoFEMFunctionReturn(0);
  }

  virtual MoFEMErrorCode calDiffK() {
    MoFEMFunctionBegin;
    SETERRQ(PETSC_COMM_SELF, MOFEM_NOT_IMPLEMENTED,
            "Not implemented how to calculate derivative of hydraulic "
            "conductivity");
    MoFEMFunctionReturn(0);
  }

  virtual MoFEMErrorCode calC() {
    MoFEMFunctionBegin;
    SETERRQ(PETSC_COMM_SELF, MOFEM_NOT_IMPLEMENTED,
            "Not implemented how to calculate capacity");
    MoFEMFunctionReturn(0);
  }

  virtual MoFEMErrorCode calDiffC() {
    MoFEMFunctionBegin;
    SETERRQ(PETSC_COMM_SELF, MOFEM_NOT_IMPLEMENTED,
            "Not implemented how to calculate capacity");
    MoFEMFunctionReturn(0);
  }

  virtual MoFEMErrorCode calTheta() {
    MoFEMFunctionBegin;
    SETERRQ(PETSC_COMM_SELF, MOFEM_NOT_IMPLEMENTED,
            "Not implemented how to calculate capacity");
    MoFEMFunctionReturn(0);
  }

  virtual MoFEMErrorCode calSe() {
    MoFEMFunctionBegin;
    SETERRQ(PETSC_COMM_SELF, MOFEM_NOT_IMPLEMENTED,
            "Not implemented how to calculate capacity");
    MoFEMFunctionReturn(0);
  }
};

/**
 * \brief Implementation of operators, problem and finite elements for
 * unsaturated flow
 */
struct UnsaturatedFlowElement : public MixTransportElement {

  DM dM; ///< Discrete manager for unsaturated flow problem

  UnsaturatedFlowElement(MoFEM::Interface &m_field)
      : MixTransportElement(m_field), dM(PETSC_NULL), lastEvalBcValEnt(0),
        lastEvalBcBlockValId(-1), lastEvalBcFluxEnt(0),
        lastEvalBcBlockFluxId(-1) {}

  ~UnsaturatedFlowElement() {
    if (dM != PETSC_NULL) {
      CHKERR DMDestroy(&dM);
      CHKERRABORT(PETSC_COMM_WORLD, ierr);
    }
  }

  typedef std::map<int, boost::shared_ptr<GenericMaterial>> MaterialsDoubleMap;
  MaterialsDoubleMap dMatMap; ///< materials database

  /**
   * \brief For given element handle get material block Id
   * @param  ent      finite element entity handle
   * @param  block_id reference to returned block id
   * @return          error code
   */
  virtual MoFEMErrorCode getMaterial(const EntityHandle ent,
                                     int &block_id) const {
    MoFEMFunctionBegin;
    for (MaterialsDoubleMap::const_iterator mit = dMatMap.begin();
         mit != dMatMap.end(); mit++) {
      if (mit->second->tEts.find(ent) != mit->second->tEts.end()) {
        block_id = mit->first;
        MoFEMFunctionReturnHot(0);
      }
    }
    SETERRQ(mField.get_comm(), MOFEM_DATA_INCONSISTENCY,
            "Element not found, no material data");
    MoFEMFunctionReturn(0);
  }

  /**
   * \brief Class storing information about boundary condition
   */
  struct BcData {
    Range eNts;
    double fixValue;
    boost::function<double(const double x, const double y, const double z)>
        hookFun;
    BcData() : hookFun(NULL) {}
  };
  typedef map<int, boost::shared_ptr<BcData>> BcMap;
  BcMap bcValueMap; ///< Store boundary condition for head capillary pressure

  EntityHandle lastEvalBcValEnt;
  int lastEvalBcBlockValId;

  /**
   * \brief Get value on boundary
   * @param  ent   entity handle
   * @param  gg    number of integration point
   * @param  x     x-coordinate
   * @param  y     y-coordinate
   * @param  z     z-coordinate
   * @param  value returned value
   * @return       error code
   */
  MoFEMErrorCode getBcOnValues(const EntityHandle ent, const int gg,
                               const double x, const double y, const double z,
                               double &value) {
    MoFEMFunctionBegin;
    int block_id = -1;
    if (lastEvalBcValEnt == ent) {
      block_id = lastEvalBcBlockValId;
    } else {
      for (BcMap::iterator it = bcValueMap.begin(); it != bcValueMap.end();
           it++) {
        if (it->second->eNts.find(ent) != it->second->eNts.end()) {
          block_id = it->first;
        }
      }
      lastEvalBcValEnt = ent;
      lastEvalBcBlockValId = block_id;
    }
    if (block_id >= 0) {
      if (bcValueMap.at(block_id)->hookFun) {
        value = bcValueMap.at(block_id)->hookFun(x, y, z);
      } else {
        value = bcValueMap.at(block_id)->fixValue;
      }
    } else {
      value = 0;
    }
    MoFEMFunctionReturn(0);
  }

  BcMap bcFluxMap;
  EntityHandle lastEvalBcFluxEnt;
  int lastEvalBcBlockFluxId;

  /**
   * \brief essential (Neumann) boundary condition (set fluxes)
   * @param  ent  handle to finite element entity
   * @param  x    coord
   * @param  y    coord
   * @param  z    coord
   * @param  flux reference to flux which is set by function
   * @return      [description]
   */
  MoFEMErrorCode getBcOnFluxes(const EntityHandle ent, const double x,
                               const double y, const double z, double &flux) {
    MoFEMFunctionBegin;
    int block_id = -1;
    if (lastEvalBcFluxEnt == ent) {
      block_id = lastEvalBcBlockFluxId;
    } else {
      for (BcMap::iterator it = bcFluxMap.begin(); it != bcFluxMap.end();
           it++) {
        if (it->second->eNts.find(ent) != it->second->eNts.end()) {
          block_id = it->first;
        }
      }
      lastEvalBcFluxEnt = ent;
      lastEvalBcBlockFluxId = block_id;
    }
    if (block_id >= 0) {
      if (bcFluxMap.at(block_id)->hookFun) {
        flux = bcFluxMap.at(block_id)->hookFun(x, y, z);
      } else {
        flux = bcFluxMap.at(block_id)->fixValue;
      }
    } else {
      flux = 0;
    }
    MoFEMFunctionReturn(0);
  }

  /**
   * \brief Evaluate boundary condition at the boundary
   */
  struct OpRhsBcOnValues
      : public MoFEM::FaceElementForcesAndSourcesCore::UserDataOperator {

    UnsaturatedFlowElement &cTx;
    boost::shared_ptr<MethodForForceScaling> valueScale;

    /**
     * \brief Constructor
     */
    OpRhsBcOnValues(UnsaturatedFlowElement &ctx, const std::string fluxes_name,
                    boost::shared_ptr<MethodForForceScaling> &value_scale)
        : MoFEM::FaceElementForcesAndSourcesCore::UserDataOperator(
              fluxes_name, UserDataOperator::OPROW),
          cTx(ctx), valueScale(value_scale) {}

    VectorDouble nF; ///< Vector of residuals

    /**
     * \brief Integrate boundary condition
     * @param  side local index of entity
     * @param  type type of entity
     * @param  data data on entity
     * @return      error code
     */
    MoFEMErrorCode doWork(int side, EntityType type,
                          DataForcesAndSourcesCore::EntData &data) {
      MoFEMFunctionBegin;
      if (data.getFieldData().size() == 0)
        MoFEMFunctionReturnHot(0);
      // Get EntityHandle of the finite element
      EntityHandle fe_ent = getNumeredEntFiniteElementPtr()->getEnt();
      // Resize and clear vector
      nF.resize(data.getIndices().size());
      nF.clear();
      // Get number of integration points
      int nb_gauss_pts = data.getN().size1();
      for (int gg = 0; gg < nb_gauss_pts; gg++) {
        double x, y, z;
        x = getCoordsAtGaussPts()(gg, 0);
        y = getCoordsAtGaussPts()(gg, 1);
        z = getCoordsAtGaussPts()(gg, 2);
        double value;
        // get value of boundary condition
        CHKERR cTx.getBcOnValues(fe_ent, gg, x, y, z, value);
        const double w = getGaussPts()(2, gg) * 0.5;
        const double beta = w * (value - z);
        noalias(nF) += beta * prod(data.getVectorN<3>(gg), getNormal());
      }
      // Scale vector if history  evaluating method is given
      Vec f = getFEMethod()->ts_F;
      if (valueScale) {
        CHKERR valueScale->scaleNf(getFEMethod(), nF);
      }
      // Assemble vector
      CHKERR VecSetValues(f, data.getIndices().size(), &data.getIndices()[0],
                          &nF[0], ADD_VALUES);
      MoFEMFunctionReturn(0);
    }
  };

  /**
   * \brief Assemble flux residual
   */
  struct OpResidualFlux
      : public MoFEM::VolumeElementForcesAndSourcesCore::UserDataOperator {

    UnsaturatedFlowElement &cTx;

    OpResidualFlux(UnsaturatedFlowElement &ctx, const std::string &flux_name)
        : MoFEM::VolumeElementForcesAndSourcesCore::UserDataOperator(
              flux_name, UserDataOperator::OPROW),
          cTx(ctx) {}

    VectorDouble divVec, nF;
    FTensor::Index<'i', 3> i;

    MoFEMErrorCode doWork(int side, EntityType type,
                          DataForcesAndSourcesCore::EntData &data) {
      MoFEMFunctionBegin;
      const int nb_dofs = data.getIndices().size();
      if (nb_dofs == 0)
        MoFEMFunctionReturnHot(0);
      nF.resize(nb_dofs, false);
      nF.clear();
      // Get EntityHandle of the finite element
      EntityHandle fe_ent = getNumeredEntFiniteElementPtr()->getEnt();
      // Get material block id
      int block_id;
      CHKERR cTx.getMaterial(fe_ent, block_id);
      // Get material block
      boost::shared_ptr<GenericMaterial> &block_data = cTx.dMatMap.at(block_id);
      // block_data->printMatParameters(block_id,"Read material");
      // Get base function
      auto t_n_hdiv = data.getFTensor1N<3>();
      // Get pressure
      auto t_h = getFTensor0FromVec(cTx.valuesAtGaussPts);
      // Get flux
      auto t_flux = getFTensor1FromMat<3>(cTx.fluxesAtGaussPts);
      // Coords at integration points
      auto t_coords = getFTensor1CoordsAtGaussPts();
      // Get integration weight
      auto t_w = getFTensor0IntegrationWeight();
      // Get volume
      double vol = getVolume();

      FTensor::Index<'i', 3> i;
      auto t_base_diff_hdiv = data.getFTensor2DiffN<3, 3>();
      divVec.resize(nb_dofs, false);

      // Get material parameters
      int nb_gauss_pts = data.getN().size1();
      for (int gg = 0; gg != nb_gauss_pts; gg++) {
        // Get divergence
        for(auto &v : divVec) {
          v = t_base_diff_hdiv(i, i);
          ++t_base_diff_hdiv;
        }

        const double alpha = t_w * vol;
        block_data->h = t_h;
        block_data->x = t_coords(0);
        block_data->y = t_coords(1);
        block_data->z = t_coords(2);
        CHKERR block_data->calK();
        const double K = block_data->K;
        const double scaleZ = block_data->scaleZ;
        const double z = t_coords(2); /// z-coordinate at Gauss pt
        // Calculate pressure gradient
        noalias(nF) -= alpha * (t_h - z * scaleZ) * divVec;
        // Calculate presure gradient from flux
        FTensor::Tensor0<double *> t_nf(&*nF.begin());
        for (int rr = 0; rr != nb_dofs; rr++) {
          t_nf += alpha * (1 / K) * (t_n_hdiv(i) * t_flux(i));
          ++t_n_hdiv; // move to next base function
          ++t_nf;     // move to next element in vector
        }
        ++t_h; // move to next integration point
        ++t_flux;
        ++t_coords;
        ++t_w;
      }
      // Assemble residual
      CHKERR VecSetValues(getFEMethod()->ts_F, nb_dofs,
                          &*data.getIndices().begin(), &*nF.begin(),
                          ADD_VALUES);
      MoFEMFunctionReturn(0);
    }
  };

  /**
   * Assemble mass residual
   */
  struct OpResidualMass
      : public MoFEM::VolumeElementForcesAndSourcesCore::UserDataOperator {

    UnsaturatedFlowElement &cTx;

    OpResidualMass(UnsaturatedFlowElement &ctx, const std::string &val_name)
        : MoFEM::VolumeElementForcesAndSourcesCore::UserDataOperator(
              val_name, UserDataOperator::OPROW),
          cTx(ctx) {}

    VectorDouble nF;

    MoFEMErrorCode doWork(int side, EntityType type,
                          DataForcesAndSourcesCore::EntData &data) {
      MoFEMFunctionBegin;
      const int nb_dofs = data.getIndices().size();
      if (nb_dofs == 0)
        MoFEMFunctionReturnHot(0);
      // Resize local element vector
      nF.resize(nb_dofs, false);
      nF.clear();
      // Get EntityHandle of the finite element
      EntityHandle fe_ent = getNumeredEntFiniteElementPtr()->getEnt();
      // Get material block id
      int block_id;
      CHKERR cTx.getMaterial(fe_ent, block_id);
      // Get material block
      boost::shared_ptr<GenericMaterial> &block_data = cTx.dMatMap.at(block_id);
      // Get pressure
      auto t_h = getFTensor0FromVec(cTx.valuesAtGaussPts);
      // Get pressure rate
      auto t_h_t = getFTensor0FromVec(*cTx.headRateAtGaussPts);
      // Flux divergence
      auto t_div_flux = getFTensor0FromVec(cTx.divergenceAtGaussPts);
      // Get integration weight
      auto t_w = getFTensor0IntegrationWeight();
      // Coords at integration points
      auto t_coords = getFTensor1CoordsAtGaussPts();
      // Scale eq.
      const double scale = block_data->sCale;
      // Get volume
      const double vol = getVolume();
      // Get number of integration points
      int nb_gauss_pts = data.getN().size1();
      for (int gg = 0; gg != nb_gauss_pts; gg++) {
        const double alpha = t_w * vol * scale;
        block_data->h = t_h;
        block_data->x = t_coords(0);
        block_data->y = t_coords(1);
        block_data->z = t_coords(2);
        CHKERR block_data->calC();
        const double C = block_data->C;
        // Calculate flux conservation
        noalias(nF) += (alpha * (t_div_flux + C * t_h_t)) * data.getN(gg);
        ++t_h;
        ++t_h_t;
        ++t_div_flux;
        ++t_coords;
        ++t_w;
      }
      // Assemble local vector
      Vec f = getFEMethod()->ts_F;
      CHKERR VecSetValues(f, nb_dofs, &*data.getIndices().begin(), &*nF.begin(),
                          ADD_VALUES);
      MoFEMFunctionReturn(0);
    }
  };

  struct OpTauDotSigma_HdivHdiv
      : public MoFEM::VolumeElementForcesAndSourcesCore::UserDataOperator {

    UnsaturatedFlowElement &cTx;

    OpTauDotSigma_HdivHdiv(UnsaturatedFlowElement &ctx,
                           const std::string flux_name)
        : MoFEM::VolumeElementForcesAndSourcesCore::UserDataOperator(
              flux_name, flux_name, UserDataOperator::OPROWCOL),
          cTx(ctx) {
      sYmm = true;
    }

    MatrixDouble nN, transNN;

    FTensor::Index<'j', 3> j;

    /**
     * \brief Assemble matrix
     * @param  row_side local index of row entity on element
     * @param  col_side local index of col entity on element
     * @param  row_type type of row entity, f.e. MBVERTEX, MBEDGE, or MBTET
     * @param  col_type type of col entity, f.e. MBVERTEX, MBEDGE, or MBTET
     * @param  row_data data for row
     * @param  col_data data for col
     * @return          error code
     */
    MoFEMErrorCode doWork(int row_side, int col_side, EntityType row_type,
                          EntityType col_type,
                          DataForcesAndSourcesCore::EntData &row_data,
                          DataForcesAndSourcesCore::EntData &col_data) {
      MoFEMFunctionBegin;
      const int nb_row = row_data.getIndices().size();
      const int nb_col = col_data.getIndices().size();
      if (nb_row == 0)
        MoFEMFunctionReturnHot(0);
      if (nb_col == 0)
        MoFEMFunctionReturnHot(0);
      nN.resize(nb_row, nb_col, false);
      nN.clear();
      // Get EntityHandle of the finite element
      EntityHandle fe_ent = getNumeredEntFiniteElementPtr()->getEnt();
      // Get material block id
      int block_id;
      CHKERR cTx.getMaterial(fe_ent, block_id);
      // Get material block
      boost::shared_ptr<GenericMaterial> &block_data = cTx.dMatMap.at(block_id);
      // Get pressure
      auto t_h = getFTensor0FromVec(cTx.valuesAtGaussPts);
      // Coords at integration points
      auto t_coords = getFTensor1CoordsAtGaussPts();
      // Get base functions
      auto t_n_hdiv_row = row_data.getFTensor1N<3>();
      // Get integration weight
      auto t_w = getFTensor0IntegrationWeight();
      // Get volume
      const double vol = getVolume();
      int nb_gauss_pts = row_data.getN().size1();
      for (int gg = 0; gg != nb_gauss_pts; gg++) {
        block_data->h = t_h;
        block_data->x = t_coords(0);
        block_data->y = t_coords(1);
        block_data->z = t_coords(2);
        CHKERR block_data->calK();
        const double K = block_data->K;
        // get integration weight and multiply by element volume
        const double alpha = t_w * vol;
        const double beta = alpha * (1 / K);
        FTensor::Tensor0<double *> t_a(&*nN.data().begin());
        for (int kk = 0; kk != nb_row; kk++) {
          auto t_n_hdiv_col = col_data.getFTensor1N<3>(gg, 0);
          for (int ll = 0; ll != nb_col; ll++) {
            t_a += beta * (t_n_hdiv_row(j) * t_n_hdiv_col(j));
            ++t_n_hdiv_col;
            ++t_a;
          }
          ++t_n_hdiv_row;
        }
        ++t_coords;
        ++t_h;
        ++t_w;
      }
      Mat a = getFEMethod()->ts_B;
      CHKERR MatSetValues(a, nb_row, &*row_data.getIndices().begin(), nb_col,
                          &*col_data.getIndices().begin(), &*nN.data().begin(),
                          ADD_VALUES);
      // matrix is symmetric, assemble other part
      if (row_side != col_side || row_type != col_type) {
        transNN.resize(nb_col, nb_row);
        noalias(transNN) = trans(nN);
        CHKERR MatSetValues(a, nb_col, &*col_data.getIndices().begin(), nb_row,
                            &*row_data.getIndices().begin(),
                            &*transNN.data().begin(), ADD_VALUES);
      }
      MoFEMFunctionReturn(0);
    }
  };

  struct OpVU_L2L2
      : public MoFEM::VolumeElementForcesAndSourcesCore::UserDataOperator {

    UnsaturatedFlowElement &cTx;

    OpVU_L2L2(UnsaturatedFlowElement &ctx, const std::string value_name)
        : MoFEM::VolumeElementForcesAndSourcesCore::UserDataOperator(
              value_name, value_name, UserDataOperator::OPROWCOL),
          cTx(ctx) {
      sYmm = true;
    }

    MatrixDouble nN;

    /**
     * \brief Assemble matrix
     * @param  row_side local index of row entity on element
     * @param  col_side local index of col entity on element
     * @param  row_type type of row entity, f.e. MBVERTEX, MBEDGE, or MBTET
     * @param  col_type type of col entity, f.e. MBVERTEX, MBEDGE, or MBTET
     * @param  row_data data for row
     * @param  col_data data for col
     * @return          error code
     */
    MoFEMErrorCode doWork(int row_side, int col_side, EntityType row_type,
                          EntityType col_type,
                          DataForcesAndSourcesCore::EntData &row_data,
                          DataForcesAndSourcesCore::EntData &col_data) {
      MoFEMFunctionBegin;
      int nb_row = row_data.getIndices().size();
      int nb_col = col_data.getIndices().size();
      if (nb_row == 0)
        MoFEMFunctionReturnHot(0);
      if (nb_col == 0)
        MoFEMFunctionReturnHot(0);
      nN.resize(nb_row, nb_col, false);
      nN.clear();
      // Get EntityHandle of the finite element
      EntityHandle fe_ent = getNumeredEntFiniteElementPtr()->getEnt();
      // Get material block id
      int block_id;
      CHKERR cTx.getMaterial(fe_ent, block_id);
      // Get material block
      boost::shared_ptr<GenericMaterial> &block_data = cTx.dMatMap.at(block_id);
      // Get pressure
      auto t_h = getFTensor0FromVec(cTx.valuesAtGaussPts);
      // // Get pressure
      // auto t_flux_residual = getFTensor0FromVec(*cTx.resAtGaussPts);
      // Get pressure rate
      auto t_h_t = getFTensor0FromVec(*cTx.headRateAtGaussPts);
      // Get integration weight
      auto t_w = getFTensor0IntegrationWeight();
      // Coords at integration points
      auto t_coords = getFTensor1CoordsAtGaussPts();
      // Scale eq.
      const double scale = block_data->sCale;
      // Time step factor
      double ts_a = getFEMethod()->ts_a;
      // get volume
      const double vol = getVolume();
      int nb_gauss_pts = row_data.getN().size1();
      // get base functions on rows
      auto t_n_row = row_data.getFTensor0N();
      for (int gg = 0; gg != nb_gauss_pts; gg++) {
        // get integration weight and multiply by element volume
        double alpha = t_w * vol * scale;
        // evaluate material model at integration points
        // to calculate capacity and tangent of capacity term
        block_data->h = t_h;
        block_data->h_t = t_h_t;
        block_data->x = t_coords(0);
        block_data->y = t_coords(1);
        block_data->z = t_coords(2);
        CHKERR block_data->calC();
        CHKERR block_data->calDiffC();
        const double C = block_data->C;
        const double diffC = block_data->diffC;
        // assemble local entity tangent matrix
        FTensor::Tensor0<FTensor::PackPtr<double *, 1>> t_a(
            &*nN.data().begin());
        // iterate base functions on rows
        for (int kk = 0; kk != nb_row; kk++) {
          // get first base function on column at integration point gg
          auto t_n_col = col_data.getFTensor0N(gg, 0);
          // iterate base functions on columns
          for (int ll = 0; ll != nb_col; ll++) {
            // assemble elements of local matrix
            t_a += (alpha * (C * ts_a + diffC * t_h_t)) * t_n_row * t_n_col;
            ++t_n_col; // move to next base function on column
            ++t_a;     // move to next element in local tangent matrix
          }
          ++t_n_row; // move to next base function on row
        }
        ++t_w;      // move to next integration weight
        ++t_coords; // move to next coordinate at integration point
        ++t_h;      // move to next capillary head at integration point
        // ++t_flux_residual;
        ++t_h_t; // move to next capillary head rate at integration point
      }
      Mat a = getFEMethod()->ts_B;
      CHKERR MatSetValues(a, nb_row, &row_data.getIndices()[0], nb_col,
                          &col_data.getIndices()[0], &*nN.data().begin(),
                          ADD_VALUES);
      MoFEMFunctionReturn(0);
    }
  };

  struct OpVDivSigma_L2Hdiv
      : public MoFEM::VolumeElementForcesAndSourcesCore::UserDataOperator {

    UnsaturatedFlowElement &cTx;

    /**
     * \brief Constructor
     */
    OpVDivSigma_L2Hdiv(UnsaturatedFlowElement &ctx,
                       const std::string &val_name_row,
                       const std::string &flux_name_col)
        : MoFEM::VolumeElementForcesAndSourcesCore::UserDataOperator(
              val_name_row, flux_name_col, UserDataOperator::OPROWCOL, false),
          cTx(ctx) {}

    MatrixDouble nN;
    VectorDouble divVec;

    /**
     * \brief Do calculations
     * @param  row_side local index of entity on row
     * @param  col_side local index of entity on column
     * @param  row_type type of row entity
     * @param  col_type type of col entity
     * @param  row_data row data structure carrying information about base
     * functions, DOFs indices, etc.
     * @param  col_data column data structure carrying information about base
     * functions, DOFs indices, etc.
     * @return          error code
     */
    MoFEMErrorCode doWork(int row_side, int col_side, EntityType row_type,
                          EntityType col_type,
                          DataForcesAndSourcesCore::EntData &row_data,
                          DataForcesAndSourcesCore::EntData &col_data) {
      MoFEMFunctionBegin;
      int nb_row = row_data.getFieldData().size();
      int nb_col = col_data.getFieldData().size();
      if (nb_row == 0)
        MoFEMFunctionReturnHot(0);
      if (nb_col == 0)
        MoFEMFunctionReturnHot(0);
      // Get EntityHandle of the finite element
      EntityHandle fe_ent = getNumeredEntFiniteElementPtr()->getEnt();
      // Get material block id
      int block_id;
      CHKERR cTx.getMaterial(fe_ent, block_id);
      // Get material block
      boost::shared_ptr<GenericMaterial> &block_data = cTx.dMatMap.at(block_id);
      nN.resize(nb_row, nb_col, false);
      divVec.resize(nb_col, false);
      nN.clear();
      // take direvatives of base functions
      FTensor::Index<'i', 3> i;
      auto t_base_diff_hdiv = col_data.getFTensor2DiffN<3, 3>();
      // Scale eq.
      const double scale = block_data->sCale;
      int nb_gauss_pts = row_data.getN().size1();
      for (int gg = 0; gg < nb_gauss_pts; gg++) {
        double alpha = getGaussPts()(3, gg) * getVolume() * scale;
        for(auto &v : divVec) {
          v = t_base_diff_hdiv(i, i);
          ++t_base_diff_hdiv;
        }
        noalias(nN) += alpha * outer_prod(row_data.getN(gg), divVec);
      }
      CHKERR MatSetValues(getFEMethod()->ts_B, nb_row,
                          &row_data.getIndices()[0], nb_col,
                          &col_data.getIndices()[0], &nN(0, 0), ADD_VALUES);
      MoFEMFunctionReturn(0);
    }
  };

  struct OpDivTauU_HdivL2
      : public MoFEM::VolumeElementForcesAndSourcesCore::UserDataOperator {

    UnsaturatedFlowElement &cTx;

    /**
     * \brief Constructor
     */
    OpDivTauU_HdivL2(UnsaturatedFlowElement &ctx,
                     const std::string &flux_name_col,
                     const std::string &val_name_row)
        : MoFEM::VolumeElementForcesAndSourcesCore::UserDataOperator(
              flux_name_col, val_name_row, UserDataOperator::OPROWCOL, false),
          cTx(ctx) {}

    MatrixDouble nN;
    VectorDouble divVec;
    FTensor::Index<'i', 3> i;

    /**
     * \brief Do calculations
     * @param  row_side local index of entity on row
     * @param  col_side local index of entity on column
     * @param  row_type type of row entity
     * @param  col_type type of col entity
     * @param  row_data row data structure carrying information about base
     * functions, DOFs indices, etc.
     * @param  col_data column data structure carrying information about base
     * functions, DOFs indices, etc.
     * @return          error code
     */
    MoFEMErrorCode doWork(int row_side, int col_side, EntityType row_type,
                          EntityType col_type,
                          DataForcesAndSourcesCore::EntData &row_data,
                          DataForcesAndSourcesCore::EntData &col_data) {
      MoFEMFunctionBegin;
      int nb_row = row_data.getFieldData().size();
      int nb_col = col_data.getFieldData().size();
      if (nb_row == 0)
        MoFEMFunctionReturnHot(0);
      if (nb_col == 0)
        MoFEMFunctionReturnHot(0);
      nN.resize(nb_row, nb_col, false);
      divVec.resize(nb_row, false);
      nN.clear();
      // Get EntityHandle of the finite element
      EntityHandle fe_ent = getNumeredEntFiniteElementPtr()->getEnt();
      // Get material block id
      int block_id;
      CHKERR cTx.getMaterial(fe_ent, block_id);
      // Get material block
      boost::shared_ptr<GenericMaterial> &block_data = cTx.dMatMap.at(block_id);
      // Get pressure
      auto t_h = getFTensor0FromVec(cTx.valuesAtGaussPts);
      // Get flux
      auto t_flux = getFTensor1FromMat<3>(cTx.fluxesAtGaussPts);
      // Coords at integration points
      auto t_coords = getFTensor1CoordsAtGaussPts();
      // Get integration weight
      auto t_w = getFTensor0IntegrationWeight();
      // Get base function
      auto t_n_hdiv_row = row_data.getFTensor1N<3>();
      // Get direvatives of base functions
      FTensor::Index<'i', 3> i;
      auto t_base_diff_hdiv = row_data.getFTensor2DiffN<3, 3>();
      // Get volume
      double vol = getVolume();
      int nb_gauss_pts = row_data.getN().size1();
      for (int gg = 0; gg != nb_gauss_pts; gg++) {
        block_data->h = t_h;
        block_data->x = t_coords(0);
        block_data->y = t_coords(1);
        block_data->z = t_coords(2);
        CHKERR block_data->calK();
        CHKERR block_data->calDiffK();
        const double K = block_data->K;
        // const double z = t_coords(2);
        const double KK = K * K;
        const double diffK = block_data->diffK;
        double alpha = t_w * vol;
        for(auto &v : divVec) {
          v = t_base_diff_hdiv(i, i);
          ++t_base_diff_hdiv;
        }
        noalias(nN) -= alpha * outer_prod(divVec, col_data.getN(gg));
        FTensor::Tensor0<double *> t_a(&*nN.data().begin());
        for (int rr = 0; rr != nb_row; rr++) {
          double beta = alpha * (-diffK / KK) * (t_n_hdiv_row(i) * t_flux(i));
          auto t_n_col = col_data.getFTensor0N(gg, 0);
          for (int cc = 0; cc != nb_col; cc++) {
            t_a += beta * t_n_col;
            ++t_n_col;
            ++t_a;
          }
          ++t_n_hdiv_row;
        }
        ++t_w;
        ++t_coords;
        ++t_h;
        ++t_flux;
      }
      CHKERR MatSetValues(getFEMethod()->ts_B, nb_row,
                          &row_data.getIndices()[0], nb_col,
                          &col_data.getIndices()[0], &nN(0, 0), ADD_VALUES);
      MoFEMFunctionReturn(0);
    }
  };

  struct OpEvaluateInitiallHead
      : public MoFEM::VolumeElementForcesAndSourcesCore::UserDataOperator {
    UnsaturatedFlowElement &cTx;
    OpEvaluateInitiallHead(UnsaturatedFlowElement &ctx,
                           const std::string &val_name)
        : MoFEM::VolumeElementForcesAndSourcesCore::UserDataOperator(
              val_name, UserDataOperator::OPROW),
          cTx(ctx) {}

    MatrixDouble nN;
    VectorDouble nF;

    MoFEMErrorCode doWork(int side, EntityType type,
                          DataForcesAndSourcesCore::EntData &data) {
      MoFEMFunctionBegin;
      if (data.getFieldData().size() == 0)
        MoFEMFunctionReturnHot(0);
      int nb_dofs = data.getFieldData().size();
      int nb_gauss_pts = data.getN().size1();
      if (nb_dofs != static_cast<int>(data.getN().size2())) {
        SETERRQ(PETSC_COMM_WORLD, MOFEM_DATA_INCONSISTENCY,
                "wrong number of dofs");
      }
      nN.resize(nb_dofs, nb_dofs, false);
      nF.resize(nb_dofs, false);
      nN.clear();
      nF.clear();

      // Get EntityHandle of the finite element
      EntityHandle fe_ent = getFEEntityHandle();
      // Get material block id
      int block_id;
      CHKERR cTx.getMaterial(fe_ent, block_id);
      // Get material block
      boost::shared_ptr<GenericMaterial> &block_data = cTx.dMatMap.at(block_id);

      // loop over integration points
      for (int gg = 0; gg < nb_gauss_pts; gg++) {
        // get coordinates at integration point
        block_data->x = getCoordsAtGaussPts()(gg, 0);
        block_data->y = getCoordsAtGaussPts()(gg, 1);
        block_data->z = getCoordsAtGaussPts()(gg, 2);
        // get weight for integration rule
        double alpha = getGaussPts()(2, gg) * getVolume();
        nN += alpha * outer_prod(data.getN(gg), data.getN(gg));
        nF += alpha * block_data->initialPcEval() * data.getN(gg);
      }

      // factor matrix
      cholesky_decompose(nN);
      // solve local problem
      cholesky_solve(nN, nF, ublas::lower());

      // set solution to vector
      CHKERR VecSetValues(cTx.D1, nb_dofs, &*data.getIndices().begin(),
                          &*nF.begin(), INSERT_VALUES);

      MoFEMFunctionReturn(0);
    }
  };

  struct OpIntegrateFluxes
      : public MoFEM::FaceElementForcesAndSourcesCore::UserDataOperator {
    UnsaturatedFlowElement &cTx;

    /**
     * \brief Constructor
     */
    OpIntegrateFluxes(UnsaturatedFlowElement &ctx,
                      const std::string fluxes_name)
        : MoFEM::FaceElementForcesAndSourcesCore::UserDataOperator(
              fluxes_name, UserDataOperator::OPROW),
          cTx(ctx) {}

    FTensor::Index<'i', 3> i;

    /**
     * \brief Integrate boundary condition
     * @param  side local index of entity
     * @param  type type of entity
     * @param  data data on entity
     * @return      error code
     */
    MoFEMErrorCode doWork(int side, EntityType type,
                          DataForcesAndSourcesCore::EntData &data) {
      MoFEMFunctionBegin;
      int nb_dofs = data.getFieldData().size();
      if (nb_dofs == 0)
        MoFEMFunctionReturnHot(0);
      // Get base function
      auto t_n_hdiv = data.getFTensor1N<3>();
      // get normal of face
      auto t_normal = getFTensor1Normal();
      // Integration weight
      auto t_w = getFTensor0IntegrationWeight();
      double flux_on_entity = 0;
      int nb_gauss_pts = data.getN().size1();
      for (int gg = 0; gg < nb_gauss_pts; gg++) {
        auto t_data = data.getFTensor0FieldData();
        for (int rr = 0; rr != nb_dofs; rr++) {
          flux_on_entity -= (0.5 * t_data * t_w) * (t_n_hdiv(i) * t_normal(i));
          ++t_n_hdiv;
          ++t_data;
        }
        ++t_w;
      }
      CHKERR VecSetValue(cTx.ghostFlux, 0, flux_on_entity, ADD_VALUES);
      MoFEMFunctionReturn(0);
    }
  };

  /**
   * Operator used to post-process results for unsaturated infiltration problem.
   * Operator should with element for post-processing results, i.e.
   * PostProcVolumeOnRefinedMesh
   */
  struct OpPostProcMaterial
      : public MoFEM::VolumeElementForcesAndSourcesCore::UserDataOperator {

    UnsaturatedFlowElement &cTx;
    moab::Interface &postProcMesh;
    std::vector<EntityHandle> &mapGaussPts;

    OpPostProcMaterial(UnsaturatedFlowElement &ctx,
                       moab::Interface &post_proc_mesh,
                       std::vector<EntityHandle> &map_gauss_pts,
                       const std::string field_name)
        : MoFEM::VolumeElementForcesAndSourcesCore::UserDataOperator(
              field_name, ForcesAndSourcesCore::UserDataOperator::OPROW),
          cTx(ctx), postProcMesh(post_proc_mesh), mapGaussPts(map_gauss_pts) {}

    MoFEMErrorCode doWork(int side, EntityType type,
                          DataForcesAndSourcesCore::EntData &data) {
      MoFEMFunctionBegin;
      int nb_dofs = data.getFieldData().size();
      if (nb_dofs == 0)
        MoFEMFunctionReturnHot(0);

      // Get EntityHandle of the finite element
      EntityHandle fe_ent = getNumeredEntFiniteElementPtr()->getEnt();
      // Get material block id
      int block_id;
      CHKERR cTx.getMaterial(fe_ent, block_id);
      // Get material block
      boost::shared_ptr<GenericMaterial> &block_data = cTx.dMatMap.at(block_id);

      // Set bloc Id
      Tag th_id;
      int def_block_id = -1;
      CHKERR postProcMesh.tag_get_handle("BLOCK_ID", 1, MB_TYPE_INTEGER, th_id,
                                         MB_TAG_CREAT | MB_TAG_SPARSE,
                                         &def_block_id);

      // Create mesh tag. Tags are created on post-processing mesh and
      // visable in post-processor, e.g. Paraview
      double zero = 0;
      Tag th_theta;
      CHKERR postProcMesh.tag_get_handle("THETA", 1, MB_TYPE_DOUBLE, th_theta,
                                         MB_TAG_CREAT | MB_TAG_SPARSE, &zero);
      Tag th_se;
      CHKERR postProcMesh.tag_get_handle("Se", 1, MB_TYPE_DOUBLE, th_se,
                                         MB_TAG_CREAT | MB_TAG_SPARSE, &zero);
      // Tag th_ks;
      // CHKERR postProcMesh.tag_get_handle(
      //   "Ks",1,MB_TYPE_DOUBLE,th_ks,
      //   MB_TAG_CREAT|MB_TAG_SPARSE,&zero
      // );
      // CHKERR postProcMesh.tag_set_data(th_ks,&fe_ent,1,&block_data->Ks);
      Tag th_k;
      CHKERR postProcMesh.tag_get_handle("K", 1, MB_TYPE_DOUBLE, th_k,
                                         MB_TAG_CREAT | MB_TAG_SPARSE, &zero);
      Tag th_c;
      CHKERR postProcMesh.tag_get_handle("C", 1, MB_TYPE_DOUBLE, th_c,
                                         MB_TAG_CREAT | MB_TAG_SPARSE, &zero);

      // Get pressure at integration points
      auto t_h = getFTensor0FromVec(cTx.valuesAtGaussPts);
      // Coords at integration points
      auto t_coords = getFTensor1CoordsAtGaussPts();

      int nb_gauss_pts = data.getN().size1();
      for (int gg = 0; gg < nb_gauss_pts; gg++) {
        block_data->h = t_h;
        block_data->x = t_coords(0);
        block_data->y = t_coords(1);
        block_data->z = t_coords(2);
        // Calculate theta (water content) and save it on mesh tags
        CHKERR block_data->calTheta();
        double theta = block_data->tHeta;
        CHKERR postProcMesh.tag_set_data(th_theta, &mapGaussPts[gg], 1, &theta);
        CHKERR block_data->calSe();
        // Calculate Se (effective saturation and save it on the mesh tags)
        double Se = block_data->Se;
        CHKERR postProcMesh.tag_set_data(th_se, &mapGaussPts[gg], 1, &Se);
        // Calculate K (hydraulic conductivity) and save it on the mesh tags
        CHKERR block_data->calK();
        double K = block_data->K;
        CHKERR postProcMesh.tag_set_data(th_k, &mapGaussPts[gg], 1, &K);
        // Calculate water capacity and save it on the mesh tags
        CHKERR block_data->calC();
        double C = block_data->C;
        CHKERR postProcMesh.tag_set_data(th_c, &mapGaussPts[gg], 1, &C);

        // Set block iD
        CHKERR postProcMesh.tag_set_data(th_id, &mapGaussPts[gg], 1, &block_id);

        ++t_h;
        ++t_coords;
      }

      MoFEMFunctionReturn(0);
    }
  };

  /**
   * Finite element implementation called by TS monitor. Element calls other
   * finite elements to evaluate material properties and save results on the
   * mesh.
   *
   * \note Element overloaded only FEMethod::postProcess methos where other
   * elements are called.
   */
  struct MonitorPostProc : public MoFEM::FEMethod {

    UnsaturatedFlowElement &cTx;
    boost::shared_ptr<PostProcVolumeOnRefinedMesh> postProc;
    boost::shared_ptr<ForcesAndSourcesCore> fluxIntegrate;

    const int fRequency;

    MonitorPostProc(UnsaturatedFlowElement &ctx,
                    boost::shared_ptr<PostProcVolumeOnRefinedMesh> &post_proc,
                    boost::shared_ptr<ForcesAndSourcesCore> flux_Integrate,
                    const int frequency)
        : cTx(ctx), postProc(post_proc), fluxIntegrate(flux_Integrate),
          fRequency(frequency) {}

    MoFEMErrorCode preProcess() {
      MoFEMFunctionBegin;
      MoFEMFunctionReturn(0);
    }

    MoFEMErrorCode operator()() {
      MoFEMFunctionBegin;
      MoFEMFunctionReturn(0);
    }

    MoFEMErrorCode postProcess() {
      MoFEMFunctionBegin;

      // Get time step
      int step;
      CHKERR TSGetTimeStepNumber(ts, &step);

      if ((step) % fRequency == 0) {
        // Post-process results and save in the file
        PetscPrintf(PETSC_COMM_WORLD, "Output results %d - %d\n", step,
                    fRequency);
        CHKERR DMoFEMLoopFiniteElements(cTx.dM, "MIX", postProc);
        CHKERR postProc->writeFile(
            string("out_") + boost::lexical_cast<std::string>(step) + ".h5m");
      }

      // Integrate fluxes on faces where pressure head is applied
      CHKERR VecZeroEntries(cTx.ghostFlux);
      CHKERR VecGhostUpdateBegin(cTx.ghostFlux, INSERT_VALUES, SCATTER_FORWARD);
      CHKERR VecGhostUpdateEnd(cTx.ghostFlux, INSERT_VALUES, SCATTER_FORWARD);
      // Run finite element to integrate fluxes
      CHKERR DMoFEMLoopFiniteElements(cTx.dM, "MIX_BCVALUE", fluxIntegrate);
      CHKERR VecAssemblyBegin(cTx.ghostFlux);
      CHKERR VecAssemblyEnd(cTx.ghostFlux);
      // accumulate errors from processors
      CHKERR VecGhostUpdateBegin(cTx.ghostFlux, ADD_VALUES, SCATTER_REVERSE);
      CHKERR VecGhostUpdateEnd(cTx.ghostFlux, ADD_VALUES, SCATTER_REVERSE);
      // scatter errors to all processors
      CHKERR VecGhostUpdateBegin(cTx.ghostFlux, INSERT_VALUES, SCATTER_FORWARD);
      CHKERR VecGhostUpdateEnd(cTx.ghostFlux, INSERT_VALUES, SCATTER_FORWARD);
      double *ghost_flux;
      CHKERR VecGetArray(cTx.ghostFlux, &ghost_flux);
      CHKERR PetscPrintf(PETSC_COMM_WORLD, "Flux at time %6.4g %6.4g\n", ts_t,
                         ghost_flux[0]);
      CHKERR VecRestoreArray(cTx.ghostFlux, &ghost_flux);

      MoFEMFunctionReturn(0);
    }
  };

  /// \brief add fields
  MoFEMErrorCode addFields(const std::string &values, const std::string &fluxes,
                           const int order) {
    MoFEMFunctionBegin;
    // Fields
    CHKERR mField.add_field(fluxes, HDIV, DEMKOWICZ_JACOBI_BASE, 1);
    CHKERR mField.add_field(values, L2, AINSWORTH_LEGENDRE_BASE, 1);
    CHKERR mField.add_field(values + "_t", L2, AINSWORTH_LEGENDRE_BASE, 1);
    // CHKERR mField.add_field(fluxes+"_residual",L2,AINSWORTH_LEGENDRE_BASE,1);

    // meshset consisting all entities in mesh
    EntityHandle root_set = mField.get_moab().get_root_set();
    // add entities to field

    for (_IT_CUBITMESHSETS_BY_SET_TYPE_FOR_LOOP_(mField, BLOCKSET, it)) {
      if (it->getName().compare(0, 4, "SOIL") != 0)
        continue;
      CHKERR mField.add_ents_to_field_by_type(dMatMap[it->getMeshsetId()]->tEts,
                                              MBTET, fluxes);
      CHKERR mField.add_ents_to_field_by_type(dMatMap[it->getMeshsetId()]->tEts,
                                              MBTET, values);
      CHKERR mField.add_ents_to_field_by_type(dMatMap[it->getMeshsetId()]->tEts,
                                              MBTET, values + "_t");
      // CHKERR mField.add_ents_to_field_by_type(
      //   dMatMap[it->getMeshsetId()]->tEts,MBTET,fluxes+"_residual"
      // );
    }

    CHKERR mField.set_field_order(root_set, MBTET, fluxes, order + 1);
    CHKERR mField.set_field_order(root_set, MBTRI, fluxes, order + 1);
    CHKERR mField.set_field_order(root_set, MBTET, values, order);
    CHKERR mField.set_field_order(root_set, MBTET, values + "_t", order);
    // CHKERR mField.set_field_order(root_set,MBTET,fluxes+"_residual",order);
    MoFEMFunctionReturn(0);
  }

  /// \brief add finite elements
  MoFEMErrorCode addFiniteElements(const std::string &fluxes_name,
                                   const std::string &values_name) {
    MoFEMFunctionBegin;

    // Define element "MIX". Note that this element will work with fluxes_name
    // and values_name. This reflect bilinear form for the problem
    CHKERR mField.add_finite_element("MIX", MF_ZERO);
    CHKERR mField.modify_finite_element_add_field_row("MIX", fluxes_name);
    CHKERR mField.modify_finite_element_add_field_col("MIX", fluxes_name);
    CHKERR mField.modify_finite_element_add_field_row("MIX", values_name);
    CHKERR mField.modify_finite_element_add_field_col("MIX", values_name);
    CHKERR mField.modify_finite_element_add_field_data("MIX", fluxes_name);
    CHKERR mField.modify_finite_element_add_field_data("MIX", values_name);
    CHKERR mField.modify_finite_element_add_field_data("MIX",
                                                       values_name + "_t");
    // CHKERR
    // mField.modify_finite_element_add_field_data("MIX",fluxes_name+"_residual");

    for (_IT_CUBITMESHSETS_BY_SET_TYPE_FOR_LOOP_(mField, BLOCKSET, it)) {
      if (it->getName().compare(0, 4, "SOIL") != 0)
        continue;
      CHKERR mField.add_ents_to_finite_element_by_type(
          dMatMap[it->getMeshsetId()]->tEts, MBTET, "MIX");
    }

    // Define element to integrate natural boundary conditions, i.e. set values.
    CHKERR mField.add_finite_element("MIX_BCVALUE", MF_ZERO);
    CHKERR mField.modify_finite_element_add_field_row("MIX_BCVALUE",
                                                      fluxes_name);
    CHKERR mField.modify_finite_element_add_field_col("MIX_BCVALUE",
                                                      fluxes_name);
    CHKERR mField.modify_finite_element_add_field_data("MIX_BCVALUE",
                                                       fluxes_name);
    CHKERR mField.modify_finite_element_add_field_data("MIX_BCVALUE",
                                                       values_name);

    for (_IT_CUBITMESHSETS_BY_SET_TYPE_FOR_LOOP_(mField, BLOCKSET, it)) {
      if (it->getName().compare(0, 4, "HEAD") != 0)
        continue;
      CHKERR mField.add_ents_to_finite_element_by_type(
          bcValueMap[it->getMeshsetId()]->eNts, MBTRI, "MIX_BCVALUE");
    }

    // Define element to apply essential boundary conditions.
    CHKERR mField.add_finite_element("MIX_BCFLUX", MF_ZERO);
    CHKERR mField.modify_finite_element_add_field_row("MIX_BCFLUX",
                                                      fluxes_name);
    CHKERR mField.modify_finite_element_add_field_col("MIX_BCFLUX",
                                                      fluxes_name);
    CHKERR mField.modify_finite_element_add_field_data("MIX_BCFLUX",
                                                       fluxes_name);
    CHKERR mField.modify_finite_element_add_field_data("MIX_BCFLUX",
                                                       values_name);

    for (_IT_CUBITMESHSETS_BY_SET_TYPE_FOR_LOOP_(mField, BLOCKSET, it)) {
      if (it->getName().compare(0, 4, "FLUX") != 0)
        continue;
      CHKERR mField.add_ents_to_finite_element_by_type(
          bcFluxMap[it->getMeshsetId()]->eNts, MBTRI, "MIX_BCFLUX");
    }

    MoFEMFunctionReturn(0);
  }

  /**
   * \brief Build problem
   * @param  ref_level mesh refinement on which mesh problem you like to built.
   * @return           error code
   */
  MoFEMErrorCode buildProblem(Range zero_flux_ents,
                              BitRefLevel ref_level = BitRefLevel().set(0)) {
    MoFEMFunctionBegin;

    // Build fields
    CHKERR mField.build_fields();
    // Build finite elements
    CHKERR mField.build_finite_elements("MIX");
    CHKERR mField.build_finite_elements("MIX_BCFLUX");
    CHKERR mField.build_finite_elements("MIX_BCVALUE");
    // Build adjacencies of degrees of freedom and elements
    CHKERR mField.build_adjacencies(ref_level);

    //  create DM instance
    CHKERR DMCreate(PETSC_COMM_WORLD, &dM);
    // setting that DM is type of DMMOFEM, i.e. MOFEM implementation manages DM
    CHKERR DMSetType(dM, "DMMOFEM");
    // mesh is portioned, each process keeps only part of problem
    CHKERR DMMoFEMSetIsPartitioned(dM, PETSC_TRUE);
    // creates problem in DM
    CHKERR DMMoFEMCreateMoFEM(dM, &mField, "MIX", ref_level);
    // discretised problem creates square matrix (that makes some optimizations)
    CHKERR DMMoFEMSetIsPartitioned(dM, PETSC_TRUE);
    // set DM options from command line
    CHKERR DMSetFromOptions(dM);
    // add finite elements
    CHKERR DMMoFEMAddElement(dM, "MIX");
    CHKERR DMMoFEMAddElement(dM, "MIX_BCFLUX");
    CHKERR DMMoFEMAddElement(dM, "MIX_BCVALUE");
    // constructor data structures
    CHKERR DMSetUp(dM);

    // remove zero flux dofs
    CHKERR mField.getInterface<ProblemsManager>()->removeDofsOnEntities(
        "MIX", "FLUXES", zero_flux_ents);

    PetscSection section;
    CHKERR mField.getInterface<ISManager>()->sectionCreate("MIX", &section);
    CHKERR DMSetDefaultSection(dM, section);
    CHKERR DMSetDefaultGlobalSection(dM, section);
    // CHKERR PetscSectionView(section,PETSC_VIEWER_STDOUT_WORLD);
    CHKERR PetscSectionDestroy(&section);

    MoFEMFunctionReturn(0);
  }

  boost::shared_ptr<ForcesAndSourcesCore>
      feFaceBc; ///< Elemnet to calculate essential bc
  boost::shared_ptr<ForcesAndSourcesCore>
      feFaceRhs; ///< Face element apply natural bc
  boost::shared_ptr<ForcesAndSourcesCore>
      feVolInitialPc; ///< Calculate inital boundary conditions
  boost::shared_ptr<ForcesAndSourcesCore>
      feVolRhs; ///< Assemble residual vector
  boost::shared_ptr<ForcesAndSourcesCore> feVolLhs; ///< Assemble tangent matrix
  boost::shared_ptr<MethodForForceScaling>
      scaleMethodFlux; ///< Method scaling fluxes
  boost::shared_ptr<MethodForForceScaling>
      scaleMethodValue;                  ///< Method scaling values
  boost::shared_ptr<FEMethod> tsMonitor; ///< Element used by TS monitor to
                                         ///< postprocess results at time step

  boost::shared_ptr<VectorDouble>
      headRateAtGaussPts; ///< Vector keeps head rate
  // boost::shared_ptr<VectorDouble> resAtGaussPts;  ///< Residual field

  /**
   * \brief Set integration rule to volume elements
   *
   */
  struct VolRule {
    int operator()(int, int, int p_data) const { return 2 * p_data + p_data; }
  };
  /**
   * \brief Set integration rule to boundary elements
   *
   */
  struct FaceRule {
    int operator()(int p_row, int p_col, int p_data) const {
      return 2 * p_data;
    }
  };

  std::vector<int> bcVecIds;
  VectorDouble bcVecVals, vecValsOnBc;

  /**
   * \brief Pre-peprocessing
   * Set head pressute rate and get inital essential boundary conditions
   */
  struct preProcessVol {
    UnsaturatedFlowElement &cTx;
    boost::shared_ptr<ForcesAndSourcesCore> fePtr;
    // std::string mArk;

    preProcessVol(
        UnsaturatedFlowElement &ctx,
        boost::shared_ptr<ForcesAndSourcesCore> &fe_ptr /*,std::string mark*/
        )
        : cTx(ctx), fePtr(fe_ptr) /*,mArk(mark)*/ {}
    MoFEMErrorCode operator()() {
      MoFEMFunctionBegin;
      // Update pressure rates
      CHKERR fePtr->mField.getInterface<VecManager>()->setOtherLocalGhostVector(
          fePtr->problemPtr, "VALUES", string("VALUES") + "_t", ROW,
          fePtr->ts_u_t, INSERT_VALUES, SCATTER_REVERSE);
      switch (fePtr->ts_ctx) {
      case TSMethod::CTX_TSSETIFUNCTION:
        CHKERR VecSetOption(fePtr->ts_F, VEC_IGNORE_NEGATIVE_INDICES,
                            PETSC_TRUE);
        if (!cTx.bcIndices.empty()) {
          double scale;
          CHKERR cTx.scaleMethodFlux->getForceScale(fePtr->ts_t, scale);
          if (cTx.bcVecIds.size() != cTx.bcIndices.size()) {
            cTx.bcVecIds.insert(cTx.bcVecIds.begin(), cTx.bcIndices.begin(),
                                cTx.bcIndices.end());
            cTx.bcVecVals.resize(cTx.bcVecIds.size(), false);
            cTx.vecValsOnBc.resize(cTx.bcVecIds.size(), false);
          }
          CHKERR VecGetValues(cTx.D0, cTx.bcVecIds.size(),
                              &*cTx.bcVecIds.begin(), &*cTx.bcVecVals.begin());
          CHKERR VecGetValues(fePtr->ts_u, cTx.bcVecIds.size(),
                              &*cTx.bcVecIds.begin(),
                              &*cTx.vecValsOnBc.begin());
          cTx.bcVecVals *= scale;
          VectorDouble::iterator vit = cTx.bcVecVals.begin();
          const NumeredDofEntity *dof_ptr;
          for (std::vector<int>::iterator it = cTx.bcVecIds.begin();
               it != cTx.bcVecIds.end(); it++, vit++) {
            if (auto dof_ptr =
                    fePtr->problemPtr->getColDofsByPetscGlobalDofIdx(*it)
                        .lock())
            dof_ptr->getFieldData() = *vit;
          }
        } else {
          cTx.bcVecIds.resize(0);
          cTx.bcVecVals.resize(0);
          cTx.vecValsOnBc.resize(0);
        }
        break;
      default:
        // don nothing
        break;
      }
      MoFEMFunctionReturn(0);
    }
  };

  /**
   * \brief Post proces method for volume element
   * Assemble vectors and matrices and apply essential boundary conditions
   */
  struct postProcessVol {
    UnsaturatedFlowElement &cTx;
    boost::shared_ptr<ForcesAndSourcesCore> fePtr;
    // std::string mArk;
    postProcessVol(
        UnsaturatedFlowElement &ctx,
        boost::shared_ptr<ForcesAndSourcesCore> &fe_ptr //,std::string mark
        )
        : cTx(ctx), fePtr(fe_ptr) /*,mArk(mark)*/ {}
    MoFEMErrorCode operator()() {
      MoFEMFunctionBegin;
      switch (fePtr->ts_ctx) {
      case TSMethod::CTX_TSSETIJACOBIAN: {
        CHKERR MatAssemblyBegin(fePtr->ts_B, MAT_FINAL_ASSEMBLY);
        CHKERR MatAssemblyEnd(fePtr->ts_B, MAT_FINAL_ASSEMBLY);
        // MatView(fePtr->ts_B,PETSC_VIEWER_DRAW_WORLD);
        // std::string wait;
        // std::cin >> wait;
        CHKERR MatZeroRowsColumns(fePtr->ts_B, cTx.bcVecIds.size(),
                                  &*cTx.bcVecIds.begin(), 1, PETSC_NULL,
                                  PETSC_NULL);
        CHKERR MatAssemblyBegin(fePtr->ts_B, MAT_FINAL_ASSEMBLY);
        CHKERR MatAssemblyEnd(fePtr->ts_B, MAT_FINAL_ASSEMBLY);
        // MatView(fePtr->ts_B,PETSC_VIEWER_DRAW_WORLD);
        // std::string wait;
        // std::cin >> wait;
      } break;
      case TSMethod::CTX_TSSETIFUNCTION: {
        CHKERR VecAssemblyBegin(fePtr->ts_F);
        CHKERR VecAssemblyEnd(fePtr->ts_F);
        if (!cTx.bcVecIds.empty()) {
          cTx.vecValsOnBc -= cTx.bcVecVals;
          CHKERR VecSetValues(fePtr->ts_F, cTx.bcVecIds.size(),
                              &*cTx.bcVecIds.begin(), &*cTx.vecValsOnBc.begin(),
                              INSERT_VALUES);
        }
        CHKERR VecAssemblyBegin(fePtr->ts_F);
        CHKERR VecAssemblyEnd(fePtr->ts_F);
      } break;
      default:
        // don nothing
        break;
      }
      MoFEMFunctionReturn(0);
    }
  };

  /**
   * \brief Create finite element instances
   * @param  vol_rule integration rule for volume element
   * @param  face_rule integration rule for boundary element
   * @return error code
   */
  MoFEMErrorCode
  setFiniteElements(ForcesAndSourcesCore::RuleHookFun vol_rule = VolRule(),
                    ForcesAndSourcesCore::RuleHookFun face_rule = FaceRule()) {
    MoFEMFunctionBegin;

    // create finite element instances
    feFaceBc = boost::shared_ptr<ForcesAndSourcesCore>(
        new FaceElementForcesAndSourcesCore(mField));
    feFaceRhs = boost::shared_ptr<ForcesAndSourcesCore>(
        new FaceElementForcesAndSourcesCore(mField));
    feVolInitialPc = boost::shared_ptr<ForcesAndSourcesCore>(
        new VolumeElementForcesAndSourcesCore(mField));
    feVolLhs = boost::shared_ptr<ForcesAndSourcesCore>(
        new VolumeElementForcesAndSourcesCore(mField));
    feVolRhs = boost::shared_ptr<ForcesAndSourcesCore>(
        new VolumeElementForcesAndSourcesCore(mField));
    // set integration rule to elements
    feFaceBc->getRuleHook = face_rule;
    feFaceRhs->getRuleHook = face_rule;
    feVolInitialPc->getRuleHook = vol_rule;
    feVolLhs->getRuleHook = vol_rule;
    feVolRhs->getRuleHook = vol_rule;
    // set function hook for finite element postprocessing stage
    feVolRhs->preProcessHook = preProcessVol(*this, feVolRhs);
    feVolLhs->preProcessHook = preProcessVol(*this, feVolLhs);
    feVolRhs->postProcessHook = postProcessVol(*this, feVolRhs);
    feVolLhs->postProcessHook = postProcessVol(*this, feVolLhs);

    // Set Piola transform
    feFaceBc->getOpPtrVector().push_back(
        new OpHOSetContravariantPiolaTransformOnFace3D(HDIV));
    feFaceRhs->getOpPtrVector().push_back(
        new OpHOSetContravariantPiolaTransformOnFace3D(HDIV));

    // create method for setting history for fluxes on boundary
    scaleMethodFlux = boost::shared_ptr<MethodForForceScaling>(
        new TimeForceScale("-flux_history", false));

    // create method for setting history for presure heads on boundary
    scaleMethodValue = boost::shared_ptr<MethodForForceScaling>(
        new TimeForceScale("-head_history", false));


    // Set operator to calculate essential boundary conditions
    feFaceBc->getOpPtrVector().push_back(
        new OpEvaluateBcOnFluxes(*this, "FLUXES"));

    // Set operator to calculate initial capillary pressure
    feVolInitialPc->getOpPtrVector().push_back(
        new OpEvaluateInitiallHead(*this, "VALUES"));

    // set residual face from Neumann terms, i.e. applied pressure
    feFaceRhs->getOpPtrVector().push_back(
        new OpRhsBcOnValues(*this, "FLUXES", scaleMethodValue));
    // set residual finite element operators
    headRateAtGaussPts = boost::make_shared<VectorDouble>();
    // resAtGaussPts = boost::make_shared<VectorDouble>();
    feVolRhs->getOpPtrVector().push_back(new OpCalculateScalarFieldValues(
        string("VALUES") + "_t", headRateAtGaussPts, MBTET));
    feVolRhs->getOpPtrVector().push_back(
        new OpValuesAtGaussPts(*this, "VALUES"));
    feVolRhs->getOpPtrVector().push_back(
        new OpFluxDivergenceAtGaussPts(*this, "FLUXES"));
    feVolRhs->getOpPtrVector().push_back(new OpResidualFlux(*this, "FLUXES"));
    feVolRhs->getOpPtrVector().push_back(new OpResidualMass(*this, "VALUES"));
    feVolRhs->getOpPtrVector().back().opType =
        ForcesAndSourcesCore::UserDataOperator::OPROW;
    // set tangent matrix finite element operators
    feVolLhs->getOpPtrVector().push_back(new OpCalculateScalarFieldValues(
        string("VALUES") + "_t", headRateAtGaussPts, MBTET));
    // feVolLhs->getOpPtrVector().push_back(
    //   new
    //   OpCalculateScalarFieldValues(string("FLUXES")+"_residual",resAtGaussPts,MBTET)
    // );
    feVolLhs->getOpPtrVector().push_back(
        new OpValuesAtGaussPts(*this, "VALUES"));
    feVolLhs->getOpPtrVector().push_back(
        new OpFluxDivergenceAtGaussPts(*this, "FLUXES"));
    feVolLhs->getOpPtrVector().push_back(
        new OpTauDotSigma_HdivHdiv(*this, "FLUXES"));
    feVolLhs->getOpPtrVector().push_back(new OpVU_L2L2(*this, "VALUES"));
    feVolLhs->getOpPtrVector().push_back(
        new OpVDivSigma_L2Hdiv(*this, "VALUES", "FLUXES"));
    feVolLhs->getOpPtrVector().push_back(
        new OpDivTauU_HdivL2(*this, "FLUXES", "VALUES"));

    // Adding finite elements to DM, time solver will ask for it to assemble
    // tangent matrices and residuals
    boost::shared_ptr<FEMethod> null;
    CHKERR DMMoFEMTSSetIFunction(dM, "MIX_BCVALUE", feFaceRhs, null, null);
    CHKERR DMMoFEMTSSetIFunction(dM, "MIX", feVolRhs, null, null);
    CHKERR DMMoFEMTSSetIJacobian(dM, "MIX", feVolLhs, null, null);

    // setting up post-processing
    boost::shared_ptr<PostProcVolumeOnRefinedMesh> post_process(
        new PostProcVolumeOnRefinedMesh(mField));
    CHKERR post_process->generateReferenceElementMesh();
    CHKERR post_process->addFieldValuesPostProc("VALUES");
    CHKERR post_process->addFieldValuesPostProc("VALUES_t");
    // CHKERR post_process->addFieldValuesPostProc("FLUXES_residual");
    CHKERR post_process->addFieldValuesPostProc("FLUXES");
    post_process->getOpPtrVector().push_back(
        new OpValuesAtGaussPts(*this, "VALUES"));
    post_process->getOpPtrVector().push_back(
        new OpPostProcMaterial(*this, post_process->postProcMesh,
                               post_process->mapGaussPts, "VALUES"));

    // Integrate fluxes on boundary
    boost::shared_ptr<ForcesAndSourcesCore> flux_integrate;
    flux_integrate = boost::shared_ptr<ForcesAndSourcesCore>(
        new FaceElementForcesAndSourcesCore(mField));
    flux_integrate->getOpPtrVector().push_back(
        new OpIntegrateFluxes(*this, "FLUXES"));
    int frequency = 1;
    CHKERR PetscOptionsBegin(PETSC_COMM_WORLD, "", "Monitor post proc", "none");
    CHKERR PetscOptionsInt(
        "-how_often_output",
        "frequency how often results are dumped on hard disk", "", frequency,
        &frequency, NULL);
    ierr = PetscOptionsEnd();
    CHKERRG(ierr);

    tsMonitor = boost::shared_ptr<FEMethod>(
        new MonitorPostProc(*this, post_process, flux_integrate, frequency));
    TsCtx *ts_ctx;
    DMMoFEMGetTsCtx(dM, &ts_ctx);
    ts_ctx->get_postProcess_to_do_Monitor().push_back(tsMonitor);
    MoFEMFunctionReturn(0);
  }

  Vec D1;        ///< Vector with inital head capillary pressure
  Vec ghostFlux; ///< Ghost Vector of integrated fluxes

  /**
   * \brief Create vectors and matrices
   * @return Error code
   */
  MoFEMErrorCode createMatrices() {
    MoFEMFunctionBegin;
    CHKERR DMCreateMatrix(dM, &Aij);
    CHKERR DMCreateGlobalVector(dM, &D0);
    CHKERR VecDuplicate(D0, &D1);
    CHKERR VecDuplicate(D0, &D);
    CHKERR VecDuplicate(D0, &F);
    int ghosts[] = {0};
    int nb_locals = mField.get_comm_rank() == 0 ? 1 : 0;
    int nb_ghosts = mField.get_comm_rank() > 0 ? 1 : 0;
    CHKERR VecCreateGhost(PETSC_COMM_WORLD, nb_locals, 1, nb_ghosts, ghosts,
                          &ghostFlux);
    MoFEMFunctionReturn(0);
  }

  /**
   * \brief Delete matrices and vector when no longer needed
   * @return error code
   */
  MoFEMErrorCode destroyMatrices() {
    MoFEMFunctionBegin;
    CHKERR MatDestroy(&Aij);
    CHKERR VecDestroy(&D);
    CHKERR VecDestroy(&D0);
    CHKERR VecDestroy(&D1);
    CHKERR VecDestroy(&F);
    CHKERR VecDestroy(&ghostFlux);
    MoFEMFunctionReturn(0);
  }

  /**
   * \brief Calculate boundary conditions for fluxes
   * @return Error code
   */
  MoFEMErrorCode calculateEssentialBc() {
    MoFEMFunctionBegin;
    // clear vectors
    CHKERR VecZeroEntries(D0);
    CHKERR VecGhostUpdateBegin(D0, INSERT_VALUES, SCATTER_FORWARD);
    CHKERR VecGhostUpdateEnd(D0, INSERT_VALUES, SCATTER_FORWARD);
    // clear essential bc indices, it could have dofs from other mesh refinement
    bcIndices.clear();
    // set operator to calculate essential boundary conditions
    CHKERR DMoFEMLoopFiniteElements(dM, "MIX_BCFLUX", feFaceBc);
    CHKERR VecGhostUpdateBegin(D0, INSERT_VALUES, SCATTER_REVERSE);
    CHKERR VecGhostUpdateEnd(D0, INSERT_VALUES, SCATTER_REVERSE);
    CHKERR VecAssemblyBegin(D0);
    CHKERR VecAssemblyEnd(D0);
    double norm2D0;
    CHKERR VecNorm(D0, NORM_2, &norm2D0);
    // CHKERR VecView(D0,PETSC_VIEWER_STDOUT_WORLD);
    PetscPrintf(PETSC_COMM_WORLD, "norm2D0 = %6.4e\n", norm2D0);
    MoFEMFunctionReturn(0);
  }

  /**
   * \brief Calculate inital pressure head distribution
   * @return Error code
   */
  MoFEMErrorCode calculateInitialPc() {
    MoFEMFunctionBegin;
    // clear vectors
    CHKERR VecZeroEntries(D1);
    CHKERR VecGhostUpdateBegin(D1, INSERT_VALUES, SCATTER_FORWARD);
    CHKERR VecGhostUpdateEnd(D1, INSERT_VALUES, SCATTER_FORWARD);
    // Calculate initial pressure head on each element
    CHKERR DMoFEMLoopFiniteElements(dM, "MIX", feVolInitialPc);
    // Assemble vector
    CHKERR VecGhostUpdateBegin(D1, INSERT_VALUES, SCATTER_REVERSE);
    CHKERR VecGhostUpdateEnd(D1, INSERT_VALUES, SCATTER_REVERSE);
    CHKERR VecAssemblyBegin(D1);
    CHKERR VecAssemblyEnd(D1);
    // Calculate norm
    double norm2D1;
    CHKERR VecNorm(D1, NORM_2, &norm2D1);
    // CHKERR VecView(D0,PETSC_VIEWER_STDOUT_WORLD);
    PetscPrintf(PETSC_COMM_WORLD, "norm2D1 = %6.4e\n", norm2D1);
    MoFEMFunctionReturn(0);
  }

  /**
   * \brief solve problem
   * @return error code
   */
  MoFEMErrorCode solveProblem(bool set_initial_pc = true) {
    MoFEMFunctionBegin;
    if (set_initial_pc) {
      // Set initial head
      CHKERR DMoFEMMeshToLocalVector(dM, D1, INSERT_VALUES, SCATTER_REVERSE);
    }

    // Initiate vector from data on the mesh
    CHKERR DMoFEMMeshToLocalVector(dM, D, INSERT_VALUES, SCATTER_FORWARD);

    // Create time solver
    TS ts;
    CHKERR TSCreate(PETSC_COMM_WORLD, &ts);
    // Use backward Euler method
    CHKERR TSSetType(ts, TSBEULER);
    // Set final time
    double ftime = 1;
    CHKERR TSSetDuration(ts, PETSC_DEFAULT, ftime);
    // Setup solver from commabd line
    CHKERR TSSetFromOptions(ts);
    // Set DM to TS
    CHKERR TSSetDM(ts, dM);
#if PETSC_VERSION_GE(3, 7, 0)
    CHKERR TSSetExactFinalTime(ts, TS_EXACTFINALTIME_STEPOVER);
#endif
    // Set-up monitor
    TsCtx *ts_ctx;
    DMMoFEMGetTsCtx(dM, &ts_ctx);
    CHKERR TSMonitorSet(ts, TsMonitorSet, ts_ctx, PETSC_NULL);

    // This add SNES monitor, to show error by fields. It is dirty trick
    // to add monitor, so code is hidden from doxygen
    CHKERR TSSetSolution(ts, D);
    CHKERR TSSetUp(ts);
    SNES snes;
    CHKERR TSGetSNES(ts, &snes);

#if PETSC_VERSION_GE(3, 7, 0)
    {
      PetscViewerAndFormat *vf;
      CHKERR PetscViewerAndFormatCreate(PETSC_VIEWER_STDOUT_WORLD,
                                        PETSC_VIEWER_DEFAULT, &vf);
      CHKERR SNESMonitorSet(
          snes,
          (MoFEMErrorCode(*)(SNES, PetscInt, PetscReal,
                             void *))SNESMonitorFields,
          vf, (MoFEMErrorCode(*)(void **))PetscViewerAndFormatDestroy);
    }
#else
    {
      CHKERR SNESMonitorSet(snes,
                            (MoFEMErrorCode(*)(SNES, PetscInt, PetscReal,
                                               void *))SNESMonitorFields,
                            0, 0);
    }
#endif

    CHKERR TSSolve(ts, D);

    // Get statistic form TS and print it
    CHKERR TSGetTime(ts, &ftime);
    PetscInt steps, snesfails, rejects, nonlinits, linits;
    CHKERR TSGetTimeStepNumber(ts, &steps);
    CHKERR TSGetSNESFailures(ts, &snesfails);
    CHKERR TSGetStepRejections(ts, &rejects);
    CHKERR TSGetSNESIterations(ts, &nonlinits);
    CHKERR TSGetKSPIterations(ts, &linits);
    PetscPrintf(PETSC_COMM_WORLD,
                "steps %D (%D rejected, %D SNES fails), ftime %g, nonlinits "
                "%D, linits %D\n",
                steps, rejects, snesfails, ftime, nonlinits, linits);

    MoFEMFunctionReturn(0);
  }
};

} // namespace MixTransport

#endif //  __UNSATURATD_FLOW_HPP__
