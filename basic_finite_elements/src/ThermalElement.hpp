/** \file ThermalElement.hpp
 \ingroup mofem_thermal_elem

 \brief Operators and data structures for thermal analysis

 Implementation of thermal element for unsteady and steady case.
 Radiation and convection blocks implemented by Xuan Meng

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

#ifndef __THERMAL_ELEMENT_HPP
#define __THERMAL_ELEMENT_HPP

/** \brief structure grouping operators and data used for thermal problems
 * \ingroup mofem_thermal_elem
 *
 * In order to assemble matrices and right hand vectors, the loops over
 * elements, entities within the element and finally loop over integration
 * points are executed.
 *
 * Following implementation separate those three types of loops and to each
 * loop attach operator.
 *
 */
struct ThermalElement {

  /// \brief  definition of volume element
  struct MyVolumeFE : public MoFEM::VolumeElementForcesAndSourcesCore {
    MyVolumeFE(MoFEM::Interface &m_field)
        : MoFEM::VolumeElementForcesAndSourcesCore(m_field) {}

    /** \brief it is used to calculate nb. of Gauss integration points
     *
     * for more details pleas look
     *   Reference:
     *
     * Albert Nijenhuis, Herbert Wilf,
     * Combinatorial Algorithms for Computers and Calculators,
     * Second Edition,
     * Academic Press, 1978,
     * ISBN: 0-12-519260-6,
     * LC: QA164.N54.
     *
     * More details about algorithm
     * http://people.sc.fsu.edu/~jburkardt/cpp_src/gm_rule/gm_rule.html
     **/
    int getRule(int order) { return 2 * (order - 1); };
  };
  MyVolumeFE feRhs; ///< cauclate right hand side for tetrahedral elements
  MyVolumeFE &getLoopFeRhs() { return feRhs; } ///< get rhs volume element
  MyVolumeFE feLhs; //< calculate left hand side for tetrahedral elements
  MyVolumeFE &getLoopFeLhs() { return feLhs; } ///< get lhs volume element

  /** \brief define surface element
   *
   * This element is used to integrate heat fluxes; convection and radiation
   */
  struct MyTriFE : public MoFEM::FaceElementForcesAndSourcesCore {
    MyTriFE(MoFEM::Interface &m_field)
        : MoFEM::FaceElementForcesAndSourcesCore(m_field) {}
    int getRule(int order) { return 2 * order; };
  };

  MyTriFE feFlux;                             //< heat flux element
  MyTriFE &getLoopFeFlux() { return feFlux; } //< get heat flux element

  MyTriFE feConvectionRhs; //< convection element
  MyTriFE feConvectionLhs;
  MyTriFE &getLoopFeConvectionRhs() {
    return feConvectionRhs;
  } //< get convection element
  MyTriFE &getLoopFeConvectionLhs() { return feConvectionLhs; }

  MyTriFE feRadiationRhs; //< radiation element
  MyTriFE feRadiationLhs;
  MyTriFE &getLoopFeRadiationRhs() {
    return feRadiationRhs;
  } //< get radiation element
  MyTriFE &getLoopFeRadiationLhs() { return feRadiationLhs; }

  MoFEM::Interface &mField;
  ThermalElement(MoFEM::Interface &m_field)
      : feRhs(m_field), feLhs(m_field), feFlux(m_field),
        feConvectionRhs(m_field), feConvectionLhs(m_field),
        feRadiationRhs(m_field), feRadiationLhs(m_field), mField(m_field) {}

  /** \brief data for calculation heat conductivity and heat capacity elements
   * \ingroup mofem_thermal_elem
   */
  struct BlockData {
    // double cOnductivity;
    MatrixDouble cOnductivity_mat; // This is (3x3) conductivity matrix
    double cApacity; // rou * c_p == material density multiple heat capacity
    double initTemp; ///< initial temperature
    Range tEts;      ///< contains elements in block set
  };
  std::map<int, BlockData>
      setOfBlocks; ///< maps block set id with appropriate BlockData

  /** \brief data for calculation heat flux
   * \ingroup mofem_thermal_elem
   */
  struct FluxData {
    HeatFluxCubitBcData dAta; ///< for more details look to BCMultiIndices.hpp
                              ///< to see details of HeatFluxCubitBcData
    Range tRis;               ///< surface triangles where hate flux is applied
  };
  std::map<int, FluxData>
      setOfFluxes; ///< maps side set id with appropriate FluxData

  /** \brief data for convection
   * \ingroup mofem_thermal_elem
   */
  struct ConvectionData {
    double cOnvection;  /*The summation of Convection coefficients*/
    double tEmperature; /*Ambient temperature of the area contains the black
                           body */
    Range tRis; ///< those will be on body skin, except this with contact with
                ///< other body where temperature is applied
  };
  std::map<int, ConvectionData>
      setOfConvection; //< maps block set id with appropriate data

  /** \brief data for radiation
   * \ingroup mofem_thermal_elem
   */
  struct RadiationData {
    double sIgma;      /* The Stefan-Boltzmann constant*/
    double eMissivity; /* The surface emissivity coefficients range = [0,1] */
    // double aBsorption; /* The surface absorption coefficients */
    double aMbienttEmp; /* The incident radiant heat flow per unit surface area;
                           or the ambient temperature of space*/
    Range tRis; ///< those will be on body skin, except this with contact with
                ///< other body where temperature is applied
  };
  std::map<int, RadiationData>
      setOfRadiation; //< maps block set id with appropriate data

  /** \brief common data used by volume elements
   * \ingroup mofem_thermal_elem
   */
  struct CommonData {
    VectorDouble temperatureAtGaussPts;
    VectorDouble temperatureRateAtGaussPts;
    MatrixDouble gradAtGaussPts;
    inline ublas::matrix_row<MatrixDouble> getGradAtGaussPts(const int gg) {
      return ublas::matrix_row<MatrixDouble>(gradAtGaussPts, gg);
    }
  };
  CommonData commonData;

  /// \brief operator to calculate temperature gradient at Gauss points
  struct OpGetGradAtGaussPts
      : public MoFEM::VolumeElementForcesAndSourcesCore::UserDataOperator {

    CommonData &commonData;
    OpGetGradAtGaussPts(const std::string field_name, CommonData &common_data)
        : MoFEM::VolumeElementForcesAndSourcesCore::UserDataOperator(
              field_name, UserDataOperator::OPROW),
          commonData(common_data) {}

    /** \brief operator calculating temperature gradients
     *
     * temperature gradient is calculated multiplying derivatives of shape
     * functions by degrees of freedom.
     */
    MoFEMErrorCode doWork(int side, EntityType type,
                          EntitiesFieldData::EntData &data);
  };

  /** \brief operator to calculate temperature  and rate of temperature at Gauss
   * points \ingroup mofem_thermal_elem
   */
  template <typename OP>
  struct OpGetFieldAtGaussPts : public OP::UserDataOperator {

    VectorDouble &fieldAtGaussPts;
    OpGetFieldAtGaussPts(const std::string field_name,
                         VectorDouble &field_at_gauss_pts)
        : OP::UserDataOperator(field_name, OP::UserDataOperator::OPROW),
          fieldAtGaussPts(field_at_gauss_pts) {}

    /** \brief operator calculating temperature and rate of temperature
     *
     * temperature temperature or rate of temperature is calculated multiplying
     * shape functions by degrees of freedom
     */
    MoFEMErrorCode doWork(int side, EntityType type,
                          EntitiesFieldData::EntData &data) {
      MoFEMFunctionBeginHot;
      try {

        if (data.getFieldData().size() == 0)
          MoFEMFunctionReturnHot(0);
        int nb_dofs = data.getFieldData().size();
        int nb_gauss_pts = data.getN().size1();

        // initialize
        fieldAtGaussPts.resize(nb_gauss_pts);
        if (type == MBVERTEX) {
          // loop over shape functions on entities always start from
          // vertices, so if nodal shape functions are processed, vector of
          // field values is zero at initialization
          std::fill(fieldAtGaussPts.begin(), fieldAtGaussPts.end(), 0);
        }

        for (int gg = 0; gg < nb_gauss_pts; gg++) {
          fieldAtGaussPts[gg] +=
              inner_prod(data.getN(gg, nb_dofs), data.getFieldData());
        }

      } catch (const std::exception &ex) {
        std::ostringstream ss;
        ss << "throw in method: " << ex.what() << std::endl;
        SETERRQ(PETSC_COMM_SELF, MOFEM_STD_EXCEPTION_THROW, ss.str().c_str());
      }

      MoFEMFunctionReturnHot(0);
    }
  };

  /** \brief operator to calculate temperature at Gauss pts
   * \ingroup mofem_thermal_elem
   */
  struct OpGetTetTemperatureAtGaussPts
      : public OpGetFieldAtGaussPts<MoFEM::VolumeElementForcesAndSourcesCore> {
    OpGetTetTemperatureAtGaussPts(const std::string field_name,
                                  CommonData &common_data)
        : OpGetFieldAtGaussPts<MoFEM::VolumeElementForcesAndSourcesCore>(
              field_name, common_data.temperatureAtGaussPts) {}
  };

  /** \brief operator to calculate temperature at Gauss pts
   * \ingroup mofem_thermal_elem
   */
  struct OpGetTriTemperatureAtGaussPts
      : public OpGetFieldAtGaussPts<MoFEM::FaceElementForcesAndSourcesCore> {
    OpGetTriTemperatureAtGaussPts(const std::string field_name,
                                  CommonData &common_data)
        : OpGetFieldAtGaussPts<MoFEM::FaceElementForcesAndSourcesCore>(
              field_name, common_data.temperatureAtGaussPts) {}
  };

  /** \brief operator to calculate temperature rate at Gauss pts
   * \ingroup mofem_thermal_elem
   */
  struct OpGetTetRateAtGaussPts
      : public OpGetFieldAtGaussPts<MoFEM::VolumeElementForcesAndSourcesCore> {
    OpGetTetRateAtGaussPts(const std::string field_name,
                           CommonData &common_data)
        : OpGetFieldAtGaussPts<MoFEM::VolumeElementForcesAndSourcesCore>(
              field_name, common_data.temperatureRateAtGaussPts) {}
  };

  /** \biref operator to calculate right hand side of heat conductivity terms
   * \ingroup mofem_thermal_elem
   */
  struct OpThermalRhs
      : public MoFEM::VolumeElementForcesAndSourcesCore::UserDataOperator {

    BlockData &dAta;
    CommonData &commonData;
    bool useTsF;
    OpThermalRhs(const std::string field_name, BlockData &data,
                 CommonData &common_data)
        : MoFEM::VolumeElementForcesAndSourcesCore::UserDataOperator(
              field_name, UserDataOperator::OPROW),
          dAta(data), commonData(common_data), useTsF(true) {}

    Vec F;
    OpThermalRhs(const std::string field_name, Vec _F, BlockData &data,
                 CommonData &common_data)
        : MoFEM::VolumeElementForcesAndSourcesCore::UserDataOperator(
              field_name, UserDataOperator::OPROW),
          dAta(data), commonData(common_data), useTsF(false), F(_F) {}

    VectorDouble Nf;

    /** \brief calculate thermal conductivity matrix
     *
     * F = int diffN^T k gard_T dOmega^2
     *
     */
    MoFEMErrorCode doWork(int side, EntityType type,
                          EntitiesFieldData::EntData &data);
  };

  /** \biref operator to calculate left hand side of heat conductivity terms
   * \ingroup mofem_thermal_elem
   */
  struct OpThermalLhs
      : public MoFEM::VolumeElementForcesAndSourcesCore::UserDataOperator {

    BlockData &dAta;
    CommonData &commonData;
    bool useTsB;
    OpThermalLhs(const std::string field_name, BlockData &data,
                 CommonData &common_data)
        : MoFEM::VolumeElementForcesAndSourcesCore::UserDataOperator(
              field_name, UserDataOperator::OPROWCOL),
          dAta(data), commonData(common_data), useTsB(true) {}

    Mat A;
    OpThermalLhs(const std::string field_name, Mat _A, BlockData &data,
                 CommonData &common_data)
        : MoFEM::VolumeElementForcesAndSourcesCore::UserDataOperator(
              field_name, UserDataOperator::OPROWCOL),
          dAta(data), commonData(common_data), useTsB(false), A(_A) {}

    MatrixDouble K, transK;

    /** \brief calculate thermal conductivity matrix
     *
     * K = int diffN^T k diffN^T dOmega^2
     *
     */
    MoFEMErrorCode doWork(int row_side, int col_side, EntityType row_type,
                          EntityType col_type,
                          EntitiesFieldData::EntData &row_data,
                          EntitiesFieldData::EntData &col_data);
  };

  /** \brief operator to calculate right hand side of heat capacity terms
   * \ingroup mofem_thermal_elem
   */
  struct OpHeatCapacityRhs
      : public MoFEM::VolumeElementForcesAndSourcesCore::UserDataOperator {

    BlockData &dAta;
    CommonData &commonData;
    OpHeatCapacityRhs(const std::string field_name, BlockData &data,
                      CommonData &common_data)
        : MoFEM::VolumeElementForcesAndSourcesCore::UserDataOperator(
              field_name, UserDataOperator::OPROW),
          dAta(data), commonData(common_data) {}

    VectorDouble Nf;

    /** \brief calculate thermal conductivity matrix
     *
     * F = int N^T c (dT/dt) dOmega^2
     *
     */
    MoFEMErrorCode doWork(int side, EntityType type,
                          EntitiesFieldData::EntData &data);
  };

  /** \brief operator to calculate left hand side of heat capacity terms
   * \ingroup mofem_thermal_elem
   */
  struct OpHeatCapacityLhs
      : public MoFEM::VolumeElementForcesAndSourcesCore::UserDataOperator {

    BlockData &dAta;
    CommonData &commonData;
    OpHeatCapacityLhs(const std::string field_name, BlockData &data,
                      CommonData &common_data)
        : MoFEM::VolumeElementForcesAndSourcesCore::UserDataOperator(
              field_name, UserDataOperator::OPROWCOL),
          dAta(data), commonData(common_data) {}

    MatrixDouble M, transM;

    /** \brief calculate heat capacity matrix
     *
     * M = int N^T c N dOmega^2
     *
     */
    MoFEMErrorCode doWork(int row_side, int col_side, EntityType row_type,
                          EntityType col_type,
                          EntitiesFieldData::EntData &row_data,
                          EntitiesFieldData::EntData &col_data);
  };

  /** \brief operator for calculate heat flux and assemble to right hand side
   * \ingroup mofem_thermal_elem
   */
  struct OpHeatFlux
      : public MoFEM::FaceElementForcesAndSourcesCore::UserDataOperator {

    FluxData &dAta;
    bool hoGeometry;
    bool useTsF;
    OpHeatFlux(const std::string field_name, FluxData &data,
               bool ho_geometry = false)
        : MoFEM::FaceElementForcesAndSourcesCore::UserDataOperator(
              field_name, UserDataOperator::OPROW),
          dAta(data), hoGeometry(ho_geometry), useTsF(true) {}

    Vec F;
    OpHeatFlux(const std::string field_name, Vec _F, FluxData &data,
               bool ho_geometry = false)
        : MoFEM::FaceElementForcesAndSourcesCore::UserDataOperator(
              field_name, UserDataOperator::OPROW),
          dAta(data), hoGeometry(ho_geometry), useTsF(false), F(_F) {}

    VectorDouble Nf;

    /** \brief calculate heat flux
     *
     * F = int_S N^T * flux dS
     *
     */
    MoFEMErrorCode doWork(int side, EntityType type,
                          EntitiesFieldData::EntData &data);
  };

  /**
   * operator to calculate radiation therms on body surface and assemble to lhs
   * of equations for the jocabian Matrix of Picard Linearization \ingroup
   * mofem_thermal_elem
   */
  struct OpRadiationLhs
      : public MoFEM::FaceElementForcesAndSourcesCore::UserDataOperator {
    CommonData
        &commonData; // get the temperature or temperature Rate from CommonData
    RadiationData &dAta;
    bool hoGeometry;
    bool useTsB;

    OpRadiationLhs(const std::string field_name, RadiationData &data,
                   CommonData &common_data, bool ho_geometry = false)
        : MoFEM::FaceElementForcesAndSourcesCore::UserDataOperator(
              field_name, UserDataOperator::OPROWCOL),
          commonData(common_data), dAta(data), hoGeometry(ho_geometry),
          useTsB(true) {}

    Mat A;
    OpRadiationLhs(const std::string field_name, Mat _A, RadiationData &data,
                   CommonData &common_data, bool ho_geometry = false)
        : MoFEM::FaceElementForcesAndSourcesCore::UserDataOperator(
              field_name, UserDataOperator::OPROWCOL),
          commonData(common_data), dAta(data), hoGeometry(ho_geometry),
          useTsB(false), A(_A) {}

    MatrixDouble N, transN;

    /** \brief calculate thermal radiation term in the lhs of equations(Tangent
     * Matrix) for transient Thermal Problem
     *
     * K = intS 4* N^T* sIgma* eMissivity* N*  T^3 dS (Reference _ see Finite
     * Element Simulation of Heat Transfer by jean-Michel Bergheau)
     */
    MoFEMErrorCode doWork(int row_side, int col_side, EntityType row_type,
                          EntityType col_type,
                          EntitiesFieldData::EntData &row_data,
                          EntitiesFieldData::EntData &col_data);
  };

  /** \brief operator to calculate radiation therms on body surface and assemble
   * to rhs of transient equations(Residual Vector) \ingroup mofem_thermal_elem
   */
  struct OpRadiationRhs
      : public MoFEM::FaceElementForcesAndSourcesCore::UserDataOperator {

    CommonData
        &commonData; // get the temperature or temperature Rate from CommonData
    RadiationData &dAta;
    bool hoGeometry;
    bool useTsF;
    OpRadiationRhs(const std::string field_name, RadiationData &data,
                   CommonData &common_data, bool ho_geometry = false)
        : MoFEM::FaceElementForcesAndSourcesCore::UserDataOperator(
              field_name, UserDataOperator::OPROW),
          commonData(common_data), dAta(data), hoGeometry(ho_geometry),
          useTsF(true) {}

    Vec F;
    OpRadiationRhs(const std::string field_name, Vec _F, RadiationData &data,
                   CommonData &common_data, bool ho_geometry = false)
        : MoFEM::FaceElementForcesAndSourcesCore::UserDataOperator(
              field_name, UserDataOperator::OPROW),
          commonData(common_data), dAta(data), hoGeometry(ho_geometry),
          useTsF(false), F(_F) {}

    VectorDouble Nf;

    /** \brief calculate Transient Radiation condition on the right hand side
     *residual
     *
     *  R=int_S N^T * sIgma * eMissivity * (Ta^4 -Ts^4) dS
     **/
    MoFEMErrorCode doWork(int side, EntityType type,
                          EntitiesFieldData::EntData &data);
  };

  /** \brief operator to calculate convection therms on body surface and
   * assemble to rhs of equations \ingroup mofem_thermal_elem
   */
  struct OpConvectionRhs
      : public MoFEM::FaceElementForcesAndSourcesCore::UserDataOperator {

    CommonData
        &commonData; // get the temperature or temperature Rate from CommonData
    ConvectionData &dAta;
    bool hoGeometry;
    bool useTsF;
    OpConvectionRhs(const std::string field_name, ConvectionData &data,
                    CommonData &common_data, bool ho_geometry = false)
        : MoFEM::FaceElementForcesAndSourcesCore::UserDataOperator(
              field_name, UserDataOperator::OPROW),
          commonData(common_data), dAta(data), hoGeometry(ho_geometry),
          useTsF(true) {}

    Vec F;
    OpConvectionRhs(const std::string field_name, Vec _F, ConvectionData &data,
                    CommonData &common_data, bool ho_geometry = false)
        : MoFEM::FaceElementForcesAndSourcesCore::UserDataOperator(
              field_name, UserDataOperator::OPROW),
          commonData(common_data), dAta(data), hoGeometry(ho_geometry),
          useTsF(false), F(_F) {}

    VectorDouble Nf;

    /** brief calculate Convection condition on the right hand side
     *  R=int_S N^T*alpha*N_f  dS **/

    MoFEMErrorCode doWork(int side, EntityType type,
                          EntitiesFieldData::EntData &data);
  };

  /// \biref operator to calculate convection therms on body surface and
  /// assemble to lhs of equations
  struct OpConvectionLhs
      : public MoFEM::FaceElementForcesAndSourcesCore::UserDataOperator {

    ConvectionData &dAta;
    bool hoGeometry;
    bool useTsB;

    OpConvectionLhs(const std::string field_name, ConvectionData &data,
                    bool ho_geometry = false)
        : MoFEM::FaceElementForcesAndSourcesCore::UserDataOperator(
              field_name, UserDataOperator::OPROWCOL),
          dAta(data), hoGeometry(ho_geometry), useTsB(true) {}

    Mat A;
    OpConvectionLhs(const std::string field_name, Mat _A, ConvectionData &data,
                    bool ho_geometry = false)
        : MoFEM::FaceElementForcesAndSourcesCore::UserDataOperator(
              field_name, UserDataOperator::OPROWCOL),
          dAta(data), hoGeometry(ho_geometry), useTsB(false), A(_A) {}

    MatrixDouble K, transK;
    /** \brief calculate thermal convection term in the lhs of equations
     *
     * K = intS N^T alpha N dS
     */
    MoFEMErrorCode doWork(int row_side, int col_side, EntityType row_type,
                          EntityType col_type,
                          EntitiesFieldData::EntData &row_data,
                          EntitiesFieldData::EntData &col_data);
  };

  /** \brief this calass is to control time stepping
   * \ingroup mofem_thermal_elem
   *
   * It is used to save data for temperature rate vector to MoFEM field.
   */
  struct UpdateAndControl : public FEMethod {

    MoFEM::Interface &mField;
    const std::string tempName;
    const std::string rateName;

    UpdateAndControl(MoFEM::Interface &m_field, const std::string temp_name,
                     const std::string rate_name)
        : mField(m_field), tempName(temp_name), rateName(rate_name) {}

    MoFEMErrorCode preProcess();
    MoFEMErrorCode postProcess();
  };

  /** \brief TS monitore it records temperature at time steps
   * \ingroup mofem_thermal_elem
   */
  struct TimeSeriesMonitor : public FEMethod {

    MoFEM::Interface &mField;
    const std::string seriesName;
    const std::string tempName;
    BitRefLevel mask;

    TimeSeriesMonitor(MoFEM::Interface &m_field, const std::string series_name,
                      const std::string temp_name)
        : mField(m_field), seriesName(series_name), tempName(temp_name) {
      mask.set();
    }

    MoFEMErrorCode postProcess();
  };

  /** \brief add thermal element on tets
   * \ingroup mofem_thermal_elem
   *
   * It get data from block set and define element in moab
   *w
   * \param field name
   * \param name of mesh nodal positions (if not defined nodal coordinates are
   *used)
   */
  MoFEMErrorCode addThermalElements(
      const std::string field_name,
      const std::string mesh_nodals_positions = "MESH_NODE_POSITIONS");

  /** \brief add heat flux element
   * \ingroup mofem_thermal_elem
   *
   * It get data from heat flux set and define element in moab. Alternatively
   * uses block set with name HEAT_FLUX.
   *
   * \param field name
   * \param name of mesh nodal positions (if not defined nodal coordinates are
   * used)
   */
  MoFEMErrorCode addThermalFluxElement(
      const std::string field_name,
      const std::string mesh_nodals_positions = "MESH_NODE_POSITIONS");

  /** \brief add convection element
   * \ingroup mofem_thermal_elem
   *
   * It get data from convection set and define element in moab. Alternatively
   * uses block set with name CONVECTION.
   *
   * \param field name
   * \param name of mesh nodal positions (if not defined nodal coordinates are
   * used)
   */
  MoFEMErrorCode addThermalConvectionElement(
      const std::string field_name,
      const std::string mesh_nodals_positions = "MESH_NODE_POSITIONS");

  /** \brief add Non-linear Radiation element
   * \ingroup mofem_thermal_elem
   *
   * It get data from Radiation set and define element in moab. Alternatively
   * uses block set with name RADIATION.
   *
   * \param field name
   * \param name of mesh nodal positions (if not defined nodal coordinates are
   * used)
   */
  MoFEMErrorCode addThermalRadiationElement(
      const std::string field_name,
      const std::string mesh_nodals_positions = "MESH_NODE_POSITIONS");

  /** \brief this function is used in case of stationary problem to set elements
   * for rhs \ingroup mofem_thermal_elem
   */
  MoFEMErrorCode setThermalFiniteElementRhsOperators(string field_name, Vec &F);

  /** \brief this function is used in case of stationary heat conductivity
   * problem for lhs \ingroup mofem_thermal_elem
   */
  MoFEMErrorCode setThermalFiniteElementLhsOperators(string field_name, Mat A);

  /** \brief this function is used in case of stationary problem for heat flux
   * terms \ingroup mofem_thermal_elem
   */
  MoFEMErrorCode setThermalFluxFiniteElementRhsOperators(
      string field_name, Vec &F,
      const std::string mesh_nodals_positions = "MESH_NODE_POSITIONS");

  /* \brief linear Steady convection terms in lhs
   */
  MoFEMErrorCode setThermalConvectionFiniteElementRhsOperators(
      string field_name, Vec &F,
      const std::string mesh_nodals_positions = "MESH_NODE_POSITIONS");

  /* \brief linear Steady convection terms in rhs
   */
  MoFEMErrorCode setThermalConvectionFiniteElementLhsOperators(
      string field_name, Mat A,
      const std::string mesh_nodals_positions = "MESH_NODE_POSITIONS");

  /** \brief set up operators for unsteady heat flux; convection; radiation
   * problem \ingroup mofem_thermal_elem
   */
  MoFEMErrorCode setTimeSteppingProblem(
      string field_name, string rate_name,
      const std::string mesh_nodals_positions = "MESH_NODE_POSITIONS");

  /** \brief set up operators for unsteady heat flux; convection; radiation
   * problem \ingroup mofem_thermal_elem
   */
  MoFEMErrorCode setTimeSteppingProblem(
      TsCtx &ts_ctx, string field_name, string rate_name,
      const std::string mesh_nodals_positions = "MESH_NODE_POSITIONS");
};

#endif //__THERMAL_ELEMENT_HPP

/**
 * \defgroup mofem_thermal_elem Thermal element
 * \ingroup user_modules
 **/
