/* \file FluidPressure.hpp
 *
 * \brief Implementation of fluid pressure element
 *
 * \todo Implement nonlinear case (consrvative force, i.e. normal follows
 * surface normal)
 *
 */



#ifndef __FLUID_PRESSURE_HPP
#define __FLUID_PRESSURE_HPP

/** \brief Fluid pressure forces

\todo Implementation for large displacements

*/
struct FluidPressure {

  MoFEM::Interface &mField;
  struct MyTriangleFE : public MoFEM::FaceElementForcesAndSourcesCore {

    MyTriangleFE(MoFEM::Interface &m_field)
        : MoFEM::FaceElementForcesAndSourcesCore(m_field) {}
    int getRule(int order) { return order + 1; };

    MoFEMErrorCode preProcess() {
      MoFEMFunctionBeginHot;
      MoFEMFunctionReturnHot(0);
    }
  };
  MyTriangleFE fe;
  MyTriangleFE &getLoopFe() { return fe; }

  FluidPressure(MoFEM::Interface &m_field) : mField(m_field), fe(mField) {}

  typedef int MeshSetId;
  struct FluidData {
    double dEnsity; ///< fluid density [kg/m^2] or any consistent unit
    VectorDouble aCCeleration; ///< acceleration [m/s^2]
    VectorDouble zEroPressure; ///< fluid level of reference zero pressure.
    Range
        tRis; ///< range of surface element to which fluid pressure is applied
    friend std::ostream &operator<<(std::ostream &os,
                                    const FluidPressure::FluidData &e);
  };
  std::map<MeshSetId, FluidData> setOfFluids;

  boost::ptr_vector<MethodForForceScaling> methodsOp;

  struct OpCalculatePressure
      : public MoFEM::FaceElementForcesAndSourcesCore::UserDataOperator {

    Vec F;
    FluidData &dAta;
    boost::ptr_vector<MethodForForceScaling> &methodsOp;
    bool allowNegativePressure; ///< allows for negative pressures
    bool hoGeometry;

    OpCalculatePressure(const std::string field_name, Vec _F, FluidData &data,
                        boost::ptr_vector<MethodForForceScaling> &methods_op,
                        bool allow_negative_pressure, bool ho_geometry)
        : MoFEM::FaceElementForcesAndSourcesCore::UserDataOperator(
              field_name, UserDataOperator::OPROW),
          F(_F), dAta(data), methodsOp(methods_op),
          allowNegativePressure(allow_negative_pressure),
          hoGeometry(ho_geometry) {}

    VectorDouble Nf;

    MoFEMErrorCode doWork(int side, EntityType type,
                          EntitiesFieldData::EntData &data);
    
  };

  MoFEMErrorCode addNeumannFluidPressureBCElements(
      const std::string field_name,
      const std::string mesh_nodals_positions = "MESH_NODE_POSITIONS");

  MoFEMErrorCode setNeumannFluidPressureFiniteElementOperators(
      string field_name, Vec F, bool allow_negative_pressure = true,
      bool ho_geometry = false);
};

std::ostream &operator<<(std::ostream &os, const FluidPressure::FluidData &e);

#endif //__FLUID_PRESSSURE_HPP
