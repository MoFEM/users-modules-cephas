/* \file FluidPressure.hpp
 *
 * \brief Implementation of fluid pressure element
 *
 * \todo Implement nonlinear case (consrvative force, i.e. normal follows
 * surface normal)
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
