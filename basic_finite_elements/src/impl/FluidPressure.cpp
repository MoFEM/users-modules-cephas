/* \file FluidPressure.cpp
 *
 * \brief Implementation of fluid pressure element
 *
 * \todo Implement nonlinear case (consrvative force, i.e. normal follows surface normal)
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

MoFEMErrorCode FluidPressure::addNeumannFluidPressureBCElements(
  const std::string field_name,const std::string mesh_nodals_positions 
) {
  MoFEMFunctionBeginHot;

  ierr = mField.add_finite_element("FLUID_PRESSURE_FE",MF_ZERO); CHKERRG(ierr);
  ierr = mField.modify_finite_element_add_field_row("FLUID_PRESSURE_FE",field_name); CHKERRG(ierr);
  ierr = mField.modify_finite_element_add_field_col("FLUID_PRESSURE_FE",field_name); CHKERRG(ierr);
  ierr = mField.modify_finite_element_add_field_data("FLUID_PRESSURE_FE",field_name); CHKERRG(ierr);
  if(mField.check_field(mesh_nodals_positions)) {
    ierr = mField.modify_finite_element_add_field_data("FLUID_PRESSURE_FE",mesh_nodals_positions); CHKERRG(ierr);
  }

  //takes skin of block of entities
  Skinner skin(&mField.get_moab());
  // loop over all blocksets and get data which name is FluidPressure
  for(_IT_CUBITMESHSETS_BY_SET_TYPE_FOR_LOOP_(mField,BLOCKSET,bit)) {

    if(bit->getName().compare(0,14,"FLUID_PRESSURE") == 0) {

      //get block attributes
      std::vector<double> attributes;
      ierr = bit->getAttributes(attributes); CHKERRG(ierr);
      if(attributes.size()<7) {
        SETERRQ1(PETSC_COMM_SELF,1,"not enough block attributes to deffine fluid pressure element, attributes.size() = %d ",attributes.size());
      }
      setOfFluids[bit->getMeshsetId()].dEnsity = attributes[0];
      setOfFluids[bit->getMeshsetId()].aCCeleration.resize(3);
      setOfFluids[bit->getMeshsetId()].aCCeleration[0] = attributes[1];
      setOfFluids[bit->getMeshsetId()].aCCeleration[1] = attributes[2];
      setOfFluids[bit->getMeshsetId()].aCCeleration[2] = attributes[3];
      setOfFluids[bit->getMeshsetId()].zEroPressure.resize(3);
      setOfFluids[bit->getMeshsetId()].zEroPressure[0] = attributes[4];
      setOfFluids[bit->getMeshsetId()].zEroPressure[1] = attributes[5];
      setOfFluids[bit->getMeshsetId()].zEroPressure[2] = attributes[6];
      //get blok tetrahedron and triangles
      Range tets;
      rval = mField.get_moab().get_entities_by_type(bit->meshset,MBTET,tets,true); CHKERRG(rval);
      Range tris;
      rval = mField.get_moab().get_entities_by_type(bit->meshset,MBTRI,setOfFluids[bit->getMeshsetId()].tRis,true); CHKERRG(rval);
      //this get triangles only on block surfaces
      Range tets_skin_tris;
      rval = skin.find_skin(0,tets,false,tets_skin_tris); CHKERRG(rval);
      setOfFluids[bit->getMeshsetId()].tRis.merge(tets_skin_tris);
      std::ostringstream ss;
      ss << setOfFluids[bit->getMeshsetId()] << std::endl;
      PetscPrintf(mField.get_comm(),ss.str().c_str());

      ierr = mField.add_ents_to_finite_element_by_type(setOfFluids[bit->getMeshsetId()].tRis,MBTRI,"FLUID_PRESSURE_FE"); CHKERRG(ierr);

    }

  }

  MoFEMFunctionReturnHot(0);
}


MoFEMErrorCode FluidPressure::setNeumannFluidPressureFiniteElementOperators(
  string field_name,Vec F,bool allow_negative_pressure,bool ho_geometry
) {
  MoFEMFunctionBeginHot;
  std::map<MeshSetId,FluidData>::iterator sit = setOfFluids.begin();
  for(;sit!=setOfFluids.end();sit++) {
    //add finite element
    fe.getOpPtrVector().push_back(new OpCalculatePressure(
      field_name,F,sit->second,methodsOp,allow_negative_pressure,ho_geometry
    ));
  }
  MoFEMFunctionReturnHot(0);
}

std::ostream& operator<<(std::ostream& os,const FluidPressure::FluidData &e) {
  os << "dEnsity " << e.dEnsity << std::endl;
  os << "aCCeleration " << e.aCCeleration << std::endl;
  os << "zEroPressure " << e.zEroPressure << std::endl;
  return os;
}
