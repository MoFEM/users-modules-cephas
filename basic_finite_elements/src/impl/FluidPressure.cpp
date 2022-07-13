/* \file FluidPressure.cpp
 *
 * \brief Implementation of fluid pressure element
 *
 * \todo Implement nonlinear case (consrvative force, i.e. normal follows
 * surface normal)
 *
 */



MoFEMErrorCode FluidPressure::OpCalculatePressure::doWork(
    int side, EntityType type, EntitiesFieldData::EntData &data) {
  MoFEMFunctionBegin;
  if (data.getIndices().size() == 0)
    MoFEMFunctionReturnHot(0);
  EntityHandle ent = getNumeredEntFiniteElementPtr()->getEnt();
  if (dAta.tRis.find(ent) == dAta.tRis.end())
    MoFEMFunctionReturnHot(0);

  const auto &dof_ptr = data.getFieldDofs()[0];
  const int rank = dof_ptr->getNbOfCoeffs();
  const int nb_row_dofs = data.getIndices().size() / rank;

  Nf.resize(data.getIndices().size());
  Nf.clear();

  for (unsigned int gg = 0; gg < data.getN().size1(); gg++) {

    VectorDouble dist;
    VectorDouble zero_pressure = dAta.zEroPressure;

    dist = ublas::matrix_row<MatrixDouble>(getCoordsAtGaussPts(), gg);
    dist -= zero_pressure;
    double dot = cblas_ddot(3, &dist[0], 1, &dAta.aCCeleration[0], 1);
    if (!allowNegativePressure)
      dot = fmax(0, dot);
    double pressure = dot * dAta.dEnsity;

    for (int rr = 0; rr < rank; rr++) {
      double force;
      if (hoGeometry) {
        force = pressure * getNormalsAtGaussPts()(gg, rr);
      } else {
        force = pressure * getNormal()[rr];
      }
      cblas_daxpy(nb_row_dofs, getGaussPts()(2, gg) * force,
                  &data.getN()(gg, 0), 1, &Nf[rr], rank);
    }
  }

  if (F == PETSC_NULL)
    F = getKSPf();

  if (F == PETSC_NULL)
    SETERRQ(PETSC_COMM_SELF, MOFEM_IMPOSIBLE_CASE, "impossible case");

  CHKERR VecSetValues(F, data.getIndices().size(), &data.getIndices()[0],
                      &Nf[0], ADD_VALUES);

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode FluidPressure::addNeumannFluidPressureBCElements(
    const std::string field_name, const std::string mesh_nodals_positions) {
  MoFEMFunctionBegin;

  CHKERR mField.add_finite_element("FLUID_PRESSURE_FE", MF_ZERO);
  CHKERR mField.modify_finite_element_add_field_row("FLUID_PRESSURE_FE",
                                                    field_name);
  CHKERR mField.modify_finite_element_add_field_col("FLUID_PRESSURE_FE",
                                                    field_name);
  CHKERR mField.modify_finite_element_add_field_data("FLUID_PRESSURE_FE",
                                                     field_name);
  if (mField.check_field(mesh_nodals_positions)) {
    CHKERR mField.modify_finite_element_add_field_data("FLUID_PRESSURE_FE",
                                                       mesh_nodals_positions);
  }

  // takes skin of block of entities
  Skinner skin(&mField.get_moab());
  // loop over all blocksets and get data which name is FluidPressure
  for (_IT_CUBITMESHSETS_BY_SET_TYPE_FOR_LOOP_(mField, BLOCKSET, bit)) {

    if (bit->getName().compare(0, 14, "FLUID_PRESSURE") == 0) {

      // get block attributes
      std::vector<double> attributes;
      CHKERR bit->getAttributes(attributes);
      if (attributes.size() < 7) {
        SETERRQ1(PETSC_COMM_SELF, 1,
                 "not enough block attributes to deffine fluid pressure "
                 "element, attributes.size() = %d ",
                 attributes.size());
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
      // get blok tetrahedron and triangles
      Range tets;
      CHKERR mField.get_moab().get_entities_by_type(bit->meshset, MBTET, tets,
                                                    true);
      Range tris;
      CHKERR mField.get_moab().get_entities_by_type(
          bit->meshset, MBTRI, setOfFluids[bit->getMeshsetId()].tRis, true);
      // this get triangles only on block surfaces
      Range tets_skin_tris;
      CHKERR skin.find_skin(0, tets, false, tets_skin_tris);
      setOfFluids[bit->getMeshsetId()].tRis.merge(tets_skin_tris);
      std::ostringstream ss;
      ss << setOfFluids[bit->getMeshsetId()] << std::endl;
      PetscPrintf(mField.get_comm(), ss.str().c_str());

      CHKERR mField.add_ents_to_finite_element_by_type(
          setOfFluids[bit->getMeshsetId()].tRis, MBTRI, "FLUID_PRESSURE_FE");
    }
  }

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode FluidPressure::setNeumannFluidPressureFiniteElementOperators(
    string field_name, Vec F, bool allow_negative_pressure, bool ho_geometry) {
  MoFEMFunctionBegin;
  std::map<MeshSetId, FluidData>::iterator sit = setOfFluids.begin();
  for (; sit != setOfFluids.end(); sit++) {
    // add finite element
    fe.getOpPtrVector().push_back(
        new OpCalculatePressure(field_name, F, sit->second, methodsOp,
                                allow_negative_pressure, ho_geometry));
  }
  MoFEMFunctionReturn(0);
}

std::ostream &operator<<(std::ostream &os, const FluidPressure::FluidData &e) {
  os << "dEnsity " << e.dEnsity << std::endl;
  os << "aCCeleration " << e.aCCeleration << std::endl;
  os << "zEroPressure " << e.zEroPressure << std::endl;
  return os;
}
