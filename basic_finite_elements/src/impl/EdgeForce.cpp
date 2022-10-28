
/** \file EdgeForce.cpp
  \ingroup mofem_static_boundary_conditions
*/



#include <MoFEM.hpp>
using namespace MoFEM;
#include <MethodForForceScaling.hpp>
#include <SurfacePressure.hpp>
#include <EdgeForce.hpp>

EdgeForce::OpEdgeForce::OpEdgeForce(
    const std::string field_name, Vec f, bCForce &data,
    boost::ptr_vector<MethodForForceScaling> &methods_op, bool use_snes_f)
    : EdgeElementForcesAndSourcesCore::UserDataOperator(field_name, OPROW),
      F(f), dAta(data), methodsOp(methods_op), useSnesF(use_snes_f) {}

MoFEMErrorCode
EdgeForce::OpEdgeForce::doWork(int side, EntityType type,
                               EntitiesFieldData::EntData &data) {
  MoFEMFunctionBegin;

  if (data.getIndices().size() == 0) {
    MoFEMFunctionReturnHot(0);
  }
  EntityHandle ent = getNumeredEntFiniteElementPtr()->getEnt();
  if (dAta.eDges.find(ent) == dAta.eDges.end()) {
    MoFEMFunctionReturnHot(0);
  }

  // Get pointer to DOF and its rank
  const auto &dof_ptr = data.getFieldDofs()[0];
  int rank = dof_ptr->getNbOfCoeffs();

  int nb_dofs = data.getIndices().size();

  Nf.resize(nb_dofs, false);
  Nf.clear();

  int nb_gauss_pts = data.getN().size1();
  wEights.resize(nb_gauss_pts, false);

  // This will work for fluxes and other fields with rank other than 3.
  for (int rr = 0; rr < rank; rr++) {

    // Get force value for each vector element from blockset data.
    double force;
    if (rr == 0) {
      force = dAta.data.data.value3 * dAta.data.data.value1;
    } else if (rr == 1) {
      force = dAta.data.data.value4 * dAta.data.data.value1;
    } else if (rr == 2) {
      force = dAta.data.data.value5 * dAta.data.data.value1;
    } else {
      SETERRQ(PETSC_COMM_SELF, MOFEM_DATA_INCONSISTENCY, "data inconsistency");
    }

    // Integrate force on the line
    for (int gg = 0; gg < nb_gauss_pts; gg++) {

      if (!rr) {
        wEights[gg] = 0;
        // This is if edge is curved, i.e. HO geometry
        for (int dd = 0; dd < 3; dd++) {
          wEights[gg] += pow(getTangentAtGaussPts()(gg, dd), 2);
        }
        wEights[gg] = std::sqrt(wEights[gg]);
        wEights[gg] *= getGaussPts()(1, gg);
      }

      cblas_daxpy(nb_dofs / rank, wEights[gg] * force, &data.getN()(gg, 0), 1,
                  &Nf[rr], rank);
    }
  }

  // I time/step varying force or calculate in arc-length control. This hack
  // scale force appropriately, and is controlled for user
  CHKERR MethodForForceScaling::applyScale(getFEMethod(), methodsOp, Nf);

  // Assemble force into right-hand vector
  Vec myF = F;
  if (useSnesF || F == PETSC_NULL) 
    myF = getKSPf();

  CHKERR VecSetValues(myF, data.getIndices().size(), &data.getIndices()[0],
                      &Nf[0], ADD_VALUES);

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode EdgeForce::addForce(const std::string field_name, Vec F,
                                   int ms_id, bool use_snes_f) {
  const CubitMeshSets *cubit_meshset_ptr;
  MeshsetsManager *mmanager_ptr;
  MoFEMFunctionBegin;
  CHKERR mField.getInterface(mmanager_ptr);
  CHKERR mmanager_ptr->getCubitMeshsetPtr(ms_id, NODESET, &cubit_meshset_ptr);
  CHKERR cubit_meshset_ptr->getBcDataStructure(mapForce[ms_id].data);
  CHKERR mField.get_moab().get_entities_by_type(
      cubit_meshset_ptr->meshset, MBEDGE, mapForce[ms_id].eDges, true);
  // Add operator for element, set data and entities operating on the data
  fe.getOpPtrVector().push_back(
      new OpEdgeForce(field_name, F, mapForce[ms_id], methodsOp, use_snes_f));
  MoFEMFunctionReturn(0);
}
