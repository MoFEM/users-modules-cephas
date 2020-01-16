/**
 * \file basic_poisson.cpp
 * \example basic_poisson.cpp
 *
 * Using Basic interface calculate the divergence of base functions, and
 * integral of flux on the boundary. Since the h-div space is used, volume
 * integral and boundary integral should give the same result.
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

#include <MoFEM.hpp>

using namespace MoFEM;

static char help[] = "...\n\n";

#include <BasicFiniteElements.hpp>

using EntData = DataForcesAndSourcesCore::EntData;
using DomainEle = FaceElementForcesAndSourcesCoreBase;
using DomainEleOp = DomainEle::UserDataOperator;
using BoundaryEle = EdgeElementForcesAndSourcesCoreBase;
using BoundaryEleOp = BoundaryEle::UserDataOperator;

#include <BaseOps.hpp>

using OpDomainGradGrad = OpTools<DomainEleOp>::OpGradGrad<2>;
using OpDomainSource = OpTools<DomainEleOp>::OpSource<2>;
using OpBoundaryMass = OpTools<BoundaryEleOp>::OpMass;
using OpBoundarySource = OpTools<BoundaryEleOp>::OpSource<2>;

struct Example {

  Example(MoFEM::Interface &m_field) : mField(m_field) {}

  MoFEMErrorCode runProblem();

private:
  MoFEM::Interface &mField;

  static double approxFunction(const double x, const double y, const double z) {
    return sin(x * 10.) * cos(y * 10.);
  }
  static double nablaFunction(const double x, const double y, const double z) {
    return 200 * sin(x * 10.) * cos(y * 10.);
  }

  static int integrationRule(int, int, int p_data) { return 2 * p_data; };

  MoFEMErrorCode setUP();
  MoFEMErrorCode createCommonData();
  MoFEMErrorCode bC();
  MoFEMErrorCode OPs();
  MoFEMErrorCode kspSolve();
  MoFEMErrorCode postProcess();
  MoFEMErrorCode checkResults();

  MatrixDouble invJac;
  struct CommonData {
    boost::shared_ptr<VectorDouble> approxVals;
    SmartPetscObj<Vec> L2Vec;
    SmartPetscObj<Vec> resVec;
  };
  boost::shared_ptr<CommonData> commonDataPtr;
  boost::shared_ptr<std::vector<bool>> boundaryMarker;

  struct OpError : public DomainEleOp {
    boost::shared_ptr<CommonData> commonDataPtr;
    OpError(boost::shared_ptr<CommonData> &common_data_ptr)
        : DomainEleOp("U", OPROW), commonDataPtr(common_data_ptr) {}
    MoFEMErrorCode doWork(int side, EntityType type, EntData &data);
  };
};

MoFEMErrorCode Example::runProblem() {
  MoFEMFunctionBegin;
  CHKERR setUP();
  CHKERR createCommonData();
  CHKERR bC();
  CHKERR OPs();
  CHKERR kspSolve();
  CHKERR postProcess();
  CHKERR checkResults();
  MoFEMFunctionReturn(0);
}

//! [Set up problem]
MoFEMErrorCode Example::setUP() {
  MoFEMFunctionBegin;
  Simple *simple = mField.getInterface<Simple>();
  // Add field
  CHKERR simple->addDomainField("U", H1, AINSWORTH_LEGENDRE_BASE, 1);
  CHKERR simple->addBoundaryField("U", H1, AINSWORTH_LEGENDRE_BASE, 1);
  constexpr int order = 5;
  CHKERR simple->setFieldOrder("U", order);
  CHKERR simple->setUp();
  MoFEMFunctionReturn(0);
}
//! [Set up problem]

//! [Create common data]
MoFEMErrorCode Example::createCommonData() {
  MoFEMFunctionBegin;
  Simple *simple = mField.getInterface<Simple>();
  commonDataPtr = boost::make_shared<CommonData>();
  commonDataPtr->resVec = smartCreateDMDVector(simple->getDM());
  commonDataPtr->L2Vec = createSmartVectorMPI(
      mField.get_comm(), (!mField.get_comm_rank()) ? 1 : 0, 1);
  commonDataPtr->approxVals = boost::make_shared<VectorDouble>();
  MoFEMFunctionReturn(0);
}
//! [Create common data]

//! [Boundary condition]
MoFEMErrorCode Example::bC() {
  MoFEMFunctionBegin;

  Simple *simple = mField.getInterface<Simple>();

  auto get_ents_on_mesh_skin = [&]() {
    Range faces;
    CHKERR mField.get_moab().get_entities_by_type(0, MBTRI, faces);
    Skinner skin(&mField.get_moab());
    Range skin_edges;
    CHKERR skin.find_skin(0, faces, false, skin_edges);
    Range skin_verts;
    CHKERR mField.get_moab().get_connectivity(skin_edges, skin_verts, true);
    skin_edges.merge(skin_verts);
    return skin_edges;
  };

  auto mark_boundary_dofs = [&](Range &&skin_edges) {
    auto problem_manager = mField.getInterface<ProblemsManager>();
    auto marker_ptr = boost::make_shared<std::vector<bool>>();
    problem_manager->markDofs(simple->getProblemName(), ROW, skin_edges,
                              *marker_ptr);
    return marker_ptr;
  };

  // Get global local vector of marked DOFs. Is global, since is set for all
  // DOFs on processor. Is local since only DOFs on processor are in the
  // vector. To access DOFs use local indices.
  boundaryMarker = mark_boundary_dofs(get_ents_on_mesh_skin());

  MoFEMFunctionReturn(0);
}
//! [Boundary condition]

//! [Push operators to pipeline]
MoFEMErrorCode Example::OPs() {
  MoFEMFunctionBegin;
  Basic *basic = mField.getInterface<Basic>();
  basic->getOpDomainLhsPipeline().push_back(
      new OpCalculateInvJacForFace(invJac));
  basic->getOpDomainLhsPipeline().push_back(new OpSetInvJacH1ForFace(invJac));
  auto beta = [](const double, const double, const double) { return 1; };
  basic->getOpDomainLhsPipeline().push_back(
      new OpDomainGradGrad("U", "U", beta, boundaryMarker));
  basic->getOpDomainRhsPipeline().push_back(
      new OpDomainSource("U", Example::nablaFunction, boundaryMarker));
  CHKERR basic->setDomainRhsIntegrationRule(integrationRule);
  CHKERR basic->setDomainLhsIntegrationRule(integrationRule);
  basic->getOpBoundaryLhsPipeline().push_back(new OpBoundaryMass("U", "U", beta));
  basic->getOpBoundaryRhsPipeline().push_back(
      new OpBoundarySource("U", approxFunction));
  CHKERR basic->setBoundaryRhsIntegrationRule(integrationRule);
  CHKERR basic->setBoundaryLhsIntegrationRule(integrationRule);
  MoFEMFunctionReturn(0);
}
//! [Push operators to pipeline]

//! [Solve]
MoFEMErrorCode Example::kspSolve() {
  MoFEMFunctionBegin;
  Simple *simple = mField.getInterface<Simple>();
  Basic *basic = mField.getInterface<Basic>();
  auto solver = basic->createKSP();
  CHKERR KSPSetFromOptions(solver);
  CHKERR KSPSetUp(solver);

  auto dm = simple->getDM();
  auto D = smartCreateDMDVector(dm);
  auto F = smartVectorDuplicate(D);

  CHKERR KSPSolve(solver, F, D);
  CHKERR VecGhostUpdateBegin(D, INSERT_VALUES, SCATTER_FORWARD);
  CHKERR VecGhostUpdateEnd(D, INSERT_VALUES, SCATTER_FORWARD);
  CHKERR DMoFEMMeshToLocalVector(dm, D, INSERT_VALUES, SCATTER_REVERSE);
  MoFEMFunctionReturn(0);
}

//! [Solve]
MoFEMErrorCode Example::postProcess() {
  MoFEMFunctionBegin;
  Basic *basic = mField.getInterface<Basic>();
  basic->getDomainLhsFE().reset();
  basic->getBoundaryLhsFE().reset();
  basic->getBoundaryRhsFE().reset();
  auto post_proc_fe = boost::make_shared<PostProcFaceOnRefinedMesh>(mField);
  post_proc_fe->generateReferenceElementMesh();
  post_proc_fe->addFieldValuesPostProc("U");
  basic->getDomainRhsFE() = post_proc_fe;
  CHKERR basic->loopFiniteElements();
  CHKERR post_proc_fe->writeFile("out_poisson.h5m");
  MoFEMFunctionReturn(0);
}
//! [Postprocess results]

//! [Solve]
MoFEMErrorCode Example::checkResults() {
  MoFEMFunctionBegin;
  Basic *basic = mField.getInterface<Basic>();
  basic->getDomainLhsFE().reset();
  basic->getDomainRhsFE().reset();
  basic->getBoundaryLhsFE().reset();
  basic->getBoundaryRhsFE().reset();
  basic->getOpDomainRhsPipeline().clear();
  basic->getOpDomainRhsPipeline().push_back(
      new OpCalculateScalarFieldValues("U", commonDataPtr->approxVals));
  basic->getOpDomainRhsPipeline().push_back(new OpError(commonDataPtr));
  CHKERR basic->setDomainRhsIntegrationRule(integrationRule);
  CHKERR basic->loopFiniteElements();
  CHKERR VecAssemblyBegin(commonDataPtr->L2Vec);
  CHKERR VecAssemblyEnd(commonDataPtr->L2Vec);
  CHKERR VecAssemblyBegin(commonDataPtr->resVec);
  CHKERR VecAssemblyEnd(commonDataPtr->resVec);
  double nrm2;
  CHKERR VecNorm(commonDataPtr->resVec, NORM_2, &nrm2);
  const double *array;
  CHKERR VecGetArrayRead(commonDataPtr->L2Vec, &array);
  if (mField.get_comm_rank() == 0)
    PetscPrintf(PETSC_COMM_SELF, "Error %6.4e Vec norm %6.4e\n", sqrt(array[0]),
                nrm2);
  CHKERR VecRestoreArrayRead(commonDataPtr->L2Vec, &array);
  constexpr double eps = 1e-8;
  if (nrm2 > eps)
    SETERRQ(PETSC_COMM_SELF, MOFEM_DATA_INCONSISTENCY,
            "Not converged solution");
  MoFEMFunctionReturn(0);
}
//! [Solver]

int main(int argc, char *argv[]) {

  MoFEM::Core::Initialize(&argc, &argv, (char *)0, help);

  try {

    //! [Register MoFEM discrete manager in PETSc]
    DMType dm_name = "DMMOFEM";
    CHKERR DMRegister_MoFEM(dm_name);
    //! [Register MoFEM discrete manager in PETSc

    //! [Create MoAB]
    moab::Core mb_instance;              ///< mesh database
    moab::Interface &moab = mb_instance; ///< mesh database interface
    //! [Create MoAB]

    //! [Create MoFEM]
    MoFEM::Core core(moab);           ///< finite element database
    MoFEM::Interface &m_field = core; ///< finite element database insterface
    //! [Create MoFEM]

    //! [Load mesh]
    Simple *simple = m_field.getInterface<Simple>();
    CHKERR simple->getOptions();
    CHKERR simple->loadFile("");
    //! [Load mesh]

    //! [Example]
    Example ex(m_field);
    CHKERR ex.runProblem();
    //! [Example]
  }
  CATCH_ERRORS;

  CHKERR MoFEM::Core::Finalize();
}

MoFEMErrorCode Example::OpError::doWork(int side, EntityType type,
                                        EntData &data) {
  MoFEMFunctionBegin;

  if (const size_t nb_dofs = data.getIndices().size()) {

    const int nb_integration_pts = getGaussPts().size2();
    auto t_w = getFTensor0IntegrationWeight();
    auto t_val = getFTensor0FromVec(*(commonDataPtr->approxVals));
    auto t_coords = getFTensor1CoordsAtGaussPts();

    VectorDouble nf(nb_dofs, false);
    nf.clear();

    FTensor::Index<'i', 3> i;
    const double volume = getMeasure();

    auto t_row_base = data.getFTensor0N();
    double error = 0;
    for (int gg = 0; gg != nb_integration_pts; ++gg) {

      const double alpha = t_w * volume;
      double diff = t_val - Example::approxFunction(t_coords(0), t_coords(1),
                                                    t_coords(2));
      error += alpha * pow(diff, 2);

      for (size_t r = 0; r != nb_dofs; ++r) {
        nf[r] += alpha * t_row_base * diff;
        ++t_row_base;
      }

      ++t_w;
      ++t_val;
      ++t_coords;
    }

    const int index = 0;
    CHKERR VecSetValue(commonDataPtr->L2Vec, index, error, ADD_VALUES);
    CHKERR VecSetValues(commonDataPtr->resVec, data, &nf[0], ADD_VALUES);
  }

  MoFEMFunctionReturn(0);
}