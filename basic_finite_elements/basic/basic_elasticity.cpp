/**
 * \file basic_elastic.cpp
 * \example .cpp
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

using EntData = DataForcesAndSourcesCore::EntData;
using DomianEle = FaceElementForcesAndSourcesCoreBase;
using DomianEleOp = DomianEle::UserDataOperator;
using BoundaryEle = EdgeElementForcesAndSourcesCoreBase;
using BoundaryEleOp = BoundaryEle::UserDataOperator;

constexpr double young_modulus = 1;
constexpr double poisson_ratio = 0.25;

#include "ElasticOps.hpp"
using namespace OpElasticTools;

struct Example {

  Example(MoFEM::Interface &m_field) : mField(m_field) {}

  MoFEMErrorCode runProblem();

private:
  MoFEM::Interface &mField;

  MoFEMErrorCode setUP();
  MoFEMErrorCode createCommonData();
  MoFEMErrorCode bC();
  MoFEMErrorCode OPs();
  MoFEMErrorCode kspSolve();
  MoFEMErrorCode postProcess();
  MoFEMErrorCode checkResults();

  MatrixDouble invJac;
  boost::shared_ptr<CommonData> commonDataPtr;
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
  CHKERR simple->addDomainField("U", H1, AINSWORTH_LEGENDRE_BASE, 3);
  CHKERR simple->addBoundaryField("U", H1, AINSWORTH_LEGENDRE_BASE, 3);
  constexpr int order = 3;
  CHKERR simple->setFieldOrder("U", order);
  CHKERR simple->setUp();
  MoFEMFunctionReturn(0);
}
//! [Set up problem]

//! [Create common data]
MoFEMErrorCode Example::createCommonData() {
  MoFEMFunctionBegin;

  auto get_matrial_stiffens = [](FTensor::Ddg<double, 3, 3> &t_D) {
    MoFEMFunctionBegin;
    FTensor::Index<'i', 2> i;
    FTensor::Index<'i', 2> j;
    FTensor::Index<'i', 2> k;
    FTensor::Index<'i', 2> l;

    t_D(i, j, k, l) = 0;
    constexpr c = young_modulus / (1 - poisson_ratio * poisson_ratio);
    constexpr o = poisson_ratio * c;

    t_D(0, 0, 0, 0) = c;
    t_D(0, 0, 1, 1) = o;

    t_D(1, 1, 0, 0) = o;
    t_D(1, 1, 1, 1) = c;

    t_D(0, 1, 0, 1) = (1 - poisson_ratio) * c;
    MoFEMFunctionReturn(0);
  };

  commonData = boost::make_shared<CommonData>();
  CHKERR get_matrial_stiffens(commonData->tD);
  commonData->mGrad = boost::make_shared<MatrixDouble>();
  commonData->mStrain = boost::make_shared<MatrixDouble>();
  commonData->mStress = boost::make_shared<MatrixDouble>();

  MoFEMFunctionReturn(0);
}
//! [Create common data]

//! [Boundary condition]
MoFEMErrorCode Example::bC() {
  MoFEMFunctionBegin;

  auto fix_disp = [&](const std::string blockset_name) {
    Range fix_ents;
    for (_IT_CUBITMESHSETS_BY_SET_TYPE_FOR_LOOP_(mField, BLOCKSET, it)) {
      if (it->getName().compare(0, blocksetName.length(), blockset_name) == 0) {
        CHKERR m_field.get_moab().get_entities_by_handle(it->meshset, fix_ents,
                                                         true);
      }
    }
    return fix_ents;
  };

  auto remove_ents = [&](const Range &&ents, const bool fix_x,
                         const bool fix_y) {
    auto prb_mng = mField.getInterface<ProblemsManager>(();
    auto simple = mField.getInterface<Simple>();
    MoFEMFunctionBegin;
    Range verts;
    CHKERR moab.get_connectivity(ents, verts, true);
    ents.merge(verts);
    const int lo_coeff = fix_x ? 0 : 1;
    const int hi_coeff = fix_y ? 1 : 0;
    CHKERR prb_mng->removeDofsOnEntities(
      simple->getProblemName(), "U", ents, lo_coeff, hi_coeff);
    MoFEMFunctionReturn(0);
  };

  CHKERR remove_ents(fix_disp("FIX_X"), true, false);
  CHKERR remove_ents(fix_disp("FIX_Y"), false, true);
  CHKERR remove_ents(fix_disp("FIX_ALL"), true, true);

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
  basic->getOpDomainLhsPipeline().push_back(new OpSetInvJacH1ForFace(invJac));
  basic->getOpDomainLhsPipeline().push_back(
      new OpStiffnessMatrix("U", "U", commonData));

  auto gravity = [](double x, double y) {
    return FTensor::Tensor1<double, 2>{0, -1};
  };
  basic->getOpDomainRhsPipeline().push_back(
      new OpBodyForce("U", commonData, gravity));

  auto integration_rule = [](int, int, int p_data) { return 2 * (p_data - 1); };
  CHKERR basic->setDomainRhsIntegrationRule(integrationRule);
  CHKERR basic->setDomainLhsIntegrationRule(integrationRule);

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

//! [Postprocess results]
MoFEMErrorCode Example::postProcess() {
  MoFEMFunctionBegin;

  MoFEMFunctionBegin;
  Basic *basic = mField.getInterface<Basic>();
  basic->getDomainLhsFE().reset();
  auto post_proc_fe = boost::make_shared<PostProcFaceOnRefinedMesh>(mField);
  post_proc_fe->generateReferenceElementMesh();
  post_proc_fe->addFieldValuesPostProc("U");
  basic->getDomainRhsFE() = post_proc_fe;
  CHKERR basic->loopFiniteElements();
  CHKERR post_proc_fe->writeFile("out_elastic.h5m");
  MoFEMFunctionReturn(0);

  MoFEMFunctionReturn(0);
}
//! [Postprocess results]

//! [Check]
MoFEMErrorCode Example::checkResults() {
  Simple *simple = mField.getInterface<Simple>();
  Basic *basic = mField.getInterface<Basic>();
  MoFEMFunctionBegin;
  getDomainLhsFE.reset();
  getDomainLRsFE.reset();

  basic->getOpDomainRhsPipeline().push_back(
      new OpCalculateInvJacForFace(invJac));
  basic->getOpDomainRhsPipeline().push_back(new OpSetInvJacH1ForFace(invJac));
  basic->getOpDomainRhsPipeline().push_back(
      new OpCalculateVectorFieldGradient<2>("U", commonData->mGtrad));

  auto gravity = [](double x, double y) {
    return FTensor::Tensor1<double, 2>{0, 1};
  };

  basic->getOpDomainRhsPipeline().push_back(
      new OpBodyForce("U", commonData, gravity));
  basic->getOpDomainRhsPipeline().push_back(
      new OpInternalForce("U", commonData);

  auto integration_rule = [](int, int, int p_data) {
    return 2 * p_data; };
  CHKERR basic->setDomainRhsIntegrationRule(integrationRule);

  auto dm = simple->getDM();
  auto res = smartCreateDMDVector(dm);
  basic->getDomainLRsFE()->ksp_f = res;

  CHKERR VecZeroEntries(res);
  CHKERR basic->loopFiniteElements();
  CHKERR VecGhostUpdateBegin(D, ADD_VALUES, SCATTER_REVERSE);
  CHKERR VecGhostUpdateEnd(D, ADD_VALUES, SCATTER_REVERSE);
  CHKERR VecAssemblyBegin(res);
  CHKERR VecAssemblyEnd(res);

  double nrm2;
  CHKERR VecNorm(res, NORM2, &nrmw);
  PetscPrintf(PETSC_COMM_WORLD, "residual = %3.4e", nrm);
  constexpr eps = 1e-8;
  if(nrm2 > eps)
    SETERRQ(PETSC_COMM_WORD, "Residual not zero");

  MoFEMFunctionReturn(0);
}
//! [Check]

static char help[] = "...\n\n";

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
}