/**
 * \file basic_approx.cpp
 * \example basic_approx.cpp
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

using VolEle = FaceElementForcesAndSourcesCoreBase;
using VolEleOp = VolEle::UserDataOperator;
using FaceEle = EdgeElementForcesAndSourcesCoreBase;
using FaceEleOp = FaceEle::UserDataOperator;
using EntData = DataForcesAndSourcesCore::EntData;

#include <BaseOps.hpp>

using OpVolGradGrad = OpTools<VolEleOp>::OpGradGrad<2>;
using OpVolGradGradResidual = OpTools<VolEleOp>::OpGradGradResidual<2>;
using OpFaceBase = OpTools<FaceEleOp>::OpBase;

constexpr double emissivity = 1;
constexpr double boltzmann_constant = 5.670367e-2;
constexpr double Beta = emissivity * boltzmann_constant;

constexpr double T_ambient = 3;
constexpr double Flux = -1.0e6;

struct Example {

  Example(MoFEM::Interface &m_field) : mField(m_field) {}

  MoFEMErrorCode runProblem();

private:
  MoFEM::Interface &mField;

  static int integrationRule(int, int, int p_data) { return 2 * p_data; };

  MoFEMErrorCode setUP();
  MoFEMErrorCode createCommonData();
  MoFEMErrorCode bC();
  MoFEMErrorCode OPs();
  MoFEMErrorCode kspSolve();
  MoFEMErrorCode postProcess();
  MoFEMErrorCode checkResults();

  MatrixDouble invJac;
  boost::shared_ptr<VectorDouble> approxVals;
  boost::shared_ptr<MatrixDouble> approxGradVals;

  struct OpRadiationLhs : public OpFaceBase {

  private:
    boost::shared_ptr<VectorDouble> approxVals;
    FTensor::Index<'i', 2> i; ///< summit Index

  public:
    OpRadiationLhs(boost::shared_ptr<VectorDouble> &approx_vals)
        : OpFaceBase("U", OpFaceBase::OPROWCOL), approxVals(approx_vals) {}

    MoFEMErrorCode iNtegrate(EntData &row_data, EntData &col_data);
  };

  struct OpRadiationRhs : public OpFaceBase {

  private:
    boost::shared_ptr<VectorDouble> approxVals;
    FTensor::Index<'i', 2> i; ///< summit Index

  public:
    OpRadiationRhs(boost::shared_ptr<VectorDouble> &approx_vals)
        : OpFaceBase("U", OpFaceBase::OPROW), approxVals(approx_vals) {}

    MoFEMErrorCode iNtegrate(EntData &row_data);
  };

  struct OpFluxRhs : public OpFaceBase {

  private:
    FTensor::Index<'i', 2> i; ///< summit Index

  public:
    OpFluxRhs() : OpFaceBase("U", OpFaceBase::OPROW) {}

    MoFEMErrorCode iNtegrate(EntData &row_data);
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
  constexpr int order = 3;
  CHKERR simple->setFieldOrder("U", order);
  CHKERR simple->setUp();
  MoFEMFunctionReturn(0);
}
//! [Set up problem]

//! [Create common data]
MoFEMErrorCode Example::createCommonData() {
  MoFEMFunctionBegin;
  approxVals = boost::make_shared<VectorDouble>();
  approxGradVals = boost::make_shared<MatrixDouble>();
  MoFEMFunctionReturn(0);
}
//! [Create common data]

//! [Boundary condition]
MoFEMErrorCode Example::bC() {
  MoFEMFunctionBegin;
  // Set initial values
  auto set_initial_temperature = [&](VectorAdaptor &&field_data, double *xcoord,
                                     double *ycoord, double *zcoord) {
    MoFEMFunctionBeginHot;
    field_data[0] = T_ambient;
    MoFEMFunctionReturnHot(0);
  };
  FieldBlas *field_blas;
  CHKERR mField.getInterface(field_blas);
  CHKERR field_blas->setVertexDofs(set_initial_temperature, "U");
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
  auto beta = [](const double r, const double, const double) {
    return 2e3 * 2 * M_PI * r;
  };
  basic->getOpDomainLhsPipeline().push_back(new OpVolGradGrad("U", beta));
  CHKERR basic->setDomainLhsIntegrationRule(integrationRule);

  basic->getOpDomainRhsPipeline().push_back(
      new OpCalculateInvJacForFace(invJac));
  basic->getOpDomainRhsPipeline().push_back(new OpSetInvJacH1ForFace(invJac));
  basic->getOpDomainRhsPipeline().push_back(
      new OpCalculateScalarFieldGradient<2>("U", approxGradVals));
  basic->getOpDomainRhsPipeline().push_back(
      new OpVolGradGradResidual("U", beta, approxGradVals));
  CHKERR basic->setDomainRhsIntegrationRule(integrationRule);

  basic->getOpBoundaryRhsPipeline().push_back(
      new OpCalculateScalarFieldValues("U", approxVals));
  basic->getOpBoundaryRhsPipeline().push_back(new OpRadiationRhs(approxVals));
  basic->getOpBoundaryRhsPipeline().push_back(new OpFluxRhs());
  CHKERR basic->setBoundaryRhsIntegrationRule(integrationRule);

  basic->getOpBoundaryLhsPipeline().push_back(
      new OpCalculateScalarFieldValues("U", approxVals));
  basic->getOpBoundaryLhsPipeline().push_back(new OpRadiationLhs(approxVals));
  CHKERR basic->setBoundaryLhsIntegrationRule(integrationRule);
  MoFEMFunctionReturn(0);
}
//! [Push operators to pipeline]

//! [Solve]
MoFEMErrorCode Example::kspSolve() {
  MoFEMFunctionBegin;
  Simple *simple = mField.getInterface<Simple>();
  Basic *basic = mField.getInterface<Basic>();
  auto ts = basic->createTS();

  double ftime = 1;
  CHKERR TSSetDuration(ts, PETSC_DEFAULT, ftime);
  CHKERR TSSetFromOptions(ts);
  CHKERR TSSetExactFinalTime(ts, TS_EXACTFINALTIME_MATCHSTEP);

  auto T = smartCreateDMDVector(simple->getDM());
  CHKERR DMoFEMMeshToLocalVector(simple->getDM(), T, INSERT_VALUES,
                                 SCATTER_FORWARD);

  CHKERR TSSolve(ts, T);
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
//! [Solve]

//! [Postprocess results]
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
  CHKERR post_proc_fe->writeFile("out_radiation.h5m");
  MoFEMFunctionReturn(0);
}
//! [Postprocess results]

//! [Check results]
MoFEMErrorCode Example::checkResults() { return 0; }
//! [Check results]

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

//! [Radiation Lhs]
MoFEMErrorCode Example::OpRadiationLhs::iNtegrate(EntData &row_data,
                                                  EntData &col_data) {
  MoFEMFunctionBegin;
  // get element volume
  const double vol = OpFaceBase::getMeasure();
  // get integration weights
  auto t_w = OpFaceBase::getFTensor0IntegrationWeight();
  // get base function gradient on rows
  auto t_row_base = row_data.getFTensor0N();
  // gat temperature at integration points
  auto t_val = getFTensor0FromVec(*(approxVals));
  // get coordinate at integration points
  auto t_coords = OpFaceBase::getFTensor1CoordsAtGaussPts();

  // loop over integration points
  for (int gg = 0; gg != OpFaceBase::nbIntegrationPts; gg++) {

    // Cylinder radius
    const double r_cylinder = t_coords(0);

    // take into account Jacobean
    const double alpha = t_w * vol * Beta * (2 * M_PI * r_cylinder);
    // loop over rows base functions
    for (int rr = 0; rr != OpFaceBase::nbRows; ++rr) {
      auto t_col_base = col_data.getFTensor0N(gg, 0);
      // loop over columns
      for (int cc = 0; cc != OpFaceBase::nbCols; cc++) {
        if (std::abs(t_coords(0)) > std::numeric_limits<double>::epsilon()) {
          locMat(rr, cc) += alpha * t_row_base * t_col_base * 4 * pow(t_val, 3);
        }
        ++t_col_base;
      }
      ++t_row_base;
    }

    ++t_val;
    ++t_coords;
    ++t_w; // move to another integration weight
  }
  MoFEMFunctionReturn(0);
}
//! [Radiation Lhs]

//! [Radiation Lhs]
MoFEMErrorCode Example::OpRadiationRhs::iNtegrate(EntData &row_data) {
  MoFEMFunctionBegin;
  // get element volume
  const double vol = OpFaceBase::getMeasure();
  // get integration weights
  auto t_w = OpFaceBase::getFTensor0IntegrationWeight();
  // get base function gradient on rows
  auto t_row_base = row_data.getFTensor0N();
  // gat temperature at integration points
  auto t_val = getFTensor0FromVec(*(approxVals));
  // get coordinate at integration points
  auto t_coords = OpFaceBase::getFTensor1CoordsAtGaussPts();

  // loop over integration points
  for (int gg = 0; gg != OpFaceBase::nbIntegrationPts; gg++) {

    // Cylinder radius
    const double r_cylinder = t_coords(0);

    // take into account Jacobean
    const double alpha = t_w * vol * Beta * (2 * M_PI * r_cylinder);
    // loop over rows base functions
    for (int rr = 0; rr != OpFaceBase::nbRows; ++rr) {
      if (std::abs(t_coords(0)) > std::numeric_limits<double>::epsilon()) {
        OpFaceBase::locF[rr] +=
            alpha * t_row_base * (pow(t_val, 4) - pow(T_ambient, 4));
      }
      ++t_row_base;
    }
    ++t_coords;
    ++t_val;
    ++t_w; // move to another integration weight
  }

  MoFEMFunctionReturn(0);
}
//! [Radiation Lhs]

//! [Flux Rhs]
MoFEMErrorCode Example::OpFluxRhs::iNtegrate(EntData &row_data) {
  MoFEMFunctionBegin;
  // get element volume
  const double vol = OpFaceBase::getMeasure();
  // get integration weights
  auto t_w = OpFaceBase::getFTensor0IntegrationWeight();
  // get base function gradient on rows
  auto t_row_base = row_data.getFTensor0N();
  // get coordinate at integration points
  auto t_coords = OpFaceBase::getFTensor1CoordsAtGaussPts();
  // get time
  const double time = OpFaceBase::getFEMethod()->ts_t;

  // loop over integration points
  for (int gg = 0; gg != OpFaceBase::nbIntegrationPts; gg++) {

    // Cylinder radius
    const double r_cylinder = t_coords(0);

    const double r = sqrt(t_coords(i) * t_coords(i));
    const double c = std::abs(t_coords(0)) / r;

    // take into account Jacobean
    const double alpha = t_w * vol * (2 * M_PI * r_cylinder);
    // loop over rows base functions
    for (int rr = 0; rr != OpFaceBase::nbRows; ++rr) {
      OpFaceBase::locF[rr] += alpha * t_row_base * c * Flux * time;
      ++t_row_base;
    }
    ++t_coords;
    ++t_w; // move to another integration weight
  }

  MoFEMFunctionReturn(0);
}
//! [Flux Rhs]
