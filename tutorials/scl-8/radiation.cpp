/**
 * \file lesson6_radiation.cpp
 * \example lesson6_radiation.cpp
 *
 * Using PipelineManager interface calculate the divergence of base functions,
 * and integral of flux on the boundary. Since the h-div space is used, volume
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

using DomainEle = PipelineManager::FaceEle2D;
using DomainEleOp = DomainEle::UserDataOperator;
using EdgeEle = PipelineManager::EdgeEle2D;
using EdgeEleOp = EdgeEle::UserDataOperator;
using EntData = DataForcesAndSourcesCore::EntData;

using OpDomainGradGrad = FormsIntegrators<DomainEleOp>::Assembly<
    PETSC>::BiLinearForm<GAUSS>::OpGradGrad<1, 1, 2>;
using OpDomainGradTimesVec = FormsIntegrators<DomainEleOp>::Assembly<
    PETSC>::LinearForm<GAUSS>::OpGradTimesTensor<1, 1, 2>;
using OpBase = OpBaseImpl<PETSC, EdgeEleOp>;

// Units 
// Temperature: Kelvins
// Length: 1 km
// Time: 1 s

constexpr double heat_conductivity = ((0.4+0.7)/2) * 1e3;

constexpr double emissivity = 1;
constexpr double boltzmann_constant = 5.670374419e-2;
constexpr double Beta = emissivity * boltzmann_constant;

constexpr double T_ambient = 3;
constexpr double Flux = -0.873e6;

struct Example {

  Example(MoFEM::Interface &m_field) : mField(m_field) {}

  MoFEMErrorCode runProblem();

private:
  MoFEM::Interface &mField;

  static int integrationRule(int, int, int p_data) { return 2 * p_data; };

  MoFEMErrorCode setupProblem();
  MoFEMErrorCode createCommonData();
  MoFEMErrorCode bC();
  MoFEMErrorCode OPs();
  MoFEMErrorCode kspSolve();
  MoFEMErrorCode postProcess();
  MoFEMErrorCode checkResults();

  MatrixDouble invJac;
  boost::shared_ptr<VectorDouble> approxVals;
  boost::shared_ptr<MatrixDouble> approxGradVals;

  struct OpRadiationLhs : public OpBase {

  private:
    boost::shared_ptr<VectorDouble> approxVals;

  public:
    OpRadiationLhs(boost::shared_ptr<VectorDouble> &approx_vals)
        : OpBase("U", "U", OpBase::OPROWCOL), approxVals(approx_vals) {
      this->sYmm = false;
    }

    MoFEMErrorCode iNtegrate(EntData &row_data, EntData &col_data);
  };

  struct OpRadiationRhs : public OpBase {

  private:
    boost::shared_ptr<VectorDouble> approxVals;
    FTensor::Index<'i', 2> i; ///< summit Index

  public:
    OpRadiationRhs(boost::shared_ptr<VectorDouble> &approx_vals)
        : OpBase("U", "U", OpBase::OPROW), approxVals(approx_vals) {}

    MoFEMErrorCode iNtegrate(EntData &row_data);
  };

  struct OpFluxRhs : public OpBase {

  private:
    FTensor::Index<'i', 2> i; ///< summit Index

  public:
    OpFluxRhs() : OpBase("U", "U", OpBase::OPROW) {}

    MoFEMErrorCode iNtegrate(EntData &row_data);
  };
};

MoFEMErrorCode Example::runProblem() {
  MoFEMFunctionBegin;
  CHKERR setupProblem();
  CHKERR createCommonData();
  CHKERR bC();
  CHKERR OPs();
  CHKERR kspSolve();
  CHKERR postProcess();
  CHKERR checkResults();
  MoFEMFunctionReturn(0);
}

//! [Set up problem]
MoFEMErrorCode Example::setupProblem() {
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
  PipelineManager *pipeline_mng = mField.getInterface<PipelineManager>();
  auto beta = [](const double r, const double, const double) {
    return heat_conductivity * (2 * M_PI * r);
  };

  pipeline_mng->getOpDomainLhsPipeline().push_back(
      new OpCalculateInvJacForFace(invJac));
  pipeline_mng->getOpDomainLhsPipeline().push_back(
      new OpSetInvJacH1ForFace(invJac));
  pipeline_mng->getOpDomainLhsPipeline().push_back(
      new OpDomainGradGrad("U", "U", beta));
  CHKERR pipeline_mng->setDomainLhsIntegrationRule(integrationRule);

  pipeline_mng->getOpDomainRhsPipeline().push_back(
      new OpCalculateInvJacForFace(invJac));
  pipeline_mng->getOpDomainRhsPipeline().push_back(
      new OpSetInvJacH1ForFace(invJac));
  pipeline_mng->getOpDomainRhsPipeline().push_back(
      new OpCalculateScalarFieldGradient<2>("U", approxGradVals));
  pipeline_mng->getOpDomainRhsPipeline().push_back(
      new OpDomainGradTimesVec("U", approxGradVals, beta));
  CHKERR pipeline_mng->setDomainRhsIntegrationRule(integrationRule);

  pipeline_mng->getOpBoundaryRhsPipeline().push_back(
      new OpCalculateScalarFieldValues("U", approxVals));
  pipeline_mng->getOpBoundaryRhsPipeline().push_back(
      new OpRadiationRhs(approxVals));
  pipeline_mng->getOpBoundaryRhsPipeline().push_back(new OpFluxRhs());
  CHKERR pipeline_mng->setBoundaryRhsIntegrationRule(integrationRule);

  pipeline_mng->getOpBoundaryLhsPipeline().push_back(
      new OpCalculateScalarFieldValues("U", approxVals));
  pipeline_mng->getOpBoundaryLhsPipeline().push_back(
      new OpRadiationLhs(approxVals));
  CHKERR pipeline_mng->setBoundaryLhsIntegrationRule(integrationRule);
  MoFEMFunctionReturn(0);
}
//! [Push operators to pipeline]

//! [Solve]
MoFEMErrorCode Example::kspSolve() {
  MoFEMFunctionBegin;
  Simple *simple = mField.getInterface<Simple>();
  PipelineManager *pipeline_mng = mField.getInterface<PipelineManager>();
  auto ts = pipeline_mng->createTS();

  double ftime = 1;
  CHKERR TSSetDuration(ts, PETSC_DEFAULT, ftime);
  CHKERR TSSetFromOptions(ts);
  CHKERR TSSetExactFinalTime(ts, TS_EXACTFINALTIME_MATCHSTEP);

  auto T = smartCreateDMVector(simple->getDM());
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
  MOFEM_LOG_C("EXAMPLE", Sev::inform,
              "steps %d (%d rejected, %d SNES fails), ftime %g, nonlinits "
              "%d, linits %d\n",
              steps, rejects, snesfails, ftime, nonlinits, linits);

  MoFEMFunctionReturn(0);
}
//! [Solve]

//! [Postprocess results]
MoFEMErrorCode Example::postProcess() {
  MoFEMFunctionBegin;
  PipelineManager *pipeline_mng = mField.getInterface<PipelineManager>();
  pipeline_mng->getDomainLhsFE().reset();
  pipeline_mng->getBoundaryLhsFE().reset();
  pipeline_mng->getBoundaryRhsFE().reset();
  auto post_proc_fe = boost::make_shared<PostProcFaceOnRefinedMesh>(mField);
  post_proc_fe->generateReferenceElementMesh();
  post_proc_fe->addFieldValuesPostProc("U");
  pipeline_mng->getDomainRhsFE() = post_proc_fe;
  CHKERR pipeline_mng->loopFiniteElements();
  CHKERR post_proc_fe->writeFile("out_radiation.h5m");
  MoFEMFunctionReturn(0);
}
//! [Postprocess results]

//! [Check results]
MoFEMErrorCode Example::checkResults() { return 0; }
//! [Check results]

int main(int argc, char *argv[]) {

  // Initialisation of MoFEM/PETSc and MOAB data structures
  const char param_file[] = "param_file.petsc";
  MoFEM::Core::Initialize(&argc, &argv, param_file, help);

  // Add logging channel for example
  auto core_log = logging::core::get();
  core_log->add_sink(
      LogManager::createSink(LogManager::getStrmWorld(), "EXAMPLE"));
  LogManager::setLog("EXAMPLE");
  MOFEM_LOG_TAG("EXAMPLE", "example");

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
  const double vol = this->getMeasure();
  // get integration weights
  auto t_w = getFTensor0IntegrationWeight();
  // get base function gradient on rows
  auto t_row_base = row_data.getFTensor0N();
  // gat temperature at integration points
  auto t_val = getFTensor0FromVec(*(approxVals));
  // get coordinate at integration points
  auto t_coords = getFTensor1CoordsAtGaussPts();

  // loop over integration points
  for (int gg = 0; gg != nbIntegrationPts; gg++) {

    // Cylinder radius
    const double r_cylinder = t_coords(0);

    // take into account Jacobean
    const double alpha = t_w * vol * Beta * (2 * M_PI * r_cylinder);
    // loop over rows base functions
    for (int rr = 0; rr != nbRows; ++rr) {
      auto t_col_base = col_data.getFTensor0N(gg, 0);
      // loop over columns
      for (int cc = 0; cc != nbCols; cc++) {
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
  const double vol = getMeasure();
  // get integration weights
  auto t_w = getFTensor0IntegrationWeight();
  // get base function gradient on rows
  auto t_row_base = row_data.getFTensor0N();
  // gat temperature at integration points
  auto t_val = getFTensor0FromVec(*(approxVals));
  // get coordinate at integration points
  auto t_coords = getFTensor1CoordsAtGaussPts();

  // loop over integration points
  for (int gg = 0; gg != nbIntegrationPts; gg++) {

    // Cylinder radius
    const double r_cylinder = t_coords(0);

    // take into account Jacobean
    const double alpha = t_w * vol * Beta * (2 * M_PI * r_cylinder);
    // loop over rows base functions
    for (int rr = 0; rr != nbRows; ++rr) {
      if (std::abs(t_coords(0)) > std::numeric_limits<double>::epsilon()) {
        locF[rr] += alpha * t_row_base * (pow(t_val, 4) - pow(T_ambient, 4));
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
  const double vol = getMeasure();
  // get integration weights
  auto t_w = getFTensor0IntegrationWeight();
  // get base function gradient on rows
  auto t_row_base = row_data.getFTensor0N();
  // get coordinate at integration points
  auto t_coords = getFTensor1CoordsAtGaussPts();
  // // get time
  const double time = getFEMethod()->ts_t;

  // loop over integration points
  for (int gg = 0; gg != nbIntegrationPts; gg++) {

    // Cylinder radius
    const double r_cylinder = t_coords(0);

    const double r = sqrt(t_coords(i) * t_coords(i));
    const double c = std::abs(t_coords(0)) / r;

    // take into account Jacobean
    const double alpha = t_w * vol * (2 * M_PI * r_cylinder);
    // loop over rows base functions
    for (int rr = 0; rr != nbRows; ++rr) {
      locF[rr] += alpha * t_row_base * c * Flux * time;
      ++t_row_base;
    }
    ++t_coords;
    ++t_w; // move to another integration weight
  }

  MoFEMFunctionReturn(0);
}
//! [Flux Rhs]
