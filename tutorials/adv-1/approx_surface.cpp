/**
 * \file approx_surface.cpp
 * \example approx_surface.cpp
 *
 */

#include <MoFEM.hpp>

using namespace MoFEM;

static char help[] = "...\n\n";

#include <BasicFiniteElements.hpp>

constexpr int FM_DIM = 2;

template <int DIM> struct ElementsAndOps {};

template <> struct ElementsAndOps<2> {
  using DomainEle = MoFEM::FaceElementForcesAndSourcesCore;
  using BoundaryEle = MoFEM::EdgeElementForcesAndSourcesCore;
  using VolumeEle = MoFEM::VolumeElementForcesAndSourcesCore;
};

using DomainEle = ElementsAndOps<FM_DIM>::DomainEle;
using DomainEleOp = DomainEle::UserDataOperator;

using BoundaryEle = ElementsAndOps<FM_DIM>::BoundaryEle;
using BoundaryEleOp = BoundaryEle::UserDataOperator;

using EntData = EntitiesFieldData::EntData;

using VolumeEle = ElementsAndOps<FM_DIM>::VolumeEle;

using AssemblyDomainEleOp =
    FormsIntegrators<DomainEleOp>::Assembly<PETSC>::OpBase;

using AssemblyBoundaryEleOp =
    FormsIntegrators<BoundaryEleOp>::Assembly<PETSC>::OpBase;


constexpr double a = 47.65;
constexpr double a2 = a * a;
constexpr double a4 = a2 * a2;

constexpr double A = 1;

FTensor::Index<'i', 3> i;
FTensor::Index<'j', 3> j;
FTensor::Index<'k', 3> k;

auto res_J = [](const double x, const double y, const double z) {
  const double res = (x * x + y * y - a2);
  return res;
};

auto res_J_dx = [](const double x, const double y, const double z) {
  const double res = res_J(x, y, z);
  return FTensor::Tensor1<double, 3>{res * (2 * x), res * (2 * y),
                                     0.};
};

auto lhs_J_dx2 = [](const double x, const double y, const double z) {
  const double res = res_J(x, y, z);
  return FTensor::Tensor2<double, 3, 3>{

      (res * 2 + (4 * x * x)),
      (4 * y * x),
      0.,

      (4 * x * y),
      (2 * res + (4 * y * y)),
      0.,

      0.,
      0.,
      0.};
};

struct OpRhs : public AssemblyDomainEleOp {

  OpRhs(const std::string field_name, boost::shared_ptr<MatrixDouble> x_ptr,
        boost::shared_ptr<MatrixDouble> dot_x_ptr, boost::shared_ptr<Range> ents_ptr = nullptr)
      : AssemblyDomainEleOp(field_name, field_name, AssemblyDomainEleOp::OPROW),
        xPtr(x_ptr), xDotPtr(dot_x_ptr), entsPtr(ents_ptr) {}

  MoFEMErrorCode iNtegrate(EntitiesFieldData::EntData &row_data) {
    MoFEMFunctionBegin;

    if (entsPtr) {
      if (entsPtr->find(this->getFEEntityHandle()) == entsPtr->end())
        MoFEMFunctionReturnHot(0);
    }

    auto t_w = getFTensor0IntegrationWeight();
    auto t_row_base = row_data.getFTensor0N();

    auto t_x0 = getFTensor1CoordsAtGaussPts();
    auto t_x = getFTensor1FromMat<3>(*xPtr);
    auto t_dot_x = getFTensor1FromMat<3>(*xDotPtr);
    auto t_normal = getFTensor1NormalsAtGaussPts();

    constexpr auto t_kd = FTensor::Kronecker_Delta<double>();

    for (int gg = 0; gg != nbIntegrationPts; gg++) {

      constexpr auto t_kd = FTensor::Kronecker_Delta<double>();
      FTensor::Tensor1<double, 3> t_n{t_x0(0), t_x0(1), t_x0(2)};
      t_n.normalize();
      FTensor::Tensor2<double, 3, 3> t_P, t_Q;
      t_P(i, j) = t_n(i) * t_n(j);
      t_Q(i, j) = t_kd(i, j) - t_P(i, j);

      auto t_J_res = res_J_dx(t_x(0), t_x(1), t_x(2));

      const double alpha = t_w;
      auto t_nf = getFTensor1FromArray<3, 3>(locF);
      double l = std::sqrt(t_normal(i) * t_normal(i));

      FTensor::Tensor1<double, 3> t_res;
      t_res(i) =
          alpha * l * ((t_P(i, k) * t_J_res(k) + t_Q(i, k) * t_dot_x(k)));

      int rr = 0;
      for (; rr != nbRows / 3; ++rr) {

        t_nf(j) += t_row_base * t_res(j);

        ++t_row_base;
        ++t_nf;
      }
      for (; rr < nbRowBaseFunctions; ++rr) {
        ++t_row_base;
      }

      ++t_w;
      ++t_x;
      ++t_dot_x;
      ++t_x0;
      ++t_normal;
    }

    MoFEMFunctionReturn(0);
  }

private:
  boost::shared_ptr<MatrixDouble> xPtr;
  boost::shared_ptr<MatrixDouble> xDotPtr;
  boost::shared_ptr<Range> entsPtr;
};

struct OpLhs : public AssemblyDomainEleOp {

  OpLhs(const std::string field_name, boost::shared_ptr<MatrixDouble> x_ptr,
        boost::shared_ptr<MatrixDouble> dot_x_ptr, boost::shared_ptr<Range> ents_ptr = nullptr)
      : AssemblyDomainEleOp(field_name, field_name,
                            AssemblyDomainEleOp::OPROWCOL),
        xPtr(x_ptr), xDotPtr(dot_x_ptr), entsPtr(ents_ptr) {
    this->sYmm = false;
  }

  MoFEMErrorCode iNtegrate(EntitiesFieldData::EntData &row_data,
                           EntitiesFieldData::EntData &col_data) {
    MoFEMFunctionBegin;

    if (entsPtr) {
      if (entsPtr->find(this->getFEEntityHandle()) == entsPtr->end())
        MoFEMFunctionReturnHot(0);
    }

    auto t_w = getFTensor0IntegrationWeight();
    auto t_row_base = row_data.getFTensor0N();

    auto t_x0 = getFTensor1CoordsAtGaussPts();
    auto t_x = getFTensor1FromMat<3>(*xPtr);
    auto t_dot_x = getFTensor1FromMat<3>(*xDotPtr);
    auto t_normal = getFTensor1NormalsAtGaussPts();

    auto get_t_mat = [&](const int rr) {
      return FTensor::Tensor2<FTensor::PackPtr<double *, 3>, 3, 3>{
          &locMat(rr + 0, 0), &locMat(rr + 0, 1), &locMat(rr + 0, 2),

          &locMat(rr + 1, 0), &locMat(rr + 1, 1), &locMat(rr + 1, 2),

          &locMat(rr + 2, 0), &locMat(rr + 2, 1), &locMat(rr + 2, 2)};
    };

    constexpr auto t_kd = FTensor::Kronecker_Delta<double>();
    const double ts_a = getTSa();

    for (int gg = 0; gg != nbIntegrationPts; gg++) {

      constexpr auto t_kd = FTensor::Kronecker_Delta<double>();
      FTensor::Tensor1<double, 3> t_n{t_x0(0), t_x0(1), t_x0(2)};
      t_n.normalize();
      FTensor::Tensor2<double, 3, 3> t_P, t_Q;
      t_P(i, j) = t_n(i) * t_n(j);
      t_Q(i, j) = t_kd(i, j) - t_P(i, j);

      auto t_J_lhs = lhs_J_dx2(t_x(0), t_x(1), t_x(2));
      double l = std::sqrt(t_normal(i) * t_normal(i));

      const double alpha = t_w;
      FTensor::Tensor2<double, 3, 3> t_lhs;
      t_lhs(i, j) =
          (alpha * l) * (t_P(i, k) * t_J_lhs(k, j) + t_Q(i, j) * ts_a);

      int rr = 0;
      for (; rr != nbRows / 3; rr++) {

        auto t_col_base = col_data.getFTensor0N(gg, 0);
        auto t_mat = get_t_mat(3 * rr);

        for (int cc = 0; cc != nbCols / 3; cc++) {

          const double rc = t_row_base * t_col_base;
          t_mat(i, j) += rc * t_lhs(i, j);

          ++t_col_base;
          ++t_mat;
        }
        ++t_row_base;
      }

      for (; rr < nbRowBaseFunctions; ++rr)
        ++t_row_base;

      ++t_w;
      ++t_x;
      ++t_x0;
      ++t_normal;
    }

    MoFEMFunctionReturn(0);
  }

private:
  boost::shared_ptr<MatrixDouble> xPtr;
  boost::shared_ptr<MatrixDouble> xDotPtr;
  boost::shared_ptr<Range> entsPtr;
};

struct OpRhsEdge : public AssemblyBoundaryEleOp {

  OpRhsEdge(const std::string field_name, boost::shared_ptr<MatrixDouble> x_ptr,
        boost::shared_ptr<MatrixDouble> dot_x_ptr, boost::shared_ptr<Range> ents_ptr = nullptr)
      : AssemblyBoundaryEleOp(field_name, field_name, AssemblyBoundaryEleOp::OPROW),
        xPtr(x_ptr), xDotPtr(dot_x_ptr), entsPtr(ents_ptr) {}

  MoFEMErrorCode iNtegrate(EntitiesFieldData::EntData &row_data) {
    MoFEMFunctionBegin;

    if (entsPtr) {
      if (entsPtr->find(this->getFEEntityHandle()) == entsPtr->end())
        MoFEMFunctionReturnHot(0);
    }

    auto t_w = getFTensor0IntegrationWeight();
    auto t_row_base = row_data.getFTensor0N();

    auto t_x0 = getFTensor1CoordsAtGaussPts();
    auto t_x = getFTensor1FromMat<3>(*xPtr);
    auto t_dot_x = getFTensor1FromMat<3>(*xDotPtr);
    auto t_normal = getFTensor1NormalsAtGaussPts();

    constexpr auto t_kd = FTensor::Kronecker_Delta<double>();

    for (int gg = 0; gg != nbIntegrationPts; gg++) {

      constexpr auto t_kd = FTensor::Kronecker_Delta<double>();
      FTensor::Tensor1<double, 3> t_n{t_x0(0), t_x0(1), t_x0(2)};
      t_n.normalize();
      FTensor::Tensor2<double, 3, 3> t_P, t_Q;
      t_P(i, j) = t_n(i) * t_n(j);
      t_Q(i, j) = t_kd(i, j) - t_P(i, j);

      auto t_J_res = res_J_dx(t_x(0), t_x(1), t_x(2));

      const double alpha = t_w;
      auto t_nf = getFTensor1FromArray<3, 3>(locF);
      double l = std::sqrt(t_normal(i) * t_normal(i));

      FTensor::Tensor1<double, 3> t_res;
      t_res(i) =
          alpha * l * ((t_P(i, k) * t_J_res(k) + t_Q(i, k) * t_dot_x(k)));

      int rr = 0;
      for (; rr != nbRows / 3; ++rr) {

        t_nf(j) += t_row_base * t_res(j);

        ++t_row_base;
        ++t_nf;
      }
      for (; rr < nbRowBaseFunctions; ++rr) {
        ++t_row_base;
      }

      ++t_w;
      ++t_x;
      ++t_dot_x;
      ++t_x0;
      ++t_normal;
    }

    MoFEMFunctionReturn(0);
  }

private:
  boost::shared_ptr<MatrixDouble> xPtr;
  boost::shared_ptr<MatrixDouble> xDotPtr;
  boost::shared_ptr<Range> entsPtr;
};

struct OpLhsEdge : public AssemblyBoundaryEleOp {

  OpLhsEdge(const std::string field_name, boost::shared_ptr<MatrixDouble> x_ptr,
        boost::shared_ptr<MatrixDouble> dot_x_ptr, boost::shared_ptr<Range> ents_ptr = nullptr)
      : AssemblyBoundaryEleOp(field_name, field_name,
                            AssemblyBoundaryEleOp::OPROWCOL),
        xPtr(x_ptr), xDotPtr(dot_x_ptr), entsPtr(ents_ptr) {
    this->sYmm = false;
  }

  MoFEMErrorCode iNtegrate(EntitiesFieldData::EntData &row_data,
                           EntitiesFieldData::EntData &col_data) {
    MoFEMFunctionBegin;

    if (entsPtr) {
      if (entsPtr->find(this->getFEEntityHandle()) == entsPtr->end())
        MoFEMFunctionReturnHot(0);
    }

    auto t_w = getFTensor0IntegrationWeight();
    auto t_row_base = row_data.getFTensor0N();

    auto t_x0 = getFTensor1CoordsAtGaussPts();
    auto t_x = getFTensor1FromMat<3>(*xPtr);
    auto t_dot_x = getFTensor1FromMat<3>(*xDotPtr);
    auto t_normal = getFTensor1NormalsAtGaussPts();

    auto get_t_mat = [&](const int rr) {
      return FTensor::Tensor2<FTensor::PackPtr<double *, 3>, 3, 3>{
          &locMat(rr + 0, 0), &locMat(rr + 0, 1), &locMat(rr + 0, 2),

          &locMat(rr + 1, 0), &locMat(rr + 1, 1), &locMat(rr + 1, 2),

          &locMat(rr + 2, 0), &locMat(rr + 2, 1), &locMat(rr + 2, 2)};
    };

    constexpr auto t_kd = FTensor::Kronecker_Delta<double>();
    const double ts_a = getTSa();

    for (int gg = 0; gg != nbIntegrationPts; gg++) {

      constexpr auto t_kd = FTensor::Kronecker_Delta<double>();
      FTensor::Tensor1<double, 3> t_n{t_x0(0), t_x0(1), t_x0(2)};
      t_n.normalize();
      FTensor::Tensor2<double, 3, 3> t_P, t_Q;
      t_P(i, j) = t_n(i) * t_n(j);
      t_Q(i, j) = t_kd(i, j) - t_P(i, j);

      auto t_J_lhs = lhs_J_dx2(t_x(0), t_x(1), t_x(2));
      double l = std::sqrt(t_normal(i) * t_normal(i));

      const double alpha = t_w;
      FTensor::Tensor2<double, 3, 3> t_lhs;
      t_lhs(i, j) =
          (alpha * l) * (t_P(i, k) * t_J_lhs(k, j) + t_Q(i, j) * ts_a);

      int rr = 0;
      for (; rr != nbRows / 3; rr++) {

        auto t_col_base = col_data.getFTensor0N(gg, 0);
        auto t_mat = get_t_mat(3 * rr);

        for (int cc = 0; cc != nbCols / 3; cc++) {

          const double rc = t_row_base * t_col_base;
          t_mat(i, j) += rc * t_lhs(i, j);

          ++t_col_base;
          ++t_mat;
        }
        ++t_row_base;
      }

      for (; rr < nbRowBaseFunctions; ++rr)
        ++t_row_base;

      ++t_w;
      ++t_x;
      ++t_x0;
      ++t_normal;
    }

    MoFEMFunctionReturn(0);
  }

private:
  boost::shared_ptr<MatrixDouble> xPtr;
  boost::shared_ptr<MatrixDouble> xDotPtr;
  boost::shared_ptr<Range> entsPtr;
};


struct OpError : public DomainEleOp {

  OpError(const std::string field_name, boost::shared_ptr<MatrixDouble> x_ptr)
      : DomainEleOp(field_name, field_name, AssemblyDomainEleOp::OPROW),
        xPtr(x_ptr) {

    std::fill(&doEntities[MBEDGE], &doEntities[MBMAXTYPE], false);
  }

  MoFEMErrorCode doWork(int side, EntityType type, EntData &data) {

    MoFEMFunctionBegin;

    auto t_w = getFTensor0IntegrationWeight();
    auto t_x = getFTensor1FromMat<3>(*xPtr);
    auto t_normal = getFTensor1NormalsAtGaussPts();
    auto nb_integration_pts = getGaussPts().size2();

    double error = 0;

    for (int gg = 0; gg != nb_integration_pts; gg++) {

      double l = std::sqrt(t_normal(i) * t_normal(i));
      error += t_w * l * std::abs((t_x(i) * t_x(i) - A * A));

      ++t_w;
      ++t_x;
      ++t_normal;
    }

    CHKERR VecSetValue(errorVec, 0, error, ADD_VALUES);

    MoFEMFunctionReturn(0);
  }

  static SmartPetscObj<Vec> errorVec;

private:
  boost::shared_ptr<MatrixDouble> xPtr;
};

SmartPetscObj<Vec> OpError::errorVec;

struct ApproxSphere {

  ApproxSphere(MoFEM::Interface &m_field) : mField(m_field) {}

  MoFEMErrorCode runProblem();

private:
  MoFEM::Interface &mField;

  MoFEMErrorCode getOptions();
  MoFEMErrorCode readMesh();
  MoFEMErrorCode setupProblem();
  MoFEMErrorCode setOPs();
  MoFEMErrorCode solveSystem();
  MoFEMErrorCode outputResults();
};

//! [Run programme]
MoFEMErrorCode ApproxSphere::runProblem() {
  MoFEMFunctionBegin;
  CHKERR getOptions();
  CHKERR readMesh();
  CHKERR setupProblem();
  CHKERR setOPs();
  CHKERR solveSystem();
  CHKERR outputResults();
  MoFEMFunctionReturn(0);
}
//! [Run programme]

MoFEMErrorCode ApproxSphere::getOptions() {
  MoFEMFunctionBeginHot;
  MoFEMFunctionReturnHot(0);
}

//! [Read mesh]
MoFEMErrorCode ApproxSphere::readMesh() {
  MoFEMFunctionBegin;
  auto simple = mField.getInterface<Simple>();
  CHKERR simple->getOptions();
  CHKERR simple->loadFile();
  MoFEMFunctionReturn(0);
}
//! [Read mesh]

//! [Set up problem]
MoFEMErrorCode ApproxSphere::setupProblem() {
MoFEMFunctionBegin;

  auto get_smoothing_entities = [&](Range &smoothing_ents) {
    MoFEMFunctionBegin;
    smoothing_ents.clear();
    string block_name = "SMOOTHING_SURFACE";

    for (_IT_CUBITMESHSETS_BY_SET_TYPE_FOR_LOOP_(mField, BLOCKSET, bit)) {
      if (bit->getName().compare(0, block_name.size(), block_name) == 0) {
        MOFEM_LOG("EXAMPLE", Sev::inform) << "Found smoothing blockset";
        Range dim_ents;
        CHKERR mField.get_moab().get_entities_by_dimension(bit->getMeshset(),
                                                             FM_DIM, dim_ents, true);
        smoothing_ents.merge(dim_ents); 
        Range ents;
        CHKERR mField.get_moab().get_adjacencies(smoothing_ents, 0, false, ents,
                                                  moab::Interface::UNION);
        smoothing_ents.merge(ents);                                                    
        if (FM_DIM == 2) {
          Range edges;
          CHKERR mField.get_moab().get_adjacencies(dim_ents, 1, false, edges,
                                                  moab::Interface::UNION);
          
          smoothing_ents.merge(edges);
        }
      }
    }

    MoFEMFunctionReturn(0);
  };

  Range smoothing_ents;
  CHKERR get_smoothing_entities(smoothing_ents);

  auto simple = mField.getInterface<Simple>();
  //is it domain or boundary?
  // CHKERR simple->addDomainField("GEOMETRY", H1, AINSWORTH_LEGENDRE_BASE, 3);
  CHKERR simple->addBoundaryField("GEOMETRY", H1, AINSWORTH_LEGENDRE_BASE, 3);
  CHKERR simple->addDataField("GEOMETRY", H1, AINSWORTH_LEGENDRE_BASE, 3);

  int order = 3;
  CHKERR PetscOptionsGetInt(PETSC_NULL, "", "-order", &order, PETSC_NULL);
  cerr << "smoothing_ents.size()  " << smoothing_ents.size() <<"\n";
  
  auto get_unused_skin = [&]() {
    Range body_ents;
    CHKERR mField.get_moab().get_entities_by_dimension(0, FM_DIM+1, body_ents);
    Skinner skin(&mField.get_moab());

    Range skin_ents = body_ents;
    // CHKERR skin.find_skin(0, body_ents, false, skin_ents);
    cerr << "asd skin_ents.size()  " << skin_ents.size() <<"\n";

    Range ents;
    CHKERR mField.get_moab().get_adjacencies(skin_ents, 0, false, ents,
                                             moab::Interface::UNION);
    skin_ents.merge(ents);
    if (FM_DIM == 2) {
      Range edges;
      CHKERR mField.get_moab().get_adjacencies(skin_ents, 1, false, edges,
                                               moab::Interface::UNION);

      skin_ents.merge(edges);

      Range faces;
      CHKERR mField.get_moab().get_adjacencies(skin_ents, 2, false, faces,
                                               moab::Interface::UNION);

      skin_ents.merge(faces);
    }
    cerr << "skin_ents.size()  " << skin_ents.size() <<"\n";
    skin_ents -= smoothing_ents;
    cerr << "2) skin_ents.size()  " << skin_ents.size() <<"\n";
    return skin_ents;
  };

  CHKERR simple->setFieldOrder("GEOMETRY", order/*, &smoothing_ents*/);
  CHKERR simple->setUp();

  // Range rm_unused_skin_ents = get_unused_skin();
  // CHKERR mField.getInterface<ProblemsManager>()->removeDofsOnEntities(
  //     simple->getProblemName(), "GEOMETRY", rm_unused_skin_ents);

  
  MoFEMFunctionReturn(0);
}
//! [Set up problem]

//! [Push operators to pipeline]
MoFEMErrorCode ApproxSphere::setOPs() {
  MoFEMFunctionBegin;
  PipelineManager *pipeline_mng = mField.getInterface<PipelineManager>();

  auto bc_mng = mField.getInterface<BcManager>();
  auto simple = mField.getInterface<Simple>();

  CHKERR bc_mng->removeBlockDOFsOnEntities(simple->getProblemName(), "REMOVE_Z",
                                           "GEOMETRY", 2, 2);

  CHKERR bc_mng->removeBlockDOFsOnEntities(simple->getProblemName(), "SMOOTHING_SURFACE",
                                           "GEOMETRY", 2, 2);

  auto integration_rule = [](int, int, int approx_order) {
    return 3 * approx_order;
  };
  CHKERR pipeline_mng->setBoundaryRhsIntegrationRule(integration_rule);
  CHKERR pipeline_mng->setBoundaryLhsIntegrationRule(integration_rule);

  auto x_ptr = boost::make_shared<MatrixDouble>();
  auto dot_x_ptr = boost::make_shared<MatrixDouble>();
  auto det_ptr = boost::make_shared<VectorDouble>();
  auto jac_ptr = boost::make_shared<MatrixDouble>();
  auto inv_jac_ptr = boost::make_shared<MatrixDouble>();

  auto def_ops = [&](auto &pipeline) {
    pipeline.push_back(
        new OpCalculateVectorFieldValues<3>("GEOMETRY", x_ptr));
    pipeline.push_back(
        new OpCalculateVectorFieldValuesDot<3>("GEOMETRY", dot_x_ptr));
  };

  def_ops(pipeline_mng->getOpBoundaryRhsPipeline());
  def_ops(pipeline_mng->getOpBoundaryLhsPipeline());

  auto get_smoothing_entities = [&](Range &smoothing_ents) {
    MoFEMFunctionBegin;
    smoothing_ents.clear();
    string block_name = "SMOOTHING_SURFACE";
    for (_IT_CUBITMESHSETS_BY_SET_TYPE_FOR_LOOP_(mField, BLOCKSET, bit)) {
      if (bit->getName().compare(0, block_name.size(), block_name) == 0) {
        MOFEM_LOG("EXAMPLE", Sev::inform) << "Found smoothing blockset";
        Range dim_ents;
        CHKERR mField.get_moab().get_entities_by_dimension(
            bit->getMeshset(), FM_DIM, dim_ents, true);
        smoothing_ents.merge(dim_ents);
        Range ents;
        CHKERR mField.get_moab().get_adjacencies(smoothing_ents, 0, false, ents,
                                                 moab::Interface::UNION);
        smoothing_ents.merge(ents);
        if (FM_DIM == 2) {
          Range edges;
          CHKERR mField.get_moab().get_adjacencies(dim_ents, 1, false, edges,
                                                   moab::Interface::UNION);

          smoothing_ents.merge(edges);
        }
      }
    }

    MoFEMFunctionReturn(0);
  };

  Range smoothing_ents;
  CHKERR get_smoothing_entities(smoothing_ents);

  auto sub_ents_ptr = boost::make_shared<Range>(smoothing_ents);
  pipeline_mng->getOpBoundaryRhsPipeline().push_back(
      new OpRhs("GEOMETRY", x_ptr, dot_x_ptr, sub_ents_ptr));
  pipeline_mng->getOpBoundaryLhsPipeline().push_back(
      new OpLhs("GEOMETRY", x_ptr, dot_x_ptr, sub_ents_ptr));

  auto get_smoothing_edges = [&](Range &smoothing_edges) {
    MoFEMFunctionBegin;
    smoothing_edges.clear();
    string block_name = "REMOVE_Z";

    for (_IT_CUBITMESHSETS_BY_SET_TYPE_FOR_LOOP_(mField, BLOCKSET, bit)) {
      if (bit->getName().compare(0, block_name.size(), block_name) == 0) {
        MOFEM_LOG("EXAMPLE", Sev::inform) << "Found smoothing edges blockset";
        Range edges_ents;
        CHKERR mField.get_moab().get_entities_by_dimension(bit->getMeshset(),
                                                             FM_DIM-1, edges_ents, true);
        smoothing_edges.merge(edges_ents); 
        Range vertex_ents;
        CHKERR mField.get_moab().get_adjacencies(smoothing_edges, 0, false, vertex_ents,
                                                  moab::Interface::UNION);
        smoothing_edges.merge(vertex_ents);                                                    
      }
    }

    
    MoFEMFunctionReturn(0);
  };

  Range smoothing_edges;
  CHKERR get_smoothing_edges(smoothing_edges);

  // auto sub_edges_ptr = boost::make_shared<Range>(smoothing_edges);
  //   pipeline_mng->getOpBoundaryRhsPipeline().push_back(
  //       new OpRhsEdge("GEOMETRY", x_ptr, dot_x_ptr, sub_edges_ptr));
  //   pipeline_mng->getOpBoundaryLhsPipeline().push_back(
  //       new OpLhsEdge("GEOMETRY", x_ptr, dot_x_ptr, sub_edges_ptr));


  MoFEMFunctionReturn(0);
}
//! [Push operators to pipeline]

//! [Solve]
MoFEMErrorCode ApproxSphere::solveSystem() {
  MoFEMFunctionBegin;

  // Project HO geometry from mesh
  Projection10NodeCoordsOnField ent_method_material(mField, "GEOMETRY");
  CHKERR mField.loop_dofs("GEOMETRY", ent_method_material);

  auto *simple = mField.getInterface<Simple>();
  auto *pipeline_mng = mField.getInterface<PipelineManager>();

  //here
    auto get_smoothing_entities = [&](Range &smoothing_ents) {
    MoFEMFunctionBegin;
    smoothing_ents.clear();
    string block_name = "SMOOTHING_SURFACE";

    for (_IT_CUBITMESHSETS_BY_SET_TYPE_FOR_LOOP_(mField, BLOCKSET, bit)) {
      if (bit->getName().compare(0, block_name.size(), block_name) == 0) {
        MOFEM_LOG("EXAMPLE", Sev::inform) << "Found smoothing blockset";
        Range dim_ents;
        CHKERR mField.get_moab().get_entities_by_dimension(bit->getMeshset(),
                                                             FM_DIM, dim_ents, true);
        smoothing_ents.merge(dim_ents); 
        Range ents;
        CHKERR mField.get_moab().get_adjacencies(smoothing_ents, 0, false, ents,
                                                  moab::Interface::UNION);
        smoothing_ents.merge(ents);                                                    
        if (FM_DIM == 2) {
          Range edges;
          CHKERR mField.get_moab().get_adjacencies(dim_ents, 1, false, edges,
                                                  moab::Interface::UNION);
          
          smoothing_ents.merge(edges);
        }
      }
    }

    MoFEMFunctionReturn(0);
  };

  Range smoothing_ents;
  CHKERR get_smoothing_entities(smoothing_ents);

auto get_smoothing_edges = [&](Range &smoothing_ents) {
    MoFEMFunctionBegin;
    smoothing_ents.clear();
    string block_name = "REMOVE_Z";

    for (_IT_CUBITMESHSETS_BY_SET_TYPE_FOR_LOOP_(mField, BLOCKSET, bit)) {
      if (bit->getName().compare(0, block_name.size(), block_name) == 0) {
        MOFEM_LOG("EXAMPLE", Sev::inform) << "Found smoothing edges blockset";
        Range edges_ents;
        CHKERR mField.get_moab().get_entities_by_dimension(bit->getMeshset(),
                                                             FM_DIM-1, edges_ents, true);
      cerr << "edges_ents.size() " << edges_ents.size() <<"\n";                                                             
        smoothing_ents.merge(edges_ents); 
        Range vertex_ents;
        CHKERR mField.get_moab().get_adjacencies(smoothing_ents, 0, false, vertex_ents,
                                                  moab::Interface::UNION);
        smoothing_ents.merge(vertex_ents);                                                    
      }
    }

    MoFEMFunctionReturn(0);
  };

  Range smoothing_edges;
  CHKERR get_smoothing_edges(smoothing_edges);
  cerr << "smoothing_edges.size() " << smoothing_edges.size() <<"\n";
  //####
for (int i = 0; i != 10; ++i){
  double ftime = 1;
  /*auto subDM_1 = createDM(mField.get_comm(), "DMMOFEM");
  CHKERR DMMoFEMCreateSubDM(subDM_1, simple->getDM(), "SUB_1");
  CHKERR DMMoFEMSetSquareProblem(subDM_1, PETSC_TRUE);
  CHKERR DMMoFEMAddElement(subDM_1, simple->getBoundaryFEName());
  auto sub_ents_edge_ptr = boost::make_shared<Range>(smoothing_edges);
  CHKERR DMMoFEMAddSubFieldRow(subDM_1, "GEOMETRY", sub_ents_edge_ptr);
  // CHKERR DMMoFEMAddSubFieldCol(subDM, "GEOMETRY", sub_ents_ptr);

  // CHKERR DMMoFEMAddElement(sub_dm, simple->getDomainFEName());
  
  CHKERR DMSetUp(subDM_1);

  //here

  // auto dm = simple->getDM();
  MoFEM::SmartPetscObj<TS> ts_1;
  ts_1 = pipeline_mng->createTSIM(subDM_1);
  //new DM
  // CHKERR TSSetDM(ts, subDM);

  
  CHKERR TSSetMaxSteps(ts_1, 1);
  CHKERR TSSetExactFinalTime(ts_1, TS_EXACTFINALTIME_MATCHSTEP);

  // auto T = createDMVector(simple->getDM());
  auto T_1 = createDMVector(subDM_1);
  // CHKERR DMoFEMMeshToLocalVector(simple->getDM(), T, INSERT_VALUES,
  //                                SCATTER_FORWARD);
  // CHKERR VecGhostUpdateBegin(T, INSERT_VALUES, SCATTER_FORWARD);
  // CHKERR VecGhostUpdateEnd(T, INSERT_VALUES, SCATTER_FORWARD);
  
  CHKERR DMoFEMMeshToLocalVector(subDM_1, T_1, INSERT_VALUES,
                                 SCATTER_FORWARD);    
  
  CHKERR TSSetSolution(ts_1, T_1);
  CHKERR TSSetFromOptions(ts_1);

  CHKERR TSSolve(ts_1, NULL);
  CHKERR TSGetTime(ts_1, &ftime);*/

  //####


  auto subDM_2 = createDM(mField.get_comm(), "DMMOFEM");
  CHKERR DMMoFEMCreateSubDM(subDM_2, simple->getDM(), "SUB_2");
  CHKERR DMMoFEMSetSquareProblem(subDM_2, PETSC_TRUE);
  CHKERR DMMoFEMAddElement(subDM_2, simple->getBoundaryFEName());
  auto sub_ents_ptr = boost::make_shared<Range>(smoothing_ents);
  CHKERR DMMoFEMAddSubFieldRow(subDM_2, "GEOMETRY", sub_ents_ptr);
  // CHKERR DMMoFEMAddSubFieldCol(subDM, "GEOMETRY", sub_ents_ptr);

  // CHKERR DMMoFEMAddElement(sub_dm, simple->getDomainFEName());
  
  CHKERR DMSetUp(subDM_2);

  //here

  // auto dm = simple->getDM();
  MoFEM::SmartPetscObj<TS> ts_2;
  ts_2 = pipeline_mng->createTSIM(subDM_2);
  //new DM
  // CHKERR TSSetDM(ts, subDM);

  //double ftime = 1;
  CHKERR TSSetMaxSteps(ts_2, 1);
  CHKERR TSSetExactFinalTime(ts_2, TS_EXACTFINALTIME_MATCHSTEP);

  // auto T = createDMVector(simple->getDM());
  auto T_2 = createDMVector(subDM_2);
  // CHKERR DMoFEMMeshToLocalVector(simple->getDM(), T, INSERT_VALUES,
  //                                SCATTER_FORWARD);
  // CHKERR VecGhostUpdateBegin(T, INSERT_VALUES, SCATTER_FORWARD);
  // CHKERR VecGhostUpdateEnd(T, INSERT_VALUES, SCATTER_FORWARD);
  
  CHKERR DMoFEMMeshToLocalVector(subDM_2, T_2, INSERT_VALUES,
                                 SCATTER_FORWARD);    
  
  CHKERR TSSetSolution(ts_2, T_2);
  CHKERR TSSetFromOptions(ts_2);

  CHKERR TSSolve(ts_2, NULL);
  CHKERR TSGetTime(ts_2, &ftime);
}
  //#######





  // CHKERR mField.getInterface<FieldBlas>()->fieldScale(A, "GEOMETRY");

  MoFEMFunctionReturn(0);
}

//! [Solve]
MoFEMErrorCode ApproxSphere::outputResults() {
  MoFEMFunctionBegin;

  auto x_ptr = boost::make_shared<MatrixDouble>();
  auto det_ptr = boost::make_shared<VectorDouble>();
  auto jac_ptr = boost::make_shared<MatrixDouble>();
  auto inv_jac_ptr = boost::make_shared<MatrixDouble>();

  auto simple = mField.getInterface<Simple>();
  auto dm = simple->getDM();

  auto post_proc_fe =
      boost::make_shared<PostProcBrokenMeshInMoab<VolumeEle>>(mField);

  post_proc_fe->getOpPtrVector().push_back(
      new OpCalculateVectorFieldValues<3>("GEOMETRY", x_ptr));

  using OpPPMap = OpPostProcMapInMoab<3, 3>;

  post_proc_fe->getOpPtrVector().push_back(

      new OpPPMap(post_proc_fe->getPostProcMesh(),
                  post_proc_fe->getMapGaussPts(),

                  {},

                  {{"GEOMETRY", x_ptr}},

                  {}, {}

                  )

  );

  CHKERR DMoFEMLoopFiniteElements(dm, "dFE", post_proc_fe);
  CHKERR post_proc_fe->writeFile("out_approx.h5m");

  auto error_fe = boost::make_shared<DomainEle>(mField);

  error_fe->getOpPtrVector().push_back(
      new OpGetHONormalsOnFace("GEOMETRY"));
  error_fe->getOpPtrVector().push_back(
      new OpCalculateVectorFieldValues<3>("GEOMETRY", x_ptr));
  error_fe->getOpPtrVector().push_back(new OpError("GEOMETRY", x_ptr));

  error_fe->preProcessHook = [&]() {
    MoFEMFunctionBegin;
    MOFEM_LOG("EXAMPLE", Sev::inform) << "Create vec ";
    OpError::errorVec = createVectorMPI(
        mField.get_comm(), (!mField.get_comm_rank()) ? 1 : 0, 1);
    VecZeroEntries(OpError::errorVec);
    MoFEMFunctionReturn(0);
  };

  error_fe->postProcessHook = [&]() {
    MoFEMFunctionBegin;
    CHKERR VecAssemblyBegin(OpError::errorVec);
    CHKERR VecAssemblyEnd(OpError::errorVec);
    double error2;
    CHKERR VecSum(OpError::errorVec, &error2);
    MOFEM_LOG("EXAMPLE", Sev::inform)
        << "Error " << std::sqrt(error2 / (4 * M_PI * A * A));
    OpError::errorVec.reset();
    MoFEMFunctionReturn(0);
  };

  CHKERR DMoFEMLoopFiniteElements(dm, "dFE", error_fe);

  CHKERR simple->deleteDM();
  CHKERR simple->deleteFiniteElements();
  if (mField.get_comm_size() > 1)
    CHKERR mField.get_moab().write_file("out_ho_mesh.h5m", "MOAB",
                                        "PARALLEL=WRITE_PART");
  else
    CHKERR mField.get_moab().write_file("out_ho_mesh.h5m");
  MoFEMFunctionReturn(0);
}
//! [Postprocess results]

int main(int argc, char *argv[]) {

  // Initialisation of MoFEM/PETSc and MOAB data structures
  const char param_file[] = "param_file.petsc";
  MoFEM::Core::Initialize(&argc, &argv, param_file, help);

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

    //! [ApproxSphere]
    ApproxSphere ex(m_field);
    CHKERR ex.runProblem();
    //! [ApproxSphere]
  }
  CATCH_ERRORS;

  MoFEM::Core::Finalize();
}
