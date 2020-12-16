/** \file arbitrary_smooth_contact_surface.cpp
 * \example arbitrary_smooth_contact_surface.cpp
 *
 * Implementation of mortar contact between surfaces with matching meshes
 *
 **/

/* MoFEM is free software: you can redistribute it and/or modify it under
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
 * License along with MoFEM. If not, see <http://www.gnu.org/licenses/>.
 */

#include <MoFEM.hpp>

using namespace MoFEM;

static char help[] = "...\n\n";

#include <BasicFiniteElements.hpp>

using EntData = DataForcesAndSourcesCore::EntData;
using FaceEle = FaceElementForcesAndSourcesCore;
using FaceEleOp = FaceEle::UserDataOperator;
using VolumeEle = VolumeElementForcesAndSourcesCore;
using VolumeEleOp = VolumeEle::UserDataOperator;

struct ArbitrarySmoothingProblem {

Vec volumeVec;
double cnVaule;

  struct SurfaceSmoothingElement : public FaceEle {
    MoFEM::Interface &mField;
    SurfaceSmoothingElement(MoFEM::Interface &m_field)
        : FaceEle(m_field), mField(m_field) {}

  int getRule(int order) { return 3 * (order + 2 ) ; };

    // Destructor
    ~SurfaceSmoothingElement() {}
  };

  struct SaveVertexDofOnTag : public MoFEM::DofMethod {
    MoFEM::Interface &mField;
    std::string tagName;
    SaveVertexDofOnTag(MoFEM::Interface &m_field, std::string tag_name)
        : mField(m_field), tagName(tag_name) {}
    Tag tHtAg;
    MoFEMErrorCode preProcess() {
      MoFEMFunctionBegin;
      if (!fieldPtr) {
        SETERRQ(mField.get_comm(), MOFEM_DATA_INCONSISTENCY,
                "Null pointer, probably field not found");
      }
      if (fieldPtr->getSpace() != H1) {
        SETERRQ(mField.get_comm(), MOFEM_DATA_INCONSISTENCY,
                "Field must be in H1 space");
      }
      std::vector<double> def_vals(fieldPtr->getNbOfCoeffs(), 0);
      rval = mField.get_moab().tag_get_handle(tagName.c_str(), tHtAg);
      // rval = mField.get_moab().tag_get_handle(fieldName.c_str(), tHfIeld);

      // if (rval != MB_SUCCESS) {
      //   CHKERR mField.get_moab().tag_get_handle(
      //       tagName.c_str(), fieldPtr->getNbOfCoeffs(), MB_TYPE_DOUBLE, tH,
      //       MB_TAG_CREAT | MB_TAG_SPARSE, &def_vals[0]);
      // }
      MoFEMFunctionReturn(0);
    }

    MoFEMErrorCode operator()() {
      MoFEMFunctionBegin;
      if (dofPtr->getEntType() != MBVERTEX)
        MoFEMFunctionReturnHot(0);
      EntityHandle ent = dofPtr->getEnt();
      int rank = dofPtr->getNbOfCoeffs();
      double tag_val[rank];
      CHKERR mField.get_moab().tag_get_data(tHtAg, &ent, 1, tag_val);
      double mag = sqrt(tag_val[0] * tag_val[0] + tag_val[1] * tag_val[1] + tag_val[2] * tag_val[2]);
      dofPtr->getFieldData() = tag_val[dofPtr->getDofCoeffIdx()];///mag;
      // cerr << "tag_passing!   " << dofPtr->getFieldData() <<"\n";
      MoFEMFunctionReturn(0);
    }

    MoFEMErrorCode postProcess() {
      MoFEMFunctionBegin;
      MoFEMFunctionReturn(0);
    }
  };

  struct ResolveAndSaveForEdges : public MoFEM::DofMethod {
    MoFEM::Interface &mField;
    std::string tagName;
    ResolveAndSaveForEdges(MoFEM::Interface &m_field, std::string tag_name)
        : mField(m_field), tagName(tag_name) {}
    Tag tHtAg;
    MoFEMErrorCode preProcess() {
      MoFEMFunctionBegin;
      if (!fieldPtr) {
        SETERRQ(mField.get_comm(), MOFEM_DATA_INCONSISTENCY,
                "Null pointer, probably field not found");
      }
      if (fieldPtr->getSpace() != H1) {
        SETERRQ(mField.get_comm(), MOFEM_DATA_INCONSISTENCY,
                "Field must be in H1 space");
      }
      std::vector<double> def_vals(fieldPtr->getNbOfCoeffs(), 0);
      rval = mField.get_moab().tag_get_handle(tagName.c_str(), tHtAg);
      // rval = mField.get_moab().tag_get_handle(fieldName.c_str(), tHfIeld);

      // if (rval != MB_SUCCESS) {
      //   CHKERR mField.get_moab().tag_get_handle(
      //       tagName.c_str(), fieldPtr->getNbOfCoeffs(), MB_TYPE_DOUBLE, tH,
      //       MB_TAG_CREAT | MB_TAG_SPARSE, &def_vals[0]);
      // }
      MoFEMFunctionReturn(0);
    }

    MoFEMErrorCode operator()() {
      MoFEMFunctionBegin;
      if (dofPtr->getEntType() != MBEDGE)
        MoFEMFunctionReturnHot(0);
      EntityHandle ent = dofPtr->getEnt();
      int rank = dofPtr->getNbOfCoeffs();
      double tag_val[rank];
      CHKERR mField.get_moab().tag_get_data(tHtAg, &ent, 1, tag_val);
      dofPtr->getFieldData() = tag_val[dofPtr->getDofCoeffIdx()];
      cerr << "tag_passing!   " << dofPtr->getFieldData() <<"\n";
      MoFEMFunctionReturn(0);
    }

    MoFEMErrorCode postProcess() {
      MoFEMFunctionBegin;
      MoFEMFunctionReturn(0);
    }
  };


  struct ResolveAndSaveForEdges2 : public DofMethod {

  MoFEM::Interface &mFieldEdges;
  Tag vertexTag;
  std::string vertexTagName;
  std::string edgesTagName;
  std::string fieldName;
  int vErbose;

  ResolveAndSaveForEdges2(MoFEM::Interface &m_field, std::string field_name,
                          std::string vertex_tag_name,
                          std::string edges_tag_name, int verb = 0)
      : mFieldEdges(m_field), fieldName(field_name),
        vertexTagName(vertex_tag_name), edgesTagName(edges_tag_name),
        vErbose(verb) {}

  MoFEMErrorCode preProcess() {
    MoFEMFunctionBeginHot;
    MoFEMFunctionReturnHot(0);
  }

  VectorDouble vecComponents;
  VectorDouble3 aveMidVector;
  VectorDouble3 midVector;
  VectorDouble3 diffVectors;
  VectorDouble3 dOf;

  MoFEMErrorCode operator()() {
    MoFEMFunctionBeginHot;
    if (dofPtr == NULL) {
      SETERRQ(PETSC_COMM_SELF, MOFEM_DATA_INCONSISTENCY, "data inconsistency");
    }
    if (dofPtr->getName() != fieldName)
      MoFEMFunctionReturnHot(0);

    if (dofPtr->getEntType() != MBEDGE) 
      MoFEMFunctionReturnHot(0);
    
    if (dofPtr->getEntDofIdx() != dofPtr->getDofCoeffIdx()) {
      MoFEMFunctionReturnHot(0);
    }

    EntityHandle edge = dofPtr->getEnt();
    if (mFieldEdges.get_moab().type_from_handle(edge) != MBEDGE) {
      SETERRQ(PETSC_COMM_SELF, MOFEM_DATA_INCONSISTENCY,
              "this method works only elements which are type of MBEDGE");
    }
    // vecComponents
    int num_nodes;
    const EntityHandle *conn;
    CHKERR mFieldEdges.get_moab().get_connectivity(edge, conn, num_nodes, false);
    
    if (num_nodes != 2) {
      SETERRQ(PETSC_COMM_SELF, MOFEM_DATA_INCONSISTENCY,
              "this method works only 4 node tets");
    }
    vecComponents.resize(num_nodes * 3);
    
    //ignatios
    // CHKERR mFieldEdges.get_moab().get_coords(conn, num_nodes,
    //                                     &*vecComponents.data().begin());

    Tag th_nb_ver;
    CHKERR mFieldEdges.get_moab().tag_get_handle(vertexTagName.c_str(), th_nb_ver);
    double *vector_values[2];
    CHKERR mFieldEdges.get_moab().tag_get_by_ptr(th_nb_ver, conn, 2,
                                             (const void **)&vector_values);

    Tag th_nb_edges;
    CHKERR mFieldEdges.get_moab().tag_get_handle(edgesTagName.c_str(), th_nb_edges);
    double *vector_values_edges;
    CHKERR mFieldEdges.get_moab().tag_get_by_ptr(th_nb_edges, &edge, 1,
                                             (const void **)&vector_values_edges);

    for(int nn = 0; nn != 2; ++nn ){
    for (int ii = 0; ii != 3; ++ii) {
      vecComponents(3 * nn + ii ) = vector_values[nn][ii]; 
    }
}

VectorDouble coords;
coords.resize(num_nodes * 3);
CHKERR mFieldEdges.get_moab().get_coords(conn, num_nodes, &*coords.data().begin());
double rad_1 = sqrt(coords(0) * coords(0) + coords(1) * coords(1)  + coords(2) * coords(2));
double rad_2 = sqrt(coords(3) * coords(3) + coords(4) * coords(4)  + coords(5) * coords(5));

// cerr << "Rad 1   " << rad_1 << "   Rad 2  " << rad_2 <<"\n";


aveMidVector.resize(3);
midVector.resize(3);

for (int dd = 0; dd != 3; ++dd) {
  aveMidVector[dd] =
      (vecComponents[0 * 3 + dd] + vecComponents[1 * 3 + dd]) * 0.5;
  midVector[dd] = vector_values_edges[dd];
    }

    double edge_shape_function_val = 0.25;
    FieldApproximationBase base = dofPtr->getApproxBase();
    switch (base) {
    case AINSWORTH_LEGENDRE_BASE:
      break;
    case AINSWORTH_LOBATTO_BASE:
      edge_shape_function_val *= LOBATTO_PHI0(0);
      break;
    default:
      SETERRQ(PETSC_COMM_SELF, MOFEM_NOT_IMPLEMENTED, "not yet implemented");
    }

    diffVectors.resize(3);
    ublas::noalias(diffVectors) = midVector - aveMidVector;
    dOf.resize(3);
    ublas::noalias(dOf) = diffVectors / edge_shape_function_val;
    if (dofPtr->getNbOfCoeffs() != 3) {
      SETERRQ(PETSC_COMM_SELF, MOFEM_DATA_INCONSISTENCY,
              "this method works only fields which are rank 3");
    }
    dofPtr->getFieldData() = dOf[dofPtr->getDofCoeffIdx()];
    MoFEMFunctionReturnHot(0);
  }

  MoFEMErrorCode postProcess() {
    MoFEMFunctionBeginHot;
    MoFEMFunctionReturnHot(0);
  }
};


  struct VolumeSmoothingElement : public VolumeEle {
    MoFEM::Interface &mField;

    VolumeSmoothingElement(MoFEM::Interface &m_field)
        : VolumeEle(m_field), mField(m_field) {}

    // Destructor
    ~VolumeSmoothingElement() {}
  };

  struct CommonSurfaceSmoothingElement {

    boost::shared_ptr<MatrixDouble> nOrmal;
    boost::shared_ptr<MatrixDouble> tAngent1;
    boost::shared_ptr<MatrixDouble> tAngent2;
    boost::shared_ptr<MatrixDouble> nOrmalField;

    boost::shared_ptr<MatrixDouble> hoGaussPointLocation;

    boost::shared_ptr<MatrixDouble> lagMult;
    boost::shared_ptr<VectorDouble> lagMultScalar;

    boost::shared_ptr<VectorDouble> areaNL;

    CommonSurfaceSmoothingElement(MoFEM::Interface &m_field) : mField(m_field) {
      nOrmal = boost::make_shared<MatrixDouble>();
      tAngent1 = boost::make_shared<MatrixDouble>();
      tAngent2 = boost::make_shared<MatrixDouble>();
      lagMult = boost::make_shared<MatrixDouble>();
      lagMultScalar = boost::make_shared<VectorDouble>();
      hoGaussPointLocation = boost::make_shared<MatrixDouble>();
      nOrmalField = boost::make_shared<MatrixDouble>();
      areaNL = boost::make_shared<VectorDouble>();
    }

  private:
    MoFEM::Interface &mField;
  };

  MoFEMErrorCode addSurfaceSmoothingElement(const string element_name,
                                            const string field_name_position,
                                            const string field_name_normal_field,
                                            const string field_lag_mult,
                                            Range &range_smoothing_elements) {
    MoFEMFunctionBegin;

    CHKERR mField.add_finite_element(element_name, MF_ZERO);

    CHKERR mField.modify_finite_element_add_field_col(element_name,
                                                      field_name_position);

    CHKERR mField.modify_finite_element_add_field_row(element_name,
                                                      field_name_position);

    CHKERR mField.modify_finite_element_add_field_data(element_name,
                                                       field_name_position);

    // CHKERR mField.modify_finite_element_add_field_col(element_name,
    //                                                   field_name_normal_field);

    // CHKERR mField.modify_finite_element_add_field_row(element_name,
    //                                                   field_name_normal_field);

    CHKERR mField.modify_finite_element_add_field_data(element_name,
                                                       field_name_normal_field);

    //field_lag_mmult
    // CHKERR mField.modify_finite_element_add_field_col(element_name,
    //                                                   field_lag_mult);

    // CHKERR mField.modify_finite_element_add_field_row(element_name,
    //                                                   field_lag_mult);

    CHKERR mField.modify_finite_element_add_field_data(element_name,
                                                       field_lag_mult);

    mField.add_ents_to_finite_element_by_type(range_smoothing_elements, MBTRI,
                                              element_name);

    MoFEMFunctionReturn(0);
  }

  MoFEMErrorCode addVolumeElement(const string element_name,
                                  const string field_name,
                                  Range &range_volume_elements) {
    MoFEMFunctionBegin;

    CHKERR mField.add_finite_element(element_name, MF_ZERO);

    CHKERR mField.modify_finite_element_add_field_data(element_name,
                                                       field_name);

    mField.add_ents_to_finite_element_by_type(range_volume_elements, MBTET,
                                              element_name);

    MoFEMFunctionReturn(0);
  }

  /// \brief Computes, for reference configuration, normal to slave face that is
  /// common to all gauss points
  struct OpGetTangentForSmoothSurf : public FaceEleOp {

    boost::shared_ptr<CommonSurfaceSmoothingElement>
        commonSurfaceSmoothingElement;
    OpGetTangentForSmoothSurf(
        const string field_name,
        boost::shared_ptr<CommonSurfaceSmoothingElement> common_data_smoothing, bool check_dofs = false)
        : FaceEleOp(field_name, UserDataOperator::OPCOL),
          commonSurfaceSmoothingElement(common_data_smoothing), checkDofs(check_dofs) {
          }

    MoFEMErrorCode doWork(int side, EntityType type, EntData &data) {
      MoFEMFunctionBegin;

      if (data.getFieldData().size() == 0)
        PetscFunctionReturn(0);

      int ngp = data.getN().size1();

      unsigned int nb_dofs = data.getFieldData().size() / 3;
      FTensor::Index<'i', 3> i;
      if (type == MBVERTEX) {
        commonSurfaceSmoothingElement->tAngent1->resize(3, ngp, false);
        commonSurfaceSmoothingElement->tAngent1->clear();

        commonSurfaceSmoothingElement->tAngent2->resize(3, ngp, false);
        commonSurfaceSmoothingElement->tAngent2->clear();
        commonSurfaceSmoothingElement->hoGaussPointLocation->resize(3, ngp, false);
        commonSurfaceSmoothingElement->hoGaussPointLocation->clear();
      }

      auto t_1 =
          getFTensor1FromMat<3>(*commonSurfaceSmoothingElement->tAngent1);
      auto t_2 =
          getFTensor1FromMat<3>(*commonSurfaceSmoothingElement->tAngent2);

      auto t_ho_pos =
          getFTensor1FromMat<3>(*commonSurfaceSmoothingElement->hoGaussPointLocation);

      for (unsigned int gg = 0; gg != ngp; ++gg) {
        auto t_N = data.getFTensor1DiffN<2>(gg, 0);
        // auto t_dof = data.getFTensor1FieldData<3>();
        FTensor::Tensor1<double *, 3> t_dof(
              &data.getFieldData()[0], &data.getFieldData()[1], &data.getFieldData()[2], 3);
        FTensor::Tensor0<double *> t_base(&data.getN()(gg, 0));

        for (unsigned int dd = 0; dd != nb_dofs; ++dd) {

          // if(checkDofs && gg == 0){
          // cerr << "type  " << type << "  t_dof   " << t_dof << "\n";
          // }
          t_1(i) += t_dof(i) * t_N(0);
          t_2(i) += t_dof(i) * t_N(1);
          
          // if(type != MBVERTEX)
            t_ho_pos(i) += t_base * t_dof(i);

          ++t_dof;
          ++t_N;
          ++t_base;
        }
        ++t_1;
        ++t_2;
        ++t_ho_pos;
      }

      MoFEMFunctionReturn(0);
    }
    private:
    bool checkDofs;
  };




    struct OpPrintRadius : public FaceEleOp {

    boost::shared_ptr<CommonSurfaceSmoothingElement>
        commonSurfaceSmoothingElement;
    OpPrintRadius(
        const string field_name,
        boost::shared_ptr<CommonSurfaceSmoothingElement> common_data_smoothing, bool check_dofs = false)
        : FaceEleOp(field_name, UserDataOperator::OPCOL),
          commonSurfaceSmoothingElement(common_data_smoothing), checkDofs(check_dofs) {
          }

    MoFEMErrorCode doWork(int side, EntityType type, EntData &data) {
      MoFEMFunctionBegin;

      if (data.getFieldData().size() == 0)
        PetscFunctionReturn(0);

      if (type == MBVERTEX) 
        PetscFunctionReturn(0);
      
      int ngp = data.getN().size1();

      unsigned int nb_dofs = data.getFieldData().size() / 3;
      FTensor::Index<'i', 3> i;

      auto t_ho_pos =
          getFTensor1FromMat<3>(*commonSurfaceSmoothingElement->hoGaussPointLocation);

      for (unsigned int gg = 0; gg != ngp; ++gg) {
        
        const double radius = sqrt(t_ho_pos(i) * t_ho_pos(i));
        // cerr << "Radius    " << radius << "\n";
        ++t_ho_pos;
      }

      MoFEMFunctionReturn(0);
    }
    private:
    bool checkDofs;
  };

  /// \brief Computes, for reference configuration, normal to slave face that is
  /// common to all gauss points
  struct OpGetNormalForSmoothSurf : public FaceEleOp {

    boost::shared_ptr<CommonSurfaceSmoothingElement>
        commonSurfaceSmoothingElement;
    OpGetNormalForSmoothSurf(
        const string field_name,
        boost::shared_ptr<CommonSurfaceSmoothingElement> common_data_smoothing)
        : FaceEleOp(field_name, UserDataOperator::OPCOL),
          commonSurfaceSmoothingElement(common_data_smoothing) {}

    MoFEMErrorCode doWork(int side, EntityType type, EntData &data) {
      MoFEMFunctionBegin;

      if (data.getFieldData().size() == 0)
        PetscFunctionReturn(0);

      if (type != MBVERTEX)
        PetscFunctionReturn(0);

      int ngp = data.getN().size1();

      FTensor::Index<'i', 3> i;
      FTensor::Index<'j', 3> j;
      FTensor::Index<'k', 3> k;
      commonSurfaceSmoothingElement->nOrmal->resize(3, ngp, false);
      commonSurfaceSmoothingElement->nOrmal->clear();
      commonSurfaceSmoothingElement->areaNL->resize(ngp, false);
      commonSurfaceSmoothingElement->areaNL->clear();

      auto t_1 =
          getFTensor1FromMat<3>(*commonSurfaceSmoothingElement->tAngent1);
      auto t_2 =
          getFTensor1FromMat<3>(*commonSurfaceSmoothingElement->tAngent2);
      
      auto t_n = getFTensor1FromMat<3>(*commonSurfaceSmoothingElement->nOrmal);
      auto t_area =
      getFTensor0FromVec(*commonSurfaceSmoothingElement->areaNL);
      for (unsigned int gg = 0; gg != ngp; ++gg) {
        t_n(i) = FTensor::levi_civita(i, j, k) * t_1(j) * t_2(k);
        const double n_mag = sqrt(t_n(i) * t_n(i));
        // t_n(i) /= n_mag;
        
        t_area = 0.5 * n_mag;
        ++t_area;
        ++t_n;
        ++t_1;
        ++t_2;
      }

      MoFEMFunctionReturn(0);
    }
  };

    struct OpCalcMeanNormal : public FaceEleOp {

    boost::shared_ptr<CommonSurfaceSmoothingElement>
        commonSurfaceSmoothingElement;
    OpCalcMeanNormal(const string field_name, const string tag_name_nb_tris,
                     const string tag_name_ave_norm, EntityType ent_type, MoFEM::Interface &m_field)
        : FaceEleOp(field_name, UserDataOperator::OPCOL),
          tagNbTris(tag_name_nb_tris), tagAveNormal(tag_name_ave_norm), entType(ent_type),
          mField(m_field) {}

    MoFEMErrorCode doWork(int side, EntityType type, EntData &data) {
      MoFEMFunctionBegin;

      // if (data.getFieldData().size() == 0)
      //   PetscFunctionReturn(0);
      FTensor::Index<'i', 3> i;
      
      if (type != entType)
        PetscFunctionReturn(0);

      const double area_m = getMeasure();
      VectorDouble3 normal = getNormal();

      double n_mag;
      n_mag = sqrt(normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]);

      VectorDouble n_unit;
      n_unit.resize(3, false);
      for (int ii = 0; ii != 3; ++ii) {
        n_unit[ii] = normal[ii] / n_mag;
      }

      EntityHandle ent = getNumeredEntFiniteElementPtr()->getEnt();
      Range entities_to_consider;
      if(type == MBVERTEX){
      CHKERR mField.get_moab().get_connectivity(&ent, 1, entities_to_consider);
      } else if (type == MBEDGE) {
      CHKERR mField.get_moab().get_adjacencies(&ent, 1, 1, false, entities_to_consider);
      }

      auto get_tensor_vec = [](VectorDouble &n) {
        return FTensor::Tensor1<double *, 3>(&n(0), &n(1), &n(2));
      };

      VectorDouble v_coords_vertices;
      v_coords_vertices.resize(3, false);
      v_coords_vertices.clear();

      FTensor::Tensor1<double, 3> t_vertex_1;
      FTensor::Tensor1<double, 3> t_vertex_2;
      FTensor::Tensor1<double, 3> t_vertex_3;
      auto t_help = get_tensor_vec(v_coords_vertices);
      
      std::vector<EntityHandle> ents_for_tags;
      int count = 0;
      for (Range::iterator it_entities_to_consider =
               entities_to_consider.begin();
           it_entities_to_consider != entities_to_consider.end();
           ++it_entities_to_consider) {
        CHKERR mField.get_moab().get_coords(&*it_entities_to_consider, 1,
                                            &*v_coords_vertices.data().begin());
        ++count;
        if(type == MBVERTEX){
        switch (count) {
        case 1:
          t_vertex_1(i) = t_help(i);
          break;
        case 2:
          t_vertex_2(i) = t_help(i);
          break;
        case 3:
          t_vertex_3(i) = t_help(i);
          break;
        default:
          SETERRQ1(PETSC_COMM_SELF, MOFEM_ATOM_TEST_INVALID,
                   "Too many coords in triangle = %6.4e instead of 3", count);
        }
        }
        v_coords_vertices.clear();
        
        ents_for_tags.push_back(*it_entities_to_consider);
      }
      
      VectorDouble3 v_weights(3);
      
      double alpha_v, sin_v, denom_v, weight_v;

      auto get_weight = [&](FTensor::Tensor1<double, 3> t_edge_a,
                            FTensor::Tensor1<double, 3> t_edge_b,
                            FTensor::Tensor1<double, 3> t_edge_c) {
        denom_v =
            sqrt(t_edge_b(i) * t_edge_b(i)) * sqrt(t_edge_a(i) * t_edge_a(i));
        sin_v = 2. * area_m / denom_v;
        alpha_v = std::acos(t_edge_a(i) * t_edge_b(i) / denom_v);
        // cerr << "sin_v  " << sqrt(t_edge_c(i) * t_edge_c(i)) <<"\n";
        weight_v = pow(sin_v / sqrt(t_edge_c(i) * t_edge_c(i)), 2) ;//* alpha_v;
        return weight_v;
      };

      FTensor::Tensor1<double, 3> t_edge_a, t_edge_b, t_edge_c; 
      t_edge_a(i) =  t_vertex_2(i)  - t_vertex_1(i);
      t_edge_b(i) =  t_vertex_3(i)  - t_vertex_1(i);
      t_edge_c(i) =  t_vertex_3(i)  - t_vertex_2(i);
      v_weights(0) = get_weight(t_edge_a, t_edge_b, t_edge_c);

      t_edge_a(i) =  t_vertex_3(i)  - t_vertex_2(i);
      t_edge_b(i) =  t_vertex_1(i)  - t_vertex_2(i);
      t_edge_c(i) =  t_vertex_3(i)  - t_vertex_1(i);
      v_weights(1) = get_weight(t_edge_a, t_edge_b, t_edge_c);

      t_edge_a(i) =  t_vertex_1(i)  - t_vertex_3(i);
      t_edge_b(i) =  t_vertex_2(i)  - t_vertex_3(i);
      t_edge_c(i) =  t_vertex_2(i)  - t_vertex_1(i);
      v_weights(2) = get_weight(t_edge_a, t_edge_b, t_edge_c);

      // cerr << "v_weights " << v_weights << "\n";


      Tag th_nb_tri;
      CHKERR mField.get_moab().tag_get_handle(tagNbTris.c_str(), th_nb_tri);
      Tag th_ave_normal;
      CHKERR mField.get_moab().tag_get_handle(tagAveNormal.c_str(), th_ave_normal);

      for (vector<EntityHandle>::iterator vit = ents_for_tags.begin();
           vit != ents_for_tags.end(); ++vit) {
        int *num_nodes;
        CHKERR mField.get_moab().tag_get_by_ptr(th_nb_tri, &*vit, 1,
                                                (const void **)&num_nodes);
        // cerr << "num_nodes !!!  " << *num_nodes << "\n";
        double *ave_normal;
        CHKERR mField.get_moab().tag_get_by_ptr(th_ave_normal, &*vit, 1,
                                                (const void **)&ave_normal);

        for (int ii = 0; ii != 3; ++ii){
          
          ave_normal[ii] += n_unit[ii] / (double)(*num_nodes);

          // ave_normal[ii] += n_unit[ii] * v_weights(ii);
        }

        // cerr << "ave_normal  " << ave_normal[0] << "  " << ave_normal[1] << "  "
        //      << ave_normal[2] << "  "
        //      << "\n";
      }

      MoFEMFunctionReturn(0);
    }

    private:
    const string tagNbTris;
    const string tagAveNormal;
    EntityType entType;
    MoFEM::Interface &mField;
  };


/// \brief Computes, for reference configuration, normal to slave face that is
  /// common to all gauss points
  struct OpPrintNormals : public FaceEleOp {

    boost::shared_ptr<CommonSurfaceSmoothingElement>
        commonSurfaceSmoothingElement;
    OpPrintNormals(
        const string field_name,
        boost::shared_ptr<CommonSurfaceSmoothingElement> common_data_smoothing)
        : FaceEleOp(field_name, UserDataOperator::OPCOL),
          commonSurfaceSmoothingElement(common_data_smoothing) {}

    MoFEMErrorCode doWork(int side, EntityType type, EntData &data) {
      MoFEMFunctionBegin;


      if (data.getFieldData().size() == 0)
        PetscFunctionReturn(0);

      if (type != MBVERTEX)
        PetscFunctionReturn(0);

      int ngp = data.getN().size1();

      FTensor::Index<'i', 3> i;
      FTensor::Index<'j', 3> j;
      FTensor::Index<'k', 3> k;
      auto t_tangent_1_tot =
          getFTensor1FromMat<3>(*commonSurfaceSmoothingElement->tAngent1);
      auto t_tangent_2_tot =
          getFTensor1FromMat<3>(*commonSurfaceSmoothingElement->tAngent2);
      // auto t_tangent_2 =
      // getFTensor1FromMat<3>(*commonSurfaceSmoothingElement->tAngentTwoField);

      auto t_n = getFTensor1FromMat<3>(*commonSurfaceSmoothingElement->nOrmal);

      FTensor::Tensor1<double, 3> t_normal_field;
      
      for (unsigned int gg = 0; gg != ngp; ++gg) {
        
        t_normal_field(i) = FTensor::levi_civita(i, j, k) * t_tangent_1_tot(j) * t_tangent_2_tot(k);
        CHKERR PetscPrintf(PETSC_COMM_WORLD, "X Normal = %e, %e, %e\n\n",
                       t_normal_field(0), t_normal_field(1), t_normal_field(2));
        CHKERR PetscPrintf(PETSC_COMM_WORLD, "Field Normal = %e, %e, %e\n\n",
                           t_normal_field(0), t_normal_field(1), t_normal_field(2));

        CHKERR PetscPrintf(PETSC_COMM_WORLD, "X Tan 1 = %e, %e, %e\n\n",
                           t_tangent_1_tot(0), t_tangent_1_tot(1),
                           t_tangent_1_tot(2));

        CHKERR PetscPrintf(PETSC_COMM_WORLD, "X Tan 2 = %e, %e, %e\n\n",
                           t_tangent_2_tot(0), t_tangent_2_tot(1),
                           t_tangent_2_tot(2));



        ++t_n;
        ++t_tangent_1_tot;
        ++t_tangent_2_tot;
      }

      MoFEMFunctionReturn(0);
    }
  };
  

    /// \brief Computes, for reference configuration, normal to slave face that is
  /// common to all gauss points
  struct OpGetNormalField : public FaceEleOp {

    boost::shared_ptr<CommonSurfaceSmoothingElement>
        commonSurfaceSmoothingElement;
    OpGetNormalField(
        const string normal_field_name,
        boost::shared_ptr<CommonSurfaceSmoothingElement> common_data_smoothing)
        : FaceEleOp(normal_field_name, UserDataOperator::OPCOL),
          commonSurfaceSmoothingElement(common_data_smoothing) {
          }

    MoFEMErrorCode doWork(int side, EntityType type, EntData &data) {
      MoFEMFunctionBegin;

      if (data.getFieldData().size() == 0)
        PetscFunctionReturn(0);

      if (type != MBVERTEX && type != MBEDGE) 
        PetscFunctionReturn(0);

      int ngp = data.getN().size1();

      unsigned int nb_dofs = data.getFieldData().size() / 3;

      FTensor::Index<'i', 3> i;
      
      if(type == MBVERTEX){
        commonSurfaceSmoothingElement->nOrmalField->resize(3, ngp, false);
        commonSurfaceSmoothingElement->nOrmalField->clear();
      }

      auto t_n_field =
          getFTensor1FromMat<3>(*commonSurfaceSmoothingElement->nOrmalField);

      for (unsigned int gg = 0; gg != ngp; ++gg) {
        FTensor::Tensor1<double *, 3> t_dof(
              &data.getFieldData()[0], &data.getFieldData()[1], &data.getFieldData()[2], 3);
        FTensor::Tensor0<double *> t_base_normal_field(&data.getN()(gg, 0));

        for (unsigned int dd = 0; dd != nb_dofs; ++dd) {
          t_n_field(i) += t_dof(i) * t_base_normal_field;
          ++t_dof;
          ++t_base_normal_field;
        }
        ++t_n_field;
      }

      MoFEMFunctionReturn(0);
    }
  };

  //
    /// \brief Computes, for reference configuration, normal to slave face that is
  /// common to all gauss points
  struct OpGetNormalisedNormalField : public FaceEleOp {

    boost::shared_ptr<CommonSurfaceSmoothingElement>
        commonSurfaceSmoothingElement;
    OpGetNormalisedNormalField(
        const string normal_field_name,
        boost::shared_ptr<CommonSurfaceSmoothingElement> common_data_smoothing)
        : FaceEleOp(normal_field_name, UserDataOperator::OPCOL),
          commonSurfaceSmoothingElement(common_data_smoothing) {
          }

    MoFEMErrorCode doWork(int side, EntityType type, EntData &data) {
      MoFEMFunctionBegin;

      if (data.getFieldData().size() == 0)
        PetscFunctionReturn(0);

      if (type != MBVERTEX && type != MBEDGE) 
        PetscFunctionReturn(0);

      int ngp = data.getN().size1();

      unsigned int nb_dofs = data.getFieldData().size() / 3;

      FTensor::Index<'i', 3> i;

      auto t_n_field =
          getFTensor1FromMat<3>(*commonSurfaceSmoothingElement->nOrmalField);

      for (unsigned int gg = 0; gg != ngp; ++gg) {
        const double n_mag = sqrt(t_n_field(i) * t_n_field(i));
        t_n_field(i) /= n_mag;
        ++t_n_field;
      }

      MoFEMFunctionReturn(0);
    }
  };


    /// \brief Computes, for reference configuration, normal to slave face that is
  /// common to all gauss points
  struct OpGetLagrangeField : public FaceEleOp {

    boost::shared_ptr<CommonSurfaceSmoothingElement>
        commonSurfaceSmoothingElement;
    OpGetLagrangeField(
        const string lagrange_field,
        boost::shared_ptr<CommonSurfaceSmoothingElement> common_data_smoothing)
        : FaceEleOp(lagrange_field, UserDataOperator::OPCOL),
          commonSurfaceSmoothingElement(common_data_smoothing) {
          }

    MoFEMErrorCode doWork(int side, EntityType type, EntData &data) {
      MoFEMFunctionBegin;

      if (data.getFieldData().size() == 0)
        PetscFunctionReturn(0);

      int ngp = data.getN().size1();

      unsigned int nb_dofs = data.getFieldData().size() / 3;
      FTensor::Index<'i', 3> i;
      if (type == MBVERTEX) {
        // commonSurfaceSmoothingElement->lagMult->resize(3, ngp, false);
        // commonSurfaceSmoothingElement->lagMult->clear();
        commonSurfaceSmoothingElement->lagMultScalar->resize(ngp, false);
        commonSurfaceSmoothingElement->lagMultScalar->clear();
      }

      // auto t_lag_mult =
      //     getFTensor1FromMat<3>(*commonSurfaceSmoothingElement->lagMult);

      auto t_lag_mult =
      getFTensor0FromVec(*commonSurfaceSmoothingElement->lagMultScalar);


      for (unsigned int gg = 0; gg != ngp; ++gg) {
        // auto t_dof = data.getFTensor1FieldData<3>();
        // FTensor::Tensor1<double *, 3> t_dof(
        //       &data.getFieldData()[0], &data.getFieldData()[1], &data.getFieldData()[2], 3);

      FTensor::Tensor0<double *> t_dof(
                &data.getFieldData()[0]);

        FTensor::Tensor0<double *> t_base_lagrange_field(&data.getN()(gg, 0));

        for (unsigned int dd = 0; dd != nb_dofs; ++dd) {
          // t_lag_mult(i) += t_dof(i) * t_base_lagrange_field;
          t_lag_mult += t_dof * t_base_lagrange_field;
          ++t_dof;
          ++t_base_lagrange_field;
        }
        ++t_lag_mult;
      }

      MoFEMFunctionReturn(0);
    }
  };

  struct OpCalSmoothingXRhs : public FaceEleOp {

    OpCalSmoothingXRhs(
        const string field_name,
        boost::shared_ptr<CommonSurfaceSmoothingElement> common_data_smoothing)
        : FaceEleOp(field_name, UserDataOperator::OPROW),
          commonSurfaceSmoothingElement(common_data_smoothing) {}

    MoFEMErrorCode doWork(int side, EntityType type, EntData &data) {
      MoFEMFunctionBegin;

      const int nb_dofs = data.getIndices().size();
      if (nb_dofs) {

        const int nb_gauss_pts = data.getN().size1();
        int nb_base_fun_col = nb_dofs / 3;

        vecF.resize(nb_dofs, false);
        vecF.clear();

        const double area_m = getMeasure(); // same area in master and slave

        FTensor::Index<'i', 3> i;

        auto t_n =
            getFTensor1FromMat<3>(*commonSurfaceSmoothingElement->nOrmal);
        
        auto t_w = getFTensor0IntegrationWeight();

        for (int gg = 0; gg != nb_gauss_pts; ++gg) {

          double val_m = t_w * area_m;

          FTensor::Tensor1<FTensor::PackPtr<double *, 3>, 3> t_assemble_m{
              &vecF[0], &vecF[1], &vecF[2]};
          auto t_base = data.getFTensor0N(gg, 0);          
          for (int bbc = 0; bbc != nb_base_fun_col; ++bbc) {
            t_assemble_m(i) += (t_n(i) ) * t_base * val_m;
            
            
            // cerr <<"Vec 2   " << t_n << "\n";

            ++t_assemble_m;
            ++t_base;
          }
          ++t_n;
          
          ++t_w;
        } // for gauss points

        CHKERR VecSetValues(getSNESf(), data, &*vecF.begin(), ADD_VALUES);
      }
      MoFEMFunctionReturn(0);
    }

  private:
    boost::shared_ptr<CommonSurfaceSmoothingElement>
        commonSurfaceSmoothingElement;
    VectorDouble vecF;
  };


  struct OpCalLagMultRhs : public FaceEleOp {

    OpCalLagMultRhs(
        const string lagrange_name,
        boost::shared_ptr<CommonSurfaceSmoothingElement> common_data_smoothing)
        : FaceEleOp(lagrange_name, UserDataOperator::OPROW),
          commonSurfaceSmoothingElement(common_data_smoothing) {}

    MoFEMErrorCode doWork(int side, EntityType type, EntData &data) {
      MoFEMFunctionBegin;

      const int nb_dofs = data.getIndices().size();
      if (nb_dofs) {

        const int nb_gauss_pts = data.getN().size1();
        int nb_base_fun_col = nb_dofs / 3;

        vecF.resize(nb_dofs, false);
        vecF.clear();

        const double area_m = getMeasure(); // same area in master and slave

        FTensor::Index<'i', 3> i;

        auto t_n =
            getFTensor1FromMat<3>(*commonSurfaceSmoothingElement->nOrmal);
        auto t_n_field =
            getFTensor1FromMat<3>(*commonSurfaceSmoothingElement->nOrmalField);

        auto t_w = getFTensor0IntegrationWeight();
      auto t_area =
      getFTensor0FromVec(*commonSurfaceSmoothingElement->areaNL);
        for (int gg = 0; gg != nb_gauss_pts; ++gg) {

          double val_m = t_w * area_m;
        // double val_m = t_w * t_area;
          FTensor::Tensor1<FTensor::PackPtr<double *, 3>, 3> t_assemble_m{
              &vecF[0], &vecF[1], &vecF[2]};
          auto t_base = data.getFTensor0N(gg, 0);          
          for (int bbc = 0; bbc != nb_base_fun_col; ++bbc) {
            t_assemble_m(i) += (t_n(i) - t_n_field(i)) * t_base * val_m;
            
            // cerr <<"Vec 2   " << t_n << "\n";

            ++t_assemble_m;
            ++t_base;
          }
          // cerr << "t_n_field   " << t_n_field <<"\n";
          // cerr << "t_n   " << t_n <<"\n";
          ++t_n;
          ++t_n_field;
          ++t_w;
          ++t_area;
        } // for gauss points

        CHKERR VecSetValues(getSNESf(), data, &*vecF.begin(), ADD_VALUES);
      }
      MoFEMFunctionReturn(0);
    }

  private:
    boost::shared_ptr<CommonSurfaceSmoothingElement>
        commonSurfaceSmoothingElement;
    VectorDouble vecF;
  };

struct OpCalPositionsRhs : public FaceEleOp {

    OpCalPositionsRhs(
        const string field_name,
        boost::shared_ptr<CommonSurfaceSmoothingElement> common_data_smoothing, double cn_value)
        : FaceEleOp(field_name, UserDataOperator::OPROW),
          commonSurfaceSmoothingElement(common_data_smoothing), cnValue(cn_value) {}

    MoFEMErrorCode doWork(int side, EntityType type, EntData &data) {
      MoFEMFunctionBegin;

      const int nb_dofs = data.getIndices().size();
      if (nb_dofs) {

        const int nb_gauss_pts = data.getN().size1();
        int nb_base_fun_col = nb_dofs / 3;

        vecF.resize(nb_dofs, false);
        vecF.clear();

        const double area_m = getMeasure(); // same area in master and slave

        FTensor::Index<'i', 3> i;
        FTensor::Index<'j', 3> j;
        FTensor::Index<'k', 3> k;

        auto make_vec_der = [&](FTensor::Tensor1<double *, 2> t_N,
                                FTensor::Tensor1<double *, 3> t_1,
                                FTensor::Tensor1<double *, 3> t_2,
                                FTensor::Tensor1<double *, 3> t_normal) {
          FTensor::Tensor2<double, 3, 3> t_n;
          FTensor::Tensor2<double, 3, 3> t_n_container;
          const double n_norm = sqrt(t_normal(i) * t_normal(i));
          t_n(i, j) = 0;
          t_n_container(i, j) = 0;
          t_n_container(i, j) +=
              FTensor::levi_civita(i, j, k) * t_2(k) * t_N(0);
          t_n_container(i, j) -=
              FTensor::levi_civita(i, j, k) * t_1(k) * t_N(1);
          t_n(i, k) = t_n_container(i, k) / n_norm - t_normal(i) * t_normal(j) *
                                                         t_n_container(j, k) /
                                                         pow(n_norm, 3);
          return t_n;
        };

        auto t_1 =
            getFTensor1FromMat<3>(*commonSurfaceSmoothingElement->tAngent1);
        auto t_2 =
            getFTensor1FromMat<3>(*commonSurfaceSmoothingElement->tAngent2);
        auto t_normal =
            getFTensor1FromMat<3>(*commonSurfaceSmoothingElement->nOrmal);

        auto t_normal_field =
            getFTensor1FromMat<3>(*commonSurfaceSmoothingElement->nOrmalField);

        // auto t_lag_mult =
        //   getFTensor1FromMat<3>(*commonSurfaceSmoothingElement->lagMult);

        auto t_w = getFTensor0IntegrationWeight();
      auto t_area =
      getFTensor0FromVec(*commonSurfaceSmoothingElement->areaNL);
      auto t_ho_pos = getFTensor1FromMat<3>(
          *commonSurfaceSmoothingElement->hoGaussPointLocation);

      auto t_lag_mult =
      getFTensor0FromVec(*commonSurfaceSmoothingElement->lagMultScalar);

      for (int gg = 0; gg != nb_gauss_pts; ++gg) {

        double val_m = t_w * area_m;
        // double val_m = t_w * t_area;

        FTensor::Tensor1<FTensor::PackPtr<double *, 3>, 3> t_assemble_m{
            &vecF[0], &vecF[1], &vecF[2]};
        auto t_base = data.getFTensor0N(gg, 0);
        auto t_N = data.getFTensor1DiffN<2>(gg, 0);

        for (int bbc = 0; bbc != nb_base_fun_col; ++bbc) {
          auto t_d_n = make_vec_der(t_N, t_1, t_2, t_normal);

          // t_assemble_m(j) += (2 *(t_normal(i) - t_normal_field(i)) +
          // t_lag_mult(i)) * t_d_n(i, j) * val_m;
          const double inv_mag = 1. / sqrt(t_normal(i) * t_normal(i));
          FTensor::Tensor1<double, 3> t_help;
          // t_help(i) = inv_mag * t_normal(i) - t_normal_field(i);

          // cerr << "Check 2   " << t_help <<"\n";
          const double g_kind = 0.25 *  (inv_mag * t_normal(i) - t_normal_field(i)) * (inv_mag * t_normal(i) - t_normal_field(i));
          // t_assemble_m(j) +=
          //     (t_lag_mult + cnValue * g_kind) * (inv_mag * t_normal(i) - t_normal_field(i)) * t_d_n(i, j) * val_m;

          t_assemble_m(j) +=
              t_normal_field(i) * (t_1(i) + t_2(i)) * t_normal_field(j) * (t_N(0) + t_N(1))  * val_m;
         


          // if(type != MBVERTEX)
          //   t_assemble_m(j) += t_ho_pos(j) * t_base;

          // cerr << "t_help    " << t_help <<"\n";
          // FTensor::Tensor1<double, 3> check;
          // check(i) = inv_mag * t_normal(i) - t_normal_field(i);
          // cerr <<"inv_mag * t_normal(i) - t_normal_field(i)    "  << check <<
          // "\n"; t_assemble_m(j) -= (t_normal_field(i) - t_normal(i)) *
          // t_d_n(i, j) * val_m;

          // t_assemble_m(i) += (t_normal_field(j) * t_normal(j) - 1. )* t_base
          // * val_m;
          ++t_assemble_m;
          ++t_base;
          ++t_N;
        }
        ++t_normal;
        ++t_w;
        ++t_1;
        ++t_2;
        // ++t_lag_mult;
        ++t_area;
        ++t_normal_field;
        ++t_lag_mult;
      } // for gauss points

        CHKERR VecSetValues(getSNESf(), data, &*vecF.begin(), ADD_VALUES);
      }
      MoFEMFunctionReturn(0);
    }

  private:
    boost::shared_ptr<CommonSurfaceSmoothingElement>
        commonSurfaceSmoothingElement;
    VectorDouble vecF;
    double cnValue;
  };


struct OpCalLagMultNewRhs : public FaceEleOp {

    OpCalLagMultNewRhs(
        const string field_name,
        boost::shared_ptr<CommonSurfaceSmoothingElement> common_data_smoothing, double cn_value)
        : FaceEleOp(field_name, UserDataOperator::OPROW),
          commonSurfaceSmoothingElement(common_data_smoothing), cnValue(cn_value) {}

    MoFEMErrorCode doWork(int side, EntityType type, EntData &data) {
      MoFEMFunctionBegin;

      const int nb_dofs = data.getIndices().size();
      if (nb_dofs) {

        const int nb_gauss_pts = data.getN().size1();
        int nb_base_fun_col = nb_dofs;

        vecF.resize(nb_dofs, false);
        vecF.clear();

        const double area_m = getMeasure(); // same area in master and slave

        FTensor::Index<'i', 3> i;

        auto t_normal =
            getFTensor1FromMat<3>(*commonSurfaceSmoothingElement->nOrmal);

        auto t_normal_field =
            getFTensor1FromMat<3>(*commonSurfaceSmoothingElement->nOrmalField);

        // auto t_lag_mult =
        //   getFTensor1FromMat<3>(*commonSurfaceSmoothingElement->lagMult);

        auto t_w = getFTensor0IntegrationWeight();
      auto t_area =
      getFTensor0FromVec(*commonSurfaceSmoothingElement->areaNL);
      auto t_ho_pos = getFTensor1FromMat<3>(
          *commonSurfaceSmoothingElement->hoGaussPointLocation);

      auto t_lag_mult =
      getFTensor0FromVec(*commonSurfaceSmoothingElement->lagMultScalar);

      for (int gg = 0; gg != nb_gauss_pts; ++gg) {

        double val_m = t_w * area_m;
        // double val_m = t_w * t_area;

        auto t_base = data.getFTensor0N(gg, 0);

        for (int bbc = 0; bbc != nb_base_fun_col; ++bbc) {
          const double inv_mag = 1. / sqrt(t_normal(i) * t_normal(i));
          FTensor::Tensor1<double, 3> t_help;
          t_help(i) = inv_mag * t_normal(i) - t_normal_field(i);
          vecF[bbc] +=
              0.5 * t_base * (inv_mag * t_normal(i) - t_normal_field(i)) *
              (inv_mag * t_normal(i) - t_normal_field(i)) * val_m;

          // ++t_assemble_m;
          ++t_base;
        }
        ++t_normal;
        ++t_w;
        ++t_area;
        ++t_normal_field;
        ++t_lag_mult;
      } // for gauss points

        CHKERR VecSetValues(getSNESf(), data, &*vecF.begin(), ADD_VALUES);
      }
      MoFEMFunctionReturn(0);
    }

  private:
    boost::shared_ptr<CommonSurfaceSmoothingElement>
        commonSurfaceSmoothingElement;
    VectorDouble vecF;
    double cnValue;
  };


struct OpCalPositionsForTanRhs : public FaceEleOp {

    OpCalPositionsForTanRhs(
        const string field_name,
        boost::shared_ptr<CommonSurfaceSmoothingElement> common_data_smoothing, double cn_value)
        : FaceEleOp(field_name, UserDataOperator::OPROW),
          commonSurfaceSmoothingElement(common_data_smoothing), cnValue(cn_value) {}

    MoFEMErrorCode doWork(int side, EntityType type, EntData &data) {
      MoFEMFunctionBegin;

      const int nb_dofs = data.getIndices().size();
      if (nb_dofs) {

        const int nb_gauss_pts = data.getN().size1();
        int nb_base_fun_col = nb_dofs / 3;

        vecF.resize(nb_dofs, false);
        vecF.clear();

        const double area_m = getMeasure(); // same area in master and slave

        FTensor::Index<'i', 3> i;
        FTensor::Index<'j', 3> j;
        FTensor::Index<'k', 3> k;
        FTensor::Index<'q', 3> q;

        auto make_vec_der = [&](FTensor::Tensor1<double *, 2> t_N,
                                FTensor::Tensor1<double *, 3> t_1,
                                FTensor::Tensor1<double *, 3> t_2,
                                FTensor::Tensor1<double *, 3> t_normal) {
          FTensor::Tensor2<double, 3, 3> t_n;
          FTensor::Tensor2<double, 3, 3> t_n_container;
          const double n_norm = sqrt(t_normal(i) * t_normal(i));
          t_n(i, j) = 0;
          t_n_container(i, j) = 0;
          t_n_container(i, j) +=
              FTensor::levi_civita(i, j, k) * t_2(k) * t_N(0);
          t_n_container(i, j) -=
              FTensor::levi_civita(i, j, k) * t_1(k) * t_N(1);
          t_n(i, k) = t_n_container(i, k) / n_norm - t_normal(i) * t_normal(j) *
                                                         t_n_container(j, k) /
                                                         pow(n_norm, 3);
          return t_n;
        };

        auto make_vec_der_1 = [&](double t_N,
                                FTensor::Tensor1<double *, 3> t_1,
                                FTensor::Tensor1<double *, 3> t_2,
                                FTensor::Tensor1<double *, 3> t_normal) {
          FTensor::Tensor2<double, 3, 3> t_n;
          FTensor::Tensor2<double, 3, 3> t_n_container;
          const double n_norm = sqrt(t_normal(i) * t_normal(i));
          t_n(i, j) = 0;
          t_n_container(i, j) = 0;
          t_n_container(i, j) +=
              FTensor::levi_civita(i, j, k) * t_2(k) * t_N;
          t_n(i, k) = t_n_container(i, k) / n_norm - t_normal(i) * t_normal(j) *
                                                         t_n_container(j, k) /
                                                         pow(n_norm, 3);
          return t_n;
        };
        

        auto t_1 =
            getFTensor1FromMat<3>(*commonSurfaceSmoothingElement->tAngent1);
        auto t_2 =
            getFTensor1FromMat<3>(*commonSurfaceSmoothingElement->tAngent2);
        
        
        auto t_normal =
            getFTensor1FromMat<3>(*commonSurfaceSmoothingElement->nOrmal);

        auto t_normal_field =
            getFTensor1FromMat<3>(*commonSurfaceSmoothingElement->nOrmalField);


        // auto t_lag_mult =
        //   getFTensor1FromMat<3>(*commonSurfaceSmoothingElement->lagMult);

        auto t_w = getFTensor0IntegrationWeight();
      auto t_area =
      getFTensor0FromVec(*commonSurfaceSmoothingElement->areaNL);

      auto t_ho_pos = getFTensor1FromMat<3>(
          *commonSurfaceSmoothingElement->hoGaussPointLocation);


      for (int gg = 0; gg != nb_gauss_pts; ++gg) {

        double val_m = t_w * area_m;

        FTensor::Tensor1<FTensor::PackPtr<double *, 3>, 3> t_assemble_m{
            &vecF[0], &vecF[1], &vecF[2]};
        auto t_base = data.getFTensor0N(gg, 0);
        auto t_N = data.getFTensor1DiffN<2>(gg, 0);
        FTensor::Tensor1<double *, 3> t_dof(&data.getFieldData()[0],
                                            &data.getFieldData()[1],
                                            &data.getFieldData()[2], 3);

        for (int bbc = 0; bbc != nb_base_fun_col; ++bbc) {
          auto t_d_n = make_vec_der(t_N, t_1, t_2, t_normal);

          // t_assemble_m(j) += (2 *(t_normal(i) - t_normal_field(i)) +
          // t_lag_mult(i)) * t_d_n(i, j) * val_m;
          double mag_tan_1 = sqrt(t_1(i) * t_1(i));
          double mag_tan_2 = sqrt(t_2(i) * t_2(i));
          double mag_t_1 = sqrt(t_1(i) * t_1(i));
          constexpr auto t_kd = FTensor::Kronecker_Delta<int>();

          // t_assemble_m(q) += ( cnValue * (t_1_unit(i) - t_tan_1_field(i)) ) *
          // FTensor::levi_civita(i, j, k) * t_normal(j) * ( t_kd(k, q)/mag_t_1
          // - t_1(k) * t_1(q)/pow(mag_t_1, 3) )  * t_N(0) * val_m;
          // t_assemble_m(q) += ( cnValue * (t_2_unit(i) - t_tan_2_field(i)) ) *
          // FTensor::levi_civita(i, j, k) * t_d_n(j, q) * t_1_unit(k) * val_m;

          // t_assemble_m(q) += ( cnValue * (t_1_unit(i) - t_tan_1_field(i)) ) *
          // FTensor::levi_civita(i, j, k) * t_normal(j) * ( t_kd(k, q)/mag_t_1
          // - t_1(k) * t_1(q)/pow(mag_t_1, 3) )  * t_N(0) * val_m;
          // t_assemble_m(q) += ( cnValue * (t_2_unit(i) - t_tan_2_field(i)) ) *
          // FTensor::levi_civita(i, j, k) * t_d_n(j, q) * t_1_unit(k) * val_m;

          // t_assemble_m(j) += cnValue * ( t_normal(i) /*- t_tan_1_field(i)*/ )
          // * t_d_n(i, j) * val_m; t_assemble_m(i) += cnValue *
          // (t_tan_1_field(j) * (t_1(j) + t_2(j) )  ) * t_tan_1_field(i) *
          // (t_N(0) + t_N(1)) * val_m; t_assemble_m(k) += cnValue *
          // (t_normal(j) * t_tan_1_field(j)  - 1. ) * t_tan_1_field(i) *
          // t_d_n(i, k) * val_m;

          // t_assemble_m(j) += cnValue * t_lag_mult_1 * (t_1(j) /- t_tan_1_field(j)) * t_N(0) * val_m;
          // t_assemble_m(j) += cnValue * t_lag_mult_2 * (t_2(j) - t_tan_2_field(j)) * t_N(1) * val_m;

          // minimum change
          // t_assemble_m(j) -= cnValue * t_ho_pos(j) * t_base * val_m;


          // cerr <<"t_assemble_m " << t_assemble_m << "\n";
          // t_assemble_m(j) += cnValue * t_1(j);
          // t_assemble_m(j) += cnValue * t_N(1);

          // t_assemble_m(q) += ( cnValue * (t_2_unit(i) - t_tan_2_field(i)) ) *
          // FTensor::levi_civita(i, j, k) * t_normal(j) * ( t_kd(k, q)/mag_t_1
          // - t_1(k) * t_1(q)/pow(mag_t_1, 3) )  * t_N(0) * val_m;
          // t_assemble_m(q) += ( cnValue * (t_1_unit(i) - t_tan_1_field(i)) ) *
          // FTensor::levi_civita(i, j, k) * t_d_n(j, q) * t_1_unit(k) * val_m;

          // t_assemble_m(i) += (t_normal_field(j) * t_normal(j) - 1. )* t_base
          // * val_m;
          ++t_assemble_m;
          ++t_base;
          ++t_N;
          ++t_dof;
        }
        ++t_normal;
        ++t_w;
        ++t_1;
        ++t_2;
        ++t_area;
        ++t_normal_field;
        ++t_ho_pos;
        } // for gauss points
        
        CHKERR VecSetValues(getSNESf(), data, &*vecF.begin(), ADD_VALUES);
      }
      MoFEMFunctionReturn(0);
    }

  private:
    boost::shared_ptr<CommonSurfaceSmoothingElement>
        commonSurfaceSmoothingElement;
    VectorDouble vecF;
    double cnValue;
  };

  struct OpCalNormalFieldRhs : public FaceEleOp {

    OpCalNormalFieldRhs(
        const string field_name_h_div,
        boost::shared_ptr<CommonSurfaceSmoothingElement> common_data_smoothing, double cn_value)
        : FaceEleOp(field_name_h_div, UserDataOperator::OPROW),
          commonSurfaceSmoothingElement(common_data_smoothing), cnValue(cn_value) {
            sYmm = false; // This will make sure to loop over all entities
          }

    MoFEMErrorCode doWork(int side, EntityType type, EntData &data) {
      MoFEMFunctionBegin;

      const int nb_dofs = data.getIndices().size();
      if (nb_dofs) {

        const int nb_gauss_pts = data.getN().size1();
        int nb_base_fun_col = nb_dofs;

        vecF.resize(nb_dofs, false);
        vecF.clear();

        const double area_m = getMeasure(); // same area in master and slave

        FTensor::Index<'i', 3> i;
        FTensor::Index<'j', 3> j;

         auto t_n_field =
          getFTensor1FromMat<3>(*commonSurfaceSmoothingElement->nOrmalField);

         auto t_n =
             getFTensor1FromMat<3>(*commonSurfaceSmoothingElement->nOrmal);

         auto t_lag_mult =
             getFTensor1FromMat<3>(*commonSurfaceSmoothingElement->lagMult);

         auto t_w = getFTensor0IntegrationWeight();

         ///
         FTensor::Index<'k', 3> k;
         FTensor::Index<'q', 2> q;

         auto t_1 = getFTensor1Tangent1();
         auto t_2 = getFTensor1Tangent2();

         const double size_1 = sqrt(t_1(i) * t_1(i));
         const double size_2 = sqrt(t_2(i) * t_2(i));
         t_1(i) /= size_1;
         t_2(i) /= size_2;

         FTensor::Tensor1<double, 3> t_normal;

         t_normal(i) = FTensor::levi_civita(i, j, k) * t_1(j) * t_2(k);

         // FTensor::Tensor2<const double *, 3, 3> t_m(
         //     &t_1(0), &t_2(0), &t_normal(0),

         //     &t_1(1), &t_2(1), &t_normal(1),

         //     &t_1(2), &t_2(2), &t_normal(2),

         //     3);

         FTensor::Tensor2<const double *, 3, 3> t_m(&t_1(0), &t_1(1), &t_1(2),

                                                    &t_2(0), &t_2(1), &t_2(2),

                                                    &t_normal(0), &t_normal(1),
                                                    &t_normal(2),

                                                    3);

         double det;
         FTensor::Tensor2<double, 3, 3> t_inv_m;
         CHKERR determinantTensor3by3(t_m, det);
         CHKERR invertTensor3by3(t_m, det, t_inv_m);
         // FTensor::Tensor2<double, 3, 2> t_container_N;
         // FTensor::Tensor2<double, 3, 2> t_transformed_N;

         // auto t_1 =
         //     getFTensor1FromMat<3>(*commonSurfaceSmoothingElement->tAngent1);
         // auto t_2 =
         //     getFTensor1FromMat<3>(*commonSurfaceSmoothingElement->tAngent2);
         auto t_area =
             getFTensor0FromVec(*commonSurfaceSmoothingElement->areaNL);

         FTensor::Tensor2<FTensor::PackPtr<double *, 6>, 3, 2> t_container_N(
             &t_inv_m(0, 0), &t_inv_m(0, 1), &t_inv_m(1, 0),

             &t_inv_m(1, 1), &t_inv_m(2, 0), &t_inv_m(2, 1));

         FTensor::Tensor1<double, 3> t_transformed_N;

         for (int gg = 0; gg != nb_gauss_pts; ++gg) {
           // const double norm_1 = sqrt(t_1(i) * t_1(i));
           // const double norm_2 = sqrt(t_2(i) * t_2(i));
           // t_1(i) /= norm_1;
           // t_2(i) /= norm_2;
           auto t_N = data.getFTensor1DiffN<2>(gg, 0);
           double val_m = t_w * area_m;
           // double val_m = t_w * t_area;
           auto t_base = data.getFTensor0N(gg, 0);

           FTensor::Tensor1<FTensor::PackPtr<double *, 3>, 3> t_assemble_m{
               &vecF[0], &vecF[1], &vecF[2]};

           for (int bbc = 0; bbc != nb_base_fun_col / 3; ++bbc) {

             // t_container_N(0,0) = t_container_N(1,0) = t_container_N(2,0) =
             // t_N(0); t_container_N(2,0) = t_container_N(2,1) =
             // t_container_N(2,1) = t_N(1);
             t_transformed_N(i) = t_N(q) * t_container_N(i, q);
             // t_transformed_N(i,k) = t_inv_m(j, i) * t_container_N(j, k);
             // t_assemble_m(0) += t_transformed_N(0, 0) * t_div_normal * val_m;
             // t_assemble_m(1) += t_transformed_N(1, 1) * t_div_normal * val_m;
           
           //Divergence
            //  t_assemble_m(i) +=  t_transformed_N(i) * t_div_normal * val_m;
           
          //  cerr << cnValue <<"\n";
             t_assemble_m(i) -= (cnValue * (t_n(i) - t_n_field(i)) /*+ t_lag_mult(i)*/ ) * t_base * val_m;
             
            //  t_assemble_m(i) += (t_n_field(i) - t_n(i) )* t_base * val_m;

             

             ++t_assemble_m;
             ++t_base;
             ++t_N;
           }

           ++t_w;
           ++t_n_field;
           ++t_lag_mult;
           // ++t_1;
           // ++t_2;
           ++t_n;
           ++t_area;
        } // for gauss points

        CHKERR VecSetValues(getSNESf(), data, &*vecF.begin(), ADD_VALUES);
      }
      MoFEMFunctionReturn(0);
    }

  private:
    boost::shared_ptr<CommonSurfaceSmoothingElement>
        commonSurfaceSmoothingElement;
    VectorDouble vecF;
    double cnValue;
  };

//   struct OpCalSmoothingNormalHdivHdivLhs : public FaceEleOp {

//     OpCalSmoothingNormalHdivHdivLhs(
//         const string field_name_h_div,
//         boost::shared_ptr<CommonSurfaceSmoothingElement> common_data_smoothing)
//         : FaceEleOp(field_name_h_div, UserDataOperator::OPROWCOL),
//           commonSurfaceSmoothingElement(common_data_smoothing) {
//             sYmm = false; // This will make sure to loop over all entities
//           }

//   MoFEMErrorCode doWork(
//     int row_side, int col_side, EntityType row_type, EntityType col_type,
//     EntData &row_data, EntData &col_data) {
//   MoFEMFunctionBegin;

//   const int nb_row = row_data.getIndices().size();
//   const int nb_col = col_data.getIndices().size();

//   if (nb_row && nb_col) {

//     const int nb_gauss_pts = row_data.getN().size1();
//     int nb_base_fun_row = row_data.getFieldData().size() / 2;
//     int nb_base_fun_col = col_data.getFieldData().size() / 2;

//     const double area = getMeasure();

//     NN.resize(2 * nb_base_fun_row, 2 * nb_base_fun_col, false);
//     NN.clear();

//     auto get_tensor_from_mat = [](MatrixDouble &m, const int r, const int c) {
//       return FTensor::Tensor2<double *, 2, 2>(&m(r + 0, c + 0), &m(r + 0, c + 1),
//                                            &m(r + 1, c + 0), &m(r + 1, c + 1));
//     };

//         FTensor::Index<'i', 3> i;
//         FTensor::Index<'j', 3> j;
//         FTensor::Index<'k', 3> k;
//         FTensor::Index<'m', 3> m;
//         FTensor::Index<'l', 3> l;
//         FTensor::Index<'q', 2> q;


// ///////


//         // FTensor::Index<'i', 3> i;
//         // FTensor::Index<'j', 3> j;
//         // FTensor::Index<'k', 3> k;
//         // FTensor::Index<'m', 3> m;
//         // FTensor::Index<'l', 3> l;
//         // FTensor::Index<'q', 2> q;
//         // auto t_divergence_n_h_div =
//         //     getFTensor0FromVec(*commonSurfaceSmoothingElement->divNormalHdiv);

//         // auto t_w = getFTensor0IntegrationWeight();

//         // auto t_1 = getFTensor1Tangent1();        
//         // auto t_2 = getFTensor1Tangent2();        

//         // const double size_1 =  sqrt(t_1(i) * t_1(i));
//         // const double size_2 =  sqrt(t_2(i) * t_2(i));
//         // t_1(i) /= size_1;
//         // t_2(i) /= size_2;

//         // auto t_t_1_div =
//         //     getFTensor1FromMat<3>(*commonSurfaceSmoothingElement->tanHdiv1);

//         // auto t_t_2_div =
//         //     getFTensor1FromMat<3>(*commonSurfaceSmoothingElement->tanHdiv2);

//         // FTensor::Tensor2<double, 3, 2> t_div_1, t_div_2;
//         // FTensor::Tensor2<double, 3, 3> t_mat_1, t_mat_2;

//         // for (int gg = 0; gg != nb_gauss_pts; ++gg) {

//         //   double val_m = t_w * area_m;

//         //   auto t_diff_base_fun = data.getFTensor2DiffN<3, 2>();
//         //   FTensor::Tensor1<FTensor::PackPtr<double *, 2>, 2> t_assemble_m{
//         //   &vecF[0], &vecF[1]};

//         //   for (int bbc = 0; bbc != nb_base_fun_col / 2; ++bbc) {
//         //     // vecF[bbc] += t_row_diff_base(i, i) * t_divergence_n_h_div * val_m;
//         //     t_mat_1(i, j) = FTensor::levi_civita(i, j, k) * t_t_1_div(k);
//         //     t_div_1(i, q) =
//         //         t_mat_1(i, j) * t_diff_base_fun(m, q) * t_1(m) * t_1(j);
//         //     t_assemble_m(0) +=
//         //         (t_div_1(0, 0) + t_div_1(1, 1)) * t_divergence_n_h_div * val_m;
//         //           t_mat_1(i, j) = FTensor::levi_civita(i, j, k) * t_t_1_div(k);

// ///////

        
//         auto t_divergence_n_h_div =
//             getFTensor0FromVec(*commonSurfaceSmoothingElement->divNormalHdiv);

//         auto t_1 = getFTensor1Tangent1();        
//         auto t_2 = getFTensor1Tangent2();        

//         const double size_1 =  sqrt(t_1(i) * t_1(i));
//         const double size_2 =  sqrt(t_2(i) * t_2(i));
//         t_1(i) /= size_1;
//         t_2(i) /= size_2;

//         FTensor::Tensor2<double, 3, 2> t_div_1_row, t_div_1_col, t_div_2_row, t_div_2_col;
//         FTensor::Tensor2<double, 3, 3> t_mat_1, t_mat_2;


//     auto t_w = getFTensor0IntegrationWeight();
//     for (int gg = 0; gg != nb_gauss_pts; ++gg) {

//       const double val_m = t_w * area;
//       auto t_row_diff_base = row_data.getFTensor2DiffN<3, 2>();
//       auto t_base_row = row_data.getFTensor1N<3>(gg, 0);

//       for (int bbr = 0; bbr != nb_base_fun_row; ++bbr) {

//         auto t_col_diff_base = col_data.getFTensor2DiffN<3, 2>();
//         auto t_base_col = col_data.getFTensor1N<3>(gg, 0);

//         for (int bbc = 0; bbc != nb_base_fun_col; ++bbc) {

//         auto t_assemble_m = get_tensor_from_mat(NN, 2 * bbr, 2 * bbc);
          
//           t_mat_1(i, j) = FTensor::levi_civita(i, j, k) * t_t_2_div(k);
//           t_div_1_row(i, q) =
//           t_mat_1(i, j) * t_row_diff_base(m, q) * t_1(m) * t_1(j);

//           t_mat_1(i, k) =
//               FTensor::levi_civita(i, j, k) * (t_base_row(m) * t_1(m) * t_1(j));
//           t_div_1_row(i, q) += t_mat_1(i, k) * t_div_tan_2(k, q);

//           t_mat_1(i, j) = FTensor::levi_civita(i, j, k) * t_t_2_div(k);
//           t_div_1_col(i, q) =
//           t_mat_1(i, j) * t_col_diff_base(m, q) * t_1(m) * t_1(j);

//           t_mat_1(i, k) =
//               FTensor::levi_civita(i, j, k) * (t_base_col(m) * t_1(m) * t_1(j));
//           t_div_1_col(i, q) += t_mat_1(i, k) * t_div_tan_2(k, q);

//           t_assemble_m(0, 0) +=
//                 (t_div_1_row(0, 0) + t_div_1_row(1, 1)) * (t_div_1_col(0, 0) + t_div_1_col(1, 1)) * val_m;

//             t_mat_2(i, k) = FTensor::levi_civita(i, j, k) * t_t_1_div(j);
//             t_div_2_row(i, q) =
//                 t_mat_2(i, k) * t_row_diff_base(m, q) * t_2(m) * t_2(k);
            
//             t_mat_2(i, j) = FTensor::levi_civita(i, j, k) * (t_base_row(m) * t_2(m) * t_2(k));
//             t_div_2_row(i, q) +=
//                 t_mat_2(i, j) * t_div_tan_1(j,q);

//             t_mat_2(i, k) = FTensor::levi_civita(i, j, k) * t_t_1_div(j);
//             t_div_2_col(i, q) =
//                 t_mat_2(i, k) * t_col_diff_base(m, q) * t_2(m) * t_2(k);
            
//             t_mat_2(i, j) = FTensor::levi_civita(i, j, k) * (t_base_col(m) * t_2(m) * t_2(k));
//             t_div_2_col(i, q) +=
//                 t_mat_2(i, j) * t_div_tan_1(j,q);

//             t_assemble_m(1, 1) +=
//                 (t_div_2_row(0, 0) + t_div_2_row(1, 1)) * (t_div_2_col(0, 0) + t_div_2_col(1, 1)) * val_m;


//           ++t_col_diff_base; // update cols
//           ++t_base_col;
//         }
//         ++t_row_diff_base; // update rows
//         ++t_base_row;
//       }
//       ++t_t_1_div;
//       ++t_t_2_div;
//       ++t_div_tan_1;
//       ++t_div_tan_2;
//       ++t_divergence_n_h_div;
//       ++t_w;
//     }

//     CHKERR MatSetValues(getSNESB(), row_data, col_data, &*NN.data().begin(),
//                         ADD_VALUES);
//   }

//   MoFEMFunctionReturn(0);
// }

//  private:
//     boost::shared_ptr<CommonSurfaceSmoothingElement>
//         commonSurfaceSmoothingElement;
//     MatrixDouble NN;
//   };

// struct OpCalSmoothingX_dnLhs : public FaceEleOp {

//     OpCalSmoothingX_dnLhs(
//         const string field_name_position, const string field_name_h_div,
//         boost::shared_ptr<CommonSurfaceSmoothingElement> common_data_smoothing)
//         : FaceEleOp(field_name_position, field_name_h_div, UserDataOperator::OPROWCOL),
//           commonSurfaceSmoothingElement(common_data_smoothing) {
//             sYmm = false; // This will make sure to loop over all entities
//           }

//   MoFEMErrorCode doWork(
//     int row_side, int col_side, EntityType row_type, EntityType col_type,
//     EntData &row_data, EntData &col_data) {
//   MoFEMFunctionBegin;

//   const int nb_row = row_data.getIndices().size();
//   const int nb_col = col_data.getIndices().size();

//   if (nb_row && nb_col) {

//     const int nb_gauss_pts = row_data.getN().size1();
//     int nb_base_fun_row = row_data.getFieldData().size() / 3;
//     int nb_base_fun_col = col_data.getFieldData().size() / 2;

//     const double area = getMeasure();

//     NN.resize(3 * nb_base_fun_row, 2 * nb_base_fun_col, false);
//     NN.clear();

//     auto get_tensor_from_mat = [](MatrixDouble &m, const int r, const int c) {
//       return FTensor::Tensor1<double *, 3>(&m(r + 0, c), &m(r + 1, c),
//                                            &m(r + 2, c));
//     };

//         auto t_1 = getFTensor1Tangent1();        
//         auto t_2 = getFTensor1Tangent2();        
       
//     FTensor::Index<'i', 3> i;
//         FTensor::Index<'j', 3> j;
//         FTensor::Index<'k', 3> k;
//         FTensor::Index<'m', 3> m;

//         const double size_1 =  sqrt(t_1(i) * t_1(i));
//         const double size_2 =  sqrt(t_2(i) * t_2(i));
//         t_1(i) /= size_1;
//         t_2(i) /= size_2;
        
//     FTensor::Tensor2<double, 3, 3> t_mat_1, t_mat_2;
//     FTensor::Tensor1<double, 3> t_vec_1, t_vec_2;

//     auto t_w = getFTensor0IntegrationWeight();
//     for (int gg = 0; gg != nb_gauss_pts; ++gg) {
     
//       const double val_m = t_w * area;

//       auto t_base_row_X = row_data.getFTensor0N(gg, 0);
//       for (int bbr = 0; bbr != nb_base_fun_row; ++bbr) {

//         // auto t_base_col_hdiv = col_data.getFTensor1N<3>(gg, 0);
//         const double m = val_m * t_base_row_X;

//         for (int bbc = 0; bbc != nb_base_fun_col; ++bbc) {
//           auto t_assemble_m = get_tensor_from_mat(NN, 3 * bbr, 2 * bbc);
//           t_mat_1(i, j) = FTensor::levi_civita(i, j, k) * t_t_1_div(k);
//           //  t_vec_1(j) = t_base_col_hdiv(m) * t_1(m) * t_1(j); 
//           t_assemble_m(i) -=  m * t_mat_1(i, j) * t_vec_1(j);

//           auto t_assemble_m_2 = get_tensor_from_mat(NN, 3 * bbr, 2 * bbc + 1);
//           t_mat_2(i, j) = FTensor::levi_civita(i, j, k) * t_t_2_div(k);
//           //  t_vec_2(j) = t_base_col_hdiv(m) * t_2(m) * t_2(j); 
//           t_assemble_m_2(i) -=  m * t_mat_2(i, j) * t_vec_2(j);

//           // ++t_base_col_hdiv; // update rows
//         }
//         ++t_base_row_X; // update cols master
//       }
//       ++t_w;
//     }

//     CHKERR MatSetValues(getSNESB(), row_data, col_data, &*NN.data().begin(),
//                         ADD_VALUES);
//   }

//   MoFEMFunctionReturn(0);
// }
// private:
//     boost::shared_ptr<CommonSurfaceSmoothingElement>
//         commonSurfaceSmoothingElement;
//     MatrixDouble NN;
// };


struct OpCalSmoothingX_dXLhs : public FaceEleOp {

    OpCalSmoothingX_dXLhs(
        const string field_name_position,
        boost::shared_ptr<CommonSurfaceSmoothingElement> common_data_smoothing, double cn_value)
        : FaceEleOp(field_name_position, UserDataOperator::OPROWCOL),
          commonSurfaceSmoothingElement(common_data_smoothing), cNvalue(cn_value) {
            sYmm = false;
          }


  MoFEMErrorCode doWork(
    int row_side, int col_side, EntityType row_type, EntityType col_type,
    EntData &row_data, EntData &col_data) {

  MoFEMFunctionBegin;


  const int row_nb_dofs = row_data.getIndices().size();
  if (!row_nb_dofs)
    MoFEMFunctionReturnHot(0);
  const int col_nb_dofs = col_data.getIndices().size();
  if (!col_nb_dofs)
    MoFEMFunctionReturnHot(0);
  const int nb_gauss_pts = row_data.getN().size1();

  const int nb_base_fun_row = row_data.getFieldData().size() / 3;
  const int nb_base_fun_col = col_data.getFieldData().size() / 3;

  NN.resize(3 * nb_base_fun_row, 3 * nb_base_fun_col, false);
  NN.clear();

  FTensor::Index<'i', 3> i;
  FTensor::Index<'j', 3> j;
  FTensor::Index<'k', 3> k;
  FTensor::Index<'m', 3> m;
  FTensor::Index<'l', 3> l;

  auto get_tensor2 = [](MatrixDouble &m, const int r, const int c) {
    return FTensor::Tensor2<double *, 3, 3>(
        &m(r + 0, c + 0), &m(r + 0, c + 1), &m(r + 0, c + 2), &m(r + 1, c + 0),
        &m(r + 1, c + 1), &m(r + 1, c + 2), &m(r + 2, c + 0), &m(r + 2, c + 1),
        &m(r + 2, c + 2));
  };

  auto make_vec_der = [&](FTensor::Tensor1<double *, 2> t_N,
                          FTensor::Tensor1<double *, 3> t_1,
                          FTensor::Tensor1<double *, 3> t_2,
                          FTensor::Tensor1<double *, 3> t_normal) {
    FTensor::Tensor2<double, 3, 3> t_n;
    FTensor::Tensor2<double, 3, 3> t_n_container;
    const double n_norm = sqrt(t_normal(i) * t_normal(i));
    t_n(i, j) = 0;
    t_n_container(i, j) = 0;
    t_n_container(i, j) += FTensor::levi_civita(i, j, k) * t_2(k) * t_N(0);
    t_n_container(i, j) -= FTensor::levi_civita(i, j, k) * t_1(k) * t_N(1);
    t_n(i, k) = t_n_container(i, k) / n_norm - t_normal(i) * t_normal(j) *
                                                   t_n_container(j, k) /
                                                   pow(n_norm, 3);
    return t_n;
  };

  auto make_vec_der_der = [&](FTensor::Tensor1<double *, 2> t_N_row,
                              FTensor::Tensor1<double *, 2> t_N_col,
                              FTensor::Tensor1<double *, 3> t_1,
                              FTensor::Tensor1<double *, 3> t_2,
                              FTensor::Tensor1<double *, 3> t_normal) {
    FTensor::Tensor3<double, 3, 3, 3> t_result;
    FTensor::Tensor3<double, 3, 3, 3> t_n_container;
    FTensor::Tensor2<double, 3, 3> t_der_col;
    FTensor::Tensor2<double, 3, 3> t_der_row;
    const double n_norm = sqrt(t_normal(i) * t_normal(i));
    const double n_norm_3 = n_norm * n_norm * n_norm;
    const double n_norm_5 = n_norm_3 * n_norm * n_norm;
    const double inv_1 = 1. / n_norm;
    const double inv_3 = 1. / n_norm_3;
    const double inv_5 = 1. / n_norm_5;
    t_result(i, j, k) = 0;
    t_der_col(i, j) = 0;
    t_der_row(i, j) = 0;
    t_der_col(i, j) += FTensor::levi_civita(i, j, k) * t_2(k) * t_N_col(0);
    t_der_col(i, j) -= FTensor::levi_civita(i, j, k) * t_1(k) * t_N_col(1);
    t_der_row(i, j) += FTensor::levi_civita(i, j, k) * t_2(k) * t_N_row(0);
    t_der_row(i, j) -= FTensor::levi_civita(i, j, k) * t_1(k) * t_N_row(1);

    t_n_container(i, j, k) = 0;
    t_n_container(i, j, k) += FTensor::levi_civita(i, j, k) * t_N_col(1) * t_N_row(0);
    t_n_container(i, j, k) -= FTensor::levi_civita(i, j, k) * t_N_col(0) * t_N_row(1);
    t_result(i, j, k) += inv_1 * t_n_container(i, j, k);
    t_result(i, j, k) -= inv_3 * t_der_row(i, j) * t_der_col(m, k) * t_normal(m);
    t_result(i, j, k) -= inv_3 * t_der_col(i, k) * t_der_row(m, j) * t_normal(m);
    t_result(i, j, k) -= inv_3 * t_normal(i) * t_normal(m) * t_n_container(m, j, k);
    t_result(i, j, k) -= inv_3 * t_normal(i) * t_der_row(m, j) * t_der_col(m, k);
    t_result(i, j, k) += 3. * inv_5 * t_normal(i) * t_normal(m) * t_der_row(m, j) * t_normal(l) * t_der_col(l, k) ;
    // cerr << "t_result     " << t_result <<"\n";
    return t_result;
  };

  const double area = getMeasure();
  auto t_w = getFTensor0IntegrationWeight();
  auto t_1 = getFTensor1FromMat<3>(*commonSurfaceSmoothingElement->tAngent1);
  auto t_2 = getFTensor1FromMat<3>(*commonSurfaceSmoothingElement->tAngent2);
  auto t_normal = getFTensor1FromMat<3>(*commonSurfaceSmoothingElement->nOrmal);
  auto t_normal_field =
      getFTensor1FromMat<3>(*commonSurfaceSmoothingElement->nOrmalField);
  auto t_area = getFTensor0FromVec(*commonSurfaceSmoothingElement->areaNL);
  auto t_lag_mult =
      getFTensor0FromVec(*commonSurfaceSmoothingElement->lagMultScalar);

  for (int gg = 0; gg != nb_gauss_pts; ++gg) {

    double val = t_w * area;
    // double val = t_w * t_area;

    auto t_N_col = col_data.getFTensor1DiffN<2>(gg, 0);

    int bbc = 0;
    for (; bbc != nb_base_fun_col; ++bbc) {

      auto t_N_row = row_data.getFTensor1DiffN<2>(gg, 0);
      auto t_d_n_col = make_vec_der(t_N_col, t_1, t_2, t_normal);

      int bbr = 0;
      for (; bbr != nb_base_fun_row; ++bbr) {

        auto t_d_n_row = make_vec_der(t_N_row, t_1, t_2, t_normal);

        auto t_assemble = get_tensor2(NN, 3 * bbr, 3 * bbc);
        // TODO: handle hoGeometry
        auto t_d_n_row_col = make_vec_der_der(t_N_row, t_N_col, t_1, t_2, t_normal);
        // t_assemble(j, k) +=
        //     t_lag_mult * val * t_d_n_row(i, j) * t_d_n_col(i, k);

        const double inv_mag = 1./ sqrt(t_normal(i) * t_normal(i)) ;
            FTensor::Tensor1<double, 3> t_help;
            t_help(i) = t_normal_field(i) * t_normal_field(i);
                    // cerr << "t_help " << t_help << "\n";
        
        const double g_kind = 0.25 * (inv_mag * t_normal(i) - t_normal_field(i)) * (inv_mag * t_normal(i) - t_normal_field(i));

        // t_assemble(j, k) +=
        //    (t_lag_mult + g_kind * cNvalue) * val * (inv_mag * t_normal(i) - t_normal_field(i)) * t_d_n_row_col(i, j, k);

        t_assemble(j, k) +=
           val * (t_N_row(0) + t_N_row(1)) * (t_N_col(0) + t_N_col(1) ) * t_normal_field(j) * t_normal_field(k);


        FTensor::Tensor2<double, 3, 3> t_n_container;
        const double n_norm = sqrt(t_normal(i) * t_normal(i));
        t_n_container(i, j) = 0;
        t_n_container(i, j) +=
            FTensor::levi_civita(i, j, k) * t_2(k) * t_N_col(0);
        t_n_container(i, j) -=
            FTensor::levi_civita(i, j, k) * t_1(k) * t_N_col(1);

        // t_assemble(j, k) +=
        //      cNvalue * (t_n_container(m, k) * (inv_mag * t_normal(m) - t_normal_field(m))) * val * (inv_mag * t_normal(i) - t_normal_field(i)) * t_d_n_row(i, j);


        // t_assemble(j, k) += 0.5 * t_w *
        //                     (inv_mag * t_normal(i) - t_normal_field(i)) *
        //                     t_d_n_row(i, j) * t_n_container(l, k) * t_normal(l) / pow(n_norm, 3);
        // t_assemble(i, k) += t_d_n_col(i, k);

        ++t_N_row;
      }
      ++t_N_col;
    }
    ++t_w;
    ++t_1;
    ++t_2;
    ++t_normal;
    ++t_normal_field;
    ++t_area;
    ++t_lag_mult;
  }

CHKERR MatSetValues(getSNESB(), row_data, col_data, &*NN.data().begin(),
                        ADD_VALUES);

  MoFEMFunctionReturn(0);
}
private:
    boost::shared_ptr<CommonSurfaceSmoothingElement>
        commonSurfaceSmoothingElement;
    MatrixDouble NN;
    double cNvalue;
};

struct OpCalSmoothingX_dLambdaLhs : public FaceEleOp {

    OpCalSmoothingX_dLambdaLhs(
        const string field_name_position, const string field_lagmult,
        boost::shared_ptr<CommonSurfaceSmoothingElement> common_data_smoothing, double cn_value)
        : FaceEleOp(field_name_position, field_lagmult, UserDataOperator::OPROWCOL),
          commonSurfaceSmoothingElement(common_data_smoothing), cNvalue(cn_value) {
            sYmm = false;
          }


  MoFEMErrorCode doWork(
    int row_side, int col_side, EntityType row_type, EntityType col_type,
    EntData &row_data, EntData &col_data) {

  MoFEMFunctionBegin;


  const int row_nb_dofs = row_data.getIndices().size();
  if (!row_nb_dofs)
    MoFEMFunctionReturnHot(0);
  const int col_nb_dofs = col_data.getIndices().size();
  if (!col_nb_dofs)
    MoFEMFunctionReturnHot(0);
  const int nb_gauss_pts = row_data.getN().size1();

  const int nb_base_fun_row = row_data.getFieldData().size() / 3;
  const int nb_base_fun_col = col_data.getFieldData().size();

  NN.resize(3 * nb_base_fun_row, nb_base_fun_col, false);
  NN.clear();
  transNN.resize(nb_base_fun_col, 3 * nb_base_fun_row, false);
  transNN.clear();

  FTensor::Index<'i', 3> i;
  FTensor::Index<'j', 3> j;
  FTensor::Index<'k', 3> k;
  FTensor::Index<'m', 3> m;
  FTensor::Index<'l', 3> l;

  auto make_vec_der = [&](FTensor::Tensor1<double *, 2> t_N,
                          FTensor::Tensor1<double *, 3> t_1,
                          FTensor::Tensor1<double *, 3> t_2,
                          FTensor::Tensor1<double *, 3> t_normal) {
    FTensor::Tensor2<double, 3, 3> t_n;
    FTensor::Tensor2<double, 3, 3> t_n_container;
    const double n_norm = sqrt(t_normal(i) * t_normal(i));
    t_n(i, j) = 0;
    t_n_container(i, j) = 0;
    t_n_container(i, j) += FTensor::levi_civita(i, j, k) * t_2(k) * t_N(0);
    t_n_container(i, j) -= FTensor::levi_civita(i, j, k) * t_1(k) * t_N(1);
    t_n(i, k) = t_n_container(i, k) / n_norm - t_normal(i) * t_normal(j) *
                                                   t_n_container(j, k) /
                                                   pow(n_norm, 3);
    return t_n;
  };

  const double area = getMeasure();
  auto t_w = getFTensor0IntegrationWeight();
  auto t_1 = getFTensor1FromMat<3>(*commonSurfaceSmoothingElement->tAngent1);
  auto t_2 = getFTensor1FromMat<3>(*commonSurfaceSmoothingElement->tAngent2);
  auto t_normal = getFTensor1FromMat<3>(*commonSurfaceSmoothingElement->nOrmal);
  auto t_normal_field =
      getFTensor1FromMat<3>(*commonSurfaceSmoothingElement->nOrmalField);
  auto t_area = getFTensor0FromVec(*commonSurfaceSmoothingElement->areaNL);

  auto get_tensor_from_mat = [](MatrixDouble &m, const int r, const int c) {
    return FTensor::Tensor1<double *, 3>(&m(r + 0, c), &m(r + 1, c),
                                         &m(r + 2, c));
  };

  for (int gg = 0; gg != nb_gauss_pts; ++gg) {

    double val = t_w * area;
    // double val = t_w * t_area;
    auto t_N_row = row_data.getFTensor1DiffN<2>(gg, 0);

    int bbr = 0;
    for (; bbr != nb_base_fun_row; ++bbr) {
      auto t_base_lambda = col_data.getFTensor0N(gg, 0);

       auto t_d_n_row = make_vec_der(t_N_row, t_1, t_2, t_normal);
      auto t_assemble_m = get_tensor_from_mat(NN, 3 * bbr, 0); 
         int bbc = 0;
    for (; bbc != nb_base_fun_col; ++bbc) {
        
        const double inv_mag = 1./ sqrt(t_normal(i) * t_normal(i)) ;

        t_assemble_m(j) +=
              t_base_lambda * (inv_mag * t_normal(i) - t_normal_field(i)) * t_d_n_row(i, j) * val;
      ++t_assemble_m;
        ++t_base_lambda;
      }
      ++t_N_row;
    }
    ++t_w;
    ++t_1;
    ++t_2;
    ++t_normal;
    ++t_normal_field;
    ++t_area;
  }

CHKERR MatSetValues(getSNESB(), row_data, col_data, &*NN.data().begin(),
                        ADD_VALUES);
noalias(transNN) = trans(NN);
CHKERR MatSetValues(getSNESB(), col_data, row_data, &*transNN.data().begin(),
                        ADD_VALUES);


  MoFEMFunctionReturn(0);
}
private:
    boost::shared_ptr<CommonSurfaceSmoothingElement>
        commonSurfaceSmoothingElement;
    MatrixDouble NN;
    MatrixDouble transNN;
    double cNvalue;
};


  MoFEMErrorCode setSmoothFaceOperatorsRhs(
      boost::shared_ptr<SurfaceSmoothingElement> fe_rhs_smooth_element,
      boost::shared_ptr<CommonSurfaceSmoothingElement>
          common_data_smooth_element,
      string field_name_position, string field_name_normal_field, string lagrange_field_name) {
    MoFEMFunctionBegin;

    fe_rhs_smooth_element->getOpPtrVector().push_back(
        new OpGetTangentForSmoothSurf(field_name_position,
                                      common_data_smooth_element));
    fe_rhs_smooth_element->getOpPtrVector().push_back(
        new OpGetNormalForSmoothSurf(field_name_position,
                                     common_data_smooth_element));
    fe_rhs_smooth_element->getOpPtrVector().push_back(new OpGetNormalField(
        field_name_normal_field, common_data_smooth_element));

    fe_rhs_smooth_element->getOpPtrVector().push_back(new OpGetNormalisedNormalField(
        field_name_normal_field, common_data_smooth_element));

    fe_rhs_smooth_element->getOpPtrVector().push_back(new OpGetLagrangeField(
        lagrange_field_name, common_data_smooth_element));


    fe_rhs_smooth_element->getOpPtrVector().push_back(new OpCalPositionsRhs(
        field_name_position, common_data_smooth_element, cnVaule));

    // fe_rhs_smooth_element->getOpPtrVector().push_back(new OpCalLagMultNewRhs(
    //     lagrange_field_name, common_data_smooth_element, cnVaule));

        



/*    fe_rhs_smooth_element->getOpPtrVector().push_back(new OpGetTangentOneField(
        field_name_tangent_one_field, common_data_smooth_element));

    fe_rhs_smooth_element->getOpPtrVector().push_back(new OpGetTangentTwoField(
        field_name_tangent_two_field, common_data_smooth_element));

     fe_rhs_smooth_element->getOpPtrVector().push_back(new OpCalNormalFieldTwo(
         field_name_tangent_one_field, common_data_smooth_element));
         
     fe_rhs_smooth_element->getOpPtrVector().push_back(new OpCalMetricTensorAndDeterminant(
         field_name_tangent_one_field, common_data_smooth_element));

     fe_rhs_smooth_element->getOpPtrVector().push_back(new OpCalPrincipalCurvatures(
         field_name_tangent_one_field, common_data_smooth_element));

     fe_rhs_smooth_element->getOpPtrVector().push_back(new OpGetLagrangeField1(
         lagrange_field_name_1, common_data_smooth_element));

     fe_rhs_smooth_element->getOpPtrVector().push_back(new OpGetLagrangeField2(
         lagrange_field_name_2, common_data_smooth_element));

     /// New
     fe_rhs_smooth_element->getOpPtrVector().push_back(new OpGetDivOneField(
         field_name_tangent_one_field, common_data_smooth_element));

     // fe_rhs_smooth_element->getOpPtrVector().push_back(new OpGetDivTwoField(
     //     field_name_tangent_two_field, common_data_smooth_element));

     // fe_rhs_smooth_element->getOpPtrVector().push_back(new
     // OpCalNormalFieldRhs(
     //     field_name_normal_field, common_data_smooth_element, cnVaule));

     //~~~~~~~~~~~~
     fe_rhs_smooth_element->getOpPtrVector().push_back(
         new OpCalTangentOneFieldRhs(field_name_tangent_one_field,
                                     common_data_smooth_element, cnVaule));

     fe_rhs_smooth_element->getOpPtrVector().push_back(
         new OpCalTangentTwoFieldRhs(field_name_tangent_two_field,
                                     common_data_smooth_element, cnVaule));


     fe_rhs_smooth_element->getOpPtrVector().push_back(
         new OpCalLagMultOneFieldRhs(lagrange_field_name_1,
                                     common_data_smooth_element, cnVaule));


     fe_rhs_smooth_element->getOpPtrVector().push_back(
         new OpCalLagMultTwoFieldRhs(lagrange_field_name_2,
                                     common_data_smooth_element, cnVaule));
     //~~~~~~~~

     // fe_rhs_smooth_element->getOpPtrVector().push_back(
     //     new OpGetNormalFT(field_name_tangent_one_field,
     //                                  common_data_smooth_element));

     // fe_rhs_smooth_element->getOpPtrVector().push_back(
     //     new OpCalLagMultRhs(lagrange_field_name,
     //     common_data_smooth_element));

     fe_rhs_smooth_element->getOpPtrVector().push_back(
         new OpCalPositionsForTanRhs(field_name_position,
                                     common_data_smooth_element, cnVaule));

     /////
     // fe_rhs_smooth_element->getOpPtrVector().push_back(
     //     new OpGetTangentHdivAtGaussPoints(field_name_normal_hdiv,
     //                                      common_data_smooth_element));

     // fe_rhs_smooth_element->getOpPtrVector().push_back(
     //     new OpGetNormalHdivAtGaussPoints(field_name_normal_hdiv,
     //                                      common_data_smooth_element));

     // Rhs
     // fe_rhs_smooth_element->getOpPtrVector().push_back(
     //     new OpCalSmoothingNormalHdivRhs(field_name_normal_hdiv,
     //                                     common_data_smooth_element));
     // fe_rhs_smooth_element->getOpPtrVector().push_back(new
     // OpCalSmoothingXRhs(
     //     field_name_position, common_data_smooth_element));
*/
     MoFEMFunctionReturn(0);
  }



 MoFEMErrorCode setSmoothFaceOperatorsLhs(
      boost::shared_ptr<SurfaceSmoothingElement> fe_lhs_smooth_element,
      boost::shared_ptr<CommonSurfaceSmoothingElement>
          common_data_smooth_element,
      string field_name_position, string field_name_normal_field, string lagrange_field_name) {
    MoFEMFunctionBegin;

    fe_lhs_smooth_element->getOpPtrVector().push_back(
        new OpGetTangentForSmoothSurf(field_name_position,
                                      common_data_smooth_element));
    fe_lhs_smooth_element->getOpPtrVector().push_back(
        new OpGetNormalForSmoothSurf(field_name_position,
                                     common_data_smooth_element));
    fe_lhs_smooth_element->getOpPtrVector().push_back(new OpGetNormalField(
        field_name_normal_field, common_data_smooth_element));

    fe_lhs_smooth_element->getOpPtrVector().push_back(
        new OpGetNormalisedNormalField(field_name_normal_field,
                                       common_data_smooth_element));

    fe_lhs_smooth_element->getOpPtrVector().push_back(new OpGetLagrangeField(
        lagrange_field_name, common_data_smooth_element));

    fe_lhs_smooth_element->getOpPtrVector().push_back(new OpCalSmoothingX_dXLhs(
        field_name_position, common_data_smooth_element, cnVaule));

    // fe_lhs_smooth_element->getOpPtrVector().push_back(new OpCalSmoothingX_dLambdaLhs(
    //     field_name_position, lagrange_field_name, common_data_smooth_element, cnVaule));

     MoFEMFunctionReturn(0);
  }


  MoFEMErrorCode setSmoothFacePostProc(
      boost::shared_ptr<SurfaceSmoothingElement> fe_smooth_post_proc,
      boost::shared_ptr<CommonSurfaceSmoothingElement>
          common_data_smooth_element,
      string field_name_position, string field_name_normal_field) {
    MoFEMFunctionBegin;
    
    fe_smooth_post_proc->getOpPtrVector().push_back(
        new OpGetTangentForSmoothSurf(field_name_position,
                                      common_data_smooth_element, true));

    fe_smooth_post_proc->getOpPtrVector().push_back(
        new OpPrintRadius(field_name_position,
                                      common_data_smooth_element, true));



    /*fe_smooth_post_proc->getOpPtrVector().push_back(
        new OpGetNormalForSmoothSurf(field_name_position,
                                     common_data_smooth_element));

    fe_smooth_post_proc->getOpPtrVector().push_back(new OpGetTangentOneField(
        field_name_tangent_one_field, common_data_smooth_element));

    fe_smooth_post_proc->getOpPtrVector().push_back(new OpGetTangentTwoField(
        field_name_tangent_two_field, common_data_smooth_element));
    // fe_smooth_post_proc->getOpPtrVector().push_back(new OpGetNormalField(
    //     field_name_normal_field, common_data_smooth_element));

    fe_smooth_post_proc->getOpPtrVector().push_back(new OpPrintNormals(
        field_name_position, common_data_smooth_element));

        fe_smooth_post_proc->getOpPtrVector().push_back(new OpGetLagrangeField1(
         "LAGMULT_1", common_data_smooth_element, true));

     fe_smooth_post_proc->getOpPtrVector().push_back(new OpGetLagrangeField2(
         "LAGMULT_2", common_data_smooth_element, true));

*/
    MoFEMFunctionReturn(0);
  }

  MoFEMErrorCode setGlobalVolumeEvaluator(
      boost::shared_ptr<VolumeSmoothingElement> fe_smooth_volume_element,
      string field_name_position, Vec volume_vec) {
    MoFEMFunctionBegin;

    fe_smooth_volume_element->getOpPtrVector().push_back(
        new VolumeCalculation(field_name_position, volume_vec));

    MoFEMFunctionReturn(0);
  }


  MoFEM::Interface &mField;
  ArbitrarySmoothingProblem(MoFEM::Interface &m_field, double c_n) : mField(m_field), cnVaule(c_n) {}
};

// double MortarContactProblem::LoadScale::lAmbda = 1;
int main(int argc, char *argv[]) {

  const string default_options = "-ksp_type fgmres \n"
                                 "-pc_type lu \n"
                                 "-pc_factor_mat_solver_type mumps \n"
                                 "-snes_type newtonls \n"
                                 "-snes_linesearch_type basic \n"
                                 "-snes_divergence_tolerance 0 \n"
                                 "-snes_max_it 50 \n"
                                 "-snes_atol 1e-8 \n"
                                 "-snes_rtol 1e-10 \n"
                                 "-snes_monitor \n"
                                 "-ksp_monitor \n"
                                 "-snes_converged_reason \n"
                                 "-my_order 2 \n"
                                 "-my_cn_value 1\n";

  string param_file = "param_file.petsc";
  if (!static_cast<bool>(ifstream(param_file))) {
    std::ofstream file(param_file.c_str(), std::ios::ate);
    if (file.is_open()) {
      file << default_options;
      file.close();
    }
  }

  enum arbitrary_smoothing_tests { EIGHT_CUBE = 1, LAST_TEST };

  // Initialize MoFEM
  MoFEM::Core::Initialize(&argc, &argv, param_file.c_str(), help);

  // Create mesh database
  moab::Core mb_instance;              // create database
  moab::Interface &moab = mb_instance; // create interface to database

  try {
    PetscBool flg_file;

    char mesh_file_name[255];
    PetscInt order = 2;
    PetscInt order_tangent_one = order;
    PetscInt order_tangent_two = order;
    PetscBool is_partitioned = PETSC_FALSE;
    PetscBool is_newton_cotes = PETSC_FALSE;
    PetscInt test_num = 0;
    PetscBool print_contact_state = PETSC_FALSE;
    PetscReal cn_value = 1.;

    CHKERR PetscOptionsBegin(PETSC_COMM_WORLD, "", "Elastic Config", "none");

    CHKERR PetscOptionsString("-my_file", "mesh file name", "", "mesh.h5m",
                              mesh_file_name, 255, &flg_file);

    CHKERR PetscOptionsInt("-my_order",
                           "approximation order of spatial positions", "", 1,
                           &order, PETSC_NULL);
    CHKERR PetscOptionsInt("-my_order_tangent_one",
                           "approximation order of spatial positions", "", 1,
                           &order_tangent_one, PETSC_NULL);
    CHKERR PetscOptionsInt("-my_order_tangent_two",
                           "approximation order of spatial positions", "", 1,
                           &order_tangent_two, PETSC_NULL);

    CHKERR PetscOptionsBool("-my_is_partitioned",
                            "set if mesh is partitioned (this result that each "
                            "process keeps only part of the mes",
                            "", PETSC_FALSE, &is_partitioned, PETSC_NULL);

    CHKERR PetscOptionsReal("-my_cn_value", "default regularisation cn value",
                            "", 1., &cn_value, PETSC_NULL);

    CHKERR PetscOptionsInt("-my_test_num", "test number", "", 0, &test_num,
                           PETSC_NULL);

    ierr = PetscOptionsEnd();
    CHKERRQ(ierr);

    // Check if mesh file was provided
    if (flg_file != PETSC_TRUE) {
      SETERRQ(PETSC_COMM_SELF, 1, "*** ERROR -my_file (MESH FILE NEEDED)");
    }

    // if (is_partitioned == PETSC_TRUE) {
    //   SETERRQ(PETSC_COMM_SELF, MOFEM_NOT_IMPLEMENTED,
    //           "Partitioned mesh is not supported");
    // }

    const char *option;
    option = "";
    CHKERR moab.load_file(mesh_file_name, 0, option);

    ParallelComm *pcomm = ParallelComm::get_pcomm(&moab, MYPCOMM_INDEX);
    if (pcomm == NULL)
      pcomm = new ParallelComm(&moab, PETSC_COMM_WORLD);

    // Create MoFEM database and link it to MoAB
    MoFEM::Core core(moab);
    MoFEM::Interface &m_field = core;

    MeshsetsManager *mmanager_ptr;
    CHKERR m_field.getInterface(mmanager_ptr);
    CHKERR mmanager_ptr->printDisplacementSet();
    CHKERR mmanager_ptr->printForceSet();
    // print block sets with materials
    CHKERR mmanager_ptr->printMaterialsSet();

    Range master_tris, slave_tris, all_tris_for_smoothing, all_tets;
    std::vector<BitRefLevel> bit_levels;

    for (_IT_CUBITMESHSETS_BY_SET_TYPE_FOR_LOOP_(m_field, BLOCKSET, it)) {
      if (it->getName().compare(0, 13, "MORTAR_MASTER") == 0) {
        CHKERR m_field.get_moab().get_entities_by_type(it->meshset, MBTRI,
                                                       master_tris, true);
      }
    }

    for (_IT_CUBITMESHSETS_BY_SET_TYPE_FOR_LOOP_(m_field, BLOCKSET, it)) {
      if (it->getName().compare(0, 12, "MORTAR_SLAVE") == 0) {
        rval = m_field.get_moab().get_entities_by_type(it->meshset, MBTRI,
                                                       slave_tris, true);
        CHKERRQ_MOAB(rval);
      }
    }

    all_tris_for_smoothing.merge(master_tris);
    all_tris_for_smoothing.merge(slave_tris);

    bit_levels.push_back(BitRefLevel().set(0));
    CHKERR m_field.getInterface<BitRefManager>()->setBitRefLevelByDim(
        0, 3, bit_levels.back());

    Range meshset_level0;
    CHKERR m_field.getInterface<BitRefManager>()->getEntitiesByRefLevel(
        bit_levels.back(), BitRefLevel().set(), meshset_level0);
    PetscSynchronizedPrintf(PETSC_COMM_WORLD, "meshset_level0 %d\n",
                            meshset_level0.size());
    PetscSynchronizedFlush(PETSC_COMM_WORLD, PETSC_STDOUT);

CHKERR m_field.getInterface<BitRefManager>()->getEntitiesByTypeAndRefLevel(
        bit_levels.back(), BitRefLevel().set(), MBTET, all_tets);

    EntityHandle meshset_surf_slave;
    CHKERR m_field.get_moab().create_meshset(MESHSET_SET, meshset_surf_slave);
    CHKERR m_field.get_moab().add_entities(meshset_surf_slave, slave_tris);

    CHKERR m_field.get_moab().write_mesh("surf_slave.vtk", &meshset_surf_slave,
                                         1);
    EntityHandle meshset_tri_slave;
    CHKERR m_field.get_moab().create_meshset(MESHSET_SET, meshset_tri_slave);

    CHKERR m_field.add_field("MESH_NODE_POSITIONS", H1,
                             AINSWORTH_LEGENDRE_BASE, 3, MB_TAG_SPARSE,
                             MF_ZERO);

    CHKERR m_field.add_field("NORMAL_FIELD", H1, AINSWORTH_LEGENDRE_BASE, 3,
                             MB_TAG_SPARSE, MF_ZERO);

    CHKERR m_field.add_field("LAGMULT", H1, AINSWORTH_LEGENDRE_BASE, 1,
                             MB_TAG_SPARSE, MF_ZERO);

    Range fixed_vertex;
    Range edges_for_meannormal;
    edges_for_meannormal.clear();
    CHKERR m_field.get_moab().get_connectivity(all_tris_for_smoothing,
                                               fixed_vertex);

    // Declare problem add entities (by tets) to the field
    CHKERR m_field.add_ents_to_field_by_type(0, MBTET,
                                             "MESH_NODE_POSITIONS");
    CHKERR m_field.set_field_order(0, MBTET, "MESH_NODE_POSITIONS", 1);
    CHKERR m_field.set_field_order(0, MBTRI, "MESH_NODE_POSITIONS", 1);
    CHKERR m_field.set_field_order(0, MBEDGE, "MESH_NODE_POSITIONS", 1);
    CHKERR m_field.set_field_order(0, MBVERTEX, "MESH_NODE_POSITIONS",
                                   1);


    if (!slave_tris.empty()) {
      CHKERR m_field.add_ents_to_field_by_type(slave_tris, MBTRI,
                                               "MESH_NODE_POSITIONS");
      CHKERR m_field.set_field_order(slave_tris, "MESH_NODE_POSITIONS", order);
      Range edges_smoothed;
      CHKERR moab.get_adjacencies(slave_tris, 1, false, edges_smoothed,
                              moab::Interface::UNION);
      // add edges
      edges_for_meannormal.merge(edges_smoothed);
      CHKERR m_field.set_field_order(edges_smoothed, "MESH_NODE_POSITIONS", order);
      Range vertices_smoothed;
      CHKERR m_field.get_moab().get_connectivity(slave_tris,
                                               vertices_smoothed);
      CHKERR m_field.set_field_order(vertices_smoothed, "MESH_NODE_POSITIONS", 1);

      //Normal
      CHKERR m_field.add_ents_to_field_by_type(slave_tris, MBTRI,
                                               "NORMAL_FIELD");
      CHKERR m_field.set_field_order(slave_tris, "NORMAL_FIELD",  1);
      CHKERR m_field.set_field_order(edges_smoothed, "NORMAL_FIELD",  2);
      CHKERR m_field.set_field_order(vertices_smoothed, "NORMAL_FIELD",  1);

            CHKERR m_field.add_ents_to_field_by_type(slave_tris, MBTRI,
                                               "LAGMULT");
      CHKERR m_field.set_field_order(slave_tris, "LAGMULT",  1);
      CHKERR m_field.set_field_order(edges_smoothed, "LAGMULT",  1);
      CHKERR m_field.set_field_order(vertices_smoothed, "LAGMULT",  1);
    }
    if (!master_tris.empty()) {
      CHKERR m_field.add_ents_to_field_by_type(master_tris, MBTRI,
                                               "MESH_NODE_POSITIONS");
      CHKERR m_field.set_field_order(master_tris, "MESH_NODE_POSITIONS", order);
      Range edges_smoothed;
      CHKERR moab.get_adjacencies(master_tris, 1, false, edges_smoothed,
                              moab::Interface::UNION);
      edges_for_meannormal.merge(edges_smoothed);
      CHKERR m_field.set_field_order(edges_smoothed, "MESH_NODE_POSITIONS", order);
      Range vertices_smoothed;
      CHKERR m_field.get_moab().get_connectivity(master_tris,
                                               vertices_smoothed);
      CHKERR m_field.set_field_order(vertices_smoothed, "MESH_NODE_POSITIONS", order);

      //Normal
      CHKERR m_field.add_ents_to_field_by_type(master_tris, MBTRI,
                                               "NORMAL_FIELD");
      CHKERR m_field.set_field_order(master_tris, "NORMAL_FIELD",  1);
      CHKERR m_field.set_field_order(edges_smoothed, "NORMAL_FIELD",  1);
      CHKERR m_field.set_field_order(vertices_smoothed, "NORMAL_FIELD",  1);

      CHKERR m_field.add_ents_to_field_by_type(master_tris, MBTRI, "LAGMULT");
      CHKERR m_field.set_field_order(master_tris, "LAGMULT", 1);
      CHKERR m_field.set_field_order(edges_smoothed, "LAGMULT", 1);
      CHKERR m_field.set_field_order(vertices_smoothed, "LAGMULT", 1);

      // const EntityHandle *conn;
      // int num_nodes;
      // CHKERR m_field.get_moab().get_connectivity(vertices_smoothed[0], conn, num_nodes, true);
      // nodes_for_tags.push_back(conn[0]);
    }

    // build field
    CHKERR m_field.build_fields();

    // Projection on "X" field
    {
      Projection10NodeCoordsOnField ent_method(m_field,
                                               "MESH_NODE_POSITIONS");
      CHKERR m_field.loop_dofs("MESH_NODE_POSITIONS", ent_method);
    }

    auto make_arbitrary_smooth_element_face = [&]() {
      return boost::make_shared<
          ArbitrarySmoothingProblem::SurfaceSmoothingElement>(m_field);
    };

    auto make_arbitrary_smooth_element_volume = [&]() {
      return boost::make_shared<
          ArbitrarySmoothingProblem::VolumeSmoothingElement>(m_field);
    };

    auto make_smooth_element_common_data = [&]() {
      return boost::make_shared<
          ArbitrarySmoothingProblem::CommonSurfaceSmoothingElement>(m_field);
    };

    auto get_smooth_face_rhs = [&](auto smooth_problem, auto make_element) {
      auto fe_rhs_smooth_face = make_element();
      auto common_data_smooth_elements = make_smooth_element_common_data();
      smooth_problem->setSmoothFaceOperatorsRhs(
          fe_rhs_smooth_face, common_data_smooth_elements,
          "MESH_NODE_POSITIONS", "NORMAL_FIELD", "LAGMULT");
      return fe_rhs_smooth_face;
    };

    auto get_smooth_volume_element_operators = [&](auto smooth_problem,
                                                   auto make_element) {
      auto fe_smooth_volumes = make_element();
      smooth_problem->setGlobalVolumeEvaluator(fe_smooth_volumes,
                                               "MESH_NODE_POSITIONS", smooth_problem->volumeVec);
      return fe_smooth_volumes;
    };

    auto get_smooth_face_lhs = [&](auto smooth_problem, auto make_element) {
      auto fe_lhs_smooth_face = make_element();
      auto common_data_smooth_elements = make_smooth_element_common_data();
      smooth_problem->setSmoothFaceOperatorsLhs(fe_lhs_smooth_face,
                                                common_data_smooth_elements,
                                                "MESH_NODE_POSITIONS", "NORMAL_FIELD", "LAGMULT");
      return fe_lhs_smooth_face;
    };

    auto get_smooth_face_post_prov = [&](auto smooth_problem, auto make_element) {
      auto fe_smooth_face_post_proc = make_element();
      auto common_data_smooth_elements = make_smooth_element_common_data();
      smooth_problem->setSmoothFacePostProc(
          fe_smooth_face_post_proc, common_data_smooth_elements,
          "MESH_NODE_POSITIONS", "NORMAL_FIELD");
      return fe_smooth_face_post_proc;
    };

    auto smooth_problem =
        boost::make_shared<ArbitrarySmoothingProblem>(m_field, cn_value);

    // add fields to the global matrix by adding the element

    smooth_problem->addSurfaceSmoothingElement(
        "SURFACE_SMOOTHING_ELEM", "MESH_NODE_POSITIONS", "NORMAL_FIELD", "LAGMULT",
        all_tris_for_smoothing);

    //addVolumeElement
    smooth_problem->addVolumeElement("VOLUME_SMOOTHING_ELEM",
                                     "MESH_NODE_POSITIONS", all_tets);

    CHKERR MetaSpringBC::addSpringElements(m_field, "MESH_NODE_POSITIONS",
                                           "MESH_NODE_POSITIONS");

    // build finite elemnts
    CHKERR m_field.build_finite_elements();

    // build adjacencies
    CHKERR m_field.build_adjacencies(bit_levels.back());

    // define problems
    CHKERR m_field.add_problem("SURFACE_SMOOTHING_PROB");

    // set refinement level for problem
    CHKERR m_field.modify_problem_ref_level_add_bit("SURFACE_SMOOTHING_PROB",
                                                    bit_levels.back());

    DMType dm_name = "DMMOFEM";
    CHKERR DMRegister_MoFEM(dm_name);

    SmartPetscObj<DM> dm;
    dm = createSmartDM(m_field.get_comm(), dm_name);

    // create dm instance
    CHKERR DMSetType(dm, dm_name);

    // set dm datastruture which created mofem datastructures
    CHKERR DMMoFEMCreateMoFEM(dm, &m_field, "SURFACE_SMOOTHING_PROB",
                              bit_levels.back());
    CHKERR DMSetFromOptions(dm);
    CHKERR DMMoFEMSetIsPartitioned(dm, is_partitioned);
    // add elements to dm
    CHKERR DMMoFEMAddElement(dm, "SURFACE_SMOOTHING_ELEM");
    CHKERR DMMoFEMAddElement(dm, "VOLUME_SMOOTHING_ELEM");
    CHKERR DMMoFEMAddElement(dm, "SPRING");
    CHKERR DMSetUp(dm);

    auto prb_mng = m_field.getInterface<ProblemsManager>();
    const int lo_coeff = 0;
    const int hi_coeff = 3;
    CHKERR m_field.getInterface<CommInterface>()->synchroniseEntities(fixed_vertex);
    CHKERR prb_mng->removeDofsOnEntities("SURFACE_SMOOTHING_PROB", "MESH_NODE_POSITIONS", fixed_vertex,
                                         lo_coeff, hi_coeff);


    // Vector of DOFs and the RHS
    auto D = smartCreateDMVector(dm);
    auto F = smartVectorDuplicate(D);

    // Stiffness matrix
    auto Aij = smartCreateDMMatrix(dm);

    CHKERR VecZeroEntries(D);
    CHKERR DMoFEMMeshToLocalVector(dm, D, INSERT_VALUES, SCATTER_FORWARD);
    CHKERR VecGhostUpdateBegin(D, INSERT_VALUES, SCATTER_FORWARD);
    CHKERR VecGhostUpdateEnd(D, INSERT_VALUES, SCATTER_FORWARD);

    CHKERR VecZeroEntries(F);
    CHKERR VecGhostUpdateBegin(F, INSERT_VALUES, SCATTER_FORWARD);
    CHKERR VecGhostUpdateEnd(F, INSERT_VALUES, SCATTER_FORWARD);

    CHKERR MatSetOption(Aij, MAT_SPD, PETSC_TRUE);
    CHKERR MatZeroEntries(Aij);

    Range fixed_boundary;
    for (_IT_CUBITMESHSETS_BY_SET_TYPE_FOR_LOOP_(m_field, BLOCKSET, it)) {
      if (it->getName().compare(0, 15, "CONSTRAIN_SHAPE") == 0) {
        CHKERR m_field.get_moab().get_entities_by_type(it->meshset, MBVERTEX,
                                                       fixed_boundary, true);
      }
    }

    cerr << "fixed_vertex    " << fixed_vertex.size() << "\n";

    EntityHandle meshset_fixed_vertices;
    CHKERR moab.create_meshset(MESHSET_SET, meshset_fixed_vertices);

    CHKERR
    moab.add_entities(meshset_fixed_vertices, fixed_vertex);

    CHKERR moab.write_mesh("fixed_vertex.vtk", &meshset_fixed_vertices, 1);
    
    //Tags on number of Tris
    std::vector<EntityHandle> nodes_for_tags;
    std::vector<EntityHandle> edges_for_tags;

    //vertices
    for (Range::iterator it_fixed_vert = fixed_vertex.begin();
         it_fixed_vert != fixed_vertex.end(); ++it_fixed_vert) {
      nodes_for_tags.push_back(*it_fixed_vert);
    }

    //vertices
    for (Range::iterator it_fixed_edge = edges_for_meannormal.begin();
         it_fixed_edge != edges_for_meannormal.end(); ++it_fixed_edge) {
      edges_for_tags.push_back(*it_fixed_edge);
    }


    double def_val = 0;
    Tag th_nb_of_sur_tris;
    CHKERR m_field.get_moab().tag_get_handle(
        "NB_TRIS", 1, MB_TYPE_DOUBLE, th_nb_of_sur_tris,
        MB_TAG_CREAT | MB_TAG_SPARSE, &def_val);

    // for (vector<EntityHandle>::iterator vit = nodes_for_tags.begin();
    //      vit != nodes_for_tags.end(); ++vit) {
    //   Range triangles;
    //   CHKERR m_field.get_moab().get_adjacencies(&*vit, 1, 2, true, triangles);
    //   triangles = intersect(all_tris_for_smoothing, triangles);
    //   int nb_tris = triangles.size();
    //   cerr << " nb_tris  " << nb_tris << "\n";
    //   CHKERR m_field.get_moab().tag_set_data(th_nb_of_sur_tris, &*vit, 1,
    //                                          &nb_tris);
    // }

    double def_val_edges = 0;
    Tag th_nb_of_sur_tris_for_edges;
    CHKERR m_field.get_moab().tag_get_handle(
        "NB_TRIS_EDGES", 1, MB_TYPE_DOUBLE, th_nb_of_sur_tris_for_edges,
        MB_TAG_CREAT | MB_TAG_SPARSE, &def_val_edges);

    // for (vector<EntityHandle>::iterator vit = edges_for_tags.begin();
    //      vit != edges_for_tags.end(); ++vit) {
    //   Range triangles;
    //   CHKERR m_field.get_moab().get_adjacencies(&*vit, 1, 2, true, triangles);
    //   triangles = intersect(all_tris_for_smoothing, triangles);
    //   int nb_tris = triangles.size();
    //   cerr << " nb_tris for edges  " << nb_tris << "\n";
    //   CHKERR m_field.get_moab().tag_set_data(th_nb_of_sur_tris_for_edges, &*vit, 1,
    //                                          &nb_tris);
    // }

    auto pass_number_of_adjacent_tris =
        [&](std::vector<EntityHandle> range_of_ents, Tag tag,
            std::string vector_tag) {
          MoFEMFunctionBegin;
          for (vector<EntityHandle>::iterator vit = range_of_ents.begin();
               vit != range_of_ents.end(); ++vit) {
            Range triangles;
            CHKERR m_field.get_moab().get_adjacencies(&*vit, 1, 2, true,
                                                      triangles);
            triangles = intersect(all_tris_for_smoothing, triangles);
            int nb_tris = triangles.size();
            // cerr << " nb_tris for edges  " << nb_tris << "\n";
            CHKERR m_field.get_moab().tag_set_data(tag,
                                                   &*vit, 1, &nb_tris);
            Tag th_nb_ver;
            CHKERR m_field.get_moab().tag_get_handle(vector_tag.c_str(),
                                                     th_nb_ver);
              // double *error_jump_ptr;
              // CHKERR m_field.get_moab().tag_get_by_ptr(th_nb_ver, &*vit, 1,
              //                                          (const void
              //                                          **)&error_jump_ptr);
              // cerr << "error_jump_ptr  " << error_jump_ptr[0] << "  "
              //      << error_jump_ptr[1] << "  " << error_jump_ptr[2] << "  "
              //      << "\n";
          }
          MoFEMFunctionReturn(0);
        };

    double def_val_vec[] = {0, 0, 0};
    Tag th_normal_vec;
    CHKERR m_field.get_moab().tag_get_handle(
        "AVE_N_VEC", 3, MB_TYPE_DOUBLE, th_normal_vec,
        MB_TAG_CREAT | MB_TAG_SPARSE, def_val_vec);

    Tag th_normal_vec_for_edges;
    CHKERR m_field.get_moab().tag_get_handle(
        "AVE_N_VEC_EDGES", 3, MB_TYPE_DOUBLE, th_normal_vec_for_edges,
        MB_TAG_CREAT | MB_TAG_SPARSE, def_val_vec);

    CHKERR pass_number_of_adjacent_tris(nodes_for_tags, th_nb_of_sur_tris, "AVE_N_VEC");
    CHKERR pass_number_of_adjacent_tris(edges_for_tags,
                                        th_nb_of_sur_tris_for_edges, "AVE_N_VEC_EDGES");

    auto read_tags_for_testing = [&](std::vector<EntityHandle> range_of_ents, std::string tag_name) {
      MoFEMFunctionBegin;
      for (vector<EntityHandle>::iterator vit = range_of_ents.begin();
           vit != range_of_ents.end(); ++vit) {
        Tag th_nb_ver;
        CHKERR m_field.get_moab().tag_get_handle(tag_name.c_str(), th_nb_ver);
        int *error_jump_ptr;
        CHKERR m_field.get_moab().tag_get_by_ptr(
            th_nb_ver, &*vit, 1, (const void **)&error_jump_ptr);
        cerr << tag_name.c_str()<<"  " << *error_jump_ptr << "\n";
      }
      MoFEMFunctionReturn(0);
    };

    // CHKERR read_tags_for_testing(nodes_for_tags, "NB_TRIS");
    // CHKERR read_tags_for_testing(edges_for_tags,
    //                                     "NB_TRIS_EDGES");


    // for (vector<EntityHandle>::iterator vit = nodes_for_tags.begin();
    //      vit != nodes_for_tags.end(); ++vit) {
    //   Tag th_nb_ver;
    //   CHKERR m_field.get_moab().tag_get_handle("NB_TRIS", th_nb_ver);
    //   int *error_jump_ptr;
    //   CHKERR m_field.get_moab().tag_get_by_ptr(th_nb_ver, &*vit, 1,
    //                                            (const void **)&error_jump_ptr);
    //   cerr << "error_jump_ptr  " << *error_jump_ptr << "\n";
    // }

    // double def_val_vec[] = {0, 0, 0};
    // Tag th_normal_vec;
    // CHKERR m_field.get_moab().tag_get_handle(
    //     "AVE_N_VEC", 3, MB_TYPE_DOUBLE, th_normal_vec,
    //     MB_TAG_CREAT | MB_TAG_SPARSE, def_val_vec);

    // Tag th_normal_vec_for_edges;
    // CHKERR m_field.get_moab().tag_get_handle(
    //     "AVE_N_VEC_EDGES", 3, MB_TYPE_DOUBLE, th_normal_vec_for_edges,
    //     MB_TAG_CREAT | MB_TAG_SPARSE, def_val_vec);

    
    // for (vector<EntityHandle>::iterator vit = nodes_for_tags.begin();
    //      vit != nodes_for_tags.end(); ++vit) {
    //   Tag th_nb_ver;
    //   CHKERR m_field.get_moab().tag_get_handle("AVE_N_VEC", th_nb_ver);
    //   double *error_jump_ptr;
    //   CHKERR m_field.get_moab().tag_get_by_ptr(th_nb_ver, &*vit, 1,
    //                                            (const void **)&error_jump_ptr);
    //   cerr << "error_jump_ptr  " << error_jump_ptr[0] << "  "
    //        << error_jump_ptr[1] << "  " << error_jump_ptr[2] << "  "
    //        << "\n";
    // }

    auto fe_pre_proc = make_arbitrary_smooth_element_face();
      // auto common_data_smooth_elements = make_smooth_element_common_data();
      fe_pre_proc->getOpPtrVector().push_back(
        new ArbitrarySmoothingProblem::OpCalcMeanNormal("MESH_NODE_POSITIONS",
                                      "NB_TRIS", "AVE_N_VEC", MBVERTEX, m_field));

      // fe_pre_proc->getOpPtrVector().push_back(
      //     new ArbitrarySmoothingProblem::OpCalcMeanNormal(
      //         "MESH_NODE_POSITIONS", "NB_TRIS_EDGES", "AVE_N_VEC_EDGES", MBEDGE,
      //         m_field));

      CHKERR DMoFEMLoopFiniteElements(dm, "SURFACE_SMOOTHING_ELEM",
                                      fe_pre_proc);

      ArbitrarySmoothingProblem::SaveVertexDofOnTag ent_method(m_field,
                                                               "AVE_N_VEC");
      ArbitrarySmoothingProblem::ResolveAndSaveForEdges2  edge_method(
          m_field, "NORMAL_FIELD",
          "AVE_N_VEC", "AVE_N_VEC_EDGES");

      CHKERR m_field.loop_dofs("NORMAL_FIELD", ent_method);
      // CHKERR m_field.loop_dofs("NORMAL_FIELD", edge_method);

      CHKERR DMMoFEMSNESSetFunction(
          dm, "SURFACE_SMOOTHING_ELEM",
          get_smooth_face_rhs(smooth_problem,
                              make_arbitrary_smooth_element_face),
          PETSC_NULL, PETSC_NULL);

      boost::shared_ptr<FEMethod> fe_null;

      CHKERR DMMoFEMSNESSetJacobian(
          dm, "SURFACE_SMOOTHING_ELEM",
          get_smooth_face_lhs(smooth_problem,
                              make_arbitrary_smooth_element_face),
          PETSC_NULL, PETSC_NULL);

      if (test_num) {
        char testing_options[] = "-ksp_type fgmres "
                                 "-pc_type lu "
                                 "-pc_factor_mat_solver_type mumps "
                                 "-snes_type newtonls "
                                 "-snes_linesearch_type basic "
                                 "-snes_max_it 20 "
                                 "-snes_atol 1e-8 "
                                 "-snes_rtol 1e-8 ";
        CHKERR PetscOptionsInsertString(NULL, testing_options);
    }

    auto snes = MoFEM::createSNES(m_field.get_comm());
    CHKERR SNESSetDM(snes, dm);
    SNESConvergedReason snes_reason;
    SnesCtx *snes_ctx;
    // create snes nonlinear solver
    {
      CHKERR SNESSetDM(snes, dm);
      CHKERR DMMoFEMGetSnesCtx(dm, &snes_ctx);
      CHKERR SNESSetFunction(snes, F, SnesRhs, snes_ctx);
      CHKERR SNESSetJacobian(snes, Aij, Aij, SnesMat, snes_ctx);
      CHKERR SNESSetFromOptions(snes);
    }
      
    CHKERR SNESSolve(snes, PETSC_NULL, D);

    CHKERR SNESGetConvergedReason(snes, &snes_reason);

    int its;
    CHKERR SNESGetIterationNumber(snes, &its);
    CHKERR PetscPrintf(PETSC_COMM_WORLD, "number of Newton iterations = %D\n\n",
                       its);

    // save on mesh
    CHKERR VecGhostUpdateBegin(D, INSERT_VALUES, SCATTER_FORWARD);
    CHKERR VecGhostUpdateEnd(D, INSERT_VALUES, SCATTER_FORWARD);
    CHKERR DMoFEMMeshToGlobalVector(dm, D, INSERT_VALUES, SCATTER_REVERSE);

    ///get_smooth_face_post_prov

     CHKERR DMoFEMLoopFiniteElements(dm, "SURFACE_SMOOTHING_ELEM",
                                      get_smooth_face_post_prov(smooth_problem, make_arbitrary_smooth_element_face));
   
    PetscPrintf(PETSC_COMM_WORLD, "Loop post proc\n");
    PetscPrintf(PETSC_COMM_WORLD, "Loop Volume\n");
    VecCreate(PETSC_COMM_WORLD,&smooth_problem->volumeVec);
    PetscInt n = 1;
    VecSetSizes(smooth_problem->volumeVec,PETSC_DECIDE,n);
    VecSetUp(smooth_problem->volumeVec);
    CHKERR DMoFEMLoopFiniteElements(dm, "VOLUME_SMOOTHING_ELEM",
                                      get_smooth_volume_element_operators(smooth_problem, make_arbitrary_smooth_element_volume));

    boost::shared_ptr<PostProcFaceOnRefinedMesh> post_proc_contact_ptr(
        new PostProcFaceOnRefinedMesh(m_field));

    CHKERR post_proc_contact_ptr->generateReferenceElementMesh();
    CHKERR post_proc_contact_ptr->addFieldValuesPostProc("MESH_NODE_POSITIONS");

    CHKERR DMoFEMLoopFiniteElements(dm, "SURFACE_SMOOTHING_ELEM",
                                    post_proc_contact_ptr);

    {
      string out_file_name;
      std::ostringstream stm;
      stm << "out_smooth"
          << ".h5m";
      out_file_name = stm.str();
      CHKERR PetscPrintf(PETSC_COMM_WORLD, "Write file %s\n",
                         out_file_name.c_str());
      CHKERR post_proc_contact_ptr->postProcMesh.write_file(
          out_file_name.c_str(), "MOAB", "PARALLEL=WRITE_PART");
    }                                      
    // PetscInt nloc;
    // VecGetLocalSize(smooth_problem->volumeVec,&nloc);
    CHKERR PetscPrintf(PETSC_COMM_WORLD, "Total Volume: ");

    // cerr << smooth_problem->volumeVec;
    CHKERR VecView(smooth_problem->volumeVec,PETSC_VIEWER_STDOUT_WORLD);
    CHKERR PetscPrintf(PETSC_COMM_WORLD, "\n");
    
    // moab_instance
    moab::Core mb_post;                   // create database
    moab::Interface &moab_proc = mb_post; // create interface to database

    mb_post.delete_mesh();
  }
  CATCH_ERRORS;

  // finish work cleaning memory, getting statistics, etc
  MoFEM::Core::Finalize();

  return 0;
}