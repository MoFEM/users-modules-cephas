/** \file PostProcOnRefMesh.cpp
 * \brief Postprocess fields on refined mesh made for 10 Node tets
 *
 * Create refined mesh, without enforcing continuity between element. Calculate
 * field values on nodes of that mesh.
 * \ingroup mofem_fs_post_proc
 */

/*
 * This file is part of MoFEM.
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
using namespace boost::numeric;
#include <PostProcOnRefMesh.hpp>

// #ifdef __cplusplus
// extern "C" {
// #endif
//   #include <gm_rule.h>
// #ifdef __cplusplus
// }
// #endif


MoFEMErrorCode PostProcCommonOnRefMesh::OpGetFieldValues::doWork(
  int side,
  EntityType type,
  DataForcesAndSourcesCore::EntData &data
) {
  MoFEMFunctionBeginHot;




  if(data.getFieldData().size()==0) MoFEMFunctionReturnHot(0);

  if(V) {
    vAlues.resize(data.getFieldData().size());
    double *a;
    ierr = VecGetArray(V,&a); CHKERRG(ierr);
    VectorDofs::iterator it,hi_it;
    it = data.getFieldDofs().begin();
    hi_it = data.getFieldDofs().end();
    for(int ii = 0;it!=hi_it;it++,ii++) {
      int local_idx = getFEMethod()->rowPtr->find((*it)->getGlobalUniqueId())->get()->getPetscLocalDofIdx();
      vAlues[ii] = a[local_idx];
    }
    ierr = VecRestoreArray(V,&a); CHKERRG(ierr);
    vAluesPtr = &vAlues;
  } else {
    vAluesPtr = &data.getFieldData();
  }

  const MoFEM::FEDofEntity *dof_ptr = data.getFieldDofs()[0].get();
  int rank = dof_ptr->getNbOfCoeffs();

  int tag_length = rank;
  FieldSpace space = dof_ptr->getSpace();
  switch(space) {
    case L2:
    case H1:
    break;
    case HCURL:
    case HDIV:
    tag_length *= 3;
    break;
    default:
    SETERRQ(PETSC_COMM_SELF,MOFEM_NOT_IMPLEMENTED,"field with that space is not implemented");
  }

  if(tag_length>1 && tag_length < 3) {
    tag_length = 3;
  } else if(tag_length > 3 && tag_length < 9) {
    tag_length = 9;
  }

  double def_VAL[tag_length];
  bzero(def_VAL,tag_length*sizeof(double));
  Tag th;
  rval = postProcMesh.tag_get_handle(
    tagName.c_str(),tag_length,MB_TYPE_DOUBLE,th,MB_TAG_CREAT|MB_TAG_SPARSE,def_VAL
  ); CHKERRG(rval);

  // zero tags, this for Vertex if H1 and TRI if Hdiv, EDGE for Hcurl
  // no need for L2
  const void* tags_ptr[mapGaussPts.size()];
  int nb_gauss_pts = data.getN().size1();
  if(mapGaussPts.size()!=(unsigned int)nb_gauss_pts) {
    // cerr << *data.getFieldDofs()[0] << endl;
    // cerr << data.getFieldData().size() << " " << data.getN().size2() << endl;
    // cerr << data.getDataOrder() << endl;
    SETERRQ2(
      PETSC_COMM_SELF,
      MOFEM_DATA_INCONSISTENCY,
      "data inconsistency %d!=%d",
      mapGaussPts.size(),nb_gauss_pts
    );
  }

  switch(space) {
    case H1:
    commonData.fieldMap[rowFieldName].resize(nb_gauss_pts);
    if(type == MBVERTEX) {
      for(int gg = 0;gg<nb_gauss_pts;gg++) {
        rval = postProcMesh.tag_set_data(th,&mapGaussPts[gg],1,def_VAL); CHKERRG(rval);
        (commonData.fieldMap[rowFieldName])[gg].resize(rank);
        (commonData.fieldMap[rowFieldName])[gg].clear();
      }
    }
    rval = postProcMesh.tag_get_by_ptr(th,&mapGaussPts[0],mapGaussPts.size(),tags_ptr); CHKERRG(rval);
    for(int gg = 0;gg<nb_gauss_pts;gg++) {
      for(int rr = 0;rr<rank;rr++) {
        const double val = cblas_ddot(
          (vAluesPtr->size()/rank),&(data.getN(gg)[0]),1,&((*vAluesPtr)[rr]),rank
        );
        (commonData.fieldMap[rowFieldName])[gg][rr] = ((double*)tags_ptr[gg])[rr] += val;
      }
    }
    break;
    case L2:
    commonData.fieldMap[rowFieldName].resize(nb_gauss_pts);
    for(int gg = 0;gg<nb_gauss_pts;gg++) {
      rval = postProcMesh.tag_set_data(th,&mapGaussPts[gg],1,def_VAL); CHKERRG(rval);
      (commonData.fieldMap[rowFieldName])[gg].resize(rank);
      (commonData.fieldMap[rowFieldName])[gg].clear();
    }
    rval = postProcMesh.tag_get_by_ptr(
      th,&mapGaussPts[0],mapGaussPts.size(),tags_ptr
    ); CHKERRG(rval);
    for(int gg = 0;gg<nb_gauss_pts;gg++) {
      bzero((double*)tags_ptr[gg],sizeof(double)*tag_length);
      for(int rr = 0;rr<rank;rr++) {
        const double val = cblas_ddot(
          (vAluesPtr->size()/rank),&(data.getN(gg)[0]),1,&((*vAluesPtr)[rr]),rank
        );
        (commonData.fieldMap[rowFieldName])[gg][rr] = ((double*)tags_ptr[gg])[rr] = val;
      }
    }
    break;
    case HCURL:
    // FIXME: fieldMap not set
    if(type == MBEDGE && side == 0) {
      for(int gg = 0;gg<nb_gauss_pts;gg++) {
        rval = postProcMesh.tag_set_data(th,&mapGaussPts[gg],1,def_VAL); CHKERRG(rval);
      }
    }
    rval = postProcMesh.tag_get_by_ptr(th,&mapGaussPts[0],mapGaussPts.size(),tags_ptr); CHKERRG(rval);
    {
      FTensor::Index<'i',3> i;
      auto t_n_hcurl = data.getFTensor1HcurlN<3>();
      for(int gg = 0;gg!=nb_gauss_pts;gg++) {
        double *ptr = &((double*)tags_ptr[gg])[0];
        int ll = 0;
        for(;ll!=static_cast<int>(vAluesPtr->size()/rank);ll++) {
          FTensor::Tensor1<double*,3> t_tag_val(ptr,&ptr[1],&ptr[2],3);
          for(int rr = 0;rr!=rank;rr++) {
            const double dof_val = (*vAluesPtr)[ll*rank+rr];
            t_tag_val(i) += dof_val*t_n_hcurl(i);
            ++t_tag_val;
          }
          ++t_n_hcurl;
        }
        for(;ll!=static_cast<int>(data.getHcurlN().size2()/3);ll++) {
          ++t_n_hcurl;
        }
      }
    }
    break;
    case HDIV:
    // FIXME: fieldMap not set
    if(type == MBTRI && side == 0) {
      for(int gg = 0;gg<nb_gauss_pts;gg++) {
        rval = postProcMesh.tag_set_data(th,&mapGaussPts[gg],1,def_VAL); CHKERRG(rval);
      }
    }
    rval = postProcMesh.tag_get_by_ptr(th,&mapGaussPts[0],mapGaussPts.size(),tags_ptr); CHKERRG(rval);
    {
      FTensor::Index<'i',3> i;
      auto t_n_hdiv = data.getFTensor1HdivN<3>();
      for(int gg = 0;gg!=nb_gauss_pts;gg++) {
        double *ptr = &((double*)tags_ptr[gg])[0];
        int ll = 0;
        for(;ll!=static_cast<int>(vAluesPtr->size()/rank);ll++) {
          FTensor::Tensor1<double*,3> t_tag_val(ptr,&ptr[1],&ptr[2],3);
          for(int rr = 0;rr!=rank;rr++) {
            const double dof_val = (*vAluesPtr)[ll*rank+rr];
            t_tag_val(i) += dof_val*t_n_hdiv(i);
            ++t_tag_val;
          }
          ++t_n_hdiv;
        }
        for(;ll!=static_cast<int>(data.getHdivN().size2()/3);ll++) {
          ++t_n_hdiv;
        }
      }
    }
    break;
    default:
    SETERRQ(PETSC_COMM_SELF,MOFEM_NOT_IMPLEMENTED,"field with that space is not implemented");
  }

  MoFEMFunctionReturnHot(0);
}

MoFEMErrorCode PostProcCommonOnRefMesh::OpGetFieldGradientValues::doWork(
  int side,
  EntityType type,
  DataForcesAndSourcesCore::EntData &data
) {
  MoFEMFunctionBeginHot;




  if(data.getFieldData().size()==0) MoFEMFunctionReturnHot(0);
  if(V) {
    vAlues.resize(data.getFieldData().size());
    double *a;
    ierr = VecGetArray(V,&a); CHKERRG(ierr);
    VectorDofs::iterator it,hi_it;
    it = data.getFieldDofs().begin();
    hi_it = data.getFieldDofs().end();
    for(int ii = 0;it!=hi_it;it++,ii++) {
      int local_idx = getFEMethod()->rowPtr->find((*it)->getGlobalUniqueId())->get()->getPetscLocalDofIdx();
      vAlues[ii] = a[local_idx];
    }
    ierr = VecRestoreArray(V,&a); CHKERRG(ierr);
    vAluesPtr = &vAlues;
  } else {
    vAluesPtr = &data.getFieldData();
  }

  const MoFEM::FEDofEntity *dof_ptr = data.getFieldDofs()[0].get();
  int rank = dof_ptr->getNbOfCoeffs();

  int tag_length = rank*3;
  FieldSpace space = dof_ptr->getSpace();
  switch(space) {
    case L2:
    case H1:
    break;
    case HCURL:
    case HDIV:
    tag_length *= 3;
    break;
    default:
    SETERRQ(PETSC_COMM_SELF,MOFEM_NOT_IMPLEMENTED,"field with that space is not implemented");
  }

  double def_VAL[tag_length];
  bzero(def_VAL,tag_length*sizeof(double));
  Tag th;
  rval = postProcMesh.tag_get_handle(tagName.c_str(),tag_length,MB_TYPE_DOUBLE,th,MB_TAG_CREAT|MB_TAG_SPARSE,def_VAL); CHKERRG(rval);

  // zero tags, this for Vertex if H1 and TRI if Hdiv, EDGE for Hcurl
  // no need for L2
  const void* tags_ptr[mapGaussPts.size()];
  int nb_gauss_pts = data.getN().size1();
  if(mapGaussPts.size()!=(unsigned int)nb_gauss_pts) {
    SETERRQ(PETSC_COMM_SELF,MOFEM_DATA_INCONSISTENCY,"data inconsistency");
  }

  try {

    switch(space) {
      case H1:
      commonData.gradMap[rowFieldName].resize(nb_gauss_pts);
      if(type == MBVERTEX) {
        for(int gg = 0;gg<nb_gauss_pts;gg++) {
          rval = postProcMesh.tag_set_data(th,&mapGaussPts[gg],1,def_VAL); CHKERRG(rval);
          (commonData.gradMap[rowFieldName])[gg].resize(rank,3);
          (commonData.gradMap[rowFieldName])[gg].clear();
        }
      }
      rval = postProcMesh.tag_get_by_ptr(th,&mapGaussPts[0],mapGaussPts.size(),tags_ptr); CHKERRG(rval);
      for(int gg = 0;gg<nb_gauss_pts;gg++) {
        for(int rr = 0;rr<rank;rr++) {
          for(int dd = 0;dd<3;dd++) {
            for(unsigned int dof = 0;dof<(vAluesPtr->size()/rank);dof++) {
              const double val = data.getDiffN(gg)(dof,dd)*(*vAluesPtr)[rank*dof+rr];
              (commonData.gradMap[rowFieldName])[gg](rr,dd) = ((double*)tags_ptr[gg])[rank*rr+dd] += val;
            }
          }
        }
      }
      break;
      case L2:
      commonData.gradMap[rowFieldName].resize(nb_gauss_pts);
      for(int gg = 0;gg<nb_gauss_pts;gg++) {
        rval = postProcMesh.tag_set_data(th,&mapGaussPts[gg],1,def_VAL); CHKERRG(rval);
        (commonData.gradMap[rowFieldName])[gg].resize(rank,3);
        (commonData.gradMap[rowFieldName])[gg].clear();
      }
      rval = postProcMesh.tag_get_by_ptr(th,&mapGaussPts[0],mapGaussPts.size(),tags_ptr); CHKERRG(rval);
      for(int gg = 0;gg<nb_gauss_pts;gg++) {
        for(int rr = 0;rr<rank;rr++) {
          for(int dd = 0;dd<3;dd++) {
            for(unsigned int dof = 0;dof<(vAluesPtr->size()/rank);dof++) {
              const double val = data.getDiffN(gg)(dof,dd)*(*vAluesPtr)[rank*dof+rr];
              (commonData.gradMap[rowFieldName])[gg](rr,dd) = ((double*)tags_ptr[gg])[rank*rr+dd] += val;
            }
          }
        }
      }
      break;
      default:
      SETERRQ(PETSC_COMM_SELF,MOFEM_NOT_IMPLEMENTED,"field with that space is not implemented");
    }

  } catch (std::exception& ex) {
    std::ostringstream ss;
    ss << "throw in method: " << ex.what() << " at line " << __LINE__ << " in file " << __FILE__;
    SETERRQ(PETSC_COMM_SELF,MOFEM_STD_EXCEPTION_THROW,ss.str().c_str());
  }

  MoFEMFunctionReturnHot(0);
}

MoFEMErrorCode PostProcFatPrismOnRefinedMesh::generateReferenceElementMesh() {
  MoFEMFunctionBeginHot;
  MoFEMFunctionReturnHot(0);
}

MoFEMErrorCode PostProcFatPrismOnRefinedMesh::setGaussPtsTrianglesOnly(int order_triangles_only) {
  MoFEMFunctionBeginHot;
  // if(gaussPtsTrianglesOnly.size1()==0 || gaussPtsTrianglesOnly.size2()==0) {
  //   SETERRQ(PETSC_COMM_SELF,MOFEM_DATA_INCONSISTENCY,"post-process mesh not generated");
  // }

  //FIXME: Refinement not implement and inefficient implementation, looks ugly.

  //
  const EntityHandle *conn;
  int num_nodes;
  EntityHandle prism;
  const int nb_on_triangle = 6;
  const int nb_through_thickness = 3;


  if(elementsMap.find(numeredEntFiniteElementPtr->getEnt())!=elementsMap.end()) {
    prism = elementsMap[numeredEntFiniteElementPtr->getEnt()];
  } else {
    MatrixDouble gauss_pts_triangles_only,gauss_pts_through_thickness;
    {
      gauss_pts_triangles_only.resize(3,3,false);
      gauss_pts_triangles_only.clear();
      gauss_pts_triangles_only(0,0) = 0;
      gauss_pts_triangles_only(1,0) = 0;
      gauss_pts_triangles_only(0,1) = 1;
      gauss_pts_triangles_only(1,1) = 0;
      gauss_pts_triangles_only(0,2) = 0;
      gauss_pts_triangles_only(1,2) = 1;
      gauss_pts_through_thickness.resize(2,2,false);
      gauss_pts_through_thickness(0,0) = 0;
      gauss_pts_through_thickness(0,1) = 1;
    }
    ublas::vector<EntityHandle> prism_conn(6);
    VectorDouble coords(3);
    for(int ggf = 0;ggf!=3;ggf++) {
      double ksi = gauss_pts_triangles_only(0,ggf);
      double eta = gauss_pts_triangles_only(1,ggf);
      coords[0] = ksi;
      coords[1] = eta;
      for(int ggt = 0;ggt!=2;ggt++) {
        double zeta = gauss_pts_through_thickness(0,ggt);
        coords[2] = zeta;
        int side = ggt*3+ggf;
        rval = postProcMesh.create_vertex(&coords[0],prism_conn[side]); CHKERRG(rval);
      }
    }
    rval = postProcMesh.create_element(MBPRISM,&prism_conn[0],6,prism); CHKERRG(rval);

    elementsMap[numeredEntFiniteElementPtr->getEnt()] = prism;
    // Range faces;
    // rval = postProcMesh.get_adjacencies(&prism,1,2,true,faces); CHKERRG(rval);
    Range edges;
    rval = postProcMesh.get_adjacencies(&prism,1,1,true,edges); CHKERRG(rval);
    EntityHandle meshset;
    rval = postProcMesh.create_meshset(MESHSET_SET,meshset); CHKERRG(rval);
    rval = postProcMesh.add_entities(meshset,&prism,1); CHKERRG(rval);
    // rval = postProcMesh.add_entities(meshset,faces); CHKERRG(rval);
    rval = postProcMesh.add_entities(meshset,edges); CHKERRG(rval);
    if(tenNodesPostProcTets) {
      rval = postProcMesh.convert_entities(meshset,true,false,false); CHKERRG(rval);
    }
    rval = postProcMesh.delete_entities(&meshset,1); CHKERRG(rval);
    rval = postProcMesh.delete_entities(edges); CHKERRG(rval);
    // rval = postProcMesh.delete_entities(faces); CHKERRG(rval);

    rval = mField.get_moab().get_connectivity(
      numeredEntFiniteElementPtr->getEnt(),conn,num_nodes,true
    ); CHKERRG(rval);
    MatrixDouble coords_prism_global;
    coords_prism_global.resize(num_nodes,3,false);
    rval = mField.get_moab().get_coords(conn,num_nodes,&coords_prism_global(0,0)); CHKERRG(rval);

    rval = postProcMesh.get_connectivity(prism,conn,num_nodes,false); CHKERRG(rval);
    MatrixDouble coords_prism_local;
    coords_prism_local.resize(num_nodes,3,false);
    rval = postProcMesh.get_coords(conn,num_nodes,&coords_prism_local(0,0)); CHKERRG(rval);

    if(gaussPtsThroughThickness.size2()!=nb_through_thickness) {
      pointsMap.clear();
      gaussPtsTrianglesOnly.resize(3,nb_on_triangle,false);
      gaussPtsTrianglesOnly.clear();
      gaussPtsThroughThickness.resize(2,nb_through_thickness,false);
      gaussPtsThroughThickness.clear();
      const double eps = 1e-6;
      int ggf = 0,ggt = 0;
      for(int nn = 0;nn!=num_nodes;nn++) {
        double ksi = coords_prism_local(nn,0);
        double eta = coords_prism_local(nn,1);
        double zeta = coords_prism_local(nn,2);
        pointsMap.insert(PointsMap3D(ksi*100,eta*100,zeta*100,nn));
        if(fabs(zeta)< eps) {
          gaussPtsTrianglesOnly(0,ggf) = ksi;
          gaussPtsTrianglesOnly(1,ggf) = eta;
          ggf++;
        }
        if(fabs(ksi)<eps&&fabs(eta)<eps) {
          gaussPtsThroughThickness(0,ggt) = zeta;
          ggt++;
        }
      }
    }

    for(int nn = 0;nn!=num_nodes;nn++) {
      const double ksi = coords_prism_local(nn,0);
      const double eta = coords_prism_local(nn,1);
      const double zeta = coords_prism_local(nn,2);
      const double n0 = N_MBTRI0(ksi,eta);
      const double n1 = N_MBTRI1(ksi,eta);
      const double n2 = N_MBTRI2(ksi,eta);
      const double e0 = N_MBEDGE0(zeta);
      const double e1 = N_MBEDGE1(zeta);
      double coords_global[3];
      for(int dd = 0;dd!=3;dd++) {
        coords_global[dd] =
        n0*e0*coords_prism_global(0,dd)+n1*e0*coords_prism_global(1,dd)+n2*e0*coords_prism_global(2,dd)+
        n0*e1*coords_prism_global(3,dd)+n1*e1*coords_prism_global(4,dd)+n2*e1*coords_prism_global(5,dd);
      }
      // cerr << coords_global[0] << " " << coords_global[1] << " " << coords_global[2] << endl;
      rval = postProcMesh.set_coords(&conn[nn],1,coords_global); CHKERRG(rval);
    }

  }

  mapGaussPts.clear();
  mapGaussPts.resize(nb_through_thickness*nb_on_triangle);
  rval = postProcMesh.get_connectivity(prism,conn,num_nodes,false); CHKERRG(rval);
  fill(mapGaussPts.begin(),mapGaussPts.end(),0);
  {
    int gg = 0;
    for(unsigned int ggf = 0;ggf!=gaussPtsTrianglesOnly.size2();ggf++) {
      const double ksi = gaussPtsTrianglesOnly(0,ggf);
      const double eta = gaussPtsTrianglesOnly(1,ggf);
      for(unsigned int ggt = 0;ggt!=gaussPtsThroughThickness.size2();ggt++,gg++) {
        const double zeta = gaussPtsThroughThickness(0,ggt);
        PointsMap3D_multiIndex::iterator it;
        it = pointsMap.find(boost::make_tuple(ksi*100,eta*100,zeta*100));
        if(it != pointsMap.end()) {
          mapGaussPts[gg] = conn[it->nN];
        }
      }
    }
  }

  MoFEMFunctionReturnHot(0);
}

MoFEMErrorCode PostProcFatPrismOnRefinedMesh::setGaussPtsThroughThickness(int order_thickness) {
  MoFEMFunctionBeginHot;
  if(gaussPtsThroughThickness.size1()==0 || gaussPtsThroughThickness.size2()==0) {
    SETERRQ(PETSC_COMM_SELF,MOFEM_DATA_INCONSISTENCY,"post-process mesh not generated");
  }
  MoFEMFunctionReturnHot(0);
}

MoFEMErrorCode PostProcFatPrismOnRefinedMesh::preProcess() {
  MoFEMFunctionBeginHot;
  // MoAB
  ParallelComm* pcomm_post_proc_mesh = ParallelComm::get_pcomm(&postProcMesh,MYPCOMM_INDEX);
  if(pcomm_post_proc_mesh != NULL) {
    delete pcomm_post_proc_mesh;
  }
  // rval = postProcMesh.delete_mesh(); CHKERRG(rval);
  MoFEMFunctionReturnHot(0);
}

MoFEMErrorCode PostProcFatPrismOnRefinedMesh::postProcess() {
  MoFEMFunctionBeginHot;
  ParallelComm* pcomm = ParallelComm::get_pcomm(&mField.get_moab(),MYPCOMM_INDEX);
  ParallelComm* pcomm_post_proc_mesh = ParallelComm::get_pcomm(&postProcMesh,MYPCOMM_INDEX);
  if(pcomm_post_proc_mesh == NULL) {
    pcomm_post_proc_mesh = new ParallelComm(&postProcMesh,mField.get_comm());
  }
  Range prims;
  rval = postProcMesh.get_entities_by_type(0,MBPRISM,prims,false);  CHKERRG(rval);
  //std::cerr << "total prims size " << prims.size() << std::endl;
  int rank = pcomm->rank();
  Range::iterator pit = prims.begin();
  for(;pit!=prims.end();pit++) {
    rval = postProcMesh.tag_set_data(
      pcomm_post_proc_mesh->part_tag(),&*pit,1,&rank
    ); CHKERRG(rval);
  }
  rval = pcomm_post_proc_mesh->resolve_shared_ents(0); CHKERRG(rval);
  MoFEMFunctionReturnHot(0);
}


MoFEMErrorCode PostProcFaceOnRefinedMesh::generateReferenceElementMesh() {
  MoFEMFunctionBeginHot;

  gaussPts.resize(3,3,false);
  gaussPts.clear();
  gaussPts(0,0) = 0;
  gaussPts(1,0) = 0;
  gaussPts(0,1) = 1;
  gaussPts(1,1) = 0;
  gaussPts(0,2) = 0;
  gaussPts(1,2) = 1;
  mapGaussPts.resize(gaussPts.size2());

  moab::Core core_ref;
  moab::Interface& moab_ref = core_ref;
  const EntityHandle *conn;
  int num_nodes;
  EntityHandle tri_conn[3];
  MatrixDouble coords(6,3);
  for(int gg = 0;gg!=3;gg++) {
    coords(gg,0) = gaussPts(0,gg);
    coords(gg,1) = gaussPts(1,gg);
    coords(gg,2) = 0;
    rval = moab_ref.create_vertex(&coords(gg,0),tri_conn[gg]); CHKERRG(rval);
  }

  EntityHandle tri;
  rval = moab_ref.create_element(MBTRI,tri_conn,3,tri); CHKERRG(rval);
  Range edges;
  rval = moab_ref.get_adjacencies(&tri,1,1,true,edges); CHKERRG(rval);
  EntityHandle meshset;
  rval = moab_ref.create_meshset(MESHSET_SET,meshset); CHKERRG(rval);
  rval = moab_ref.add_entities(meshset,&tri,1); CHKERRG(rval);
  rval = moab_ref.add_entities(meshset,edges); CHKERRG(rval);
  if(sixNodePostProcTris) {
    rval = moab_ref.convert_entities(meshset,true,false,false); CHKERRG(rval);
  }
  rval = moab_ref.get_connectivity(tri,conn,num_nodes,false); CHKERRG(rval);
  rval = moab_ref.get_coords(conn,num_nodes,&coords(0,0));

  gaussPts.resize(3,num_nodes,false);
  gaussPts.clear();
  for(int nn = 0;nn<3;nn++) {
    gaussPts(0,nn) = coords(nn,0);
    gaussPts(1,nn) = coords(nn,1);
    gaussPts(0,3+nn) = coords(3+nn,0);
    gaussPts(1,3+nn) = coords(3+nn,1);
  }

  MoFEMFunctionReturnHot(0);
}

MoFEMErrorCode PostProcFaceOnRefinedMesh::setGaussPts(int order) {
  MoFEMFunctionBeginHot;
  if(gaussPts.size1()==0) {
    SETERRQ(PETSC_COMM_SELF,MOFEM_DATA_INCONSISTENCY,"post-process mesh not generated");
  }

  //
  const EntityHandle *conn;
  int num_nodes;
  EntityHandle tri;

  if(elementsMap.find(numeredEntFiniteElementPtr->getEnt())!=elementsMap.end()) {
    tri = elementsMap[numeredEntFiniteElementPtr->getEnt()];
  } else {
    ublas::vector<EntityHandle> tri_conn(3);
    MatrixDouble coords_tri(3,3);
    VectorDouble coords(3);
    rval = mField.get_moab().get_connectivity(numeredEntFiniteElementPtr->getEnt(),conn,num_nodes,true); CHKERRG(rval);
    rval = mField.get_moab().get_coords(conn,num_nodes,&coords_tri(0,0));
    for(int gg = 0;gg!=3;gg++) {
      double ksi = gaussPts(0,gg);
      double eta = gaussPts(1,gg);
      double n0 = N_MBTRI0(ksi,eta);
      double n1 = N_MBTRI1(ksi,eta);
      double n2 = N_MBTRI2(ksi,eta);
      double x = n0*coords_tri(0,0)+n1*coords_tri(1,0)+n2*coords_tri(2,0);
      double y = n0*coords_tri(0,1)+n1*coords_tri(1,1)+n2*coords_tri(2,1);
      coords[0] = x;
      coords[1] = y;
      coords[2] = 0;
      rval = postProcMesh.create_vertex(
        &coords[0],tri_conn[gg]
      ); CHKERRG(rval);
    }
    rval = postProcMesh.create_element(MBTRI,&tri_conn[0],3,tri); CHKERRG(rval);
    elementsMap[numeredEntFiniteElementPtr->getEnt()] = tri;
    Range edges;
    rval = postProcMesh.get_adjacencies(&tri,1,1,true,edges); CHKERRG(rval);
    EntityHandle meshset;
    rval = postProcMesh.create_meshset(MESHSET_SET,meshset); CHKERRG(rval);
    rval = postProcMesh.add_entities(meshset,&tri,1); CHKERRG(rval);
    rval = postProcMesh.add_entities(meshset,edges); CHKERRG(rval);
    if(sixNodePostProcTris) {
      rval = postProcMesh.convert_entities(meshset,true,false,false); CHKERRG(rval);
    }
    rval = postProcMesh.delete_entities(&meshset,1); CHKERRG(rval);
    rval = postProcMesh.delete_entities(edges); CHKERRG(rval);
  }

  // Set values which map nodes with integration points on the prism
  {
    rval = postProcMesh.get_connectivity(tri,conn,num_nodes,false); CHKERRG(rval);
    mapGaussPts.resize(num_nodes);
    for(int nn = 0;nn<num_nodes;nn++) {
      mapGaussPts[nn] = conn[nn];
    }
  }

  MoFEMFunctionReturnHot(0);
}

MoFEMErrorCode PostProcFaceOnRefinedMesh::preProcess() {
  MoFEMFunctionBeginHot;
  ParallelComm* pcomm_post_proc_mesh = ParallelComm::get_pcomm(&postProcMesh,MYPCOMM_INDEX);
  if(pcomm_post_proc_mesh != NULL) {
    delete pcomm_post_proc_mesh;
  }
  MoFEMFunctionReturnHot(0);
}

MoFEMErrorCode PostProcFaceOnRefinedMesh::postProcess() {
  MoFEMFunctionBeginHot;
  ParallelComm* pcomm = ParallelComm::get_pcomm(&mField.get_moab(),MYPCOMM_INDEX);
  ParallelComm* pcomm_post_proc_mesh = ParallelComm::get_pcomm(&postProcMesh,MYPCOMM_INDEX);
  if(pcomm_post_proc_mesh == NULL) {
    pcomm_post_proc_mesh = new ParallelComm(&postProcMesh,mField.get_comm());
  }
  Range tris;
  rval = postProcMesh.get_entities_by_type(0,MBTRI,tris,false);  CHKERRG(rval);
  int rank = pcomm->rank();
  Range::iterator pit = tris.begin();
  for(;pit!=tris.end();pit++) {
    rval = postProcMesh.tag_set_data(
      pcomm_post_proc_mesh->part_tag(),&*pit,1,&rank
    ); CHKERRG(rval);
  }
  rval = pcomm_post_proc_mesh->resolve_shared_ents(0); CHKERRG(rval);
  MoFEMFunctionReturnHot(0);
}
