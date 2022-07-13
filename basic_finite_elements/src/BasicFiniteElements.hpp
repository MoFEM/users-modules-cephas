/** \file BasicFiniteElements.hpp
  \ingroup Header file for basic finite elements implementation
*/



#ifndef __BASICFINITEELEMENTS_HPP__
#define __BASICFINITEELEMENTS_HPP__

#include <MoFEM.hpp>
using namespace MoFEM;

#ifdef WITH_ADOL_C
#include <adolc/adolc.h>
#endif // WITH_ADOL_C

extern "C" {
  void tetcircumcenter_tp(double a[3],double b[3],double c[3], double d[3],
    double circumcenter[3],double *xi,double *eta,double *zeta);
  void tricircumcenter3d_tp(double a[3],double b[3],double c[3],
    double circumcenter[3],double *xi,double *eta);
}

#include <boost/program_options.hpp>
namespace po = boost::program_options;

#include <cholesky.hpp>

#include <MethodForForceScaling.hpp>
#include <DirichletBC.hpp>
#include <ArcLengthTools.hpp>
#include <BodyForce.hpp>
#include <ConstrainMatrixCtx.hpp>
#include <EdgeForce.hpp>
#include <FieldApproximation.hpp>
#include <FluidPressure.hpp>
#include <NodalForce.hpp>
#include <PostProcOnRefMesh.hpp>
#ifdef WITH_ADOL_C
  #include <NonLinearElasticElement.hpp>
  #include <KelvinVoigtDamper.hpp>
  #include <PostProcStresses.hpp>
  #include <Smoother.hpp>
  #include <ConvectiveMassElement.hpp>
#endif
#include <SurfacePressureComplexForLazy.hpp>
#include <PCMGSetUpViaApproxOrders.hpp>
#include <PostProcHookStresses.hpp>
#include <SurfacePressure.hpp>
#include <SurfaceSlidingConstrains.hpp>
#include <ThermalElement.hpp>
#include <ThermalStressElement.hpp>
#include <TimeForceScale.hpp>
#include <VolumeCalculation.hpp>
#include <AnalyticalDirichlet.hpp>
#include <SaveVertexDofOnTag.hpp>
#include <HookeElement.hpp>
#include <SpringElement.hpp>
#include <NavierStokesElement.hpp>
#include <SimpleRodElement.hpp>

// generic interfaces
#include <GenericElementInterface.hpp>

//FIXME: fix organisation of these headers
// #include <ElasticMaterials.hpp>
// #include <NonlinearElasticElementInterface.hpp>

#include <BasicBoundaryConditionsInterface.hpp>

using namespace BasicFiniteElements;

#endif // __BASICFINITEELEMENTS_HPP__
