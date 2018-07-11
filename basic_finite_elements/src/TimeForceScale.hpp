/* \brief TimeForceScale.hpp
 *
 * Edited and modified by Hassan.
 *
 * This is not exactly procedure for linear elastic dynamics, since jacobian is
 * evaluated at every time step and SNES procedure is involved. However it is
 * implemented like that, to test methodology for general nonlinear problem.
 *
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

#ifndef __TIMEFORCESCALE_HPP__
#define __TIMEFORCESCALE_HPP__

/** \brief Force scale operator for reading two columns
  */
struct TimeForceScale: public MethodForForceScaling {
  //Hassan: This function to read data file (once) and save it in a pair vector ts

  std::map<double,double> tSeries;
  int readFile,debug;
  string nAme;
  bool errorIfFileNotGiven;

  TimeForceScale(
    string name = "-my_time_data_file",
    bool error_if_file_not_given = true
  ):
  readFile(0),
  debug(1),
  nAme(name),
  errorIfFileNotGiven(error_if_file_not_given) {
    
    ierr = timeData(); CHKERRABORT(PETSC_COMM_WORLD,ierr);
  }

  
  
  PetscBool fLg;

  MoFEMErrorCode timeData() {
    MoFEMFunctionBeginHot;
    char time_file_name[255];
    ierr = PetscOptionsGetString(PETSC_NULL,PETSC_NULL,nAme.c_str(),time_file_name,255,&fLg); CHKERRG(ierr);
    if(!fLg && errorIfFileNotGiven) {
      SETERRQ1(PETSC_COMM_SELF,1,"*** ERROR %s (time_data FILE NEEDED)",nAme.c_str());
    }
    if(!fLg) {
      MoFEMFunctionReturnHot(0);
    }
    FILE *time_data = fopen(time_file_name,"r");
    if(time_data == NULL) {
      SETERRQ1(PETSC_COMM_SELF,1,"*** ERROR data file < %s > open unsuccessful",time_file_name);
    }
    double no1 = 0.0, no2 = 0.0;
    tSeries[no1] = no2;
    while(!feof(time_data)){
      int n = fscanf(time_data,"%lf %lf",&no1,&no2);
      if((n <= 0)||((no1==0)&&(no2==0))) {
        fgetc(time_data);
        continue;
      }
      if(n != 2){
        SETERRQ1(PETSC_COMM_SELF,1,"*** ERROR read data file error (check input time data file) { n = %d }",n);
      }
      tSeries[no1] = no2;
    }
    int r = fclose(time_data);
    if(debug) {
      std::map<double, double>::iterator tit = tSeries.begin();
      for(;tit!=tSeries.end();tit++) {
        PetscPrintf(PETSC_COMM_WORLD,"** read time series %3.2e time %3.2e\n",tit->first,tit->second);
      }
    }
    if(r!=0) {
      SETERRQ(PETSC_COMM_SELF,1,"*** ERROR file close unsuccessful");
    }
    readFile=1;
    MoFEMFunctionReturnHot(0);
  }

  MoFEMErrorCode getForceScale(const double ts_t,double &scale) {
    MoFEMFunctionBeginHot;
    if(!fLg) {
      scale = 1; // not scale at all, no history file
      MoFEMFunctionReturnHot(0);
    }
    if(readFile==0) {
      SETERRQ(PETSC_COMM_SELF,1,"data file not read");
    }
    scale = 0;
    double t0 = 0,t1,s0 = tSeries[0],s1,dt;
    std::map<double, double>::iterator tit = tSeries.begin();
    for(;tit!=tSeries.end();tit++) {
      if(tit->first > ts_t) {
        t1 = tit->first;
        s1 = tit->second;
        dt = ts_t - t0;
        scale = s0 + ( (s1-s0)/(t1-t0) )*dt;
        break;
      }
      t0 = tit->first;
      s0 = tit->second;
      scale = s0;
    }
    MoFEMFunctionReturnHot(0);
  }

  //Hassan: this function will loop over data in pair vector ts to find load
  //scale based on ts_t
  MoFEMErrorCode scaleNf(const FEMethod *fe,VectorDouble &Nf) {
    MoFEMFunctionBeginHot;
    double scale;
    const double ts_t = fe->ts_t;
    ierr = getForceScale(ts_t,scale); CHKERRG(ierr);
    Nf *= scale;
    MoFEMFunctionReturnHot(0);
  }
};

struct TimeAccelerogram: public MethodForForceScaling {

  std::map<double,VectorDouble > tSeries;
  int readFile,debug;
  string nAme;

  TimeAccelerogram(string name = "-my_accelerogram"):
  readFile(0),
  debug(1),
  nAme(name) {
    
    ierr = timeData(); CHKERRABORT(PETSC_COMM_WORLD,ierr);
  }

  
  

  MoFEMErrorCode timeData() {
    MoFEMFunctionBeginHot;
    char time_file_name[255];
    PetscBool flg = PETSC_TRUE;
    ierr = PetscOptionsGetString(PETSC_NULL,PETSC_NULL,nAme.c_str(),time_file_name,255,&flg); CHKERRG(ierr);
    if(flg != PETSC_TRUE) {
      SETERRQ1(PETSC_COMM_SELF,1,"*** ERROR %s (time_data FILE NEEDED)",nAme.c_str());
    }
    FILE *time_data = fopen(time_file_name,"r");
    if(time_data == NULL) {
      SETERRQ1(PETSC_COMM_SELF,1,"*** ERROR data file < %s > open unsuccessful",time_file_name);
    }
    double no1 = 0.0;
    VectorDouble no2(3);
    tSeries[no1] = no2;
    while(! feof (time_data)){
      int n = fscanf(time_data,"%lf %lf %lf %lf",&no1,&no2[0],&no2[1],&no2[2]);
      if(n < 0) {
        fgetc(time_data);
        continue;
      }
      if(n != 4){
        SETERRQ1(PETSC_COMM_SELF,1,"*** ERROR read data file error (check input time data file) { n = %d }",n);
      }
      tSeries[no1] = no2;
    }
    int r = fclose(time_data);
    if(debug) {
      std::map<double,VectorDouble >::iterator tit = tSeries.begin();
      for(;tit!=tSeries.end();tit++) {
        PetscPrintf(
          PETSC_COMM_WORLD,
          "** read accelerogram %3.2e time %3.2e %3.2e %3.2e\n",
          tit->first,tit->second[0],tit->second[1],tit->second[2]
        );
      }
    }
    if(r!=0) {
      SETERRQ(PETSC_COMM_SELF,1,"*** ERROR file close unsuccessful");
    }
    readFile=1;
    MoFEMFunctionReturnHot(0);
  }

  MoFEMErrorCode scaleNf(const FEMethod *fe,VectorDouble &Nf) {
    MoFEMFunctionBeginHot;
    if(readFile==0) {
      SETERRQ(PETSC_COMM_SELF,1,"data file not read");
    }
    double ts_t = fe->ts_t;
    VectorDouble acc(3);
    VectorDouble acc0 = tSeries[0],acc1(3);
    double t0 = 0,t1,dt;
    std::map<double,VectorDouble >::iterator tit = tSeries.begin();
    for(;tit!=tSeries.end();tit++) {
      if(tit->first > ts_t) {
        t1 = tit->first;
        acc1 = tit->second;
        dt = ts_t - t0;
        acc = acc0 + ( (acc1-acc0)/(t1-t0) )*dt;
        break;
      }
      t0 = tit->first;
      acc0 = tit->second;
      acc = acc0;
    }
    Nf += acc;
    MoFEMFunctionReturnHot(0);
  }

};

#endif // __TIMEFORCESCALE_HPP__
