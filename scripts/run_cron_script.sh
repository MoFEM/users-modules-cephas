#!/bin/sh

export LIBRARY_PATH=/opt/local_boost_1_65_1/lib

CTEST_SCRIPTS_FILE_PATH=/home/lukasz/tmp/cephas_users_modules/users_modules/cmake
CTSET_SCRIPT=CTestScript_rdb-srv1.cmake
CWD=`pwd`

cd $CTEST_SCRIPTS_FILE_PATH
/opt/local/bin/ctest -VV --http1.0 -S $CTSET_SCRIPT >> /home/lukasz/tests_users_modules.log 2>&1
cd $CWD
