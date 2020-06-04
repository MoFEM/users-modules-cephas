How to add a new module and program {#how_to_add_new_module_and_program}
==========================================================

[TOC]

This tutorial assumes that you have installed MoFEM for developer using
the script provided in \ref install_spack. And therefore, you will have paths
for the source code and the binary files (e.g. for release version) of the Core
Library and the Default User Module as follows

- Core Library
  - Source code: *$HOME/mofem_install/mofem-cephas/*
  - Binary files (build directory): *$HOME/mofem_install/lib_release*
- Default User Module
  - Source code: *$HOME/mofem_install/mofem-cephas/mofem/users_modules*
  - Binary files (build directory): *$HOME/mofem_install/um/build_release*

# How to add a new module

## Module developed by someone else

Example would be MoFEM Fracture Module

If you follow the steps described above, you should see the paths for MoFEM
Fracture Module are as follows

- MoFEM Fracture Module
  - Source code: *$HOME/mofem_install/mofem-cephas/mofem/users_modules/mofem_um_fracture_mechanics*
  - Binary files (build directory): *$HOME/mofem_install/um/build_release/mofem_um_fracture_mechanics*

## New module for your own purpose

Example would be something like this mofem_um_basics

If you follow the steps described above, you should see the paths for your own
module are as follows

- Your own module
  - Source code: *$HOME/mofem_install/mofem-cephas/mofem/users_modules/mofem_um_basics*
  - Binary files (build directory): *$HOME/mofem_install/um/build_release/mofem_um_basics*

After creating your own module, you may wish to create a *git* repository for
version control of your module's source code and store it safely online. You can
do so by running following commands

```
cd $HOME/mofem_install/mofem-cephas/mofem/users_modules/mofem_um_basics
git init
git add --all
git commit -m "Initial commit"
```

Then follow the instructions on how to import your local repository to a hosting
service such as [Bitbucket](https://confluence.atlassian.com/bitbucketserver/importing-code-from-an-existing-project-776640909.html#Importingcodefromanexistingproject-Importanexisting,unversionedcodeprojectintoBitbucketServer) or [Github](https://help.github.com/en/github/importing-your-projects-to-github/adding-an-existing-project-to-github-using-the-command-line).

# How to add a new program

Example would be copy and paste program for Poisson's equation with homogeneous
boundary condition. Then build and test run it.

