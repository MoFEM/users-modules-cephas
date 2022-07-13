#

set(permissions_default 
   OWNER_WRITE 
   OWNER_READ
   GROUP_READ)
 
 set(permissions_execute 
   OWNER_WRITE 
   OWNER_READ
   GROUP_READ
   OWNER_EXECUTE 
   GROUP_EXECUTE)
 
 tutorials_copy_and_install("*.msh" "${permissions_default}")
 tutorials_copy_and_install("*.cfg" "${permissions_default}")
 tutorials_copy_and_install("*.cub" "${permissions_default}")
 tutorials_copy_and_install("*.h5m" "${permissions_default}")
 tutorials_copy_and_install("*.sh" "${permissions_execute}")
 tutorials_copy_and_install("*.geo" "${permissions_execute}")
 tutorials_copy_and_install("README" "${permissions_execute}")
 tutorials_copy_and_install("param_file.petsc" "${permissions_execute}")
 
    