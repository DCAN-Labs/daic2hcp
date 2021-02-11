#!/bin/bash

# Set up FSL (if not already done so in the running environment)
# Uncomment the following 2 lines (remove the leading #) and correct the FSLDIR setting for your setup
export FSLDIR=/etc/fsl/5.0
. ${FSLDIR}/fsl.sh > /dev/null 2>&1


# Let FreeSurfer know what version of FSL to use
# FreeSurfer uses FSL_DIR instead of FSLDIR to determine the FSL version
export FSL_DIR="${FSLDIR}"


# Set up FreeSurfer (if not already done so in the running environment)
# Uncomment the following 2 lines (remove the leading #) and correct the FREESURFER_HOME setting for your setup
export FREESURFER_HOME=/opt/freesurfer
. ${FREESURFER_HOME}/SetUpFreeSurfer.sh > /dev/null 2>&1


# Set up specific environment variables for the HCP Pipeline
export HCPPIPEDIR=/app/HCPmod
export SCRATCHDIR=/tmp

export HCPPIPEDIR_Templates=${HCPPIPEDIR}/global/templates
export HCPPIPEDIR_Config=${HCPPIPEDIR}/global/config
export HCPPIPEDIR_PostFS=${HCPPIPEDIR}/scripts
export HCPPIPEDIR_fMRISurf=${HCPPIPEDIR}/scripts
export HCPPIPEDIR_Global=${HCPPIPEDIR}/global/scripts
export MSMCONFIGDIR=${HCPPIPEDIR}/MSMConfig
export MSMBin=${HCPPIPEDIR}/MSMBinaries


# Set up DCAN Environment Variables
export EXECSUMDIR=/opt/dcan-tools/executivesummary

# binary dependencies
export ANTSPATH=/usr/lib/ants
export C3DPATH=/opt/c3d/bin
export MSMBINDIR=/
export CARET7DIR=/usr/bin
export WORKBENCHDIR=${CARET7DIR}
export OMP_NUM_THREADS=1
export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=1
export TMPDIR=/tmp
export LANG="en_US.UTF-8"
