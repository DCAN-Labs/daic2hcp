#!/bin/bash
set -ex

# Requirements for this script
#  installed versions of: FSL (version 5.0.6), FreeSurfer (version 5.3.0-HCP), gradunwarp (HCP version 1.0.1)
#  environment: FSLDIR , FREESURFER_HOME , HCPPIPEDIR , CARET7DIR , PATH (for gradient_unwarp.py)

########################################## PIPELINE OVERVIEW ########################################## 

#TODO

########################################## OUTPUT DIRECTORIES ########################################## 

#TODO

# --------------------------------------------------------------------------------
#  Load Function Libraries
# --------------------------------------------------------------------------------

source $HCPPIPEDIR/global/scripts/log.shlib  # Logging related functions
source $HCPPIPEDIR/global/scripts/opts.shlib # Command line option functions

########################################## SUPPORT FUNCTIONS ########################################## 

# --------------------------------------------------------------------------------
#  Usage Description Function
# --------------------------------------------------------------------------------

show_usage() {
    echo "Usage information To Be Written"
    exit 1
}

# --------------------------------------------------------------------------------
#   Establish tool name for logging
# --------------------------------------------------------------------------------
log_SetToolName "PostFreeSurferPipeline.sh"

################################################## OPTION PARSING #####################################################

opts_ShowVersionIfRequested $@

if opts_CheckForHelpRequest $@; then
    show_usage
fi

log_Msg "Parsing Command Line Options"

# Input Variables
StudyFolder=`opts_GetOpt1 "--path" $@`
Subject=`opts_GetOpt1 "--subject" $@`
SurfaceAtlasDIR=`opts_GetOpt1 "--surfatlasdir" $@`
GrayordinatesSpaceDIR=`opts_GetOpt1 "--grayordinatesdir" $@`
GrayordinatesResolutions=`opts_GetOpt1 "--grayordinatesres" $@`
HighResMesh=`opts_GetOpt1 "--hiresmesh" $@`
LowResMeshes=`opts_GetOpt1 "--lowresmesh" $@`
SubcorticalGrayLabels=`opts_GetOpt1 "--subcortgraylabels" $@`
FreeSurferLabels=`opts_GetOpt1 "--freesurferlabels" $@`
ReferenceMyelinMaps=`opts_GetOpt1 "--refmyelinmaps" $@`
CorrectionSigma=`opts_GetOpt1 "--mcsigma" $@`
RegName=`opts_GetOpt1 "--regname" $@`
InflateExtraScale=`opts_GetOpt1 "--inflatescale" $@`
useT2=`opts_GetOpt1 "--useT2" $@`

# Extra arguments for ANTs based Atlas Registration
useStudyTemplate=`opts_GetOpt1 "--useStudyTemplate" $@`
StudyTemplate=`opts_GetOpt1 "--studytemplate" $@`
StudyTemplateBrain=`opts_GetOpt1 "--studytemplatebrain" $@`
T1wTemplate=`opts_GetOpt1 "--t1template" $@`
T1wTemplateBrain=`opts_GetOpt1 "--t1templatebrain" $@`
T1wTemplate2mm=`opts_GetOpt1 "--t1template2mm" $@`
T1wTemplate2mmBrain=`opts_GetOpt1 "--t1template2mmbrain" $@`
T2wTemplate=`opts_GetOpt1 "--t2template" $@`
T2wTemplateBrain=`opts_GetOpt1 "--t2templatebrain" $@`
T2wTemplate2mm=`opts_GetOpt1 "--t2template2mm" $@`
TemplateMask=`opts_GetOpt1 "--templatemask" $@`
Template2mmMask=`opts_GetOpt1 "--template2mmmask" $@`

#################### ABIDE FIX ########################
Reference2mm=`opts_GetOpt1 "--reference2mm" $@`
Reference2mmMask=`opts_GetOpt1 "--reference2mmmask" $@`
FNIRTConfig=`opts_GetOpt1 "--config" $@`
#######################################################

log_Msg "RegName: ${RegName}"

# default parameters
CorrectionSigma=`opts_DefaultOpt $CorrectionSigma $(echo "sqrt ( 200 )" | bc -l)`
RegName=`opts_DefaultOpt $RegName FS`
InflateExtraScale=`opts_DefaultOpt $InflateExtraScale 1`

PipelineScripts=${HCPPIPEDIR_PostFS}

#Naming Conventions
T1wImage="T1w_acpc_dc"
T1wFolder="T1w" #Location of T1w images
T2wFolder="T2w" #Location of T1w images
T2wImage="T2w_acpc_dc" 
AtlasSpaceFolder="MNINonLinear"
NativeFolder="Native"
FreeSurferFolder="$Subject"
FreeSurferInput="T1w_acpc_dc_restore_1mm"
AtlasTransform="acpc_dc2standard"
InverseAtlasTransform="standard2acpc_dc"
AtlasSpaceT1wImage="T1w_restore"
AtlasSpaceT2wImage="T2w_restore"
T1wRestoreImage="T1w_acpc_dc_restore"
T2wRestoreImage="T2w_acpc_dc_restore"
OrginalT1wImage="T1w"
OrginalT2wImage="T2w"
T1wImageBrainMask="brainmask_fs"
InitialT1wTransform="acpc.mat"
dcT1wTransform="T1w_dc.nii.gz"
InitialT2wTransform="acpc.mat"
dcT2wTransform="T2w_reg_dc.nii.gz"
FinalT2wTransform="${Subject}/mri/transforms/T2wtoT1w.mat"
BiasField="BiasField_acpc_dc"
OutputT1wImage="T1w_acpc_dc"
OutputT1wImageRestore="T1w_acpc_dc_restore"
OutputT1wImageRestoreBrain="T1w_acpc_dc_restore_brain"
OutputMNIT1wImage="T1w"
OutputMNIT1wImageRestore="T1w_restore"
OutputMNIT1wImageRestoreBrain="T1w_restore_brain"
OutputT2wImage="T2w_acpc_dc"
OutputT2wImageRestore="T2w_acpc_dc_restore"
OutputT2wImageRestoreBrain="T2w_acpc_dc_restore_brain"
OutputMNIT2wImage="T2w"
OutputMNIT2wImageRestore="T2w_restore"
OutputMNIT2wImageRestoreBrain="T2w_restore_brain"
OutputOrigT1wToT1w="OrigT1w2T1w.nii.gz"
OutputOrigT1wToStandard="OrigT1w2standard.nii.gz" #File was OrigT2w2standard.nii.gz, regnerate and apply matrix
OutputOrigT2wToT1w="OrigT2w2T1w.nii.gz" #mv OrigT1w2T2w.nii.gz OrigT2w2T1w.nii.gz
OutputOrigT2wToStandard="OrigT2w2standard.nii.gz"
BiasFieldOutput="BiasField"
Jacobian="NonlinearRegJacobians.nii.gz"

T1wFolder="$StudyFolder"/"$T1wFolder" 
T2wFolder="$StudyFolder"/"$T2wFolder" 
AtlasSpaceFolder="$StudyFolder"/"$AtlasSpaceFolder"
FreeSurferFolder="$T1wFolder"/"$FreeSurferFolder"
AtlasTransform="$AtlasSpaceFolder"/xfms/"$AtlasTransform"
InverseAtlasTransform="$AtlasSpaceFolder"/xfms/"$InverseAtlasTransform"
FSAtlasTransform="$AtlasSpaceFolder"/xfms/fs2standard
InverseFSAtlasTransform="$AtlasSpaceFolder"/xfms/standard2fs

#Conversion of FreeSurfer Volumes and Surfaces to NIFTI and GIFTI and Create Caret Files and Registration
log_Msg "Conversion of FreeSurfer Volumes and Surfaces to NIFTI and GIFTI and Create Caret Files and Registration"
log_Msg "RegName: ${RegName}"

log_Msg "Atlas Registration was taken out of PreFreeSurfer and reimplemented here in PostFreeSurfer because the new ANTs based method is improved when using the brain mask generated in FreeSurfer as opposed to the one generated in PreFreeSurfer"


#Create FreeSurfer Brain Mask. (Removed from FreeSurfer2CaretConvertAndRegisterNonlinear.sh)
# mri_convert --out_orientation RAS -rt nearest -rl "$T1wFolder"/"$T1wRestoreImage".nii.gz "$FreeSurferFolder"/mri/wmparc.mgz "$T1wFolder"/wmparc_1mm.nii.gz
fslmaths "$T1wFolder"/wmparc_1mm.nii.gz -bin -dilD -dilD -dilD -ero -ero "$T1wFolder"/"$T1wImageBrainMask"_1mm.nii.gz
${CARET7DIR}/wb_command -volume-fill-holes "$T1wFolder"/"$T1wImageBrainMask"_1mm.nii.gz "$T1wFolder"/"$T1wImageBrainMask"_1mm.nii.gz
fslmaths "$T1wFolder"/"$T1wImageBrainMask"_1mm.nii.gz -bin "$T1wFolder"/"$T1wImageBrainMask"_1mm.nii.gz
applywarp --rel --interp=nn -i "$T1wFolder"/"$T1wImageBrainMask"_1mm.nii.gz -r "$T1wFolder"/"$T1wRestoreImage".nii.gz --premat=$FSLDIR/etc/flirtsch/ident.mat -o "$T1wFolder"/"$T1wImageBrainMask".nii.gz
convertwarp --relout --rel --ref="$T1wFolder"/"$T1wImageBrainMask" --premat="$T1wFolder"/xfms/"$InitialT1wTransform" --warp1="$T1wFolder"/xfms/"$dcT1wTransform" --out="$T1wFolder"/xfms/"$OutputOrigT1wToT1w"
#applywarp --rel --interp=nn -i "$T1wFolder"/"$T1wImageBrainMask"_1mm.nii.gz -r "$T1wFolder"/"$T1wRestoreImage".nii.gz -w "$AtlasTransform" -o "$AtlasSpaceFolder"/"$T1wImageBrainMask".nii.gz
applywarp --rel --interp=spline -i "$T1wFolder"/"$OrginalT1wImage" -r "$T1wFolder"/"$T1wImageBrainMask" -w "$T1wFolder"/xfms/"$OutputOrigT1wToT1w" -o "$T1wFolder"/"$OutputT1wImage"
fslmaths "$T1wFolder"/"$OutputT1wImage" -abs "$T1wFolder"/"$OutputT1wImage" -odt float
fslmaths "$T1wFolder"/"$OutputT1wImage" -div "$T1wFolder"/"$BiasField" "$T1wFolder"/"$OutputT1wImageRestore"


# Run ANTS Atlas Registration from PreFreeSurfer using the freesurfer mask (brainmask_fs.nii.gz)

if ${useStudyTemplate:-false}; then

        # ------------------------------------------------------------------------------
        #  Atlas Registration to MNI152: ANTs with Intermediate Template
        #  Also applies registration to T1w and T2w images
        #  Modified 20170330 by EF to include the option for a native mask in registration
        # ------------------------------------------------------------------------------

        log_Msg "Performing Atlas Registration to MNI152 (ANTs based with intermediate template)"

        "${PipelineScripts}"/AtlasRegistrationToMNI152_ANTsIntermediateTemplate.sh \
            --workingdir=${AtlasSpaceFolder} \
            --t1=${T1wFolder}/${T1wImage} \
            --t1rest=${T1wFolder}/${T1wImage}_restore \
            --t1restbrain=${T1wFolder}/${T1wImage}_restore_brain \
            --t1mask=${T1wFolder}/brainmask_fs \
            --t2=${T1wFolder}/${T2wImage} \
            --t2rest=${T1wFolder}/${T2wImage}_restore \
            --t2restbrain=${T1wFolder}/${T2wImage}_restore_brain \
            --studytemplate=${StudyTemplate} \
            --studytemplatebrain=${StudyTemplateBrain} \
            --ref=${T1wTemplate} \
            --refbrain=${T1wTemplateBrain} \
            --refmask=${TemplateMask} \
            --ref2mm=${T1wTemplate2mm} \
            --ref2mmbrain=${T1wTemplate2mmBrain} \
            --ref2mmmask=${Template2mmMask} \
            --owarp=${AtlasSpaceFolder}/xfms/acpc_dc2standard.nii.gz \
            --oinvwarp=${AtlasSpaceFolder}/xfms/standard2acpc_dc.nii.gz \
            --ot1=${AtlasSpaceFolder}/${OutputMNIT1wImage} \
            --ot1rest=${AtlasSpaceFolder}/${OutputMNIT1wImageRestore} \
            --ot1restbrain=${AtlasSpaceFolder}/${OutputMNIT1wImageRestoreBrain} \
            --ot2=${AtlasSpaceFolder}/${OutputMNIT2wImage} \
            --ot2rest=${AtlasSpaceFolder}/${OutputMNIT2wImageRestore} \
            --ot2restbrain=${AtlasSpaceFolder}/${OutputMNIT2wImageRestoreBrain} \
            --useT2=${useT2} \
            --T1wFolder=${T1wFolder}

        log_Msg "Completed"

else

        # ------------------------------------------------------------------------------
        #  Atlas Registration to MNI152: FLIRT + FNIRT
        #  Also applies registration to T1w and T2w image
        #  Modified 20170330 by EF to include the option for a native mask in registration
        # ------------------------------------------------------------------------------

        log_Msg "Performing Atlas Registration to MNI152 (ANTs based)"

        "$PipelineScripts"/AtlasRegistrationToMNI152_ANTsbased.sh \
            --workingdir=${AtlasSpaceFolder} \
            --t1=${T1wFolder}/${T1wImage} \
            --t1rest=${T1wFolder}/${T1wImage}_restore \
            --t1restbrain=${T1wFolder}/${T1wImage}_restore_brain \
            --t1mask=${T1wFolder}/brainmask_fs \
            --t2=${T1wFolder}/${T2wImage} \
            --t2rest=${T1wFolder}/${T2wImage}_restore \
            --t2restbrain=${T1wFolder}/${T2wImage}_restore_brain \
            --ref=${T1wTemplate} \
            --refbrain=${T1wTemplateBrain} \
            --refmask=${TemplateMask} \
            --ref2mm=${T1wTemplate2mm} \
            --ref2mmbrain=${T1wTemplate2mmBrain} \
            --ref2mmmask=${Template2mmMask} \
            --owarp=${AtlasSpaceFolder}/xfms/acpc_dc2standard.nii.gz \
            --oinvwarp=${AtlasSpaceFolder}/xfms/standard2acpc_dc.nii.gz \
            --ot1=${AtlasSpaceFolder}/${OutputMNIT1wImage} \
            --ot1rest=${AtlasSpaceFolder}/${OutputMNIT1wImageRestore} \
            --ot1restbrain=${AtlasSpaceFolder}/${OutputMNIT1wImageRestoreBrain} \
            --ot2=${AtlasSpaceFolder}/${OutputMNIT2wImage} \
            --ot2rest=${AtlasSpaceFolder}/${OutputMNIT2wImageRestore} \
            --ot2restbrain=${AtlasSpaceFolder}/${OutputMNIT2wImageRestoreBrain} \
            --fnirtconfig=${FNIRTConfig} \
            --useT2=${useT2} \
            --T1wFolder=${T1wFolder} \
            --usemask=${usemask} \
            --useAntsReg=${useAntsReg}

        log_Msg "Completed"

fi

####### @ DAIC EDIT @ #######
# create fs2standard
convertwarp --relout --rel --ref="${AtlasSpaceFolder}/${OutputMNIT1wImageRestoreBrain}" --premat="$T1wFolder"/xfms/"$InitialT1wTransform" --warp1="${AtlasSpaceFolder}/xfms/acpc_dc2standard.nii.gz" --out="$FSAtlasTransform".nii.gz
convert_xfm -omat "$T1wFolder"/xfms/invacpc.mat -inverse "$T1wFolder"/xfms/"$InitialT1wTransform"
convertwarp --relout --rel --ref="$T1wFolder"/T1w.nii.gz --postmat="$T1wFolder"/xfms/invacpc.mat --warp1="${AtlasSpaceFolder}/xfms/standard2acpc_dc.nii.gz" --out="$InverseFSAtlasTransform".nii.gz
#############################

"$PipelineScripts"/FreeSurfer2CaretConvertAndRegisterNonlinear.sh "$StudyFolder" "$Subject" "$T1wFolder" "$AtlasSpaceFolder" "$NativeFolder" "$FreeSurferFolder" "$FreeSurferInput" "$T1wRestoreImage" "$T2wRestoreImage" "$SurfaceAtlasDIR" "$HighResMesh" "$LowResMeshes" "$FSAtlasTransform" "$InverseFSAtlasTransform" "$AtlasSpaceT1wImage" "$AtlasSpaceT2wImage" "$T1wImageBrainMask" "$FreeSurferLabels" "$GrayordinatesSpaceDIR" "$GrayordinatesResolutions" "$SubcorticalGrayLabels" "$RegName" "$InflateExtraScale" "$useT2"

#Create FreeSurfer ribbon file at full resolution
log_Msg "Create FreeSurfer ribbon file at full resolution"
"$PipelineScripts"/CreateRibbon.sh "$StudyFolder" "$Subject" "$T1wFolder" "$AtlasSpaceFolder" "$NativeFolder" "$AtlasSpaceT1wImage" "$T1wRestoreImage" "$FreeSurferLabels"

#Myelin Mapping
log_Msg "Myelin Mapping"
log_Msg "RegName: ${RegName}"

"$PipelineScripts"/CreateMyelinMaps.sh "$StudyFolder" "$Subject" "$AtlasSpaceFolder" "$NativeFolder" "$T1wFolder" "$HighResMesh" "$LowResMeshes" "$T1wFolder"/"$OrginalT1wImage" "$T2wFolder"/"$OrginalT2wImage" "$T1wFolder"/"$T1wImageBrainMask" "$T1wFolder"/xfms/"$InitialT1wTransform" "$T1wFolder"/xfms/"$dcT1wTransform" "$T2wFolder"/xfms/"$InitialT2wTransform" "$T1wFolder"/xfms/"$dcT2wTransform" "$T1wFolder"/"$FinalT2wTransform" "$AtlasTransform" "$T1wFolder"/"$BiasField" "$T1wFolder"/"$OutputT1wImage" "$T1wFolder"/"$OutputT1wImageRestore" "$T1wFolder"/"$OutputT1wImageRestoreBrain" "$AtlasSpaceFolder"/"$OutputMNIT1wImage" "$AtlasSpaceFolder"/"$OutputMNIT1wImageRestore" "$AtlasSpaceFolder"/"$OutputMNIT1wImageRestoreBrain" "$T1wFolder"/"$OutputT2wImage" "$T1wFolder"/"$OutputT2wImageRestore" "$T1wFolder"/"$OutputT2wImageRestoreBrain" "$AtlasSpaceFolder"/"$OutputMNIT2wImage" "$AtlasSpaceFolder"/"$OutputMNIT2wImageRestore" "$AtlasSpaceFolder"/"$OutputMNIT2wImageRestoreBrain" "$T1wFolder"/xfms/"$OutputOrigT1wToT1w" "$T1wFolder"/xfms/"$OutputOrigT1wToStandard" "$T1wFolder"/xfms/"$OutputOrigT2wToT1w" "$T1wFolder"/xfms/"$OutputOrigT2wToStandard" "$AtlasSpaceFolder"/"$BiasFieldOutput" "$AtlasSpaceFolder"/"$T1wImageBrainMask" "$AtlasSpaceFolder"/xfms/"$Jacobian" "$ReferenceMyelinMaps" "$CorrectionSigma" "$RegName" "$useT2"

log_Msg "Completed"
