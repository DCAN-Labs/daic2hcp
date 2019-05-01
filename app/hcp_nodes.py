import os

from nipype.interfaces.base import (traits, TraitedSpec, CommandLineInputSpec,
                                    CommandLine, File, Directory)

here = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'HCPmod')


class PostFreeSuferInputSpec(CommandLineInputSpec):
    # required inputs
    in_t1 = File(desc="T1w restored image file", exists=True)
    in_t1_dc = File(desc="T1w dc (not restored) image file", exists=True)
    in_t2_to_t1 = File(desc="T2w to T1w rigid", exists=True)
    in_biasfield = File(desc="in bias field", exists=True)
    in_warpfield = File(desc="in distortion field", exists=True)
    in_t2warpfield = File(desc="in t2 distortion field", exists=True)
    in_wmparc = File(desc='freesurfer wmparc at 1mm iso', exists=True)
    in_t1brain = File(desc="T1w brain image file", exists=True)
    in_t2brain = File(desc="in T2w brain image file", exists=True)

    path = Directory(desc="base HCP files directory", exists=True,
                     mandatory=True, argstr='--path=%s')
    subject = traits.Str(argstr='--subject=%s')
    surfatlasdir = Directory(exists=True, mandatory=True,
                             argstr='--surfatlasdir=%s')
    grayordinatesdir = Directory(exists=True, mandatory=True,
                                 argstr='--grayordinatesdir=%s')
    grayordinatesres = traits.Str(mandatory=True,
                                  argstr='--grayordinatesres=%s')
    hiresmesh = traits.Str(mandatory=True, argstr='--hiresmesh=%s')
    lowresmesh = traits.Str(mandatory=True, argstr='--lowresmesh=%s')
    subcortgraylabels = File(exists=True, mandatory=True,
                             argstr='--subcortgraylabels=%s')
    freesurferlabels = File(exists=True, mandatory=True,
                            argstr='--freesurferlabels=%s')
    refmyelinmaps = File(exists=True, mandatory=True,
                         argstr='--refmyelinmaps=%s')
    regname = traits.Str(mandatory=True, argstr='--regname=%s')
    t1template = File(exists=True, mandatory=True, argstr='--t1template=%s')
    t1templatebrain = File(exists=True, mandatory=True,
                           argstr='--t1templatebrain=%s')
    t1template2mm = File(exists=True, mandatory=True,
                         argstr='--t1template2mm=%s')
    template2mmmask = File(exists=True, mandatory=True,
                           argstr='--template2mmmask=%s')
    t2template = File(exists=True, mandatory=True,  argstr='--t2template=%s')
    t2templatebrain = File(exists=True, mandatory=True,
                           argstr='--t2templatebrain=%s')
    t2template2mm = File(exists=True, mandatory=True,
                         argstr='--t2template2mm=%s')
    templatemask = File(exists=True, mandatory=True, argstr='--templatemask=%s')
    useT2 = traits.Str(argstr='--useT2=%s')
    usestudytemplate = traits.Str(argstr='--useStudyTemplate=%s')
    out_warp = File()


class PostFreeSurferOutputSpec(TraitedSpec):
    # out_surface = File()
    out_warp = File(exists=True)


class PostFreeSurfer(CommandLine):
    _cmd = os.path.join(here, 'PostFreeSurferPipeline.sh')
    input_spec = PostFreeSuferInputSpec
    output_spec = PostFreeSurferOutputSpec

    def _list_outputs(self):
        outputs = self.output_spec().get()
        path = self.inputs.path
        out_warp = self.inputs.out_warp
        if not out_warp:
            out_warp = os.path.join(path, 'MNINonLinear', 'xfms',
                                    'fs2standard.nii.gz')
        outputs["out_warp"] = out_warp
        return outputs


class SurfaceInputSpec(CommandLineInputSpec):
    in_fmri = File(desc="input fmri time series", exists=True, mandatory=True)
    in_sbref = File(desc="input single fmri volume reference", exists=True,
                    mandatory=True)
    #in_surface = File(desc='input midthickness file (just a formality for the '
    #                       'graph)')
    path = Directory(desc="base HCP files directory", exists=True,
                     mandatory=True, argstr='--path=%s')
    subject = traits.Str(desc='subject id', mandatory=True,
                         argstr='--subject=%s')
    fmriname = traits.Str(desc='basename of fmri', argstr='--fmriname=%s',
                          mandatory=True)
    lowresmesh = traits.Str(desc='lower mesh resolution in thousands',
                            argstr='--lowresmesh=%s', mandatory=True)
    fmrires = traits.Str(desc='fmri resolution as decimal',
                         argstr='--fmrires=%s', mandatory=True)
    smoothingFWHM = traits.Str(
        desc='gauss smoothing kernel full width at half max',
        argstr='--smoothingFWHM=%s', mandatory=True)
    grayordinatesres = traits.Str(desc='fmri resolution as decimal',
                         argstr='--grayordinatesres=%s', mandatory=True)
    regname = traits.Str(desc='fmri resolution as decimal',
                         argstr='--regname=%s', mandatory=True)


class SurfaceOutputSpec(TraitedSpec):
    out_file = File()


class FMRISurface(CommandLine):
    _cmd = os.path.join(here, 'GenericfMRISurfaceProcessingPipeline.sh')
    input_spec = SurfaceInputSpec
    output_spec = SurfaceOutputSpec


class ExecutiveSummaryInputSpec(CommandLineInputSpec):
    in_files = traits.List()
    in_unprocessed = Directory(exists=True, argstr='--unproc_root=%s')
    in_processed = Directory(exists=True, argstr='--deriv_root=%s')
    in_subjectid = traits.Str(argstr='--subject_id=%s')
    in_executivesummary = traits.Str(argstr='--ex_summ_dir=%s')


class ExecutiveSummaryOutputSpec(TraitedSpec):
    pass


class ExecutiveSummary(CommandLine):
    _cmd = os.path.join(here, 'executivesummary', 'summary_tools',
                        'executivesummary_wrapper.sh')
    input_spec = ExecutiveSummaryInputSpec
    output_spec = ExecutiveSummaryOutputSpec
