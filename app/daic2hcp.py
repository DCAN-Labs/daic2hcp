#!/usr/bin/env python3
__doc__ = """converts from DAIC fMRI preprocessing outputs to surface-based HCP 
    ouputs.

    converts all FreeSurfer outputs into CIFTIs, computes a nonlinear mapping 
    to MNI space using ANTs as well as surface-based Conte69/fsLR and projects 
    fMRI data onto fsLR/MNI grayordinates."""
__version__ = 'v0.0.0'

import argparse
import os
import re
import shutil
from glob import glob

import nipype
import nipype.pipeline.engine as pe
from nipype.interfaces import (freesurfer, fsl, utility)

from hcp_nodes import (PostFreeSurfer, FMRISurface, ExecutiveSummary)


def generate_parser():
    """
    creates argparser for module
    :return: argparse.ArgumentParser
    """
    parser = argparse.ArgumentParser(
        prog='daic2hcp',
        description=__doc__,
        epilog="""example: daic2hcp.py /home/user/foobar/daic_convert 
        --mriproc=/home/user/foobar/MRIPROC 
        --boldproc=/home/user/foobar/BOLDPROC 
        --fsurf=/home/user/foobar/FSURF --ncpus 10"""
    )
    parser.add_argument('output_dir', default='./files', help='path to outputs')
    req = parser.add_argument_group('required argumenets')
    req.add_argument('--mriproc', type=str, required=True,
                        help='daic mriproc directory')
    req.add_argument('--boldproc', type=str, required=True,
                        help='daic boldproc directory')
    req.add_argument('--fsurf', type=str, required=True,
                        help='daic freesurfer directory')
    parser.add_argument('--subject-id', type=str, default='subjectid',
                        help='optional freesurfer subject id. Defaults to '
                             '"subjectid"')
    parser.add_argument('--ncpus', type=int, default=1,
                        help='number of cores to use for parallel processing.')
    parser.add_argument('--tmpfs', action='store_true',
                        help='use temp space for intermediate file creation')

    return parser


def main():
    """
    main script
    """
    # read in args
    parser = generate_parser()
    args = parser.parse_args()

    output_dir = args.output_dir
    mriprocdir = os.path.abspath(args.mriproc)
    freesurfer_dir = os.path.abspath(args.fsurf)
    boldprocdir = os.path.abspath(args.boldproc)
    if args.tmpfs:
        base_dir = None
    else:
        os.chdir(output_dir)
        base_dir = output_dir

    # naming specifications from DAIC
    # anatomicals assumed to be in register already
    t1w_file = os.path.join(mriprocdir, './MPR_res.mgz')
    t2w_file = os.path.join(mriprocdir, './T2w_res.mgz')
    mask_file = os.path.abspath(os.path.join(freesurfer_dir, 'mri',
                                             'wmparc.mgz'))
    brain_file = os.path.abspath(os.path.join(freesurfer_dir, 'mri',
                                             'brain.mgz'))
    fmri_files = sorted(map(os.path.abspath, glob(os.path.join(
        boldprocdir, 'BOLD[0-9]*_for_corr_resBOLD.mgz'))))
    rs_files = sorted(map(os.path.abspath, glob(os.path.join(
        boldprocdir, 'rsBOLD_analysis', 'rsBOLD_data_scan[0-9].mgz'))))

    # check inputs
    for path in mriprocdir, freesurfer_dir, boldprocdir, t1w_file, t2w_file, \
                mask_file, brain_file:
        assert os.path.exists(path), '%s not found!' % path
    assert len(fmri_files), 'files matching %s pattern were not found' % \
                            os.path.join(boldprocdir,
                                         'BOLD[0-9]*_for_corr_resBOLD.mgz')

    # generate daic2hcp workflow
    wf = generate_workflow(workflow_name=parser.prog, subjectid=args.subject_id,
                           output_dir=output_dir, t1w_file=t1w_file,
                           t2w_file=t2w_file, fmri_files=fmri_files,
                           rs_files=rs_files, mask_file=mask_file,
                           brain_file=brain_file,
                           fs_source_dir=freesurfer_dir, base_dir=base_dir)
    if args.ncpus > 1:
        wf.run(plugin='MultiProc', plugin_args={'n_procs': args.ncpus})
    else:
        wf.run()


def create_hcp_nodes(output_dir, subject_id):
    """
    generates PostFreeSurfer and fMRISurface Nodes for connecting HCP cifti
    processing.
    :param output_dir: path to output files.
    :param subject_id: freesurfer subject id.
    :return:
    """
    # parameters:
    surfatlasdir = os.environ['HCPPIPEDIR_Templates'] + "/standard_mesh_atlases"
    grayordinatesdir = os.environ[
                           'HCPPIPEDIR_Templates'] + "/91282_Greyordinates"
    grayordinatesres = '2'
    fmrires = '2.0'
    smoothingFWHM = '2'
    hiresmesh = "164"
    lowresmesh = "32"
    subcortgraylabels = os.environ['HCPPIPEDIR_Config'] + \
        "/FreeSurferSubcorticalLabelTableLut.txt"
    freesurferlabels = os.environ['HCPPIPEDIR_Config'] + "/FreeSurferAllLut.txt"
    refmyelinmaps = os.environ['HCPPIPEDIR_Templates'] + \
        "/standard_mesh_atlases/Conte69.MyelinMap_BC.164k_fs_LR.dscalar.nii"
    regname = "FS"
    t1template2mm = os.environ['HCPPIPEDIR_Templates'] + "/MNI152_T1_2mm.nii.gz"
    template2mmmask = os.environ['HCPPIPEDIR_Templates'] + \
        "/MNI152_T1_2mm_brain_mask_dil.nii.gz"
    useT2 = 'true'
    t1template = os.environ['HCPPIPEDIR_Templates'] + "/MNI152_T2_1mm.nii.gz"
    t1templatebrain = os.environ['HCPPIPEDIR_Templates'] + \
        "/MNI152_T1_1mm_brain.nii.gz"
    t2template = os.environ['HCPPIPEDIR_Templates'] + "/MNI152_T2_1mm.nii.gz"
    t2templatebrain = os.environ['HCPPIPEDIR_Templates'] + \
        "/MNI152_T2_1mm_brain.nii.gz"
    t2template2mm = os.environ['HCPPIPEDIR_Templates'] + "/MNI152_T2_2mm.nii.gz"
    templatemask = os.environ['HCPPIPEDIR_Templates'] + \
        "/MNI152_T1_1mm_brain_mask.nii.gz"
    usestudytemplate = 'false'
    # connecting filenames
    out_warp = os.path.join(output_dir, 'MNINonLinear', 'xfms',
                            'fs2standard.nii.gz')

    postfreesurfer = PostFreeSurfer(
        path=output_dir,
        subject=subject_id,
        surfatlasdir=surfatlasdir,
        grayordinatesdir=grayordinatesdir,
        grayordinatesres=grayordinatesres,
        hiresmesh=hiresmesh,
        lowresmesh=lowresmesh,
        subcortgraylabels=subcortgraylabels,
        freesurferlabels=freesurferlabels,
        refmyelinmaps=refmyelinmaps,
        regname=regname,
        t1template=t1template,
        t1templatebrain=t1templatebrain,
        t1template2mm=t1template2mm,
        template2mmmask=template2mmmask,
        t2template=t2template,
        t2templatebrain=t2templatebrain,
        t2template2mm=t2template2mm,
        templatemask=templatemask,
        useT2=useT2,
        usestudytemplate=usestudytemplate,
        out_warp=out_warp
    )
    fmrisurface = FMRISurface(
        path=output_dir,
        subject=subject_id,
        lowresmesh=lowresmesh,
        fmrires=fmrires,
        smoothingFWHM=smoothingFWHM,
        grayordinatesres=grayordinatesres,
        regname=regname
    )
    postfreesurfer = pe.Node(postfreesurfer, name='PostFreeSurfer')
    fmrisurface = pe.Node(fmrisurface, name='FMRISurface')

    return postfreesurfer, fmrisurface


def generate_workflow(**inputs):
    """
    generates computation graph for daic2hcp
    :param t1w_file: t1w mgz file in some space
    :param t2w_file: t2w mgz file aligned to t1w
    :param mask_file: mask or brain mgz file aligned to t1w
    :param brain_file: brain mgz used to align to reference brain
    :param fmri_files: list of fmri mgz files
    :param rs_files: list of fc-preprocessed resting state fmri mgz files
    :param output_dir: desired output "HCP" directory
    :return: nipype workflow
    """
    # setup variables
    subject_id = inputs['subjectid']
    reference = os.path.join(os.environ['HCPPIPEDIR_Templates'],
                             'MNI152_T1_1mm.nii.gz')
    referencebrain = os.path.join(os.environ['HCPPIPEDIR_Templates'],
                                  'MNI152_T1_1mm_brain.nii.gz')
    reference2mm = os.path.join(os.environ['HCPPIPEDIR_Templates'],
                                "MNI152_T1_2mm.nii.gz")

    # io spec
    input_spec = pe.Node(nipype.IdentityInterface(
        fields=['t1w_file', 't2w_file', 'mask_file', 'brain_file']),
        name='input_spec'
    )
    input_func_spec = pe.Node(nipype.IdentityInterface(fields=['fmri_file']),
                              name='input_func_spec')
    hcp_spec = pe.Node(nipype.IdentityInterface(
        fields=['t1w', 't2w', 't1w_acpc_xfm', 't2w_acpc_xfm',
                't2w_to_t1w_xfm', 't1w_distortion', 't2w_distortion',
                'bias_field', 't1w_res', 't2w_res', 't2w_dc', 't1w_res_brain',
                't2w_res_brain', 'wmparc', 'wmparc_1mm', 'fs2standard',
                'standard2fs']),
        name='hcp_spec'
    )

    # connect input DAIC files
    input_spec.inputs.t1w_file = inputs['t1w_file']
    input_spec.inputs.t2w_file = inputs['t2w_file']
    input_spec.inputs.mask_file = inputs['mask_file']
    input_spec.inputs.brain_file = inputs['brain_file']
    input_func_spec.iterables = ('fmri_file', inputs['fmri_files'] + inputs[
        'rs_files'])

    # setup HCP directory specification
    output_dir = os.path.abspath(inputs['output_dir'])
    subjects_dir = os.path.join(output_dir, 'T1w')
    freesurfer_dir = os.path.join(subjects_dir, subject_id)
    native_xfms_dir = os.path.join(subjects_dir, 'xfms')
    t2w_dir = os.path.join(output_dir, 'T2w')
    t2w_xfms_dir = os.path.join(t2w_dir, 'xfms')
    nonlinear_dir = os.path.join(output_dir, 'MNINonLinear')
    results_dir = os.path.join(nonlinear_dir, 'Results')
    nonlin_xfms_dir = os.path.join(nonlinear_dir, 'xfms')
    fs_transforms = os.path.join(freesurfer_dir, 'mri', 'transforms')

    # create directory tree
    for directory in [output_dir, subjects_dir, native_xfms_dir, t2w_dir,
                      t2w_xfms_dir, results_dir, nonlin_xfms_dir]:
        os.makedirs(directory, exist_ok=True)
    if not os.path.isdir(freesurfer_dir):
        shutil.copytree(inputs['fs_source_dir'], freesurfer_dir)
    os.makedirs(fs_transforms, exist_ok=True)

    def get_name(x):
        boldid = re.compile(r'(BOLD[0-9]*).*')
        number = re.compile(r'rsBOLD_data_scan([0-9]*).*')
        basename = os.path.basename(x).split('.')[0]
        match = boldid.match(basename)
        if match is not None:
            name = match.groups()[0]
        else:
            name = 'fcproc%s' % number.match(basename).groups()[0]
        taskname = 'task-%s' % name
        return taskname

    fmrinames = map(get_name, inputs['fmri_files'] + inputs['rs_files'])
    for fmriname in fmrinames:
        directory = os.path.join(results_dir, fmriname)
        os.makedirs(directory, exist_ok=True)

    # HCP filename specification
    hcp_spec.inputs.t1w = os.path.join(subjects_dir, 'T1w.nii.gz')
    hcp_spec.inputs.t2w = os.path.join(t2w_dir, 'T2w.nii.gz')
    hcp_spec.inputs.t1w_acpc_xfm = os.path.join(native_xfms_dir, 'acpc.mat')
    hcp_spec.inputs.t2w_acpc_xfm = os.path.join(t2w_xfms_dir, 'acpc.mat')
    hcp_spec.inputs.t2w_to_t1w_xfm = os.path.join(fs_transforms, 'T2wtoT1w.mat')
    hcp_spec.inputs.t1w_distortion = os.path.join(
        native_xfms_dir, 'T1w_dc.nii.gz')
    hcp_spec.inputs.t2w_distortion = os.path.join(
        native_xfms_dir, 'T2w_reg_dc.nii.gz')
    hcp_spec.inputs.bias_field = os.path.join(
        subjects_dir, 'BiasField_acpc_dc.nii.gz')
    hcp_spec.inputs.t1w_res = os.path.join(
        subjects_dir, 'T1w_acpc_dc_restore.nii.gz')
    hcp_spec.inputs.t1w_res_brain = os.path.join(
        subjects_dir, 'T1w_acpc_dc_restore_brain.nii.gz')
    hcp_spec.inputs.t2w_res = os.path.join(
        subjects_dir, 'T2w_acpc_dc_restore.nii.gz')
    hcp_spec.inputs.t2w_dc = os.path.join(
        subjects_dir, 'T2w_acpc_dc.nii.gz')
    hcp_spec.inputs.t2w_res_brain = os.path.join(
        subjects_dir, 'T2w_acpc_dc_restore_brain.nii.gz')
    hcp_spec.inputs.wmparc = os.path.join(subjects_dir, 'wmparc.nii.gz')
    hcp_spec.inputs.wmparc_1mm = os.path.join(subjects_dir, 'wmparc_1mm.nii.gz')
    hcp_spec.inputs.warp = os.path.join(nonlin_xfms_dir, 'fs2standard.nii.gz')

    ## create workflow components

    # utility
    def rename_func(in_file, path):
        # conditional renaming for func, fcpreproc data handled separately
        import os
        import re
        import shutil
        func_pattern = re.compile(r'(?P<name>BOLD[0-9]*).*')
        fc_pattern = re.compile(r'rsBOLD_data_scan(?P<number>[0-9]*).*')
        basename = os.path.basename(in_file)
        name = func_pattern.match(basename)
        number = fc_pattern.match(basename)
        if name:
            taskname = 'task-%s' % name.groupdict()['name']
        elif number:
            taskname = 'task-fcproc%s' % number.groupdict()['number']
        out_file = os.path.join(path, 'MNINonLinear', 'Results',
                               taskname, taskname + '.nii.gz')
        shutil.copyfile(in_file, out_file)
        return out_file
    rename = pe.Node(utility.Function(input_names=['in_file', 'path'],
                                      output_names=['out_file'],
                                      function=rename_func),
                     name='rename')
    rename.inputs.path = os.path.abspath(output_dir)
    copy_str = 'def f(src, dest): shutil.copy(src, dest); return dest'
    copy = pe.Node(
        utility.Function(
            input_names=['src', 'dest'], output_names=['dest'],
            imports=['import shutil'], function_str=copy_str
        ),
        name='copy'
    )
    basename_str = 'def f(path): return osp.basename(path).split(".")[0]'
    basename = pe.Node(
        utility.Function(
            input_names=['path'], output_names=['out_name'],
            imports=['import os.path as osp'], function_str=basename_str
        ),
        name='basename'
    )

    # mri convert
    convert_t1 = pe.Node(freesurfer.MRIConvert(out_type='niigz',
                                               out_orientation='RAS'),
                         name='convert_t1')
    convert_t2 = pe.Node(freesurfer.MRIConvert(out_type='niigz',
                                               out_orientation='RAS'),
                         name='convert_t2')
    convert_mask = pe.Node(freesurfer.MRIConvert(out_type='niigz',
                                                 resample_type='nearest',
                                                 out_orientation='RAS'),
                           name='convert_mask')
    convert_brain = pe.Node(freesurfer.MRIConvert(out_type='niigz',
                                                  out_orientation='RAS'),
                           name='convert_brain')
    convert_func = pe.Node(freesurfer.MRIConvert(out_type='niigz',
                                                 out_orientation='RAS'),
                           name='convert_func')

    # acpc alignment
    calc_acpc = pe.Node(
        fsl.FLIRT(reference=referencebrain, dof=6, interp='spline', no_search=True),
        name='calc_acpc'
    )
    copy_xfm = copy.clone(name='copy_xfm')
    apply_acpc = pe.Node(
        fsl.FLIRT(reference=reference, apply_xfm=True, interp='spline'),
        name='apply_acpc'
    )
    copy_t2w_res = copy.clone(name='copy_t2w_res')
    apply_acpc_nn = pe.Node(
        fsl.FLIRT(reference=reference, apply_xfm=True,
                  interp='nearestneighbour'),
        name='apply_acpc_nn'
    )
    mask_t1w = pe.Node(fsl.ApplyMask(), name='mask_t1w')
    mask_t2w = pe.Node(fsl.ApplyMask(), name='mask_t2w')
    resample_mask = pe.Node(
        fsl.FLIRT(apply_isoxfm=1, interp='nearestneighbour'),
        name='resample_mask'
    )

    # functional transforms
    select_first = pe.JoinNode(
        utility.Select(index=[0]),
        joinsource='input_func_spec',
        joinfield='inlist',
        name='select_first'
    )
    fs_to_fmri = pe.Node(fsl.FLIRT(cost='mutualinfo', dof=6), name='fs_to_func')
    fmri_to_fs = pe.Node(fsl.ConvertXFM(invert_xfm=True), name='func_to_fs')
    concat_warps = pe.Node(
        fsl.ConvertWarp(relwarp=True, out_relwarp=True, reference=reference2mm),
        name='concat_warps'
    )
    warp_mask = pe.Node(
        fsl.ApplyWarp(ref_file=reference2mm, interp='nn', relwarp=True),
        name='warp_mask'
    )
    mask_func = pe.Node(fsl.ApplyMask(), name='apply_mask')
    apply_warpfield = pe.Node(
        fsl.ApplyWarp(ref_file=reference2mm, interp='spline', relwarp=True),
        name='apply_warpfield'
    )
    timeseries_mean = pe.Node(
        fsl.MeanImage(dimension='T'), name='timeseries_mean'
    )
    renamesb = pe.Node(
        utility.Rename(parse_string=r'task-(?P<name>.*)_.*',
                       format_string='%(path)s/MNINonLinear/Results/'
                                     'task-%(name)s/task-%(name)s_SBRef.nii.gz',
                       path=os.path.abspath(output_dir)),
        name='renamesb'
    )
    renamesb.inputs.path = os.path.abspath(output_dir)

    # identity transforms
    identity_matrix = pe.Node(fsl.preprocess.FLIRT(),
                              name='identity_matrix')  # t1 -> t1 matrix
    zeroes = pe.Node(fsl.BinaryMaths(operation='mul', operand_value=0),
                     name='zeroes')
    repeat_zeroes = pe.Node(utility.Merge(3), name='repeat_zeroes')
    identity_warpfield = pe.Node(fsl.Merge(dimension='t'),
                                 name='identity_warpfield')  # 3D warp in t-dim
    identity_biasfield = pe.Node(fsl.BinaryMaths(operation='add',
                                                 operand_value=1),
                                 name='identity_biasfield')  # bias 1 everywhere
    copy_warpfield = copy.clone(name='copy_warpfield')

    # hcp nodes
    postfreesurfer, fmrisurface = create_hcp_nodes(output_dir, subject_id)
    executivesummary = pe.JoinNode(
        ExecutiveSummary(in_processed=os.path.abspath(output_dir),
                         in_subjectid=subject_id),
        joinfield='in_files',
        joinsource='input_func_spec',
        name='executivesummary'
    )

    ## workflow DAG
    wf = pe.Workflow(name=inputs['workflow_name'], base_dir=inputs['base_dir'])

    # convert to nii.gz
    wf.connect(
        [(input_spec, convert_t1, [('t1w_file', 'in_file')]),
         (input_spec, convert_t2, [('t2w_file', 'in_file')]),
         (input_spec, convert_mask, [('mask_file', 'in_file')]),
         (input_spec, convert_brain, [('brain_file', 'in_file')])]
    )
    # rigid align to acpc/MNI, apply mask
    wf.connect(
        [(convert_brain, calc_acpc, [('out_file', 'in_file')]),
         (calc_acpc, copy_xfm, [('out_matrix_file', 'src')]),
         (copy_xfm, apply_acpc, [('dest', 'in_matrix_file')]),
         (calc_acpc, apply_acpc_nn, [('out_matrix_file', 'in_matrix_file')]),
         (convert_t2, apply_acpc, [('out_file', 'in_file')]),
         (apply_acpc, copy_t2w_res, [('out_file', 'src')]),
         (convert_mask, apply_acpc_nn, [('out_file', 'in_file')]),
         (calc_acpc, mask_t1w, [('out_file', 'in_file')]),
         (apply_acpc_nn, mask_t1w, [('out_file', 'mask_file')]),
         (apply_acpc, mask_t2w, [('out_file', 'in_file')]),
         (apply_acpc_nn, mask_t2w, [('out_file', 'mask_file')]),
         (apply_acpc_nn, resample_mask, [('out_file', 'in_file'),
                                         ('out_file', 'reference')])]
    )
    # create identity transformations for data flow
    wf.connect(
        [(calc_acpc, identity_matrix, [('out_file', 'in_file'),
                                       ('out_file', 'reference')]),
         (calc_acpc, zeroes, [('out_file', 'in_file')]),
         (zeroes, repeat_zeroes, [('out_file', 'in1'), ('out_file', 'in2'),
                                  ('out_file', 'in3')]),
         (repeat_zeroes, identity_warpfield, [('out', 'in_files')]),
         (zeroes, identity_biasfield, [('out_file', 'in_file')]),
         (identity_warpfield, copy_warpfield, [('merged_file', 'src')])]
    )
    # connect postfreesurfer
    # there are more implicit connections, but these suffice dependency graph
    wf.connect(
        [(calc_acpc, postfreesurfer, [('out_file', 'in_t1')]),
         (copy_t2w_res, postfreesurfer, [('dest', 'in_t1_dc')]),
         (identity_warpfield, postfreesurfer, [('merged_file', 'in_warpfield')]),
         (identity_biasfield, postfreesurfer, [('out_file', 'in_biasfield')]),
         (copy_warpfield, postfreesurfer, [('dest', 'in_t2warpfield')]),
         (resample_mask, postfreesurfer, [('out_file', 'in_wmparc')]),
         (mask_t1w, postfreesurfer, [('out_file', 'in_t1brain')]),
         (mask_t2w, postfreesurfer, [('out_file', 'in_t2brain')]),
         (identity_matrix, postfreesurfer, [('out_file', 'in_t2_to_t1')])]
    )
    # transform functionals to final space
    # @TODO leverage SELECT and RENAME utilities with Don's information. In
    #  the interim, functional data is simply named as task-BOLD##
    wf.connect(
        [(input_func_spec, convert_func, [('fmri_file', 'in_file')]),
         (convert_func, select_first, [('out_file', 'inlist')]),
         (convert_t1, fs_to_fmri, [('out_file', 'in_file')]),
         (select_first, fs_to_fmri, [('out', 'reference')]),
         (fs_to_fmri, fmri_to_fs, [('out_matrix_file', 'in_file')]),
         (postfreesurfer, concat_warps, [('out_warp', 'warp1')]),
         (fmri_to_fs, concat_warps, [('out_file', 'premat')]),
         (concat_warps, apply_warpfield, [('out_file', 'field_file')]),
         (convert_func, apply_warpfield, [('out_file', 'in_file')]),
         (convert_mask, warp_mask, [('out_file', 'in_file')]),
         (postfreesurfer, warp_mask, [('out_warp', 'field_file')]),
         (warp_mask, mask_func, [('out_file', 'mask_file')]),
         (apply_warpfield, mask_func, [('out_file', 'in_file')])]
    )
    # connect fmrisurface
    # there are more implicit connections, but these suffice dependency graph
    wf.connect(
        [(mask_func, rename, [('out_file', 'in_file')]),
         (rename, timeseries_mean, [('out_file', 'in_file')]),
         (rename, fmrisurface, [('out_file', 'in_fmri')]),
         (rename, basename, [('out_file', 'path')]),
         (basename, fmrisurface, [('out_name', 'fmriname')]),
         (timeseries_mean, renamesb, [('out_file', 'in_file')]),
         (renamesb, fmrisurface, [('out_file', 'in_sbref')])]
    )
    # connect executivesummary
    wf.connect(
        [(fmrisurface, executivesummary, [('out_file', 'in_files')])]
    )

    # draw workflow: output/daic2hcp/graph.png
    wf.write_graph(graph2use='orig', dotfilename='graph.dot')

    # connect intermediates to hcp filename specifications (not shown in graph)
    wf.connect(
        [(hcp_spec, convert_t1, [('t1w', 'out_file')]),
         (hcp_spec, convert_t2, [('t2w', 'out_file')]),
         (hcp_spec, convert_mask, [('wmparc', 'out_file')]),
         (hcp_spec, calc_acpc, [('t1w_acpc_xfm', 'out_matrix_file')]),
         (hcp_spec, identity_matrix, [('t2w_to_t1w_xfm', 'out_matrix_file')]),
         (hcp_spec, identity_warpfield, [('t1w_distortion', 'merged_file')]),
         (hcp_spec, identity_biasfield, [('bias_field', 'out_file')]),
         (hcp_spec, copy_warpfield, [('t2w_distortion', 'dest')]),
         (hcp_spec, calc_acpc, [('t1w_res', 'out_file')]),
         (hcp_spec, apply_acpc, [('t2w_res', 'out_file')]),
         (hcp_spec, copy_xfm, [('t2w_acpc_xfm', 'dest')]),
         (hcp_spec, copy_t2w_res, [('t2w_dc', 'dest')]),
         (hcp_spec, mask_t1w, [('t1w_res_brain', 'out_file')]),
         (hcp_spec, mask_t2w, [('t2w_res_brain', 'out_file')]),
         (hcp_spec, resample_mask, [('wmparc_1mm', 'out_file')]),
         (hcp_spec, concat_warps, [('warp', 'out_file')])]
    )

    return wf


if __name__ == '__main__':
    main()
