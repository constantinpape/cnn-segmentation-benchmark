#! /g/kreshuk/pape/Work/software/conda/miniconda3/bin/python
# TODO would be better to use system python, but this won't work in py2

import os
import sys
import inspect
import subprocess
from datetime import datetime

# four hours in minutes
FOUR_HOURS = str(4 * 60)

# default environment
DEFAULT_ENV = 'torch13'


def write_slurm_template(script, out_path, gpu_type,
                         n_threads=7, mem_limit='64G',
                         time_limit=FOUR_HOURS, qos='normal',
                         env_name=DEFAULT_ENV):
    slurm_template = ("#!/bin/bash\n"
                      "#SBATCH -A kreshuk\n"
                      "#SBATCH -N 1\n"
                      "#SBATCH -c %s\n"
                      "#SBATCH --mem %s\n"
                      "#SBATCH -t %s\n"
                      "#SBATCH --qos=%s\n"
                      "#SBATCH -p gpu\n"
                      "#SBATCH -C gpu=%s\n"
                      "#SBATCH --gres=gpu:1\n"
                      "\n"
                      "source activate %s\n"  # TODO conda activate?
                      "export TRAIN_ON_CLUSTER=1\n"  # we set this env variable, so that the script knows we're on slurm
                      "python %s $@") % (n_threads, mem_limit, time_limit,
                                         qos, gpu_type, env_name, script)
    with open(out_path, 'w') as f:
        f.write(slurm_template)


def submit_slurm(script, input_, gpu_type='2080Ti',
                 n_threads='7', mem_limit='64G',
                 time_limit=FOUR_HOURS, qos='normal',
                 env_name=DEFAULT_ENV):
    """ Submit running a the python script with given inputs on a slurm gpu node.
    """

    assert gpu_type in ('2080Ti', '1080Ti')

    tmp_folder = os.path.expanduser('~/.submit_slurm')
    os.makedirs(tmp_folder, exist_ok=True)

    print("Submitting training script %s to cluster" % script)
    print("with arguments %s" % " ".join(input_))

    script_name = os.path.split(script)[1]
    dt = datetime.now().strftime('%Y_%m_%d_%M')
    tmp_name = os.path.splitext(script_name)[0] + dt
    batch_script = os.path.join(tmp_folder, '%s.sh' % tmp_name)
    log = os.path.join(tmp_folder, '%s.log' % tmp_name)
    err = os.path.join(tmp_folder, '%s.err' % tmp_name)

    print("Batch script saved at", batch_script)
    print("Log will be written to %s, error log to %s" % (log, err))
    write_slurm_template(script, batch_script, gpu_type,
                         n_threads, mem_limit,
                         time_limit, qos, env_name)

    cmd = ['sbatch', '-o', log, '-e', err, '-J', script_name,
           batch_script]
    cmd.extend(input_)
    # print(cmd)
    subprocess.run(cmd)


def scrape_kwargs(input_):
    params = inspect.signature(submit_slurm).parameters
    kwarg_names = [name for name in params
                   if params[name].default != inspect._empty]
    kwarg_positions = [i for i, inp in enumerate(input_)
                       if inp in kwarg_names]

    kwargs = {input_[i]: input_[i + 1] for i in kwarg_positions}

    kwarg_positions += [i + 1 for i in kwarg_positions]
    input_ = [inp for i, inp in enumerate(input_) if i not in kwarg_positions]

    return input_, kwargs


if __name__ == '__main__':
    script = os.path.realpath(os.path.abspath(sys.argv[1]))
    input_ = sys.argv[2:]
    # scrape the additional arguments (n_threads, mem_limit, etc. from the input)
    input_, kwargs = scrape_kwargs(input_)
    submit_slurm(script, input_, **kwargs)
