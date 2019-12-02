import os
import json
import argparse
import luigi
from cluster_tools.inference import InferenceLocal, InferenceSlurm


def evaluate_time(tmp_folder):
    pass


def run_prediction(weigth_folder, gpu, local, mixed_precision):
    task = InferenceLocal if local else InferenceSlurm
    name = os.path.split(weigth_folder)[1]
    tmp_folder = os.path.join('tmp_folders', name)

    config_dir = os.path.join(tmp_folder, 'config')
    os.makedirs(config_dir, exist_ok=True)

    block_shape = []
    shebang = '/g/kreshuk/pape/Work/software/conda/miniconda3/envs/torch13/bin/python'
    conf = {'block_shape': block_shape, 'shebang': shebang}
    with open(os.path.join(config_dir, 'global.config'), 'w') as f:
        json.dump(conf, f)

    gpu_type = name.split('_')
    assert gpu_type in ('1080Ti', '2080Ti', 'P40')

    input_path = '../data/sample_A_padded_20160501.hdf'
    output_path = os.path.join(tmp_folder, 'out.n5')

    t = task(max_jobs=1, tmp_folder=tmp_folder, config_dir=config_dir,
             input_path=input_path, input_key='',
             output_path=output_path)
    luigi.build([t], local_scheduler=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('weight_folder', type=str)
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--local', default=1, type=int)
    parser.add_argument('--mixed_precision', default=0, type=int)
    args = parser.parse_args()
    run_prediction(args.weigth_folder, args.gpu,
                   bool(args.local), bool(args.mixed_precision))
