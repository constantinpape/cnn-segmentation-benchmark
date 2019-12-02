import os


def set_gpu_env(gpus):
    """ Set gpu environment variables for the current training.
    """
    # don't set cuda vis devices on cluster! so we set env variable in
    # the submisssion script to nwow that we are on the cluster
    train_on_cluster = bool(os.environ.get('TRAIN_ON_CLUSTER', False))
    if not train_on_cluster:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpus))
        gpus = list(range(len(gpus)))
    return gpus
