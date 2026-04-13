import random, time, os, shutil
from .operator import load_model_from_checkpoint, _check_existance_checkpoint
import torch

from typing import Any, Callable

def _remove_partial_run_folders(history_file_name: str, krun: int, completed_runs: int):

    # get run directory prefix from history file name
    idx = history_file_name.find('_history')
    assert idx >= 0, f'History file name {history_file_name} does not contain _history'
    run_dir_prefix = history_file_name[:idx]

    for run_id in range(krun):
        run_dir = f'{run_dir_prefix}{run_id}'
        if not os.path.exists(run_dir) and run_id < completed_runs:
            raise FileNotFoundError(f'Run folder {run_dir} not found, but it should be completed')
        if os.path.exists(run_dir) and run_id >= completed_runs:
            try:
                shutil.rmtree(run_dir)
                print(f'> Removed partial run folder: {run_dir}')
            except Exception as exc:
                print(f'> Warning: failed to remove partial run folder {run_dir}: {exc}')

def _backup_history_file(history_file_name: str):
    if history_file_name is None or (not os.path.exists(history_file_name)):
        return None
    ts = time.strftime('%Y%m%d_%H%M%S')
    backup_path = f'{history_file_name}.bak.{ts}'
    shutil.copy2(history_file_name, backup_path)
    print(f'> Backup history file to: {backup_path}')
    return backup_path


def K_fold(dataset_len: int, func_train: Callable, func_eval: Callable = None, k=10, krun=-1, num_train=-1, history_file_name=None):
    '''
    K-fold validation training pipline for models. The training and evaluation process is assigned with to
    Callable function, `func_train` and `func_eval`. The current function's major task is to split dataset
    and run the training / evaluating

    ### paras

    - `dataset_len`: (`int`):    the dataset length. The index for each training and evaluating 
    will be selected from `range(dataset_len)`. Remind to use `airfoil_num` for shapewise = True.

    - `func_train`: (`Callable`):   the function for training the model. Its inputs and outputs must be 
    exactly the same as follows to run the function
        - **args** for function
            - `irun`:   (`int`)   number of run
            - `itrain`: (`List[int]`)    the indice for this run (use `dataset.subset(itrain)` to get subset)
        - **return** for function
            - `model`: (`nn.Module`)  the trained model
        - remarks
            - the dataset is well set by k-fold function, and in most cases there's no need for validation
            during each training. So, the argument `split_train_ratio` for `ModelOperator` in `func_train` 
            to train the model is Recommonded to set to `1.0`

    - `func_test`: (`Callable`, default = `None`): the function for evaluating the model. Its inputs and 
    outputs must be exactly the same as follows to run the function. If is `None`,  there will not be 
    evaluation
        - **args** for function
            - `irun`:   (`int`)   number of run
            - `model`: (`nn.Module`)  the trained model
            - `itrain`: (`List[int]`) the index number of training samples in `fldata` for this run 
            - `itest`: (`List[int]`) the index number of samples not involved in training for this run 
        - **return** for function
            - `errors`: (`np.ndarray`) an array of error for each sample, the size should be (`Ni x Nc`) where Ni
            is number of samples, and Nc is number of error values for each sample (can be the used assigned 
            value)
            - `errstats`: (`np.ndarray`) an array of overall error statistics for training and testing sample
            of this run. The size should be (`2 x Nc'`), where 2 stands for train and test, Nc' is also the number
            of error values
    - `k`: (`int`)  number of folds for k-fold run
    - `krun`: (`int`, default `-1` which means the same as `k`) number of folds actually run
    - `number_train`: (`int`, default `-1` which means all samples for each fold) if is a number < size of 
    the training folds for each run, the `number_train` samples will be randomly selected from training folds
    for each run to train the model with.

    ### retures:

    returns is a dictionary with keys:

    - `train_index`:   (`List[List[int]]`) index of training samples in `fldata` for each k-fold run
    - `test_index`:   (`List[List[int]]`) index of not-training samples in `fldata` for each k-fold run
    - `errors`: (`List[np.ndarray]`) a list of all `errors` return by each run
    - `errstats`: (`List[np.ndarray]`) a list of all `errstats` return by each run

    '''
    
    avg = int(dataset_len / k)
    fold_data_number = [avg+(1, 0)[i > dataset_len - avg * k] for i in range(k)]

    if krun <= 0:   krun = k

    print('---------------------------------------')
    print('Start K-fold training with k = %d, krun = %d' % (k, krun))

    history = {'train_index': [],
               'test_index': [],
               'errors': [],
               'errstats': [], 
               'train_time': []}

    if history_file_name is not None and os.path.exists(history_file_name):
        _backup_history_file(history_file_name)
        history = torch.load(history_file_name, weights_only=False)
        print(f'> Resume from existing history file: {history_file_name}')

    completed_runs = min(len(history['train_index']), len(history['test_index']))
    if completed_runs >= krun:
        print(f'> Found {completed_runs} completed runs, no remaining runs to train.')
        return history
        
    # roll back a potentially interrupted fold at the tail of history
    # _remove_partial_run_folders(history_file_name, krun, completed_runs)
    
    # Fresh run: keep old behavior
    if completed_runs == 0:
        print(f'make {dataset_len} indexs random ')
        all_dataset_indexs = random.sample(range(dataset_len), dataset_len)
    else:
        # Resume with non-overlap test sets
        all_dataset_indexs = []

        for old_irun, old_test in enumerate(history['test_index'][:completed_runs]):
            assert len(old_test) == fold_data_number[old_irun], \
                'Resume run has different number of test samples than previous runs.'
            all_dataset_indexs.extend(old_test)
        
        assert len(all_dataset_indexs) == len(set(all_dataset_indexs)), \
            'Resume run has duplicate test samples.'

        remaining_ids = [idx for idx in range(dataset_len) if idx not in all_dataset_indexs]
        random.shuffle(remaining_ids)
        all_dataset_indexs.extend(remaining_ids)

        print(f'> Resume mode, completed runs = {completed_runs}, remaining runs = {krun - completed_runs}')
        print(f'> remaining samples = {len(remaining_ids)}')
    
    for irun in range(completed_runs, krun):
        print('---------------------------------------')
        print('')
        print('K-fold Run %d' % irun)

        t0 = time.time()

        idx1 = sum(fold_data_number[:irun])
        idx2 = sum(fold_data_number[:irun+1])
        testing_indexs = all_dataset_indexs[idx1:idx2]
        training_indexs = all_dataset_indexs[:idx1] + all_dataset_indexs[idx2:]

        if num_train > 0 and num_train < len(training_indexs):
            # for training dataset is not the whole
            training_indexs = random.sample(training_indexs, num_train)

        history['train_index'].append(training_indexs)
        history['test_index'].append(testing_indexs)
        
        print('    training:  %d from  0 ~ %d, %d ~ end' % (len(training_indexs), idx1, idx2))
        print('    testing :  %d from  %d ~ %d' % (len(testing_indexs), idx1, idx2))

        # training_dataset = Subset(fldata, training_indexs)
        # testing_dataset = Subset(fldata, testing_indexs)

        trained_model = func_train(irun, training_indexs, testing_indexs)

        t1 = time.time()

        print('    Training finish in %.2f sec. ' % (t1 - t0))
        history['train_time'].append((t1 - t0))

        if func_eval is not None:
            print('  Evaluating results...')
            trained_model.eval()
            error, errstat = func_eval(irun, trained_model, training_indexs, testing_indexs)
            history['errors'].append(error)
            history['errstats'].append(errstat)

        if history_file_name is not None:
            torch.save(history, history_file_name)
    # if func_eval is not None:
    #     errors   = np.array(errors)
    #     errstats = np.array(errstats)
    return history

def K_fold_evaluate(func_load, func_eval, history, history_file_name=None):

    k = len(history['train_index'])

    history['errors'] = []
    history['errstats'] = []

    for irun in range(k):

        trained_model = func_load(irun, history['train_index'][irun], history['test_index'][irun])

        print('  Evaluating results...')
        trained_model.eval()
        error, errstat = func_eval(irun, trained_model, history['train_index'][irun], history['test_index'][irun])
        history['errors'].append(error)
        history['errstats'].append(errstat)
        
        if history_file_name is not None:
            torch.save(history, history_file_name)
    return history