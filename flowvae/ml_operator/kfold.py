import random, time, os
from .operator import load_model_from_checkpoint, _check_existance_checkpoint

from typing import Callable


def K_fold(dataset_len: int, func_train: Callable, func_eval: Callable = None, k=10, krun=-1, num_train=-1):
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
    
    print(f'make {dataset_len} indexs random ')
    all_dataset_indexs = random.sample(range(dataset_len), dataset_len)

    if krun <= 0:   krun = k

    print('---------------------------------------')
    print('Start K-fold training with k = %d, krun = %d' % (k, krun))

    history = {'train_index': [],
               'test_index': [],
               'errors': [],
               'errstats': [], 
               'train_time': []}

    for irun in range(krun):

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

    # if func_eval is not None:
    #     errors   = np.array(errors)
    #     errstats = np.array(errstats)
    return history

def K_fold_evaluate(fldata, model, func_eval, folder, history, device):

    k = len(history['train_index'])

    history['errors'] = []
    history['errstats'] = []

    for irun in range(k):

        load_model_from_checkpoint(model, epoch=299, folder=os.path.join('save', folder + str(irun)), device=device)
        error, errstat = func_eval(irun, model, fldata, history['train_index'][irun], history['test_index'][irun])
        history['errors'].append(error)
        history['errstats'].append(errstat)

    return history