from core.data_provider import gta

datasets_map = {
    'gta' : gta,
}


def data_provider(dataset_name, train_data_paths, valid_data_paths, batch_size,
                  img_width, seq_length, index_stride, test_step, is_training=True):
    if dataset_name not in datasets_map:
        raise ValueError('Name of dataset unknown %s' % dataset_name)
    valid_data_list = valid_data_paths.split(',')

    if dataset_name == 'gta':
        input_param = {'paths': valid_data_list,
                       'image_width': img_width,
                       'minibatch_size': batch_size,
                       'seq_length': seq_length,
                       'stride' : index_stride,
                       'test_step' : test_step,
                       'input_data_type': 'float32',
                       'name': dataset_name + ' iterator'}
        input_handle = datasets_map[dataset_name].DataProcess(input_param)
        if is_training:
            train_input_handle = input_handle.get_train_input_handle()
            train_input_handle.begin(do_shuffle=True)
            test_input_handle = input_handle.get_test_input_handle()
            test_input_handle.begin(do_shuffle=False)
            return train_input_handle, test_input_handle
        else:
            test_input_handle = input_handle.get_test_input_handle()
            test_input_handle.begin(do_shuffle=False)
            return test_input_handle
