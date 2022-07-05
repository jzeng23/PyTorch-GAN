import json
import numpy as np

epoch = 9600

json_file_name = 'barcodes/loss_mse/ExponentialLR/0.0002/start_at_epoch_2000/gamma_0.9996/epoch_' + str(epoch) + '.json'
ground_truth_file_name = 'barcodes/ground_truth_mini_dataset.json'
file = open(json_file_name)
ground_truth_file = open(ground_truth_file_name)
obj = json.load(file)
gt = json.load(ground_truth_file)
# obj['foo.png'][i][j][k] = foo.png, ith dimensional barcode, jth bar, birth(if k==0) or death(if k==1). Note: everything is a nested list

def save_json(dict, path, create_file = True):
    json_obj = json.dumps(dict)
    if create_file:
        file = open(path, 'x')
    else:
        file = open(path, 'w')
    file.write(json_obj)



result_num_bars = {}
result_avg_length = {}
new_barcodes_all = {}

target_betas = np.loadtxt('../../data/betas_mini.csv', delimiter=',')

for j in range(target_betas.shape[0]):

    key = 'data_' + str(j) + '.png'

    barcode0_gt = np.asarray(gt[key][0])
    barcode0 = np.asarray(obj[key][0])
    
    gt_num_bars_0 = target_betas[j, 0]
    gt_sum_0 = 0
    num_bars_0 = 0
    sum_0 = 0

    new_barcodes = {}

    new_barcode0 = []

    for i in range(barcode0.shape[0]):
        bar_length_0 = barcode0[i, 0] - barcode0[i, 1]
        if bar_length_0 != 0:
            new_barcode0.append([barcode0[i, 0], barcode0[i, 1]])
            if bar_length_0 != np.inf:
                num_bars_0 += 1
                sum_0 += bar_length_0
        
        gt_bar_length_0 = barcode0_gt[i, 0] - barcode0_gt[i, 1]
        if gt_bar_length_0 != 0 and gt_bar_length_0 != np.inf:
            gt_sum_0 += gt_bar_length_0
    
    new_barcodes['dim0'] = new_barcode0 

    entry_num_bars = {}
    entry_avg_length = {}

    dim0 = {}
    dim0['target'] = gt_num_bars_0
    dim0['output'] = num_bars_0
    dim0['difference_between_output_and_target'] = num_bars_0 - gt_num_bars_0
    entry_num_bars['dim0'] = dim0

    dim0_l = {}
    dim0_l['target'] = 1
    dim0_l['output'] = sum_0 / num_bars_0
    dim0_l['difference_between_output_and_target'] = sum_0 / num_bars_0 - 1
    entry_avg_length['dim0'] = dim0_l

    barcode1_gt = np.asarray(gt[key][1])
    barcode1 = np.asarray(obj[key][1])

    gt_num_bars_1 = target_betas[j, 1]
    gt_sum_1 = 0
    num_bars_1 = 0
    sum_1 = 0

    new_barcode1 = []

    for i in range(barcode1.shape[0]):
        bar_length_1 = barcode1[i, 0] - barcode1[i, 1]
        if bar_length_1 != 0:
            new_barcode1.append([barcode1[i, 0], barcode1[i, 1]])
            if bar_length_1 != np.inf:
                num_bars_1 += 1
                sum_1 += bar_length_1
        
        gt_bar_length_1 = barcode1_gt[i, 0] - barcode1_gt[i, 1]
        if gt_bar_length_1 != 0 and gt_bar_length_1 != np.inf:
            gt_sum_1 += gt_bar_length_1

    new_barcodes['dim1'] = new_barcode1 

    dim1 = {}
    dim1['target'] = gt_num_bars_1
    dim1['output'] = num_bars_1
    dim1['difference_between_output_and_target'] = num_bars_1 - gt_num_bars_1
    entry_num_bars['dim1'] = dim1

    dim1_l = {}
    dim1_l['target'] = 1
    dim1_l['output'] = sum_1 / num_bars_1
    dim1_l['difference_between_output_and_target'] = sum_1 / num_bars_1 - 1
    entry_avg_length['dim1'] = dim1_l

    result_num_bars[key] = entry_num_bars
    result_avg_length[key] = entry_avg_length
    new_barcodes_all[key] = new_barcodes

result = {}
result['number_of_bars'] = result_num_bars
result['avg_bar_length'] = result_avg_length
save_path = 'barcodes/compare_epoch_' + str(epoch) + '.json'
save_json(result, save_path, create_file=False)
save_path = 'barcodes/num_bars.json'
save_json(result_num_bars, save_path)
save_path = 'barcodes/avg_bar_length.json'
save_json(result_avg_length, save_path)
save_path = 'barcodes/epoch_' + str(epoch) + '_cleaned.json'
save_json(new_barcodes_all, save_path, create_file=False)
