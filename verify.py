from src.dataloader import Dataset
from src.augmentation import get_preprocessing
from visualization.vis import vis
from src.stats_util import get_bounding_box
from skimage.feature import peak_local_max

from src.stats_util import (
    get_dice_1,
    get_fast_aji,
    get_fast_aji_plus,
    get_pq,
    remap_label,
)
import config
from src.model import SC_Net

import torch
import numpy as np
from scipy import ndimage
from skimage.segmentation import watershed
from tqdm import tqdm


def get_stats(dataid, file_name,inst_map, pred_inst_map,metrics ,print_img_stats=True):
    true = inst_map.astype("int32")

    pred = pred_inst_map.astype("int32")

    # to ensure that the instance numbering is contiguous
    pred = remap_label(pred, by_size=False)
    true = remap_label(true, by_size=False)

    try:
        pq_info = get_pq(true, pred, match_iou=0.5)[0]
        dice_score = get_dice_1(true, pred)
        aji = get_fast_aji(true, pred)
        aji_plus = get_fast_aji_plus(true, pred)

        metrics[0].append(dice_score)
        metrics[1].append(aji)
        metrics[2].append(pq_info[0])  # dq
        metrics[3].append(pq_info[1])  # sq
        metrics[4].append(pq_info[2])  # pq
        metrics[5].append(aji_plus)

        #####write to file####
        file.write('%04d-%16s: Dice=%f | aji=%f | dq=%f | sq=%f | pq=%f\n' % (dataid, file_name,dice_score, aji, pq_info[0],pq_info[1],pq_info[2]))
    except:
        pass

    if print_img_stats:
        for scores in metrics:
            print("%f " % scores[-1], end="  ")
        print()

def get_min_values(feature_map, coordinates): #获取连线上最小值
    # Initialize an array to store minimum values for each line segment
    min_values = []

    # Iterate through each pair of coordinates
    for i in range(len(coordinates)):
        for j in range(i + 1, len(coordinates)):
            x1, y1 = coordinates[i]
            x2, y2 = coordinates[j]

            num_points = max(abs(x2 - x1), abs(y2 - y1)) + 1

            # Generate all points between (x1, y1) and (x2, y2)
            line_points = list(zip(np.linspace(x1, x2, num_points, endpoint=True).round().astype(int),
                                   np.linspace(y1, y2, num_points, endpoint=True).round().astype(int)))

            # Find minimum value along the line in the feature map
            line_min_value = np.inf
            for px, py in line_points:
                if 0 <= px < feature_map.shape[0] and 0 <= py < feature_map.shape[1]:
                    if feature_map[px, py] < line_min_value:
                        line_min_value = feature_map[px, py]

            # Add the minimum value for this line to the result
            min_values.append(line_min_value)

    min_value = min(min_values)
    return min_values,min_value


def find_max_coordinates(array):
    flat_index = np.argmax(array)
    max_coordinates = np.unravel_index(flat_index, array.shape)
    return max_coordinates
def apply_watershed(distance_transform=None,
                    prob_map=None,
                    foreground_mask=None):
    marker = ndimage.label(prob_map)[0]
    pred_inst_map = watershed(distance_transform,
                              markers=marker,
                              mask=foreground_mask,
                              compactness=0.01)
    return pred_inst_map

def assign_value_where_true(boolean_array, value):
    #Create an array of zeros or with other initialization methods that has the same shape as boolean_array
    result_array = np.zeros(boolean_array.shape, dtype=np.int32)
    # Assign a specified value to the positions where the value is True
    result_array[boolean_array] = value
    return result_array

def apply_mask(feature_map, mask):
    # Ensure the mask is a boolean array
    mask = mask.astype(bool)

    # Create an output array initialized to zero
    output = np.zeros_like(feature_map)

    # Copy values from feature_map to output where mask is True
    output[mask] = feature_map[mask]

    return output

# Remove small connected components
def remove_small_objects(image, min_size):
    # Label connected components.
    labeled_array, num_features = ndimage.label(image)
    for i in range(1, num_features + 1):
        # dentify the pixels for each connected component
        component = (labeled_array == i)
        # Remove connected components that are smaller than min_size
        if np.sum(component) < min_size:
            image[component] = 0
    return image


def do_seg(_train_image, mask ,file_name,model, is_save_experiment = True,docut = False,crop_params=''):

    train_image = torch.from_numpy(_train_image).to(config.device).unsqueeze(0)
    # predict mask
    total_mask = model(train_image).squeeze().cpu().detach().numpy()


    pred_mask = total_mask[:3, :, :]
    # predict scale
    pred_scale = total_mask[3, :, :]      # 512*512
    pred_temp_scale_map = pred_scale + 0  # Scale-aware feature map

    nuclei_map = pred_mask[:2, :, :].argmax(axis=0)  # Binarization
    prob_map = pred_mask[2, :, :]  # Centroid probability map
    temp_prob_map = prob_map + 0   # Centroid probability map
    temp_nuclei_map = nuclei_map + 0  # Semantic segmentation connectivity map


    temp_prob_marker = np.where(temp_prob_map > config.watershed_threshold, 1, 0)
    gt_inst_map = mask[:, :, 2]  # GT
    pred_inst_map = apply_watershed(
        distance_transform=temp_prob_map,  #
        prob_map=temp_prob_marker,  # Key regions of the centroid probability map"
        foreground_mask=temp_nuclei_map,  # Semantic segmentation connectivity map
    )  # Instance segmentation map
    # Instance index to be added
    pre_inst_ids = np.unique(pred_inst_map)  # Indices are not necessarily consecutive
    current_inst_id = np.max(pre_inst_ids) + 1

    # GT scale_map
    gt_scale_map = mask[:, :, 3]
    tmp_gt_scale_map = gt_scale_map + 0

    temp_pred_inst_map = pred_inst_map + 0
    # 2.Traverse each connected component of the instance segmentation and refine the nuclei segmentation based on the scale-aware feature map.
    pre_inst_ids = np.unique(temp_pred_inst_map)  # Indices are not necessarily consecutive
    for id in pre_inst_ids:
        if id == 0:
            continue
        pred_mask_inst_map = temp_pred_inst_map == id  # Connected component mask
        rmin1, rmax1, cmin1, cmax1 = get_bounding_box(pred_mask_inst_map)

        # BEGIN
        pred_inst_mask_crop = pred_mask_inst_map[rmin1:rmax1, cmin1:cmax1]  # Prediction 0: Scale prediction map for the current ID

        # pred_inst_crop = temp_pred_inst_map[rmin1:rmax1, cmin1:cmax1]  # Prediction 1: Instance segmentation map after applying the watershed algorithm
        # pred_inst_crop = apply_mask(pred_inst_crop, pred_inst_mask_crop)

        pred_prob_map_crop = temp_prob_map[rmin1:rmax1, cmin1:cmax1]
        pred_prob_map_crop = apply_mask(pred_prob_map_crop, pred_inst_mask_crop)  # Prediction 2: Predicted centroid probability map

        pred_scale_map_crop = pred_temp_scale_map[rmin1:rmax1, cmin1:cmax1]  # Prediction 2:scale feature map
        pred_scale_map_crop = apply_mask(pred_scale_map_crop, pred_inst_mask_crop)

        pred_nuclei_map_crop = temp_nuclei_map[rmin1:rmax1, cmin1:cmax1]  # Prediction 4: Semantic segmentation connectivity graph
        pred_nuclei_map_crop = apply_mask(pred_nuclei_map_crop, pred_inst_mask_crop)  # Semantic segmentation connectivity graph focusing only on the scale mask

        gt_scale_map_crop = tmp_gt_scale_map[rmin1:rmax1, cmin1:cmax1]  # Ground truth scale prediction map
        gt_scale_map_crop = apply_mask(gt_scale_map_crop, pred_inst_mask_crop)
        gt_inst_map_crop = gt_inst_map[rmin1:rmax1, cmin1:cmax1]  # Ground truth instance segmentation map
        gt_inst_map_crop = apply_mask(gt_inst_map_crop, pred_inst_mask_crop)

        # Obtain the maximum value from pred_scale_map_crop scale prediction and compute the scale using the inverse logarithm.

        pre_scale = np.max(pred_scale_map_crop)

        # 使用 peak_local_max 找到局部最大值
        coordinates_max_scale = peak_local_max(pred_scale_map_crop, min_distance=1, threshold_abs=0.1)

        # 从原始数据中提取局部最大值
        peak_values = pred_scale_map_crop[coordinates_max_scale[:, 0], coordinates_max_scale[:, 1]]

        if len(peak_values) == 0:
            continue


        mean = np.mean(peak_values)
        std = np.std(peak_values)

        threshold = 1.2 * std

        filtered_data = [x for x in peak_values if abs(x - mean) <= threshold]

        mean = np.mean(filtered_data)
        pre_scale = mean

        scale_of_inst = np.sum(pred_inst_mask_crop)
        # pre_scale = np.max(filtered_data)
        pre_scale = np.exp(pre_scale)

        if pre_scale < scale_of_inst * 0.90:  # 容忍度0.9
            # "If the size of the coarse segmentation is larger than the predicted scale, it indicates insufficient segmentation, and finer segmentation is required.

            coordinates = peak_local_max(pred_prob_map_crop, min_distance=7, threshold_abs=0.1)


            if len(coordinates) > 1:
                min_thredhold = 0.30
                temp_prob_marker = np.zeros_like(pred_prob_map_crop, dtype=np.int32)
                for i, coord in enumerate(coordinates):
                    temp_threshold = pred_prob_map_crop[coord[0], coord[1]]
                    if min_thredhold > temp_threshold:
                        min_thredhold = pred_prob_map_crop[coord[0], coord[1]]


                _, min_value = get_min_values(pred_prob_map_crop, coordinates)

                temp_threshold = min_thredhold / 1.2
                if temp_threshold < min_value and min_thredhold > min_value:
                    temp_threshold = (min_thredhold - min_value) * 0.2 + min_value  # pq=0.652494
                temp_prob_marker = np.where(pred_prob_map_crop > temp_threshold, 1, 0)

                # np.save('test_api/array.npy', pred_prob_map_crop)

                pred_inst_map_crop = apply_watershed(
                    distance_transform=pred_prob_map_crop,
                    prob_map=temp_prob_marker,
                    foreground_mask=pred_nuclei_map_crop,
                )

                ids, label_counts = np.unique(pred_inst_map_crop, return_counts=True)

                fix_id = 0
                label_counts_max = 0

                if len(ids) > 2:
                    for label, count in zip(ids, label_counts):
                        if label == 0:
                            continue
                        if count > label_counts_max:
                            fix_id = label
                            label_counts_max = count
                            instance_scale = count

                if fix_id != 0:
                    mask = pred_inst_map_crop == fix_id
                    pred_scale_map_crop_scale = apply_mask(pred_scale_map_crop, mask)
                    pre_scale = np.max(pred_scale_map_crop_scale)
                    pre_scale = np.exp(pre_scale)


                    if instance_scale > pre_scale:
                        pred_prob_map_crop_mask = apply_mask(pred_prob_map_crop, mask)
                        coord_centor = find_max_coordinates(pred_prob_map_crop_mask)



                        height, width = pred_prob_map_crop.shape


                        grid_y, grid_x = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')


                        distances = np.sqrt((grid_y - coord_centor[0]) ** 2 + (grid_x - coord_centor[1]) ** 2)
                        distances = apply_mask(distances, mask)  # mask_distance


                        num_largest = int(instance_scale - pre_scale)  #
                        flat_indices = np.argsort(distances, axis=None)[-num_largest:]
                        indices = np.unravel_index(flat_indices, distances.shape)


                        remove_mask = np.zeros_like(distances, dtype=bool)
                        remove_mask[indices] = True
                        pred_inst_map_crop[remove_mask] = -1



                min_size = 20
                num_labels = pred_inst_map_crop.max()

                for label_val in range(1, num_labels + 1):
                    component_mask = (pred_inst_map_crop == label_val).astype(np.uint8)
                    cleaned_component = remove_small_objects(component_mask, min_size)
                    pred_inst_map_crop[pred_inst_map_crop == label_val] = 0
                    pred_inst_map_crop[cleaned_component == 1] = label_val

                ids = np.unique(pred_inst_map_crop)

                for id in ids:
                    if id == 0:
                        continue
                    mask = (pred_inst_map_crop == id)
                    if id == -1:
                        pred_inst_map_crop[mask] = id
                        continue
                    pred_inst_map_crop[mask] = current_inst_id
                    current_inst_id += 1


                mask = pred_inst_map_crop != 0
                pred_inst_map[rmin1:rmax1, cmin1:cmax1][mask] = pred_inst_map_crop[mask]
                pred_inst_map[rmin1:rmax1, cmin1:cmax1][pred_inst_map[rmin1:rmax1, cmin1:cmax1] == -1] = 0

            elif pre_scale < scale_of_inst * 0.5:
                pass
    # visualization
    if is_save_experiment:
        save_path = config.scgccm_experiment
    else:
        save_path = ''
    if config.vis:
        vis(np.transpose((_train_image * 256).squeeze(), (1, 2, 0)), gt_inst_map, pred_inst_map, file_name, save_path,docut,crop_params)
    try:
        get_stats(idx, file_name, gt_inst_map, pred_inst_map, metrics, print_img_stats=False)
    except:
        pass

if __name__ == '__main__':

    file = open(config.scgccm_experiment + 'test_score.txt', 'w')
    dataset = Dataset(config.test_dir, preprocessing=get_preprocessing(None), mode="test")

    model = SC_Net(device=config.device)
    weight_dir = "./{}/{}".format(config.checkpoints_dir,
                                        config.inference_para_weights)
    print("test_dir:",config.test_dir)
    print("weight_dir:",weight_dir)
    model.load_state_dict(torch.load(weight_dir))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    metrics = [[], [], [], [], [], []]
    for idx in tqdm(range(len(dataset)), total=len(dataset)):
        _train_image, mask, file_name = dataset[idx]

        if int(np.unique(mask[:, :, 2]).shape[0]) == 1:
            continue
        do_seg(_train_image, mask, file_name,model)

    metrics = np.array(metrics)
    metrics_avg = np.mean(metrics, axis=-1)

    np.set_printoptions(formatter={"float": "{: 0.5f}".format})
    print('Total: Dice=%f | aji=%f | dq=%f | sq=%f  | pq=%f \n' % (metrics_avg[0], metrics_avg[1],metrics_avg[2],metrics_avg[3],metrics_avg[4]))
    file.write('Total: Dice=%f | aji=%f | dq=%f | sq=%f  | pq=%f \n' % (metrics_avg[0], metrics_avg[1],metrics_avg[2],metrics_avg[3],metrics_avg[4]))
    file.close()