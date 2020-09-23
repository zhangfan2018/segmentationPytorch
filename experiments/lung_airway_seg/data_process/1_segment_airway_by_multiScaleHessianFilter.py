"""Segmentation of airway by multi-scale hessian Filter.
"""

import os
import itk
import numpy as np
import SimpleITK as sitk


def segmentAirwayByMultiScaleHessainFilter(image_path, save_dir):
    ori_img = itk.imread(image_path)
    ShortImageType = itk.Image[itk.SS, 3]
    ImageType = itk.Image[itk.F, 3]
    CastImageFilterType = itk.CastImageFilter[ShortImageType, ImageType].New()
    CastImageFilterType.SetInput(ori_img)
    CastImageFilterType.Update()
    out_image = CastImageFilterType.GetOutput()
    print("Convert image type complete!")

    # smoothFilter = itk.SmoothingRecursiveGaussianImageFilter[ImageType, ImageType].New()
    # smoothFilter.SetInput(out_image)
    # smoothFilter.SetSigma(0.5)
    # smoothFilter.Update()
    # smooth_image = smoothFilter.GetOutput()

    # iterNumber = 10
    # timeStep = 0.01
    # conductPara = 3.0
    # smoothing = itk.CurvatureAnisotropicDiffusionImageFilter[ImageType, ImageType].New()
    # smoothing.SetTimeStep(timeStep)
    # smoothing.SetNumberOfIterations(iterNumber)
    # smoothing.SetConductanceParameter(conductPara)
    # smoothing.SetInput(out_image)
    # smoothing.Update()
    # smooth_image = smoothing.GetOutput()
    print("Gaussian smooth complete!")

    img_arr = itk.GetArrayFromImage(out_image)
    output = itk.GetImageFromArray(img_arr)
    output.SetOrigin(ori_img.GetOrigin())
    output.SetSpacing(ori_img.GetSpacing())
    output.SetDirection(ori_img.GetDirection())
    save_path = os.path.join(save_dir, "smooth_image.nii.gz")
    itk.imwrite(output, save_path)

    # get reversed airway img
    airway_img_win = [-1024, -800]
    airway_img_arr = img_arr.copy()
    airway_img_arr[airway_img_arr > airway_img_win[1]] = airway_img_win[1]
    reverse_airway_img_arr = airway_img_win[1] - airway_img_arr
    reverse_airway_img = itk.GetImageFromArray(reverse_airway_img_arr)
    print("Normalization image complete!")

    output = itk.GetImageFromArray(reverse_airway_img_arr)
    output.SetOrigin(ori_img.GetOrigin())
    output.SetSpacing(ori_img.GetSpacing())
    output.SetDirection(ori_img.GetDirection())
    save_path = os.path.join(save_dir, "reverse_image.nii.gz")
    itk.imwrite(output, save_path)

    # multi scale hessian filter
    print("Multi scale hessian filter start...")
    sigma_min = 0.2
    sigma_max = 10
    num_sigma_step = 5
    input_img = reverse_airway_img
    input_img_type = type(reverse_airway_img)
    hessian_pixel_type = itk.SymmetricSecondRankTensor[itk.D, 3]
    hessian_image_type = itk.Image[hessian_pixel_type, 3]
    object_filter = itk.HessianToObjectnessMeasureImageFilter[hessian_image_type, input_img_type].New()
    object_filter.SetAlpha(0.5)
    object_filter.SetBeta(0.5)
    object_filter.SetGamma(5.0)
    hessian_filter = itk.MultiScaleHessianBasedMeasureImageFilter[input_img_type, hessian_image_type, input_img_type].New()
    hessian_filter.SetInput(input_img)
    hessian_filter.SetHessianToMeasureFilter(object_filter)
    hessian_filter.SetSigmaStepMethodToLogarithmic()
    hessian_filter.SetSigmaMinimum(sigma_min)
    hessian_filter.SetSigmaMaximum(sigma_max)
    hessian_filter.SetNumberOfSigmaSteps(num_sigma_step)
    hessian_filter.Update()
    airwayness = hessian_filter.GetOutput()
    airwayness_arr = itk.GetArrayFromImage(airwayness)
    print("Multi scale hessian filter end!")

    output = itk.GetImageFromArray(airwayness_arr)
    output.SetOrigin(ori_img.GetOrigin())
    output.SetSpacing(ori_img.GetSpacing())
    output.SetDirection(ori_img.GetDirection())
    save_path = os.path.join(save_dir, "enhanced_image.nii.gz")
    itk.imwrite(output, save_path)

    # threshold airwayness to form airway mask
    airway_mask_win = [8, 100]
    rough_airway_mask_arr = airwayness_arr.copy()
    rough_airway_mask_arr[rough_airway_mask_arr < airway_mask_win[0]] = 0
    rough_airway_mask_arr[rough_airway_mask_arr != 0] = 1

    # extract 10 largest connected components as airway mask candidates
    n_max_cc = 10
    rough_mask_arr = rough_airway_mask_arr
    rough_mask_sitk = sitk.GetImageFromArray(rough_mask_arr)
    rough_mask_sitk = sitk.Cast(rough_mask_sitk, 2)
    cc = sitk.ConnectedComponent(rough_mask_sitk)
    stats = sitk.LabelIntensityStatisticsImageFilter()
    stats.Execute(cc, rough_mask_sitk)

    labels = list()
    num_pixels = list()
    for i in stats.GetLabels():
        label = i
        num_pixel = stats.GetNumberOfPixels(i)
        labels.append(label)
        num_pixels.append(num_pixel)

    num_pixels_arr = np.array(num_pixels)
    sorted_num_pixels_arr = num_pixels_arr.copy()
    sorted_num_pixels_arr.sort()
    max_10ccs = np.flip(sorted_num_pixels_arr[-n_max_cc:])

    max_10cc_labels = list()
    for index, max_10cc in enumerate(max_10ccs):
        for label, num_pixel in zip(labels, num_pixels):
            if num_pixel == max_10cc:
                max_10cc_labels.append(label)

    # save top k connected components from which to choose airway mask
    top_k_max_cc = 6
    cc_arr = sitk.GetArrayFromImage(cc)
    cc_mask_arr = np.zeros(rough_mask_arr.shape, dtype=np.float32)
    for i in range(top_k_max_cc):
        sub_cc_mask_arr = np.zeros(rough_mask_arr.shape, dtype=np.float32)
        num_pixel = max_10ccs[i]
        label = max_10cc_labels[i]
        print('index: {}, num_pixel: {}, label: {}'.format(i, num_pixel, label))
        sub_cc_mask_arr[cc_arr == label] = i + 1
        cc_mask_arr += sub_cc_mask_arr
    output = itk.GetImageFromArray(cc_mask_arr)
    output.SetOrigin(ori_img.GetOrigin())
    output.SetSpacing(ori_img.GetSpacing())
    output.SetDirection(ori_img.GetDirection())
    save_path = os.path.join(save_dir, "mask_6.nii.gz")
    itk.imwrite(output, save_path)

    # form airway mask from target k connected component
    target_k_max_cc = 0
    cc_arr = sitk.GetArrayFromImage(cc)
    mask_arr = np.zeros(rough_mask_arr.shape, dtype=np.float32)

    num_pixel = max_10ccs[target_k_max_cc]
    label = max_10cc_labels[target_k_max_cc]
    mask_arr[cc_arr == label] = 1

    output = itk.GetImageFromArray(mask_arr)
    output.SetOrigin(ori_img.GetOrigin())
    output.SetSpacing(ori_img.GetSpacing())
    output.SetDirection(ori_img.GetDirection())
    save_path = os.path.join(save_dir, "mask_0.nii.gz")
    itk.imwrite(output, save_path)


image_path = "/fileser/zhangfan/DataSet/airway_segment_data/image/NII/1.2.840.113704.1.111.1316.1389761145.9.nii.gz"
dest_path = "/fileser/zhangfan/DataSet/airway_segment_data/mask_refine/1.2.840.113704.1.111.1316.1389761145.9/"
if not os.path.exists(dest_path):
    os.mkdir(dest_path)
segmentAirwayByMultiScaleHessainFilter(image_path, dest_path)

