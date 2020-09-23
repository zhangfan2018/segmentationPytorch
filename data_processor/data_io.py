""" Data IO, read and write medical data in dicom, eg. nii.gz, nrrd and mha format.
method:
load_nii_image
load_dicom_series
save_medical_info_and_data
save_medical_image
mask_array_to_sitk
save_sitk_image
"""

import SimpleITK as sitk


class DataIO(object):
    def __init__(self):
        super(DataIO, self).__init__()

    def load_nii_image(self, file_path):
        """load medical image, eg. nii, mha"""
        """
        Args:
            file_path(str): input file path.
        Return:
            res(dict): image information and data.
        """

        res = {}
        sitk_image = sitk.ReadImage(file_path)
        res["sitk_image"] = sitk_image
        res["image"] = sitk.GetArrayFromImage(sitk_image)
        res["origin"] = sitk_image.GetOrigin()
        res["spacing"] = sitk_image.GetSpacing()
        res["direction"] = sitk_image.GetDirection()

        return res

    def load_dicom_series(self, file_path):
        """load dicom series."""
        """
        Args:
            file_path(str): input file path.
        Return:
            res(dict): image information and data.
        """
        res = {}
        reader = sitk.ImageSeriesReader()
        series_Ids = reader.GetGDCMSeriesIDs(file_path)
        dcm_series = reader.GetGDCMSeriesFileNames(file_path, series_Ids[0])
        reader.SetFileNames(dcm_series)
        sitk_image = reader.Execute()
        res["sitk_image"] = sitk_image
        res["image"] = sitk.GetArrayFromImage(sitk_image)
        res["origin"] = sitk_image.GetOrigin()
        res["spacing"] = sitk_image.GetSpacing()
        res["direction"] = sitk_image.GetDirection()

        return res

    def save_medical_info_and_data(self, image_array, origin, spacing, direction, save_path):
        """save medical information and image data."""
        """
        Args:
            image_array(numpy array): data array.
            origin:
            direction:
            spacing:
            save_path: save data dir.
        Return:
            None
        """
        savedImg = sitk.GetImageFromArray(image_array)
        savedImg.SetOrigin(origin)
        savedImg.SetSpacing(spacing)
        savedImg.SetDirection(direction)
        sitk.WriteImage(savedImg, save_path)

    def save_medical_image(self, image_array, save_path):
        """save medical image data"""
        """
        Args:
            image_array(numpy array): data array.
            save_path: save data dir.
        """
        sitk_image = sitk.GetImageFromArray(image_array)
        sitk.WriteImage(sitk_image, save_path)

    def mask_array_to_sitk(self, mask_array, data_info, label=7, sitk_type=sitk.sitkUInt8):
        """convert mask array to sitk image"""
        """
        Args:
            mask_array(numpy array): data array.
            data_info(dict): origin, spacing, direction.
            label(int): mask label.
            sitk_type: data type.
        """
        mask_array[mask_array != 0] = label
        sitk_mask = sitk.GetImageFromArray(mask_array)
        sitk_mask = sitk.Cast(sitk_mask, sitk_type)
        if "origin" in data_info: sitk_mask.SetOrigin(data_info["origin"])
        if "spacing" in data_info: sitk_mask.SetSpacing(data_info["spacing"])
        if "direction" in data_info: sitk_mask.SetDirection(data_info["direction"])

        return sitk_mask

    def save_sitk_image(self, sitk_image, file_path):
        """save sitk image."""
        """
        Args:
            sitk_image(sitk type): sitk image.
            file_path(str): file path.
        """
        itkWriter = sitk.ImageFileWriter()
        itkWriter.SetUseCompression(True)
        itkWriter.SetFileName(file_path)
        itkWriter.Execute(sitk_image)

    def save_mask_compress(self, ori_path, dest_path):
        """Save mask compress"""
        """
        Args:
        ori_path(str): path to original file.
        dest_path(str): path to destination file.
        """
        mask = sitk.ReadImage(ori_path)
        new_mask = sitk.Cast(mask, sitk.sitkUInt8)
        sitk.WriteImage(new_mask, dest_path, True)
