import os
import cv2 as cv
import numpy as np
import torch
import pydicom
from tqdm.auto import tqdm
from skimage import exposure
from dicompylercore import dicomparser
import rt_utils
import datetime
import torchvision.transforms as T
import random
from rt_utils import RTStructBuilder
from PIL import Image


def parse_folder(folder_path):
    # Load DICOM files
    image_files = {}
    rtstruct_files = {}
    rtdose_files = {}
    rtplan_files = {}
    
    dates = {}

    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        if file_name.lower().endswith('.dcm'):
            ds = dicomparser.DicomParser(file_path)
            try:
                if ds.GetSOPClassUID() == 'rtss' and 'predicted' not in file_name:
                    rtstruct_files[ds.GetFrameOfReferenceUID()] = ds
                elif ds.GetSOPClassUID() == 'rtdose' and 'predicted' not in file_name:
                    rtdose_files[ds.GetFrameOfReferenceUID()] = ds
                elif ds.GetSOPClassUID() == 'rtplan' and 'predicted' not in file_name:
                    rtplan_files[ds.GetFrameOfReferenceUID()] = ds
                elif 'predicted' not in file_name:
                    ds = pydicom.dcmread(file_path)
                    if ds.FrameOfReferenceUID in image_files:
                        image_files[ds.FrameOfReferenceUID].append(ds)
                    else:
                        image_files[ds.FrameOfReferenceUID] = [ds]
                        dates[ds.InstanceCreationDate] = ds.FrameOfReferenceUID
            except:
                print(file_name)        

    return image_files, dates, rtstruct_files, rtdose_files, rtplan_files


def sort_referenceframes(dates):
    sorted_referenceframes = list(dict(sorted(dates.items())).values())
    
    return sorted_referenceframes

def read_images(image_files, sorted_referenceframes):
    dicom_files = image_files.copy()
    
    for referenceframe in sorted_referenceframes:
        image_files[referenceframe] = sorted(image_files[referenceframe], key=lambda x: float(x.SliceLocation))[::-1]
       
        image_files[referenceframe] = [dicomparser.DicomParser(x) for x in image_files[referenceframe]]
    
    
    for referenceframe in sorted_referenceframes:
        dicom_files[referenceframe] = sorted(dicom_files[referenceframe], key=lambda x: float(x.SliceLocation))[::-1]
    
    # Access information from image files
    images = []
    for referenceframe in sorted_referenceframes:
        images_referenceframe = []
        for image_file in image_files[referenceframe]:
            image_array = image_file.GetPixelArray()
            image_array = torch.from_numpy(image_array.astype(np.int64))     
            image_array = torch.where(torch.isnan(image_array), torch.zeros_like(image_array), image_array)
            #max_grey_value = torch.max(image_array)
            #min_grey_value = torch.min(image_array)
            #image_array = (image_array - min_grey_value) / (max_grey_value - min_grey_value) 
            images_referenceframe.append(image_array)
        images.append(np.stack(images_referenceframe, 0))
    
    images = torch.from_numpy(np.stack(images, 0))
    return image_files, dicom_files, images

def get_slice_position(series_slice):
    _, _, slice_direction = get_slice_directions(series_slice)
    return np.dot(slice_direction, series_slice.ds.ImagePositionPatient)

def get_spacing_between_slices(series_data):
    if len(series_data) > 1:
        first = get_slice_position(series_data[0])
        last = get_slice_position(series_data[-1])
        return (last - first) / (len(series_data) - 1)

    # Return nonzero value for one slice just to make the transformation matrix invertible
    return 1.0

def get_slice_directions(series_slice):
    orientation = series_slice.ds.ImageOrientationPatient
    row_direction = np.array(orientation[:3])
    column_direction = np.array(orientation[3:])
    slice_direction = np.cross(row_direction, column_direction)

    if not np.allclose(
        np.dot(row_direction, column_direction), 0.0, atol=1e-3
    ) or not np.allclose(np.linalg.norm(slice_direction), 1.0, atol=1e-3):
        raise Exception("Invalid Image Orientation (Patient) attribute")

    return row_direction, column_direction, slice_direction

def get_patient_to_pixel_transformation_matrix(series_data):
    first_slice = series_data[0]

    offset = np.array(first_slice.ds.ImagePositionPatient)
    row_spacing, column_spacing = first_slice.ds.PixelSpacing
    slice_spacing = get_spacing_between_slices(series_data)
    row_direction, column_direction, slice_direction = get_slice_directions(first_slice)

    linear = np.identity(3, dtype=np.float32)
    linear[0, :3] = row_direction / row_spacing
    linear[1, :3] = column_direction / column_spacing
    linear[2, :3] = slice_direction / slice_spacing

    mat = np.identity(4, dtype=np.float32)
    mat[:3, :3] = linear
    mat[:3, 3] = offset.dot(-linear.T)

    return mat

def apply_transformation_to_3d_points(points: np.ndarray, transformation_matrix: np.ndarray):
    """
    * Augment each point with a '1' as the fourth coordinate to allow translation
    * Multiply by a 4x4 transformation matrix
    * Throw away added '1's
    """
    vec = np.concatenate((points, np.ones((points.shape[0], 1))), axis=1)
    return vec.dot(transformation_matrix.T)[:, :3]

def get_contour_data(rtstruct_files, sorted_referenceframes, masks):
    # Access information from RTSTRUCT file
    structure_data = {}
    if bool(rtstruct_files):
        for referenceframe in sorted_referenceframes:
            if referenceframe in rtstruct_files:
                structures = rtstruct_files[referenceframe].GetStructures()
                structure_data_referenceframe = {}
                for structure_id in structures:
                    if structures[structure_id]['name'] in masks:
                        structure_data_referenceframe[structures[structure_id]['name']] = rtstruct_files[referenceframe].GetStructureCoordinates(structure_id)                    
                    elif structures[structure_id]['name'] == "RING":
                        structure_data_referenceframe[structures[structure_id]['name']] = rtstruct_files[referenceframe].GetStructureCoordinates(structure_id)            

                structure_data[referenceframe] = structure_data_referenceframe
            else:
                structure_data[referenceframe] = None  
    return structure_data

def get_filled_contours(structure_data, image_files, images, sorted_referenceframes, masks, folder):
    labels = []
    rings = []
    for order, referenceframe in enumerate(sorted_referenceframes):
        ScalingFactor = image_files[referenceframe][0].ds.PixelSpacing[0]
        CenterZ = len(image_files[referenceframe])//2
        labels_referenceframe = torch.zeros((len(masks)+1, images[order].shape[0], images[order].shape[1], images[order].shape[2]))
        rings_referenceframe = torch.zeros((images[order].shape[0], images[order].shape[1], images[order].shape[2]))
        transformation_matrix = get_patient_to_pixel_transformation_matrix(image_files[referenceframe])
        
        if structure_data[referenceframe] != None:
            for mask in structure_data[referenceframe]:
                for slice_depth in structure_data[referenceframe][mask]:
                    slice_mask = np.zeros((300,334))#labels_referenceframe.shape[2], labels_referenceframe.shape[3]))
                    z = int(CenterZ + float(slice_depth)/ScalingFactor)
                
                    for structure in range(len(structure_data[referenceframe][mask][slice_depth])):
                        polygons=[]
                        for contour_coords in structure_data[referenceframe][mask][slice_depth][structure]['data']:
                            contour_coords = np.reshape(contour_coords, [len(contour_coords) // 3, 3])
                            contour_coords = apply_transformation_to_3d_points(contour_coords, transformation_matrix)
                            polygon = [np.around(contour_coords[:, :2]).astype(np.int32)]
                            polygon = np.array(polygon).squeeze()
                            polygons.append(polygon)
                        polygons = np.array(polygons, dtype=np.int32)
                        cv.fillPoly(img=slice_mask, pts=[polygons], color=1)
                    if folder != 'ProSeg':
                        slice_mask = slice_mask[62:-62,31:-31]


                    if mask == "RING":
                        rings_referenceframe[z,:,:] = torch.from_numpy(slice_mask)
                    else:
                        labels_referenceframe[masks.index(mask)+1, z, :, :] = torch.from_numpy(slice_mask)

            labels_referenceframe = torch.where(torch.isnan(labels_referenceframe), torch.zeros_like(labels_referenceframe), labels_referenceframe)
            rings_referenceframe = torch.where(torch.isnan(rings_referenceframe), torch.zeros_like(rings_referenceframe), rings_referenceframe)

            #Exclude CTV from the other labels
            if "CTV" in masks:
                for mask in range(len(masks)):
                    for slice in range(labels_referenceframe.shape[1]):
                        if not masks[mask]=="CTV":
                            labels_referenceframe[mask+1, slice, :, :] = labels_referenceframe[mask+1, slice, :, :] - labels_referenceframe[mask+1, slice, :, :] * labels_referenceframe[masks.index("CTV")+1, slice, :, :]
                        else:
                            labels_referenceframe[mask+1, slice, :, :] = labels_referenceframe[mask+1, slice, :, :]

            if "Skin" in masks: 
                for label in range(len(masks)):                        
                    for slice in range(labels_referenceframe.shape[1]):
                        if masks[mask] != "Skin":
                            labels_referenceframe[masks.index("Skin")+1, slice, :, :] = labels_referenceframe[masks.index("Skin")+1, slice, :, :] - labels_referenceframe[mask+1, slice, :, :]*labels_referenceframe[masks.index("Skin")+1, slice, :, :]
                        else:
                            labels_referenceframe[masks.index("Skin")+1, slice, :, :] = labels_referenceframe[masks.index("Skin")+1, slice, :, :]

            if order>0:
                for mask in range(len(masks)):
                    if masks[mask] != "Skin":
                        labels_referenceframe[mask+1, :, :, :] = labels_referenceframe[mask+1, :, :, :] * rings_referenceframe

        labels_referenceframe = torch.argmax(labels_referenceframe,dim=0)
        labels_referenceframe = labels_referenceframe.long()
        rings_referenceframe = rings_referenceframe.long()
        labels_referenceframe = torch.where(labels_referenceframe>len(masks), torch.tensor(0), labels_referenceframe)
        rings_referenceframe = torch.where(rings_referenceframe>1, torch.tensor(0), rings_referenceframe)
        labels.append(labels_referenceframe)
        rings.append(rings_referenceframe)

    labels = torch.stack(labels, dim=0).flip(1)
    rings = torch.stack(rings, dim=0).flip(1)        
    #labels = torch.stack(labels, dim=0).flip(2)
    #rings = torch.stack(rings, dim=0).flip(2)
    return labels, rings

def get_label_info(info, RT, masks):
    info['label_info'] = {}
    for structure_roi in RT.ds.StructureSetROISequence:
        if structure_roi.ROIName in masks:
            for roi_info in RT.ds.ROIContourSequence:
                if roi_info.ReferencedROINumber == structure_roi.ROINumber:
                    info['label_info'][(masks.index(structure_roi.ROIName)+1)] = [structure_roi.ROIName, roi_info.ROIDisplayColor]
                    break
        if structure_roi.ROIName == "RING":
            for roi_info in RT.ds.ROIContourSequence:
                if roi_info.ReferencedROINumber == structure_roi.ROINumber:
                    info['ring_info'] = [structure_roi.ROIName, roi_info.ROIDisplayColor]
                    break
    return info

def read_labels(rtstruct_files, image_files, images, sorted_referenceframes, masks, folder):
    structure_data = get_contour_data(rtstruct_files, sorted_referenceframes, masks)
    
    labels = get_filled_contours(structure_data, image_files, images, sorted_referenceframes, masks, folder)
           
    return labels

def read_doses(rtdose_files, images, sorted_referenceframes):
    # Access information from RTDOSE file
    doses = []
    for referenceframe in sorted_referenceframes:
        if referenceframe in rtdose_files:
            dose_array = rtdose_files[referenceframe].GetPixelArray()
            dose_array = dose_array*rtdose_files[referenceframe].ds.DoseGridScaling*100
            doses.append(dose_array)
        else:
            doses.append(torch.zeros_like(images[0]))    
            
    doses = torch.from_numpy(np.stack(doses, 0)).flip(1)
    
    return doses

def crop(images, labels, rings, doses, info, size, folder):
    depth = images.shape[1]
    start_z = depth//2 - size[0]//2
    end_z = depth//2 + size[0]//2
    
    height = images.shape[2]
    start_y = height//2 - size[1]//2
    end_y = height//2 + size[1]//2
    
    width = images.shape[3]
    start_x = width//2 - size[2]//2
    end_x = width//2 + size[2]//2
    
    info['size_info'] = {}
    info['size_info']['original_size'] = images.shape[1:]
    info['size_info']['crop_center'] = (depth//2,height//2,width//2)  #140
    info['size_info']['new_size'] = size
    
    if folder!= 'ProSeg':
        images = images[:, start_z:end_z, :, :]
        labels = labels[:, start_z:end_z, :, :]
        doses = doses[:, start_z:end_z, :,:]
        rings = rings[:, start_z:end_z, :, :]
    else:
        images = images[:, start_z:end_z, start_y:end_y, start_x:end_x]
        labels = labels[:, start_z:end_z, start_y:end_y, start_x:end_x]
        doses = doses[:, start_z:end_z, start_y:end_y, start_x:end_x]
        rings = rings[:, start_z:end_z, start_y:end_y, start_x:end_x]

    return images, labels, rings, doses, info

def equalize(images, equalization_mode):
    if equalization_mode == "histogram":
        for fraction in range(images.shape[0]):
            for z in range(images.shape[1]):
                images[fraction, z, :, :] = torch.from_numpy(np.reshape(exposure.equalize_hist(images[fraction, z, :, :].numpy().flatten()), images[fraction, z, :, :].shape))     
    return images

def sort_dict(dictionary, sorted_referenceframes):
    sorted_values = []  # List to store sorted values

    for frame in sorted_referenceframes:
        if frame in dictionary:
            sorted_values.append(dictionary[frame])
        else:
            sorted_values.append(None)
    return sorted_values

def extract_basisplan_and_fractions(folder_path, masks, size, equalization_mode, folder):
    info = {}

    image_files, dates, rtstruct_files, rtdose_files, rtplan_files = parse_folder(folder_path)
            
    sorted_referenceframes = sort_referenceframes(dates)
    
    image_files, dicom_files, images = read_images(image_files, sorted_referenceframes)

    labels, rings = read_labels(rtstruct_files, image_files, images, sorted_referenceframes, masks, folder)
    
    info = get_label_info(info, rtstruct_files[list(rtstruct_files.keys())[0]], masks)
    
    doses = read_doses(rtdose_files, images, sorted_referenceframes)
            
    images, labels, rings, doses, info = crop(images, labels, rings, doses, info, size, folder)
    
    images = equalize(images, equalization_mode)

    dicom_files = sort_dict(dicom_files, sorted_referenceframes)
    rtstruct_files = sort_dict(rtstruct_files, sorted_referenceframes)
    rtdose_files = sort_dict(rtdose_files, sorted_referenceframes)
        
    return images, dicom_files, labels, rings, rtstruct_files, doses, rtdose_files, info

class DataSet(torch.utils.data.Dataset):
    def __init__(self, patients, masks, size, folder, load_fractions=False, threed=False, load_dose=False, equalization_mode="None", augment=False):
        self.load_dose = load_dose
        self.folder = folder
        self.augment = augment
        self.images = []
        self.labels = []
        self.doses = []

        for patient in tqdm(patients, leave=False):
            basis = os.path.join(patient, 'Basisplan', self.folder)
            images, _, labels, rings, _, doses, _, self.info = extract_basisplan_and_fractions(basis, masks, size, equalization_mode, self.folder)
            if not load_fractions:
                self.images.append(images[0].unsqueeze(0))
                self.labels.append(labels[0])
                self.doses.append(doses[0])
                if self.augment:
                    images_augment, labels_augment = apply_augmentation(images.unsqueeze(0), labels, [add_shift, add_rotate] )
                    self.images.append(images_augment[0])
                    self.labels.append(labels_augment[0])
                    self.doses.append(doses[0])
            else:
                if labels.shape[0] > 1:
                    images = images[1:]
                    labels = labels[1:]
                    for fraction in range(images.shape[0]):
                        self.images.append(images[fraction].unsqueeze(0))
                        self.labels.append(labels[fraction])
            
        if not threed:
            if not load_dose:
                self.images = torch.cat(self.images, dim=1).permute(1,0,2,3)
                self.labels = torch.cat(self.labels, dim=0)
                self.doses = torch.cat(self.doses, dim=0)
            else:
                self.images = torch.cat(self.images, dim=1).permute(1,0,2,3)
                self.labels = torch.cat(self.labels, dim=0)
                self.labels = torch.mul(torch.nn.functional.one_hot(self.labels, len(masks)+1).permute(0,3,1,2)[:,1,:,:], 3625)
                self.doses = torch.cat(self.doses, dim=0) 
        else:
            if not load_dose:
                self.images = torch.stack(self.images)
                self.labels = torch.stack(self.labels)
                #self.doses = torch.stack(self.doses)
            else:
                self.images = torch.stack(self.images)
                self.labels = torch.stack(self.labels)
                self.labels = torch.mul(torch.nn.functional.one_hot(self.labels, len(masks)+1).permute(0,4,1,2,3)[:,1,:,:,:], 3625)
                self.doses = torch.stack(self.doses)     
                
    def __len__(self):
        return self.images.shape[0]
    
    def __getitem__(self, index):
        if not self.load_dose:
            #inputs = self.images[index]
            labels = self.labels[index]
        else:
            inputs = self.labels[index]
            labels = self.doses[index]
        return  labels

    def get_info(self):
        return self.info
    
def pad_around_original_center(image, desired_size, center):
    # Calculate the padding amounts for each dimension
        
    fill_image = torch.zeros(desired_size)
    
    start_z = center[0] - image.shape[0]//2
    end_z = center[0] + image.shape[0]//2
    
    start_y = center[1] - image.shape[1]//2
    end_y = center[1] + image.shape[1]//2
    
    start_x = center[2] - image.shape[2]//2
    end_x = center[2] + image.shape[2]//2
    
    fill_image[start_z:end_z, start_y:end_y, start_x:end_x] = image

    return fill_image

# class RTStructBuilder:
#     def create_new(dicom_series):
#         """
#         Method to generate a new rt struct from a DICOM series
#         """
#         ds = rt_utils.ds_helper.create_rtstruct_dataset(dicom_series)
#         referenceframe = dicom_series[0].FrameOfReferenceUID
#         ds.FrameOfReferenceUID = dicom_series[0].FrameOfReferenceUID
#         ds.ImagePositionPatient = dicom_series[0].ImagePositionPatient
#         ds.PixelSpacing = dicom_series[0].PixelSpacing
#         ds.SliceThickness = dicom_series[0].SliceThickness
#         return rt_utils.RTStruct(dicom_series, ds), referenceframe

def save_to_rtstruct(dicom_files, labels, info, folder, prefix):
    #rtstruct, referenceframe = RTStructBuilder.create_new(dicom_files)
    rtstruct = RTStructBuilder.create_new(dicom_files)
    labels = pad_around_original_center(labels, info['size_info']['original_size'], info['size_info']['crop_center'])
    print(labels.shape)
    #rings = pad_around_original_center(rings, info['size_info']['original_size'], info['size_info']['crop_center'])
    labels = torch.nn.functional.one_hot(labels.long(), len(info['label_info'])+1).permute(3,1,2,0)
    #rings = rings.permute(1,2,0)
    
    for label in range(labels.shape[0]):
        if label > 0:
            name = info['label_info'][label][0]
            color = list(info['label_info'][label][1])
            mask = labels[label].bool().numpy()
            rtstruct.add_roi(mask=mask, name=name, color=color)
    #rtstruct.add_roi(mask=rings.bool().numpy(), name=info['ring_info'][0], color=list(info['ring_info'][1]))
    rtstruct.save(os.path.join(folder, f"{prefix}_RTSTRUCT"))
    return os.path.join(folder, f"{prefix}_RTSTRUCT")
    
def save_to_rtdose(dicom_series, doses, info, folder, prefix="predicted"):
    rtdose = generate_base_dataset()
    add_study_and_series_information(rtdose, dicom_series)
    add_patient_information(rtdose, dicom_series)
    add_refd_frame_of_ref_sequence(rtdose, dicom_series)
    doses = pad_around_original_center(doses, info['size_info']['original_size'], info['size_info']['crop_center'])
    add_doses(rtdose, doses)
    rtdose.save_as(os.path.join(folder, f'{prefix}_RTDOSE{rtdose.FrameOfReferenceUID}'))
    return os.path.join(folder, f'{prefix}_RTDOSE{rtdose.FrameOfReferenceUID}')

def generate_base_dataset() -> pydicom.dataset.FileDataset:
    file_name = "rtdose"
    file_meta = get_file_meta()
    rtdose = pydicom.dataset.FileDataset(file_name, {}, file_meta=file_meta, preamble=b"\0" * 128)
    add_required_elements_to_ds(rtdose)
    return rtdose

def get_file_meta() -> pydicom.dataset.FileMetaDataset:
    file_meta = pydicom.dataset.FileMetaDataset()
    file_meta.FileMetaInformationGroupLength = 294
    file_meta.FileMetaInformationVersion = b"\x00\x00"
    file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian
    file_meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.481.2"
    file_meta.MediaStorageSOPInstanceUID = (
        pydicom.uid.generate_uid()
    )  # TODO find out random generation is fine
    file_meta.ImplementationClassUID = rt_utils.utils.SOPClassUID.RTSTRUCT_IMPLEMENTATION_CLASS
    file_meta.ImplementationVersionName = "DicomObjects.NET"
    file_meta.PrivateInformationCreatorUID = rt_utils.utils.SOPClassUID.RTSTRUCT_IMPLEMENTATION_CLASS
    file_meta.PrivateInformation = b"\x00\x00"
    return file_meta

def add_required_elements_to_ds(ds: pydicom.dataset.FileDataset):
    dt = datetime.datetime.now()
    # Append data elements required by the DICOM standarad
    ds.SpecificCharacterSet = "ISO_IR 100"
    ds.InstanceCreationDate = dt.strftime("%Y%m%d")
    ds.InstanceCreationTime = dt.strftime("%H%M%S.%f")
    ds.Modality = "RTDOSE"
    ds.Manufacturer = "ProSeg"
    ds.ManufacturerModelName = "ProSeg"
    ds.InstitutionName = "Anonymized Hospital"
    ds.InstitutionAdress = "Anonymized Address"
    ds.ReferringPhysiansName = "Dr. Anonymous"
    ds.StationName = "Any Station"
    ds.DoseUnits = "GY"
    ds.DoseType = "PHYSICAL"
    ds.DoseSummationType = "PLAN"
    # Set the transfer syntax
    ds.is_little_endian = True
    ds.is_implicit_VR = True
    # Set values already defined in the file meta
    ds.SOPClassUID = ds.file_meta.MediaStorageSOPClassUID
    ds.SOPInstanceUID = ds.file_meta.MediaStorageSOPInstanceUID

def add_study_and_series_information(ds: pydicom.dataset.FileDataset, dicom_series):
    reference_ds = dicom_series[0]  # All elements in series should have the same data
    ds.StudyDate = reference_ds.StudyDate
    ds.SeriesDate = getattr(reference_ds, "SeriesDate", "")
    ds.StudyTime = reference_ds.StudyTime
    ds.SeriesTime = getattr(reference_ds, "SeriesTime", "")
    ds.StudyDescription = getattr(reference_ds, "StudyDescription", "")
    ds.SeriesDescription = getattr(reference_ds, "SeriesDescription", "")
    ds.StudyInstanceUID = reference_ds.StudyInstanceUID
    ds.SeriesInstanceUID = pydicom.uid.generate_uid()  # TODO: find out if random generation is ok
    ds.StudyID = reference_ds.StudyID
    ds.SeriesNumber = "1"  # TODO: find out if we can just use 1 (Should be fine since its a new series)
    ds.PixelSpacing = reference_ds.PixelSpacing
    ds.ImageOrientationPatient = reference_ds.ImageOrientationPatient
    ds.ImagePositionPatient = [-x if x > 0 else x for x in reference_ds.ImagePositionPatient]
    ds.Rows = getattr(reference_ds, "Rows", "")
    ds.Columns = getattr(reference_ds, "Columns", "")
    ds.NumberOfFrames = len(dicom_series)
    ds.PixelRepresentation = reference_ds.PixelRepresentation
    ds.SliceThickness = reference_ds.SliceThickness
    ds.PhotometricInterpretation = "MONOCHROME2"
    values = [i * ds.SliceThickness for i in range(ds.NumberOfFrames)]
    ds.add_new(pydicom.tag.Tag(0x3004, 0x000C), 'DS', values)
    ds.add_new(pydicom.tag.Tag(0x0028, 0x0009), 'AT', [(0x3004, 0x000C)])
    ds.add_new(pydicom.tag.Tag(0x0020, 0x0052), 'UI', reference_ds.FrameOfReferenceUID)
    
def add_patient_information(ds: pydicom.dataset.FileDataset, dicom_series):
    reference_ds = dicom_series[0]  # All elements in series should have the same data
    ds.PatientName = getattr(reference_ds, "PatientName", "")
    ds.PatientID = getattr(reference_ds, "PatientID", "")
    ds.PatientBirthDate = getattr(reference_ds, "PatientBirthDate", "")
    ds.PatientSex = getattr(reference_ds, "PatientSex", "")
    ds.PatientAge = getattr(reference_ds, "PatientAge", "")
    ds.PatientSize = getattr(reference_ds, "PatientSize", "")
    ds.PatientWeight = getattr(reference_ds, "PatientWeight", "")

def add_refd_frame_of_ref_sequence(ds: pydicom.dataset.FileDataset, series_data):
    ds.FrameOfReferenceUID=series_data[0].FrameOfReferenceUID
    
def add_doses(ds: pydicom.dataset.FileDataset, doses):
    doses = (doses/0.00059785837802)/100
    max_val_dose = torch.max(doses)
    if max_val_dose > 65535:
        doses=doses/(max_val_dose/65535)
    ds.PixelData = doses.numpy().astype(np.uint16).tobytes()
    ds.DoseGridScaling = 0.00059785837802
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.SamplesPerPixel = 1
    
def add_shift(image, label, widthshift=0.02, heightshift=0.03):#0.02, 0.03
    threed = len(label.shape) > 3
    if not threed:
        height, width = image.shape[1], image.shape[2]

        max_dy = heightshift * height
        max_dx = widthshift * width

        width = np.round(np.random.uniform(-max_dx, max_dx))
        height = np.round(np.random.uniform(-max_dy, max_dy))

        image = T.functional.affine(image,0,[height, width],1,0)
        label = T.functional.affine(label,0,[height, width],1,0)
    else:
        height, width = image.shape[2], image.shape[3]
        max_dy = heightshift * height
        max_dx = widthshift * width

        width = np.round(np.random.uniform(-max_dx, max_dx))
        height = np.round(np.random.uniform(-max_dy, max_dy))
        
        for z in range(image.shape[1]):
            image[:, z] = T.functional.affine(image[:, z],0,[height, width],1,0)
            label[:, z] = T.functional.affine(label[:, z],0,[height, width],1,0) 

    return image, label



def add_rotate(image, label, minangle = -3, maxangle = 3):  #3
    threed = len(label.shape) > 3
    # Get a random rotation angle between the min and max angle
    angle = random.randint(minangle, maxangle)
    if not threed:
        # Rotate both image and label with the same angle
        image = T.functional.rotate(image, angle)
        label = T.functional.rotate(label, angle)
    else:
        for z in range(image.shape[1]):
            image[:, z] = T.functional.rotate(image[:, z], angle)
            label[:, z] = T.functional.rotate(label[:, z], angle)
    return image, label

    


def apply_augmentation(images, labels, augmentation_functions):    
    new_images = images.clone()
    images = None
    new_labels = labels.clone()
    labels = None
    
    if len(augmentation_functions) > 0:
        for i in range(new_images.shape[0]):
            new_image = new_images[i]
            new_label = new_labels[i].unsqueeze(0)
            functions_to_apply = random.sample(augmentation_functions, random.randint(1,len(augmentation_functions)))
            for function in functions_to_apply:
                new_image, new_label = function(new_image, new_label)            
            new_images[i] = new_image
            new_labels[i] = new_label.squeeze(0)
            
    return new_images, new_labels

def plot_slice_with_contours(images, labels, label_info):
    images = images[:, 0].numpy()*255
    #images = np.clip(images,0,1)
    drawn_images = []
    for image in range(images.shape[0]):
        img = images[image].astype(np.uint8)
        label = labels[image]
        img = cv.cvtColor(img, cv.COLOR_GRAY2RGB)

        overlay = img.copy()
        for label_num, info in label_info.items():
            mask = torch.where(label==label_num, torch.tensor(1), torch.tensor(0)).numpy().astype(np.uint8)
            mask = mask.astype(np.uint8)
            color = (0,255,0)#info[1]
            contours, hierarchy = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
            cv.drawContours(img, contours, -1, color, 1, 1)
            cv.drawContours(overlay, contours, -1, color, cv.FILLED)

        img = cv.addWeighted(overlay, 0.3, img, 0.7, 0)

        img = torch.from_numpy(img)
        drawn_images.append(img)
    drawn_images = torch.stack(drawn_images, dim=0)
    return drawn_images