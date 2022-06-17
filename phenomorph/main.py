#%% imports

clean_namespace = dir()

import copy
import numpy as np
import os
import random
import shutil
from dataclasses import make_dataclass

import cv2
import dlib
import xml.etree.ElementTree as ET
from xml.dom import minidom
from PIL import Image
from tqdm import tqdm as _tqdm

from phenopype import core as pp_core
from phenopype import main as pp_main
from phenopype import settings as pp_settings
from phenopype import utils as pp_utils
from phenopype import utils_lowlevel as pp_utils_lowlevel

from phenomorph import utils

#%% classes

class Model(object):
    def __init__(
            self,
            rootdir,
            tag=None,
            overwrite=False
            ):
        """


        Parameters
        ----------
        rootdir : TYPE
            DESCRIPTION.
        tag : TYPE, optional
            DESCRIPTION. The default is None.
        overwrite : TYPE, optional
            DESCRIPTION. The default is False.

        Returns
        -------
        None.

        """

        ## init at root
        self.rootdir = os.path.abspath(rootdir)
        print(f"Initialized ml-morph at {self.rootdir}")

        ## make other directories if necessary
        self.imagedir = os.path.join(self.rootdir, "images")
        self.modeldir = os.path.join(self.rootdir, "models")
        self.configdir = os.path.join(self.rootdir, "config")
        self.xmldir = os.path.join(self.rootdir, "xml")
        for directory in [self.rootdir, self.imagedir, self.configdir, self.xmldir]:
            if not os.path.exists(directory):
                os.makedirs(directory)
                print("- created {}".format(os.path.basename(directory)))

        ## load existing components
        if not tag.__class__.__name__ == "NoneType":
            ret = pp_utils_lowlevel._file_walker(self.xmldir, include=[tag], pype_mode=True)[0]
            if len(ret) > 0:
                print('- found training and test datasets "test_{}.xml" and "train_{}.xml"'.format(tag, tag))
            configpath = os.path.join(self.configdir, 'config_{}.yaml'.format(tag))
            if os.path.isfile(configpath):
                self.configpath = configpath
                print('- loaded config "config_{}.yaml"'.format(tag))
            model_path = os.path.join(self.modeldir, 'predictor_{}.dat'.format(tag))
            if os.path.isfile(model_path):
                self.model_path = model_path
                print('- loaded model "predictor_{}.dat"'.format(tag))


    def create_training_data(
            self,
            tag,
            images,
            landmarks,
            bboxes=None,
            parameters=None,
            random_seed=42,
            percentage=0.8,
            flip=False,
            prop_train=None,
            prop_test=None,
            n_train=None,
            n_test=None,
            overwrite=False,
            debug=False,
            verbose=True,
            **kwargs
            ):
        """


        Parameters
        ----------
        tag : TYPE
            DESCRIPTION.
        images : TYPE
            DESCRIPTION.
        landmarks : TYPE
            DESCRIPTION.
        bboxes : TYPE, optional
            DESCRIPTION. The default is None.
        parameters : TYPE, optional
            DESCRIPTION. The default is None.
        random_seed : TYPE, optional
            DESCRIPTION. The default is 42.
        percentage : TYPE, optional
            DESCRIPTION. The default is 0.8.
        flip : TYPE, optional
            DESCRIPTION. The default is False.
        prop_train : TYPE, optional
            DESCRIPTION. The default is None.
        prop_test : TYPE, optional
            DESCRIPTION. The default is None.
        n_train : TYPE, optional
            DESCRIPTION. The default is None.
        n_test : TYPE, optional
            DESCRIPTION. The default is None.
        overwrite : TYPE, optional
            DESCRIPTION. The default is False.
        debug : TYPE, optional
            DESCRIPTION. The default is False.
        verbose : TYPE, optional
            DESCRIPTION. The default is True.
        **kwargs : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """

        # =============================================================================
        # setup

        ## define flags
        flags = make_dataclass(
            cls_name="flags", fields=[
                ("overwrite", bool, overwrite),
                ("debug", bool, debug),
                ("verbose", bool, verbose),
                ]
        )

        ## check image list
        if not images.__class__.__name__ == "list":
            images = [images]

        imageList, imagenameList = [], []
        for idx, item in enumerate(images):
            imageDict = {}

            ## directory case
            if item.__class__.__name__ == "str":
                if os.path.isdir(item):
                    dirpath = item
                    for imagename in os.listdir(dirpath):
                        imagenameList.append(imagename)
                        imagepath = os.path.join(dirpath, imagename)
                        if os.path.isfile(imagepath):
                            imageDict[imagename] = imagepath
                        else:
                            print("WARNING: {} does not exist".format(imagepath))

            ## list of paths case
            elif item.__class__.__name__ == "list":
                for imagepath in item:
                    if os.path.isfile(imagepath):
                        imagename = os.path.basename(imagepath)
                        imagenameList.append(imagename)
                        imageDict[imagename] = imagepath
                    else:
                        print("WARNING: {} does not exist".format(imagepath))

            ## phenopype project - for overwrite skip
            elif item.__class__.__name__ == "Project":
                imageDict = copy.deepcopy(item.file_names)

            ## wrong format
            else:
                print('ERROR: cannot read postion {} from supplied image list - skipping.'.format(idx))
                continue

            ## create single dictionary per provded directory or list
            imageList.append(imageDict)

        # =============================================================================
        ## training data overwrite check

        ret = pp_utils_lowlevel._file_walker(self.xmldir, include=[tag], pype_mode=True)[0]
        if len(ret) > 0 and not flags.overwrite:
            print('Files "test_{}.xml" and "train_{}.xml" already exist (overwrite=False)'.format(tag,tag))
        else:

        # =============================================================================

            ## create landmark dict
            if not landmarks.__class__.__name__ == "list":
                landmarks = [landmarks]

            landmarksDict = {}
            for idx, item in enumerate(landmarks):

                ## file case
                if item.__class__.__name__ == "str":
                    if os.path.isfile(item):
                        landmark = utils.read_csv(item)
                        for imagename, coords in zip(landmark["im"], landmark["coords"]):
                            if imagename in imagenameList:
                                landmarksDict[imagename] = coords
                            else:
                                print("WARNING: landmarks not matching image names.")

                ## dictionary of landmarks case
                elif item.__class__.__name__ == "dict":
                    for imagename, coords in item.items():
                        if imagename in imagenameList:
                            landmarksDict[imagename] = coords
                        else:
                            print("WARNING: landmarks not matching image names.")

                ## wrong format
                else:
                    print('ERROR: cannot read postion {} from supplied landmark list - skipping.'.format(idx))
                    continue

            ## check bounding boxes
            if not bboxes.__class__.__name__ == "NoneType":
                if not bboxes.__class__.__name__ == "list":
                    bboxes = [bboxes]
            else:
                bboxes = [{}] * len(images)
            bboxesList = bboxes

            ## check parameters
            if not parameters.__class__.__name__ == "NoneType":
                if not parameters.__class__.__name__ == "list":
                    parameters = [parameters]
            else:
                parameters = [{}] * len(images)

            if not len(images) == len(landmarks) == len(bboxes) == len(parameters):
                print("ERROR: images, landmarks, bboxes or parameters have different lengths - aborting.")
                return

            ## parameter checks
            parameter_checks = {
                "percentage": percentage,
                "flip": flip,
                "prop_train":prop_train,
                "prop_test":prop_test,
                "n_train":n_train,
                "n_test":n_test,
                }
            for parameter in parameters:
                for parameter_name, parameter_value in parameter_checks.items():
                    if not parameter_name in parameter.keys():
                        parameter[parameter_name] = parameter_value

            ## init global xml files
            train_root, test_root = utils.init_xml_elements(n=2)

            ## set up random seed
            random.seed(random_seed)

            # =============================================================================
            # iterate over data


            ## loop over datasets
            for listIdx, images in enumerate(imageList):

                ## dataset specific xml files
                if len(imageList) > 1:
                    testSub_root  = utils.init_xml_elements()

                ## pull image name keys as list
                imageNames = list(images)

                ## folder specific splits and shuffling
                random.shuffle(imageNames)
                n_total = len(imageNames)
                val_warning_msg = "WARNING - specified amount of training images equal \
                      or larger than dataset. You need images for validation!"
                test_warning_msg = "No test images specified - using remaining portion"


                ## pull parameters:
                ## hierarchy: n_train > prop_train > percentage
                parameter = parameters[listIdx]

                if parameter["n_train"]:
                    if parameter["n_train"] >= n_total:
                        split = n_total
                        print(val_warning_msg)
                    else:
                        split = parameter["n_train"]
                    if parameter["n_test"]:
                        end = parameter["n_train"] + parameter["n_test"]
                    else:
                        print(test_warning_msg)
                        end = n_total
                    if end > n_total:
                        end = n_total
                elif parameter["prop_train"]:
                    if parameter["prop_train"] == 1:
                        print(val_warning_msg)
                    split = int(parameter["prop_train"] * n_total)
                    if parameter["prop_test"]:
                        end = split + int(parameter["prop_test"] * n_total)
                    if end > n_total:
                        end = n_total
                elif parameter["percentage"]:
                    split = int(parameter["percentage"] * n_total)
                    end = n_total

                for part in ["train","test"]:

                    if part == "train":
                        start, stop = 0, split
                    elif part == "test":
                        start, stop = split, end

                    for idx, filename in enumerate(imageNames[start:stop]):

                        filepath = imageList[listIdx][filename]
                        imageWidth, imageHeight = Image.open(filepath).size

                        ## feedback
                        if flags.verbose:
                            if len(imageList) > 1:
                                print("Preparing {}ing data for dataset {} - {} ({}/{})".format(part, listIdx+1, filename, idx+1, str(len(imageNames[start:stop]))))
                            else:
                                print("Preparing {}ing data for {} ({}/{})".format(part, filename, idx+1, str(len(imageNames[start:stop]))))

                        try:
                            ## get landmarks
                            coords = landmarksDict[filename]

                            ## bounding boxes
                            if not len(bboxesList[listIdx]) == 0:
                                rx, ry, rw, rh = bboxesList[listIdx][filename]
                            else:
                                rx, ry, rw, rh = 1, 1, imageWidth , imageHeight

                            ## flipping
                            if parameter["flip"]:

                                image = pp_utils.load_image(filepath)
                                image = cv2.flip(image, 1)
                                if not rx == 1:
                                    rx = imageWidth - (rx + rw)

                                coords_new = []
                                for coord in coords:
                                    coords_new.append((imageWidth - coord[0], coord[1]))
                                coords = pp_utils_lowlevel._convert_tup_list_arr(coords_new)[0]

                                pp_utils.save_image(image, dir_path=self.imagedir, file_name=filename)
                                filepath = os.path.relpath(os.path.join(self.imagedir,filename), self.xmldir)

                            else:
                                filepath = os.path.relpath(filepath, self.xmldir)

                            ## xml part
                            if part == "train":
                                train_root[2].append(utils.add_image_element(coords, (rx, ry, rw, rh), path=filepath))
                            elif part == "test":
                                test_root[2].append(utils.add_image_element(coords, (rx, ry, rw, rh), path=filepath))
                                if len(imageList) > 1:
                                    testSub_root[2].append(utils.add_image_element(coords, (rx, ry, rw, rh), path=filepath))
                        except:
                            if flags.debug:
                                raise
                            else:
                                print("something went wrong for {}".format(filename))

                    ## project specific actions after completing loop
                    if len(imageList) > 1:
                        if part == "test":
                            et = ET.ElementTree(testSub_root)
                            xmlstr = minidom.parseString(ET.tostring(et.getroot())).toprettyxml(indent="   ")
                            with open(os.path.join(self.xmldir, part + "_" + str(listIdx+1) + "_" + tag + ".xml"), "w") as f:
                                f.write(xmlstr)

                ## format final XML output
                for root, part in zip([train_root, test_root],["train","test"]):
                    et = ET.ElementTree(root)
                    xmlstr = minidom.parseString(ET.tostring(et.getroot())).toprettyxml(indent="   ")
                    with open(os.path.join(self.xmldir, part + "_" + tag + ".xml"), "w") as f:
                        f.write(xmlstr)

        # =============================================================================
        # checking saved data

        for listIdx, images in enumerate(imageList):

            train_path = os.path.join(self.xmldir, "train_{}.xml".format(tag))
            if len(imageList) > 1:
                test_path = os.path.join(self.xmldir, "test_{}_{}.xml".format(listIdx+1, tag))
            else:
                test_path = os.path.join(self.xmldir, "test_{}.xml".format(tag))


            n_total = len(images)
            n_train_imgs = utils.xml_element_counter(train_path, "image", images)
            n_test_imgs = utils.xml_element_counter(test_path, "image")

            print("\n")

            if len(imageList) > 1:
                print("Prepared train/test datasets for \"{}\" from dataset \"{}\":".format(tag, listIdx+1))
            else:
                print("Prepared train/test datasets for \"{}\":".format(tag))
            print("-----------------------------------------------------------------")
            print("total available: {} images - training: {} images - testing: {} images".format(n_total, n_train_imgs, n_test_imgs))


    def load_config(
            self,
            tag,
            configpath,
            overwrite=False,
            verbose=True,
            ):
        """


        Parameters
        ----------
        tag : TYPE
            DESCRIPTION.
        configpath : TYPE
            DESCRIPTION.
        overwrite : TYPE, optional
            DESCRIPTION. The default is False.
        verbose : TYPE, optional
            DESCRIPTION. The default is True.
         : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """

        if os.path.isfile(configpath):

            self.configpath = os.path.join(self.configdir, "config_{}.yaml".format(tag))

            if not os.path.isfile(self.configpath):
                shutil.copyfile(configpath, self.configpath)
                print("Found config file - saved a copy at {}".format(self.configpath))
            else:
                print("Found config file at {} - loading (overwrite=False)".format(self.configpath))

                cfg = pp_utils_lowlevel._load_yaml(configpath)
                options = dlib.shape_predictor_training_options()
                options.num_trees_per_cascade_level = cfg["train"]["num_trees"]
                options.nu = cfg["train"]["regularization"]
                options.num_threads = cfg["train"]["threads"]
                options.tree_depth = cfg["train"]["tree_depth"]
                options.cascade_depth = cfg["train"]["cascade_depth"]
                options.feature_pool_size = cfg["train"]["feature_pool"]
                options.num_test_splits = cfg["train"]["test_splits"]
                options.oversampling_amount = cfg["train"]["oversampling"]
                options.be_verbose = cfg["train"]["verbose"]
                self.options = options

        else:
            print("- {} does not exist!".format(configpath))

    def train_model(
            self,
            tag,
            overwrite=False
            ):
        """


        Parameters
        ----------
        tag : TYPE
            DESCRIPTION.
        overwrite : TYPE, optional
            DESCRIPTION. The default is False.

        Returns
        -------
        None.

        """

        assert self.options is not None, print(
            "Please load a ml-morph config file first"
        )

        train_xml = os.path.join(self.rootdir, "xml", f"train_{tag}.xml")
        assert os.path.exists(train_xml), print(
            f"No train xml found at {train_xml}. Please make sure to run preprocess_folder first"
        )

        if not os.path.exists(self.modeldir):
            os.makedirs(self.modeldir)

        predictor_path = os.path.join(self.modeldir, f"predictor_{tag}.dat")

        if os.path.exists(predictor_path) and overwrite is False:
            print("Model already exists. Please set overwrite=True to overwrite")
        else:
            dlib.train_shape_predictor(train_xml, predictor_path, self.options)
        error = dlib.test_shape_predictor(train_xml, predictor_path)
        print(f"Training error (average pixel deviation): {error}")

    def test_model(
            self,
            tag
            ):

        """


        Parameters
        ----------
        tag : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """

        ## assemble test-file list
        test_global_file, test_sub_files = os.path.join(self.xmldir, "test_{}.xml".format(tag)), []
        for xml_file in os.listdir(self.xmldir):
            if all([
                    "test" in xml_file,
                    tag in xml_file,
                    not xml_file == "test_{}.xml".format(tag)
                    ]):
                xml_file_path = os.path.join(self.xmldir, xml_file)
                test_sub_files.append(xml_file_path)


        predictor_path = os.path.join(self.modeldir, f"predictor_{tag}.dat")

        if os.path.isfile(predictor_path):
            error = dlib.test_shape_predictor(test_global_file, predictor_path)
            if len(test_sub_files) > 1:
                print("Testing global error (average pixel deviation): {}".format(error))
                print("-----------------------------------------------------------------")
                for idx, xml_file_path in enumerate(test_sub_files):
                    print("Testing error (average pixel deviation) on dataset {}: {}".format(idx+1, error))
                    error = dlib.test_shape_predictor(xml_file_path, predictor_path)
            else:
                print("Testing error (average pixel deviation): {}".format(error))
        else:
            print("Cannot find shape prediction model at {}".format(predictor_path))

    def predict_dir(
            self,
            tag,
            dirpath,
            print_csv=False
            ):
        """


        Parameters
        ----------
        tag : TYPE
            DESCRIPTION.
        dirpath : TYPE
            DESCRIPTION.
        print_csv : TYPE, optional
            DESCRIPTION. The default is False.

        Returns
        -------
        df : TYPE
            DESCRIPTION.

        """
        predictor_path = os.path.join(self.modeldir, f"predictor_{tag}.dat")
        assert os.path.exists(
            predictor_path
        ), f"Cannot find shape prediction model at {predictor_path}"
        assert os.path.exists(dirpath), "No image directory found at {dir_path}"
        output_xml = os.path.join(dirpath, f"predicted_{tag}.xml")
        utils.predictions_to_xml(predictor_path, dirpath, None, output_xml)
        df = utils.dlib_xml_to_pandas(output_xml, print_csv)
        os.remove(output_xml)
        return df

    def predict_image(
            self,
            tag,
            img,
            bbox_coords=None,
            plot=False,
            colour=None
            ):
        """


        Parameters
        ----------
        tag : TYPE
            DESCRIPTION.
        img : TYPE
            DESCRIPTION.
        bbox_coords : TYPE, optional
            DESCRIPTION. The default is None.
        plot : TYPE, optional
            DESCRIPTION. The default is False.
        colour : TYPE, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        landmark_tuple_list : TYPE
            DESCRIPTION.

        """
        predictor_path = os.path.join(self.modeldir, f"predictor_{tag}.dat")
        print("using model: {}".format(predictor_path))
        assert os.path.exists(
            predictor_path
        ), f"Cannot find shape prediction model at {predictor_path}"
        if type(img) == str:
            assert os.path.exists(img), "No image found at {image_path}"
            img = pp_utils.load_image(img)
        elif type(img) == np.ndarray:
            img = copy.deepcopy(img)
        if bbox_coords:
            rx, ry, rw, rh = bbox_coords
            rect = dlib.rectangle(
                left=rx, top=ry, right=rx+rw, bottom=ry+rh
            )
        else:
            rect = dlib.rectangle(
                left=1, top=1, right=img.shape[1] - 1, bottom=img.shape[0] - 1
            )
        predictor = dlib.shape_predictor(predictor_path)

        ## weird order
        shape = predictor(img, rect)
        num_parts = range(0, shape.num_parts)
        points_dict = {}
        for item, idx in enumerate(sorted(num_parts, key=str), 0):
            x, y = shape.part(item).x, shape.part(item).y
            points_dict[idx] = (x,y)

        ## fixed order
        landmark_tuple_list = []
        for key, value in sorted(points_dict.items()):
            landmark_tuple_list.append(value)

        ## optional plotting
        if plot:
            if not colour:
                colour = pp_utils_lowlevel._get_bgr("red")
            else:
                colour = pp_utils_lowlevel._get_bgr(colour)
            for idx, coords in enumerate(landmark_tuple_list, 0):
                cv2.circle(img, coords, pp_utils_lowlevel._auto_point_size(img), colour, -1)
                cv2.putText(img, str(idx + 1), coords, cv2.FONT_HERSHEY_SIMPLEX, pp_utils_lowlevel._auto_text_width(
                    img), colour, pp_utils_lowlevel._auto_text_size(img), cv2.LINE_AA)
            pp_utils.show_image(img)

        return landmark_tuple_list



class PhenopypeModel(Model):
    def __init__(
            self,
            rootdir,
            projects,
            tag=None,
            overwrite=False,
            **kwargs,
            ):
        """


        Parameters
        ----------
        rootdir : TYPE
            DESCRIPTION.
        projects : TYPE
            DESCRIPTION.
        tag : TYPE, optional
            DESCRIPTION. The default is None.
        overwrite : TYPE, optional
            DESCRIPTION. The default is False.
        **kwargs : TYPE
            DESCRIPTION.
         : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """

        super().__init__(rootdir, tag, overwrite)

        ## list check and attach projects
        self.phenopype_projects = {}
        if not projects.__class__.__name__ == "list":
            projects = [projects]

        for project in projects:
            if project.__class__.__name__ == "str":
                if os.path.isdir(project):
                    project = pp_main.Project(project)
                else:
                    print("wrong directory path - couldn't find {}".format(project))
                    return
            project_name = os.path.basename(project.root_dir)
            self.phenopype_projects[project_name] = project



    def create_training_data(
            self,
            tag,
            phenopype_tag=None,
            landmark_id=None,
            mask=False,
            mask_id=None,
            phenopype_parameters=None,
            overwrite=False,
            debug=False,
            **kwargs

            ):
        """


        Parameters
        ----------
        tag : TYPE
            DESCRIPTION.
        phenopype_tag : TYPE, optional
            DESCRIPTION. The default is None.
        landmark_id : TYPE, optional
            DESCRIPTION. The default is None.
        mask : TYPE, optional
            DESCRIPTION. The default is False.
        mask_id : TYPE, optional
            DESCRIPTION. The default is None.
        verbose : TYPE, optional
            DESCRIPTION. The default is False.
        **kwargs : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        # =============================================================================
        # setup

        ## define flags
        flags = make_dataclass(
            cls_name="flags", fields=[
                ("overwrite", bool, overwrite),
                ("debug", bool, debug),
                ])

        ## check phenopype_parameters type
        if not phenopype_parameters.__class__.__name__ == "NoneType":
            if not phenopype_parameters.__class__.__name__ == "list":
                phenopype_parameters = [phenopype_parameters]
        else:
            phenopype_parameters = [{}] * len(self.phenopype_projects)

        ## phenopype_parameters checks
        parameter_checks = {
            "phenopype_tag": tag,
            "landmark_id": landmark_id,
            "mask":mask,
            "mask_id":mask_id,
            }
        for parameter in phenopype_parameters:
            for parameter_name, parameter_value in parameter_checks.items():
                if not parameter_name in parameter.keys():
                    parameter[parameter_name] = parameter_value


        # =============================================================================
        # collect data from projects



        ## training data overwrite check
        ret = pp_utils_lowlevel._file_walker(self.xmldir, include=[tag], pype_mode=True)[0]
        if len(ret) > 0 and not flags.overwrite:
            images, landmarks, bboxes = list(self.phenopype_projects.values()), [], []
        else:
            images, landmarks, bboxes = [], [], []

            for projIdx, project_name in enumerate(self.phenopype_projects.keys()):

                project = self.phenopype_projects[project_name]
                projectImages, projectLandmarks, projectBboxes = [], {}, {}

                for dirpath in _tqdm(
                        project.dir_paths,
                        "- loading data for project {}:".format(project_name),
                        total=len(project.dir_paths),
                        ):

                    try:
                        ## load image metadata and annotations
                        attributes = pp_utils_lowlevel._load_yaml(os.path.join(dirpath, "attributes.yaml"))
                        annotations = pp_core.export.load_annotation(os.path.join(dirpath, "annotations_" + phenopype_parameters[projIdx]["phenopype_tag"] + ".json"), verbose=False)

                        ## load image paths
                        filename = attributes["image_phenopype"]["filename"]
                        if attributes["image_phenopype"]["mode"] == "link":
                            filepath = os.path.abspath(os.path.join(dirpath, attributes["image_phenopype"]["filepath"]))
                        else:
                            filepath = attributes["image_phenopype"]["filepath"]
                        projectImages.append(filepath)

                        ## load landmarks
                        if not phenopype_parameters[projIdx]["landmark_id"]:
                            annotation_id = pp_utils_lowlevel._get_annotation_id(annotations, pp_settings._landmark_type, verbose=False)
                        else:
                            annotation_id = phenopype_parameters[projIdx]["landmark_id"]
                        annotation = pp_utils_lowlevel._get_annotation2(annotations, pp_settings._landmark_type, annotation_id)
                        coords = annotation["data"][pp_settings._landmark_type]
                        projectLandmarks[filename] = np.asarray(coords, dtype="int32")

                        ## load mask
                        if phenopype_parameters[projIdx]["mask"]:
                            ## load landmarks
                            if not phenopype_parameters[projIdx]["mask_id"]:
                                annotation_id = pp_utils_lowlevel._get_annotation_id(annotations, pp_settings._mask_type, verbose=False)
                            else:
                                annotation_id = phenopype_parameters[projIdx]["mask_id"]
                            annotation = pp_utils_lowlevel._get_annotation2(annotations, pp_settings._mask_type, annotation_id)
                            coords = annotation["data"][pp_settings._mask_type]
                            projectBboxes[filename] = cv2.boundingRect(np.asarray(coords, dtype="int32"))


                    except:
                        if flags.debug:
                            raise
                        else:
                            print("something went wrong for {}".format(filename))

                ## create lists to pass on to super class
                images.append(projectImages)
                landmarks.append(projectLandmarks)
                bboxes.append(projectBboxes)

            # =============================================================================
            # supply data to super

        print("\n")
        super().create_training_data(
            tag=tag,
            images=images,
            landmarks=landmarks,
            bboxes=bboxes,
            overwrite=overwrite,
            debug=debug,
            **kwargs)

#%% functions

def predict_image(
        img,
        model_path,
        bbox_coords=None,
        plot=False,
        colour=None
        ):
    """


    Parameters
    ----------
    tag : TYPE
        DESCRIPTION.
    img : TYPE
        DESCRIPTION.
    bbox_coords : TYPE, optional
        DESCRIPTION. The default is None.
    plot : TYPE, optional
        DESCRIPTION. The default is False.
    colour : TYPE, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    landmark_tuple_list : TYPE
        DESCRIPTION.

    """

    predictor_path = model_path

    print("using model: {}".format(predictor_path))
    assert os.path.exists(
        predictor_path
    ), f"Cannot find shape prediction model at {predictor_path}"
    if type(img) == str:
        assert os.path.exists(img), "No image found at {image_path}"
        img = pp_utils.load_image(img)
    elif type(img) == np.ndarray:
        img = copy.deepcopy(img)
    if bbox_coords:
        rx, ry, rw, rh = bbox_coords
        rect = dlib.rectangle(
            left=rx, top=ry, right=rx+rw, bottom=ry+rh
        )
    else:
        rect = dlib.rectangle(
            left=1, top=1, right=img.shape[1] - 1, bottom=img.shape[0] - 1
        )
    predictor = dlib.shape_predictor(predictor_path)

    ## weird order
    shape = predictor(img, rect)
    num_parts = range(0, shape.num_parts)
    points_dict = {}
    for item, idx in enumerate(sorted(num_parts, key=str), 0):
        x, y = shape.part(item).x, shape.part(item).y
        points_dict[idx] = (x,y)

    ## fixed order
    landmark_tuple_list = []
    for key, value in sorted(points_dict.items()):
        landmark_tuple_list.append(value)

    ## optional plotting
    if plot:
        if not colour:
            colour = pp_utils_lowlevel._get_bgr("red")
        else:
            colour = pp_utils_lowlevel._get_bgr(colour)
        for idx, coords in enumerate(landmark_tuple_list, 0):
            cv2.circle(img, coords, pp_utils_lowlevel._auto_point_size(img), colour, -1)
            cv2.putText(img, str(idx + 1), coords, cv2.FONT_HERSHEY_SIMPLEX, pp_utils_lowlevel._auto_text_width(
                img), colour, pp_utils_lowlevel._auto_text_size(img), cv2.LINE_AA)
        pp_utils.show_image(img)

    return landmark_tuple_list
    
