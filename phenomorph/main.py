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

from phenopype import core as pp_core
from phenopype import main as pp_main
from phenopype import settings as pp_settings
from phenopype import utils as pp_utils
from phenopype import utils_lowlevel as pp_utils_lowlevel

from phenomorph import utils


class GenericModel(object):
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
                print("- found training and test datasets \"test_{}.xml\" and \"train_{}.xml\"".format(tag, tag))
            configpath = os.path.join(self.configdir, "config_{}.yaml".format(tag))
            if os.path.isfile(configpath):
                self.configpath = configpath
                print("- loaded config \"config_{}.yaml\"".format(tag))
            model_path = os.path.join(self.modeldir, "predictor_{}.dat".format(tag))
            if os.path.isfile(model_path):
                self.model_path = model_path
                print("- loaded model \"predictor_{}.dat\"".format(tag))
          

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
            
            **kwargs
            ):
        
        
        
        
        ## basic checks        
        if not images.__class__.__name__ == "list":
            images = [images]
        if not landmarks.__class__.__name__ == "list":
            landmarks = [landmarks]
        if not bboxes.__class__.__name__ == "NoneType":
            if not bboxes.__class__.__name__ == "list":
                bboxes = [bboxes]
        else:
            bboxes = [None] * len(images)
        if not parameters.__class__.__name__ == "NoneType":
            if not parameters.__class__.__name__ == "list":
                parameters = [parameters]
        else:
            parameters = [None] * len(images)

        if not len(images) == len(landmarks) == len(bboxes) == len(parameters):
            print("ERROR: images, landmarks, bboxes or parameters have different length - aborting.")
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
        
        parameters_updated = []
        for parameter in parameters:
            for parameter_name, parameter_value in parameter_checks.items():
                if not parameter_name in parameter.keys():
                    parameter[parameter_name] = parameter_value      
            parameters_updated.append(parameter)
            
        self.parameters = parameters    
            
        ## init global xml files
        train_root, train_image_e = utils.init_xml_elements()
        test_root, test_image_e = utils.init_xml_elements()        
        train_xml = os.path.join(self.xmldir, f"train_{tag}.xml")
        test_xml = os.path.join(self.xmldir, f"test_{tag}.xml")
        
        ## set up random seed
        random.seed(random_seed)

        
        ## loop over datasets
        for imagepaths, csvpaths, bboxes in zip(imagepathsList, csvpathsList, bboxesList):
            
            ## fetch project and set up project specific info
            dirname = os.path.basename(imagedir)
            parameter = parameters[project_name]
            feedback_dict[project_name] = {}
            
            if not os.path.isdir(imagedir):
                print('ERROR: "{}" does not exist - aborting.'.format(imagedir))
                return
            if not os.path.isdir(csvpath):
                print('ERROR: "{}" does not exist - aborting'.format(csvpath))
                return
            
            ## dataset specific xml files
            testSub_root, testSub_image_e = utils.init_xml_elements()        
            test_xml = os.path.join(self.xmldir, "test_{}_{}.xml".format(dirname, tag))

            for filename in imagedir:
                
                ## project specific splits
                proj_dirpaths_shuffled = copy.deepcopy(self.projects[project_name].dir_paths)
                random.shuffle(proj_dirpaths_shuffled)
                n_total = len(proj_dirpaths_shuffled)
                
                val_warning_msg = "WARNING - specified amount of training images equal \
                      or larger than dataset. You need images for validation!"
                test_warning_msg = "No test images specified - using remaining portion"
                            
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
                elif parameter["split"]:
                    split = int(parameter["split"] * n_total)
                    end = n_total
    
    
                for part in ["train","test"]:
            
                    if part == "train":
                        start, stop = 0, split
                    elif part == "test":
                        start, stop = split, end
        
                    for idx1, dirpath in enumerate(proj_dirpaths_shuffled[start:stop]):
                        
                        image = None
                
                        ## load data
                        attributes = pp_utils_lowlevel._load_yaml(os.path.join(dirpath, "attributes.yaml"))
                        annotations = pp_core.export.load_annotation(os.path.join(dirpath, "annotations_" + parameter["project_tag"] + ".json"), verbose=False)
                        filename = attributes["image_original"]["filename"]
                        filepath = attributes["image_phenopype"]["filepath"]
                        image_width, image_height= attributes["image_phenopype"]["width"],  attributes["image_phenopype"]["height"]
    
                        ## potentially not needed, because img-dirs are on the same level as xml dirs
                        image_phenopype_path = os.path.abspath(os.path.join(dirpath, attributes["image_phenopype"]["filepath"]))
                        filepath = os.path.relpath(image_phenopype_path, self.xmldir)
            
                        ## feedback
                        print("Preparing {} data for project {}: {} ({}/{})".format(part, project_name, filename, idx1+1, str(len(proj_dirpaths_shuffled[start:stop]))))       
            
                        ## checks and feedback
                        if annotations.__class__.__name__ == "NoneType":
                            print("No annotations found for {}".format(filename))
                            continue
                        if not annotation_type in annotations:
                            print("No annotation of type {} found for {}".format(annotation_type, filename))
                            continue
                        if annotation_id.__class__.__name__ == "NoneType":
                            annotation_id = max(list(annotations[annotation_type].keys()))
            
                        ## load landmarks
                        data = annotations[annotation_type][annotation_id]["data"][annotation_type]
            
                        ## masking
                        if parameter["mask"]:
                            if pp_settings._mask_type in annotations:
                                pass
                            else:
                                print("No annotation of type {} found for {}".format(pp_settings._mask_type, filename))
                                continue
                        
                            ## select last mask if no id is given
                            if mask_id.__class__.__name__ == "NoneType":
                                mask_id = max(list(annotations[pp_settings._mask_type].keys()))
            
                            ## get bounding rectangle and crop image to mask coords
                            coords = annotations[pp_settings._mask_type][mask_id]["data"][pp_settings._mask_type][0]
                            rx, ry, rw, rh = cv2.boundingRect(np.asarray(coords, dtype="int32"))
                        else:
                            rx, ry, rw, rh = 1, 1, image_width, image_height 
                            
                        ## flipping
                        if parameter["flip"]:
                                                    
                            image = pp_utils.load_image(dirpath)                       
                            image = cv2.flip(image, 1)
                            if not rx == 1:
                                rx = image_width - (rx + rw)
                                
                            parameter["mode"] = "save"
                            
                            data_new = []
                            for coord in data:
                                data_new.append((image_width - coord[0], coord[1]))
                            data = data_new
                        
                        ## saving
                        if parameter["mode"] == "save":
                            if image.__class__.__name__ == "NoneType":
                                image = pp_utils.load_image(dirpath)                       
                            pp_utils.save_image(image, dir_path=self.imagedir, file_name=filename)
                            filepath = os.path.relpath(os.path.join(self.imagedir,filename), self.xmldir)
                            
                        ## xml part
                        if part == "train":
                            train_images_e.append(utils.add_image_element(pp_utils_lowlevel._convert_tup_list_arr(data)[0], (rx, ry, rw, rh), path=filepath))
                        elif part == "test":
                            test_global_images_e.append(utils.add_image_element(pp_utils_lowlevel._convert_tup_list_arr(data)[0], (rx, ry, rw, rh), path=filepath))
                            test_sub_images_e.append(utils.add_image_element(pp_utils_lowlevel._convert_tup_list_arr(data)[0], (rx, ry, rw, rh), path=filepath))
    
            
                
            
            
            
            
            
            
        csv = utils.read_csv(self.csvpath)

        #     images_e.append(add_image_element(utils_lowlevel._convert_tup_list_arr(data)[0], (rx, ry, rw, rh), path=filepath))
    

        
        train_set, test_set = utils.split_train_test(csv, percentage)


        if os.path.exists(train_xml) and overwrite is False:
            print(
                "Train/Test split already exists. Please set overwrite=True to overwrite"
            )
        else:
            utils.generate_dlib_xml(train_set, self.rootdir, out_file=train_xml)
            utils.generate_dlib_xml(test_set, self.rootdir, out_file=test_xml)
            print(
                f"Train/Test split generated. Train dataset has {len(train_set['im'])} images, while Test dataset has {len(test_set['im'])} images"
            )
            
            

    def load_config(
            self, 
            configpath, 
            verbose=True
            ):
        """
        

        Parameters
        ----------
        configpath : TYPE
            DESCRIPTION.
        verbose : TYPE, optional
            DESCRIPTION. The default is True.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        
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
        
        if verbose:
            return print(f"Loaded ml-morph config file: {configpath}")


    def train_model(self, tag, overwrite=False):
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

    def test_model(self, tag):
        predictor_path = os.path.join(self.modeldir, f"predictor_{tag}.dat")
        test_xml = os.path.join(self.rootdir, "xml", f"test_{tag}.xml")
        assert os.path.exists(
            predictor_path
        ), f"Cannot find shape prediction model at {predictor_path}"
        assert os.path.exists(test_xml), f"Cannot find test xml file at {test_xml}"
        error = dlib.test_shape_predictor(test_xml, predictor_path)
        print(f"Testing error (average pixel deviation): {error}")

    def predict_dir(self, tag, dir_path, print_csv=False):
        predictor_path = os.path.join(self.modeldir, f"predictor_{tag}.dat")
        assert os.path.exists(
            predictor_path
        ), f"Cannot find shape prediction model at {predictor_path}"
        assert os.path.exists(dir_path), "No image directory found at {dir_path}"
        output_xml = os.path.join(dir_path, f"predicted_{tag}.xml")
        utils.predictions_to_xml(predictor_path, dir_path, None, output_xml)
        df = utils.dlib_xml_to_pandas(output_xml, print_csv)
        os.remove(output_xml)
        return df

    def predict_image(self, tag, img, bbox_coords=None, plot=False, colour=None):
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



class PhenopypeModel(GenericModel):
    def __init__(
            self, 
            rootdir,
            projects, 
            tag=None,
            overwrite=False,
            **kwargs,
            ):
        
        # =============================================================================
        # setup
        
        super().__init__(rootdir, tag, overwrite)

        ## list check and attach projects
        self.projects = {}
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
            self.projects[project_name] = project



    def create_training_data(
            self,
            tag,
            mode="link",
            project_tag=None,
            overwrite=False,
            landmark_id=None,
            mask=False,
            mask_id=None,
            flip=False,
            random_seed=42,
            split=0.8,
            prop_train=None,
            prop_test=None,
            n_train=None,
            n_test=None,
            parameters=None,
            ):
    
        # =============================================================================
        # setup
    
        ## define flags
        flags = make_dataclass(
            cls_name="flags", fields=[
                ("overwrite", bool, overwrite),
                ]
        )
    
        annotation_type = pp_settings._landmark_type
        annotation_id = landmark_id
        
        ## overwrite check
        ret = pp_utils_lowlevel._file_walker(self.xmldir, include=[tag], pype_mode=True)[0]
        if len(ret) > 0 and not flags.overwrite:
            print("test_{}.xml and train_{}.xml already exit (overwrite=False)\n".format(tag,tag))
        else:
            ## parameters 
            if not project_tag:
                project_tag = tag
                print("No project tag provided - using ml-morph tag.")            
                
            parameter_checks = {
                "project_tag": project_tag,
                "mode": mode,
                "mask": mask,
                "flip": 0,
                "split": split,
                "prop_train":prop_train,
                "prop_test":prop_test,
                "n_train":n_train,
                "n_test":n_test,
                }
            
            if parameters.__class__.__name__ == "NoneType":
                parameters = {}
            if len(parameters) == 0:
                for project_name in self.projects.keys():
                    parameters[project_name] = {}
            for project_name in self.projects.keys():
                if not project_name in parameters:
                    parameters[project_name] = {}
                for parameter_name, parameter_value in parameter_checks.items():
                    if not parameter_name in parameters[project_name].keys():
                        parameters[project_name][parameter_name] = parameter_value
            
            ## set up xml stuff        
            train_root = ET.Element('dataset')
            train_root.append(ET.Element('name'))
            train_root.append(ET.Element('comment'))
            train_images_e = ET.Element('images')
            train_root.append(train_images_e)
            
            test_global_root = ET.Element('dataset')
            test_global_root.append(ET.Element('name'))
            test_global_root.append(ET.Element('comment'))
            test_global_images_e = ET.Element('images')
            test_global_root.append(test_global_images_e)
            
            # =============================================================================
            # loop through images
            
            feedback_dict = {}
            
            for project_name in self.projects.keys():
    
                ## fetch project and set up project specific info
                parameter = parameters[project_name]
                feedback_dict[project_name] = {}
                
                ## project specific test-xml files
                test_sub_root = ET.Element('dataset')
                test_sub_root.append(ET.Element('name'))
                test_sub_root.append(ET.Element('comment'))
                test_sub_images_e = ET.Element('images')
                test_sub_root.append(test_sub_images_e)
                
                ## project specific splits
                random.seed(random_seed)
                proj_dirpaths_shuffled = copy.deepcopy(self.projects[project_name].dir_paths)
                random.shuffle(proj_dirpaths_shuffled)
                n_total = len(proj_dirpaths_shuffled)
                
                val_warning_msg = "WARNING - specified amount of training images equal \
                      or larger than dataset. You need images for validation!"
                test_warning_msg = "No test images specified - using remaining portion"
                            
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
                elif parameter["split"]:
                    split = int(parameter["split"] * n_total)
                    end = n_total
    
    
                for part in ["train","test"]:
            
                    if part == "train":
                        start, stop = 0, split
                    elif part == "test":
                        start, stop = split, end
        
                    for idx1, dirpath in enumerate(proj_dirpaths_shuffled[start:stop]):
                        
                        image = None
                
                        ## load data
                        attributes = pp_utils_lowlevel._load_yaml(os.path.join(dirpath, "attributes.yaml"))
                        annotations = pp_core.export.load_annotation(os.path.join(dirpath, "annotations_" + parameter["project_tag"] + ".json"), verbose=False)
                        filename = attributes["image_original"]["filename"]
                        filepath = attributes["image_phenopype"]["filepath"]
                        image_width, image_height= attributes["image_phenopype"]["width"],  attributes["image_phenopype"]["height"]
    
                        ## potentially not needed, because img-dirs are on the same level as xml dirs
                        image_phenopype_path = os.path.abspath(os.path.join(dirpath, attributes["image_phenopype"]["filepath"]))
                        filepath = os.path.relpath(image_phenopype_path, self.xmldir)
            
                        ## feedback
                        print("Preparing {} data for project {}: {} ({}/{})".format(part, project_name, filename, idx1+1, str(len(proj_dirpaths_shuffled[start:stop]))))       
            
                        ## checks and feedback
                        if annotations.__class__.__name__ == "NoneType":
                            print("No annotations found for {}".format(filename))
                            continue
                        if not annotation_type in annotations:
                            print("No annotation of type {} found for {}".format(annotation_type, filename))
                            continue
                        if annotation_id.__class__.__name__ == "NoneType":
                            annotation_id = max(list(annotations[annotation_type].keys()))
            
                        ## load landmarks
                        data = annotations[annotation_type][annotation_id]["data"][annotation_type]
            
                        ## masking
                        if parameter["mask"]:
                            if pp_settings._mask_type in annotations:
                                pass
                            else:
                                print("No annotation of type {} found for {}".format(pp_settings._mask_type, filename))
                                continue
                        
                            ## select last mask if no id is given
                            if mask_id.__class__.__name__ == "NoneType":
                                mask_id = max(list(annotations[pp_settings._mask_type].keys()))
            
                            ## get bounding rectangle and crop image to mask coords
                            coords = annotations[pp_settings._mask_type][mask_id]["data"][pp_settings._mask_type][0]
                            rx, ry, rw, rh = cv2.boundingRect(np.asarray(coords, dtype="int32"))
                        else:
                            rx, ry, rw, rh = 1, 1, image_width, image_height 
                            
                        ## flipping
                        if parameter["flip"]:
                                                    
                            image = pp_utils.load_image(dirpath)                       
                            image = cv2.flip(image, 1)
                            if not rx == 1:
                                rx = image_width - (rx + rw)
                                
                            parameter["mode"] = "save"
                            
                            data_new = []
                            for coord in data:
                                data_new.append((image_width - coord[0], coord[1]))
                            data = data_new
                        
                        ## saving
                        if parameter["mode"] == "save":
                            if image.__class__.__name__ == "NoneType":
                                image = pp_utils.load_image(dirpath)                       
                            pp_utils.save_image(image, dir_path=self.imagedir, file_name=filename)
                            filepath = os.path.relpath(os.path.join(self.imagedir,filename), self.xmldir)
                            
                        ## xml part
                        if part == "train":
                            train_images_e.append(utils.add_image_element(pp_utils_lowlevel._convert_tup_list_arr(data)[0], (rx, ry, rw, rh), path=filepath))
                        elif part == "test":
                            test_global_images_e.append(utils.add_image_element(pp_utils_lowlevel._convert_tup_list_arr(data)[0], (rx, ry, rw, rh), path=filepath))
                            test_sub_images_e.append(utils.add_image_element(pp_utils_lowlevel._convert_tup_list_arr(data)[0], (rx, ry, rw, rh), path=filepath))
    
                    ## project specific actions after completing loop
                    feedback_dict[project_name][part] = len(proj_dirpaths_shuffled[start:stop])
                    if part == "test":
                        et = ET.ElementTree(test_sub_root)
                        xmlstr = minidom.parseString(ET.tostring(et.getroot())).toprettyxml(indent="   ")
                        with open(os.path.join(self.rootdir,"xml", part + "_" + project_name + "_" + tag + ".xml"), "w") as f:
                            f.write(xmlstr)
    
            ## format final XML output
            for root, part in zip([train_root, test_global_root],["train","test"]):
                et = ET.ElementTree(root)
                xmlstr = minidom.parseString(ET.tostring(et.getroot())).toprettyxml(indent="   ")
                with open(os.path.join(self.rootdir,"xml", part + "_" + tag + ".xml"), "w") as f:
                    f.write(xmlstr)
                    
        ## feedback on training data
        feedback_dict = {}

        #     train_path = os.path.join(self.xmldir, "train_{}.xml".format(tag))
        #     test_path = os.path.join(self.xmldir, "test_{}.xml".format(tag))
        #     n_train_imgs = xml_element_counter(train_path, "image")
        #     n_test_imgs = xml_element_counter(test_path, "image")
        #     print("Datasets set up for \"{}\":".format(tag))
        #     print("Total: {} Training data: {} images".format(n_train_imgs))
        #     print("Test data: {} images".format(n_test_imgs))
        # else:
            
        for project_name, project in self.projects.items():
            
            if len(self.projects) == 1:
                test_tag = tag
            else:
                test_tag = project_name + "_" + tag

            train_path = os.path.join(self.xmldir, "train_{}.xml".format(tag))
            test_path = os.path.join(self.xmldir, "test_{}.xml".format(test_tag))
            
            n_total = len(project.file_names)
            n_train_imgs = utils.xml_element_counter(train_path, "image", project)     
            n_test_imgs = utils.xml_element_counter(test_path, "image")
            
            print("Prepared datasets for \"{}\" from project \"{}\":".format(tag, project_name))
            print("total available: {} images - training: {} images - testing: {} images".format(n_total, n_train_imgs, n_test_imgs))
                
    def create_config(
            self,
            tag,
            configpath,
            overwrite=False,
            ):

        if os.path.isfile(configpath):
            self.configpath = os.path.join(self.configdir, "config_{}.yaml".format(tag))
            if not os.path.isfile(self.configpath):
                shutil.copyfile(configpath, self.configpath)
                super().load_config(self.configpath)
                print("- saved a copy at {}".format(self.configpath))
            else:
                super().load_config(self.configpath, verbose=False)
                print("- found config file at {} - loading (overwrite=False)".format(self.configpath))               
        else:
            print("- {} does not exist!".format(configpath))


    def train_model(self, tag, overwrite=False):

        ## load config to update recent changes
        super().load_config(self.configpath)

        print("- training using the following options:\n")
        config = pp_utils_lowlevel._load_yaml(self.configpath)
        pp_utils_lowlevel._show_yaml(config["train"])
        print(" ")
        
        ## train model
        self.model.train_model(tag, overwrite)

    def test_model(self, tag):
        
        if len(self.projects) == 1:
            print("Testing global predictor performance:")
            self.model.test_model(tag)
        else:
            for project_name in self.projects.keys():
                print("Testing predictor performance on project {}:".format(project_name))
                self.model.test_model(tag=tag, test_tag=project_name + "_" + tag)


#  create function to get error per landmark
