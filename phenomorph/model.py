from phenomorph import utils
from phenopype.utils_lowlevel import _load_yaml
from phenopype import show_image, load_image
import dlib
from cv2 import circle
import os


class Model:
    def __init__(self, rootdir):
        self.rootdir = os.path.abspath(rootdir)
        self.imagedir = os.path.join(self.rootdir, "images")
        assert os.path.exists(self.imagedir), print(
            f"No image directory found at {self.imagedir}"
        )
        self.modeldir = os.path.join(self.rootdir, "models")
        self.xmldir = os.path.join(self.rootdir, "xml")
        self.options = None
        print(f"Initialized ml-morph at {self.rootdir}")

    def preprocess_folder(self, tag, overwrite=False, percentage=0.8):
        if not os.path.exists(self.xmldir):
            os.makedirs(self.xmldir)

        cvsFile = os.path.join(self.rootdir, f"landmarks_ml-morph_{tag}.csv")
        dict_csv = utils.read_csv(cvsFile)
        train_set, test_set = utils.split_train_test(dict_csv, percentage)
        train_xml = os.path.join(self.xmldir, f"train_{tag}.xml")
        test_xml = os.path.join(self.xmldir, f"test_{tag}.xml")

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

    def load_config(self, cfgpath):
        cfg = _load_yaml(cfgpath)
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
        return print(f"Loaded ml-morph config file: {cfgpath}")

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

    def predict_image(self, tag, img_path):
        predictor_path = os.path.join(self.modeldir, f"predictor_{tag}.dat")
        assert os.path.exists(
            predictor_path
        ), f"Cannot find shape prediction model at {predictor_path}"
        assert os.path.exists(img_path), "No image found at {image_path}"
        img = load_image(img_path)
        rect = dlib.rectangle(
            left=1, top=1, right=img.shape[1] - 1, bottom=img.shape[0] - 1
        )
        predictor = dlib.shape_predictor(predictor_path)
        shape = predictor(img, rect)
        num_parts = range(0, shape.num_parts)
        for item, i in enumerate(sorted(num_parts, key=str)):
            x, y = shape.part(item).x, shape.part(item).y
            circle(img, (x, y), int(img.shape[0] / 50), (0, 0, 255), -1)
        show_image(img)


#  create function to get error per landmark
