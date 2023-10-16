from pathlib import Path

#TODO: NAMING OF THESE FUNCTIONS NEEDS REVIEW
#TODO: SHOULD AGGREGATE COMMON TEMPLATES

def create_patches_dir(study_name: str, magnification: float, patch_size: int) -> Path:
    patches_dir = Path("patches") / study_name
    patches_subdir = f"mag_{magnification}-size_{patch_size}"
    return patches_dir / patches_subdir

def create_patches_meta_path(study_name: str, magnification: float, patch_size: int) -> Path:
    patches_meta_dir = Path("meta") / study_name
    patches_meta_file = f"patches_meta-mag_{magnification}-size_{patch_size}.pickle"
    return patches_meta_dir / patches_meta_file

def create_slide_meta_dir(study_name: str, magnification: float, patch_size: int) -> Path:
    slide_meta_dir = Path("meta") / study_name / f"mag_{magnification}-size_{patch_size}"
    return slide_meta_dir

def create_slide_meta_path(study_name: str, magnification: float, patch_size: int, slide_name: str) -> Path:
    slide_meta_dir = create_slide_meta_dir(study_name, magnification, patch_size)
    slide_meta_file = f"{slide_name}.pickle"
    return slide_meta_dir / slide_meta_file

def create_features_dir(study_name: str, magnification: float, patch_size: int, num_layers: int) -> Path:
    features_dir = Path("features") / study_name / f"mag_{magnification}-size_{patch_size}" / f"resnet_{num_layers}"
    return features_dir

def create_data_dir(study_name: str, magnification: float, patch_size: int, num_layers: int, slide_name: str) -> Path:
    data_dir = Path("data") / study_name / slide_name / f"mag_{magnification}-size_{patch_size}" / f"resnet_{num_layers}"
    return data_dir
