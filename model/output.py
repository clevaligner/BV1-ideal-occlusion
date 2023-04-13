#!/usr/bin/env python3

import os
import json
from typing import List, Union
import numpy as np
from glob import glob

import open3d as o3d
from loguru import logger

from .helpers import (
    Cloud,
    Vector,
    PointCloud,
    TriangleMesh,
)
from .config import OUTPUT_DIR


# Helper Functions


def is_upper_jaw(position: str):
    return position.lower() in ["haut", "upper"]


def is_lower_jaw(position: str):
    return position.lower() in ["bas", "lower"]


# Directory Functions


def upper_or_lower_dir(position: str) -> str:
    if is_upper_jaw(position):
        return "/upper"
    elif is_lower_jaw(position):
        return "/lower"


def get_client_name(client: str) -> str:
    if "/output" in client:
        return "root_estimation"

    return os.path.basename(os.path.dirname(client))


def get_client_dir(client: str, directory: str = OUTPUT_DIR, **kwargs) -> str:
    if directory != OUTPUT_DIR:
        return directory

    return f"{directory}/{get_client_name(client)}"


def ensure_dir_exists(dir_name: str) -> None:
    os.makedirs(dir_name, exist_ok=True)


# JSON Processing Functions


def dump_json(file_name: str, to_output: Union[List, dict]) -> None:
    with open(file_name, "w") as f:
        f.write(json.dumps(to_output, indent=4, sort_keys=False))


def read_and_combine_json_files(directory: str) -> List:
    combined_data = []
    input_files = glob(f"{directory}/*.json")

    for file_name in input_files:
        with open(file_name, "r") as json_file:
            combined_data.append(json.load(json_file))

    return combined_data


def save_json_matrix(client_dir: str, tooth_num: int, pcd: PointCloud) -> None:
    client_dir = f"{client_dir}/matrices"
    ensure_dir_exists(client_dir)

    # Get the transformation matrix
    R = np.asarray(pcd.get_rotation_matrix_from_xyz([0, 0, 0]))
    t = np.asarray(pcd.get_center())

    # Create the transformation matrix
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t

    file_name = f"{client_dir}/{tooth_num}.json"
    logger.info(f"Saving JSON matrices to {file_name}")

    dump_json(file_name, T.tolist())


def save_json_vector(client_dir: str, tooth_num: int, vector: Vector) -> None:
    client_dir = f"{client_dir}/orientation"
    ensure_dir_exists(client_dir)

    file_name = f"{client_dir}/{tooth_num}.json"
    logger.info(f"Saving JSON vector to {file_name}")

    dump_json(file_name, vector.tolist())


def points_to_xyz(points: Cloud) -> list:
    return [{"x": x, "y": y, "z": z} for x, y, z in points.tolist()]


def save_json_curve(
    client_dir: str, tooth_num: int, position: str, base: Cloud
) -> None:
    client_dir = f"{client_dir}/curves"
    client_dir += upper_or_lower_dir(position)
    ensure_dir_exists(client_dir)

    file_name = f"{client_dir}/tooth_bound_json_{tooth_num}.json"
    logger.info(f"Saving json curve to {file_name}")

    dump_json(file_name, points_to_xyz(base))


def combine_json_curves(client_dir: str, position: str) -> None:
    client_dir = f"{client_dir}/curves"
    client_dir += upper_or_lower_dir(position)
    logger.info(f"Combining raw ndarray curves to JSON in {client_dir}")

    input_file_names = glob(f"{client_dir}/*.dat")
    curves = []
    for file_name in input_file_names:
        curves.append(np.loadtxt(file_name))

    # Now iterate over our merged curves and
    # output them to json. We convert to x,y,z dictionaries
    #
    for i, file_name in enumerate(input_file_names):
        # Split the path into the directory path and the file name
        dir_path, file_name = os.path.split(file_name)

        # Split the file name into the name and the extension
        name, _ = os.path.splitext(file_name)

        # NOTE: for some reason I have been told that JSON
        #       must be in the filename for the web to read it?
        name = name.replace("bound_", "bound_json_")

        # Find our new filename (with extension)
        full_path = os.path.join(dir_path, f"{name}.json")

        dump_json(full_path, points_to_xyz(curves[i]))


def combine_json_vectors(client_dir: str, position: str) -> None:

    client_dir = f"{client_dir}/orientation"
    input_files = glob(f"{client_dir}/*.json")

    combined_data = read_and_combine_json_files(client_dir)

    for file_name in input_files:
        if "orientation" in file_name:
            continue

        os.remove(file_name)

    if is_upper_jaw(position):
        combined_file = f"{client_dir}/orientation_upper.json"
    elif is_lower_jaw(position):
        combined_file = f"{client_dir}/orientation_lower.json"

    dump_json(combined_file, combined_data)


def combine_json_matrix(client_dir: str, position: str) -> None:
    client_dir = f"{client_dir}/matrices"
    input_files = glob(f"{client_dir}/*.json")

    combined_data = read_and_combine_json_files(client_dir)

    for file_name in input_files:
        if "matrices" in file_name:
            continue

        os.remove(file_name)

    if is_upper_jaw(position):
        combined_file = f"{client_dir}/upper_transformation_matrices.json"
    elif is_lower_jaw(position):
        combined_file = f"{client_dir}/lower_transformation_matrices.json"

    dump_json(combined_file, combined_data)


def save_mesh_to_file(
    dir_name: str, file_name: str, mesh: TriangleMesh
) -> None:
    """
    Output the mesh to a specified file
    """
    ensure_dir_exists(dir_name)

    mesh.compute_vertex_normals()

    o3d.io.write_triangle_mesh(f"{file_name}.stl", mesh)
