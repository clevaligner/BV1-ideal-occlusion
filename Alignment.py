# -*- coding: utf-8 -*-
"""
Created on Wed Dec 21 15:32:38 2022

@author: jonat
"""

import numpy as np
import open3d as o3d
import math as m
import json
import os
import click
from sklearn.neighbors import NearestNeighbors

import model.output as output
from model.config import OUTPUT_DIR, DIRECTORY_CLIENT
from loguru import logger


def plane_through_3p(dict_three_points):
    p1 = dict_three_points['posterior1']
    p2 = dict_three_points['incisal_edges_mid']
    p3 = dict_three_points['posterior2']

    v1 = p3 - p1
    v2 = p2 - p1
    normal = np.cross(v1, v2)
    a, b, c = normal
    d = - np.dot(normal, p3)
    # d2 = - (a*p3[0] + b*p3[1] + c*p3[2])
    plane = np.array([a, b, c, d])

    return plane

def create_plane_points(points, plane):
    a, b, c, d = plane

    x_plane = np.linspace(np.min(points[:, 0]) * 1.8, np.max(points[:, 0]) * 1.8, 100)
    y_plane = np.linspace(np.min(points[:, 1]) * 1.8, np.max(points[:, 1]) * 1.8, 100)
    X, Y = np.meshgrid(x_plane, y_plane)
    Z = (d - a * X - b * Y) / c
    plane_points = np.hstack((X.flatten().reshape(-1, 1), Y.flatten().reshape(-1, 1), Z.flatten().reshape(-1, 1)))

    return plane_points

def occlusal_plane_V2(left_molars_avg,incisal_edges_mid,right_molars_avg):
    """create occlusl plane from mid edge of incisors, average of molar cusps in left side and in right side"""

    # surfaces_dict, cusps_dict = find_cusps(points, to_plot_cusps_2d, is_lower)
    # three_points = three_control_points_V2(points, surfaces_dict, cusps_dict, to_plot_three_point, is_lower)
    # three_points = dict({""}:)
    three_points = {'posterior1':left_molars_avg,'incisal_edges_mid':incisal_edges_mid,'posterior2':right_molars_avg}
    plane = plane_through_3p(three_points)

    return plane, three_points

def norm(x):
    """
    the magnitude / size / distance from tail to head, of the vector x.
    Args:
        x: a vector (x,y,z,...)

    Returns:
        the magnitude of the vector x.
    """
    return np.sqrt(dot_product(x, x))


def normalize(x):
    """
    get the unit vector of the vector x. vector dividing by it's magnitude.
    Args:
        x: a vector (x,y,z,...)

    Returns:
        the unit vector of the vector x.
    """
    return [x[i] / norm(x) for i in range(len(x))]

def move_occlusal_plane_to_three_points(plane_points, plane, three_points, epsilon=0.0001):
    neigh = NearestNeighbors(n_neighbors=1)
    neigh.fit(plane_points)
    distance, index = neigh.kneighbors(list(three_points.values())[0].reshape(1, -1), return_distance=True)
    closest_point = (plane_points[index]).reshape(-1,)
    normal = np.array(normalize(plane[:3]))
    x, y, z = list(three_points.values())[0]
    if (closest_point[2] > z):  # above
        plane_points = plane_points + distance * -normal
    elif (closest_point[2] < z):  # below
        plane_points = plane_points + distance * normal

    return plane_points

def occ_plane(low_pcd, left_molars_avg,incisal_edges_mid,right_molars_avg):
    """create occlusl plane from mid-edge of incisors, average of molar cusps in left side and in right side -
    of the lower jaw"""

    plane, three_points = occlusal_plane_V2(left_molars_avg,incisal_edges_mid,right_molars_avg)
    plane_points = create_plane_points(low_pcd, plane)
    plane_points = move_occlusal_plane_to_three_points(plane_points, plane, three_points)

    return plane, plane_points

def dot_product(x, y):
    """
    the dot product is the sum of the products of the corresponding entries of the two sequences of numbers /
    euclidean distance between vector x and vector y.
    Args:
        x: the first vector (x,y,z,...).
        y: the second vector (x,y,z,...).

    Returns:
        a single number, the euclidean distance between vector x and vector y.
    """
    return sum([x[i] * y[i] for i in range(len(x))])

def pcd_o3d(array,rgb):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(array)
    pcd.paint_uniform_color(rgb)
    pcd.estimate_normals()
    return pcd


def rotation_matrix(alpha,beta,gamma):
    def Rx(alpha):
        return np.matrix([[ 1, 0, 0],
                          [ 0, m.cos(alpha),-m.sin(alpha)],
                          [0, m.sin(alpha), m.cos(alpha)]])
    def Ry(beta):
        return np.matrix([[m.cos(beta), 0, m.sin(beta)],
                          [0, 1, 0],
                          [-m.sin(beta), 0, m.cos(beta)]])
    def Rz(gamma):
        return np.matrix([[m.cos(gamma), -m.sin(gamma), 0],
                          [m.sin(gamma), m.cos(gamma) ,0],
                          [0, 0, 1]])
    transf = Rz(np.radians(gamma)).dot(Ry(np.radians(beta))).dot(Rx(np.radians(alpha)))
    return transf

def arr_to_json(arr, file_name):
    """
    get ndarray, transform it to list,
    create a JSON object from it,
    and save it as JSON file in the same directory of the python file.
    Args:
        arr: ndarray. that we want to same as JSON file.
        file_name: string of the name of the file you want to save.
    Returns:
        -
    """
    arr_tolist = arr.tolist()
    json_object = json.dumps(arr_tolist, indent=4)
    file_name = file_name + '.json'
    with open(file_name, "w") as outfile:
        outfile.write(json_object)


def alignment(client: str, kwargs):
    """
    This function takes a client name and a set of keyword arguments
    and returns a tuple containing the target point cloud, the merged
    point cloud, the orientation vector of the tooth, and the cutting
    plane of the crown (if applicable).

    Args:
        client (str): The name of the client.
        **kwargs: Keyword arguments specifying
            various options for the function.

    Returns:
        Tuple[PointCloud, Vector, Plane]: A tuple containing the target
        point cloud, the merged point cloud, the orientation vector of
        the tooth, and the cutting plane of the crown (if applicable).
    """

    if kwargs["position"] == "upper":
        haut_bas = "haut"
    else:
        haut_bas = "bas"

    print("===========================================================================")
    print("========================= IMPORTING FILES =================================")
    print("===========================================================================")

    directory1 = os.path.dirname(os.path.realpath(__file__)) + '/inputs/423/'
    directory2 = directory1 + f"dispatching_{haut_bas}/"
    directory3 = directory1 + f"filing_teeth_{haut_bas}/"

    txt_filename = f"seg_output_{haut_bas}.txt"
    # txt_filename = "full_teeth_array.txt"
    # dental = np.loadtxt(directory3 + txt_filename)
    dental = np.loadtxt(directory1 + txt_filename)
    dental_for_plan = dental[dental[:, 3] < 16]


    # each_tooth_centroid = np.loadtxt(directory2+"each_tooth_centroid.txt")
    plane_blue = np.loadtxt(directory2 + "plane_green.txt")
    plane_green = np.loadtxt(directory2 + "plane_blue.txt")

    # plane_blue_update_size = np.loadtxt(directory2 + "plane_green_center_mass.txt")
    plane_concat = np.concatenate((plane_blue,plane_green),axis=0)

    # points_relavant = np.loadtxt(directory2 + "relevant_green.txt")
    plane_blue_pcd = pcd_o3d(plane_blue[:,:-1],rgb=[0,1,0])
    plane_green_pcd = pcd_o3d(plane_green[:,:-1],rgb=[1,0,0])

    dental_pcd = pcd_o3d(dental_for_plan[:,:-1],rgb=[1,1,0])

    # o3d.visualization.draw_geometries([plane_blue_pcd,plane_green_pcd])



    # plane_blue_pcd = plane_blue_pcd.voxel_down_sample(voxel_size=2)
    # relevant_blue_pcd = pcd_o3d(points_relavant,rgb=[1,0,0])
    # o3d.visualization.draw_geometries([relevant_blue_pcd])
    # o3d.visualization.draw_geometries([plane_blue_pcd,plane_green_pcd])

    # fc = open(directory1 +'export_selection_13.9_42.json')#bon
    fc = open(directory1 + 'export_20.18_17.json')
    data_all = json.load(fc)
    try:
        datac = data_all['dentArray']
    except:
        datac = data_all


    new_teeth = []
    plane_blue_update = []
    plane_green_update = []
    transformation_matrix = []
    transformation_matrix_m = []

    print("===========================================================================")
    print("========================= APPLYING 2D MOVEMENTS ===========================")
    print("===========================================================================")

    for nl in range(1,15):

        if haut_bas == "haut":
            offsetc = [i for i in range(15,0,-1)] #upper
            offsetc.append(0) #upper
            data = datac[offsetc[int(nl)]]
        else :
            offsetc=16 # lower
            data = datac[int(nl)+offsetc]

        plane_blue_tooth = plane_blue[plane_blue[:,-1] == int(nl)][:,:-1]
        plane_green_tooth = plane_green[plane_green[:,-1] == int(nl)][:,:-1]


        tooth  = dental[dental[:,-1] == int(nl)][:,:-1]

        center_free_edge = np.array([data["ORIGIN"]["POSITION"]["X"],data["ORIGIN"]["POSITION"]["Y"],0])

        plane_blue_tooth = plane_blue_tooth - center_free_edge
        plane_green_tooth = plane_green_tooth - center_free_edge


        tooth = tooth - center_free_edge
        T1 = np.identity(4)
        T1[:-1,-1] = - center_free_edge


        rot_yehiel = -data["MODIFIED"]["ANGLE"] + data["ORIGIN"]["ANGLE"]
        rot_matrix = rotation_matrix(0,0,rot_yehiel)
        tooth = tooth.dot(rot_matrix)
        T2 = np.identity(4)
        T2[:-1,:-1] = rot_matrix.T


        plane_blue_tooth = plane_blue_tooth.dot(rotation_matrix(0,0,rot_yehiel))
        plane_blue_tooth = np.array(plane_blue_tooth)

        plane_green_tooth = plane_green_tooth.dot(rotation_matrix(0,0,rot_yehiel))
        plane_green_tooth = np.array(plane_green_tooth)

        tooth = np.array(tooth)
        plane_blue_tooth = plane_blue_tooth + np.array([data["MODIFIED"]["POSITION"]["X"],data["MODIFIED"]["POSITION"]["Y"],0])
        plane_green_tooth = plane_green_tooth + np.array([data["MODIFIED"]["POSITION"]["X"],data["MODIFIED"]["POSITION"]["Y"],0])

        tooth = tooth + np.array([data["MODIFIED"]["POSITION"]["X"],data["MODIFIED"]["POSITION"]["Y"],0])
        T3 = np.identity(4)
        T3[:-1,-1] = np.array([data["MODIFIED"]["POSITION"]["X"],data["MODIFIED"]["POSITION"]["Y"],0])

        plane_blue_update.append(plane_blue_tooth)
        plane_green_update.append(plane_green_tooth)



        T4 = np.identity(4)
        rot_matrix_180 = rotation_matrix(0,180,0)
        T4[:-1,:-1] = rot_matrix_180.T



        if haut_bas == "haut" :
            transfo_m = T4@T3@T2@T1
            # tooth = tooth.dot(rot_matrix_180)
            # tooth = np.array(tooth)
        else :
            transfo_m = T3@T2@T1

        new_teeth.append(tooth)
        transformation_matrix_m.append(transfo_m)




    ##################### OCCLUSION PLANE ###########################################################

    # left_molars_avg = dental[dental[:,-1] == int(1)][:,:-1].mean(axis=0)
    # incisal_edges_mid = dental[dental[:,-1] == int(7)][:,:-1].mean(axis=0)
    # right_molars_avg = dental[dental[:,-1] == int(14)][:,:-1].mean(axis=0)

    # occ_plane, occ_plane_points = occ_plane(dental_for_plan, left_molars_avg,incisal_edges_mid,right_molars_avg)

    # pcd_plane = pcd_o3d(occ_plane_points,rgb=[0,0,1])
    # o3d.visualization.draw_geometries([dental_pcd, pcd_plane])







    plane_blue_pcd2 = pcd_o3d(np.concatenate(plane_blue_update),rgb=[1,0,0])
    plane_green_pcd2 = pcd_o3d(np.concatenate(plane_green_update),rgb=[0,1,0])

    # np.savetxt("comparaison_ortal/" + f"{haut_bas}_updated_after_yehiel.txt",np.concatenate(new_teeth))

    teeth_pcd = pcd_o3d(np.concatenate(new_teeth),rgb=[0,0,1])
    # o3d.visualization.draw_geometries([dental_pcd, plane_blue_pcd])
    # o3d.visualization.draw_geometries([plane_blue_pcd,plane_blue_pcd2])
    # o3d.visualization.draw_geometries([plane_blue_pcd2,teeth_pcd])
    # o3d.visualization.draw_geometries([plane_blue_pcd2,plane_green_pcd2])


    transformation_matrix_array = np.array(transformation_matrix_m)

    print("===========================================================================")
    print("========================= SAVING THE TRANSFORMATION MATRIX ================")
    print("===========================================================================")
    arr_to_json(transformation_matrix_array, "outputs/"+ f"transfo_test_{haut_bas}")


@click.command()
@click.option(
    "--position",
    default=None,
    type=click.Choice(["upper", "haut", "lower", "bas", None]),
    help="Is this the upper or lower jaw",
)
@click.option(
    "--directory",
    default=OUTPUT_DIR,
    help="Which directory to save models",
    type=str,
)
@click.argument(
    "client",
    type=str,
    nargs=1,
    default=os.path.join(
        DIRECTORY_CLIENT, "patient-4-17/seg_output_bas_new.txt"
    ),
)
def main(client: str, **kwargs):
    """
    Point cloud alignment and calculating the transformation matrix for the alignment
    """
    # Automatically detect if this is an upper or lower
    # client jaw file. Easier cli processing
    position = kwargs.get("position")
    if position is None:
        if "haut" in client:
            kwargs["position"] = "upper"
        elif "bas" in client:
            kwargs["position"] = "lower"

    try:
        alignment(client, kwargs)

    except Exception as e:
        logger.catch(e)
        raise e


if __name__ == "__main__":
    main()