#!/usr/bin/env python3

from typing import List, Union, Tuple

import numpy as np

from sklearn.cluster import DBSCAN
from scipy.interpolate import CubicSpline, splev, splprep

import open3d as o3d


Point = np.ndarray
Cloud = np.ndarray
Plane = np.ndarray
Vector = np.ndarray
NDArray = np.ndarray
PointCloud = o3d.geometry.PointCloud
Vector3V = o3d.utility.Vector3dVector
TriangleMesh = o3d.geometry.TriangleMesh


def plot_points_3d(*points_sets: Tuple[Cloud, str, int], **kwargs) -> None:
    """
    Plots 3D points in a scatter plot.
    Input:
        points: 3D numpy array of shape (N, 3),
            where N is the number of points.
    """
    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    for item in points_sets:
        points = item[0]
        color = "b"
        size = 20

        if len(item) > 1:
            color = item[1]
        if len(item) > 2:
            size = item[2]

        if points is None or len(points) < 1:
            continue

        if len(points) > 1000:
            points = sample_points_uniformly(points, 1000)

        ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=color, s=size)

    if kwargs.get("title"):
        # Add a title to the plot
        plt.title(kwargs.get("title"))

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.show()


def sample_points_uniformly(points: Cloud, num_samples: int) -> Cloud:
    """
    Samples a specified number of points uniformly from an array of points.

    Args:
        points (numpy.ndarray): An array of shape (N, 3)
            representing the points to sample from.
        num_samples (int): The number of points to sample.

    Returns:
        numpy.ndarray: An array of shape (num_samples, 3)
            representing the sampled points.

    Raises:
        ValueError: If `num_samples` is greater than the
            number of points in `points`.

    """
    num_points = len(points)
    if num_samples > num_points:
        raise ValueError(
            "num_samples must be less than or equal to the number of points."
        )

    indices = np.random.choice(num_points, size=num_samples, replace=False)
    return points[indices]


def find_centroid(points: Cloud) -> Union[Point, None]:
    """
    Returns the most central point in a point cloud.

    Args:
        points (NDArray): array of shape (n, 3) representing a point cloud

    Returns:
        NDArray: array of shape (3,) representing
        the point with the most central x,y,z-values
    """
    min_indices = np.argmin(points, axis=0)
    max_indices = np.argmax(points, axis=0)
    centroid = np.mean(points, axis=0)

    for i in range(3):
        if centroid[i] < points[min_indices[i], i]:
            centroid[i] = points[min_indices[i], i]
        elif centroid[i] > points[max_indices[i], i]:
            centroid[i] = points[max_indices[i], i]

    return centroid


def find_global_maxima(points: Cloud) -> Union[Point, None]:
    """
    Returns the point in a point cloud with the highest z-value.

    Args:
        points (NDArray): array of shape (n, 3) representing a point cloud

    Returns:
        NDArray: array of shape (3,) representing
        the point with the highest z-value
    """
    if len(points) == 0:
        return None

    max_index = np.argmax(points[:, 2])
    return points[max_index]


def find_global_minima(points: Cloud) -> Union[Point, None]:
    """
    Returns the point in a point cloud with the lowest z-value.

    Args:
        points (NDArray): array of shape (n, 3) representing a point cloud

    Returns:
        NDArray: array of shape (3,)
        representing the point with the lowest z-value
    """
    if len(points) == 0:
        return None

    min_index = np.argmin(points[:, 2])
    return points[min_index]


def move_towards(points, final_point):
    """
    Moves each point in a 3D point cloud towards a final
    point in a linear manner.

    Args:
        points (ndarray): Array of shape (n, 3) representing the point cloud.
        final_point (ndarray): Array of shape (3,) or (1, 3)
            representing the final point.

    Returns:
        ndarray: Array of shape (n, 3) representing the moved point cloud.
    """
    t = np.linspace(2, 0, len(points))
    interp_factor = (1 - t) ** 2
    interp_points = (
        #
        interp_factor[:, np.newaxis] * points
        + t[:, np.newaxis] * np.squeeze(np.atleast_1d(final_point))
    )

    return interp_points


def scale_points_from_centroid(points: Cloud, scale_factor: float) -> Cloud:
    """
    Scale an array of <n, 3> points from the centroid by a given scale factor.

    Args:
        points: A numpy array of shape <n, 3> representing the
            points to be scaled.
        scale_factor: A float representing the scale factor to be
            applied to the points.

    Returns:
        A numpy array of shape <n, 3> representing the new
            positions of the scaled points.
    """
    # compute the centroid
    centroid = np.mean(points, axis=0)

    # subtract the centroid to get the vectors
    vectors = points - centroid

    # scale the vectors
    scaled_vectors = vectors * scale_factor

    # add the scaled vectors back to the centroid to get the new points
    new_points = centroid + scaled_vectors

    return new_points


def interpolate(factor: float, point_a: Point, point_b: Point) -> Point:
    # Interpolate along each axis separately
    interp_a_x = np.interp(factor, [0, 1.0], [point_a[0], point_b[0]])
    interp_a_y = np.interp(factor, [0, 1.0], [point_a[1], point_b[1]])
    interp_a_z = np.interp(factor, [0, 1.0], [point_a[2], point_b[2]])

    # Combine the interpolated values into a new point
    return np.array([interp_a_x, interp_a_y, interp_a_z])


def get_interpolate(vals: np.ndarray) -> np.ndarray:
    # How far we go below the bottom (0 = at the bottom)
    at_bottom = 2

    # How much we retrace the model (1 = at the model)
    at_top = -2

    interp_vals = np.linspace(at_bottom, at_top, num=len(vals))

    return np.clip(interp_vals, -1, 1)


def make_2d(array, num_cols=3):
    return np.tile(array[:, np.newaxis], (1, num_cols))


def weight_arrays(a: np.array, b: np.array) -> np.array:
    weight = make_2d(get_interpolate(a))
    return weight * a + (1 - weight) * b


def merge_point_clouds(point_clouds: List[Cloud]) -> Cloud:
    """
    Merges a list of point clouds into a single point cloud.

    Args:
    point_clouds (List[np.ndarray[np.ndarray]]): A list of point clouds,
        where each point cloud is an array of points

    Returns:
    np.ndarray[np.ndarray]: A single point cloud array containing all
        the points from the input point clouds
    """

    # Concatenate all the point clouds into a single array
    concatenated_cloud = np.concatenate(point_clouds, axis=0)

    # Remove duplicate points
    unique_points, _ = np.unique(concatenated_cloud, axis=0, return_index=True)

    # Return the unique points array
    return unique_points


def average_points(point_clouds: List[Cloud]) -> List[Cloud]:
    """
    Averages each point of each slice between the two or
        three closest points in that slice.

    Args:
    point_clouds (List[np.ndarray[np.ndarray]]): A list of point clouds,
        where each point cloud is an array of points

    Returns:
    List[np.ndarray[np.ndarray]]: A new list of point clouds,
        where each point in each slice is averaged with its two
        or three closest points in that slice
    """

    result = []

    for cloud in point_clouds:
        # Sort the points by their z coordinate
        sorted_points = cloud[cloud[:, 2].argsort()]
        num_points = len(sorted_points)

        # Loop over each slice (i.e., each unique z coordinate)
        for i in range(num_points):
            z = sorted_points[i][2]
            slice_points = sorted_points[sorted_points[:, 2] == z]
            num_slice_points = len(slice_points)

            # Loop over each point in the slice
            for j in range(num_slice_points):
                point = slice_points[j]

                # Find the two or three closest points in the slice
                distances = np.linalg.norm(slice_points - point, axis=1)
                sorted_distances_indices = distances.argsort()
                nearest_indices = sorted_distances_indices[
                    1:3
                ]  # Choose the second and third closest points
                nearest_points = slice_points[nearest_indices]

                # Calculate the average of the point and its nearest neighbors
                average_point = np.mean(
                    np.vstack([point, nearest_points]), axis=0
                )

                # Replace the point with the average point
                slice_points[j] = average_point

            # Replace the slice in the sorted points array
            #   with the updated slice
            sorted_points[sorted_points[:, 2] == z] = slice_points

        # Append the updated point cloud to the result list
        result.append(sorted_points)

    return result


def remove_lonely_points(points, threshold):
    """
    Removes points from an array that are more than a
    given distance from any other point in the array.

    Parameters:
    points (ndarray): A 2D NumPy array of shape (N, D)
        representing N points in D dimensions.
    threshold (float): The minimum distance between points
        required for a point to be considered "close" to another point.

    Returns:
    ndarray: A 2D NumPy array of shape (M, D)
        representing the remaining M points in D dimensions.
    """
    from sklearn.neighbors import KDTree

    tree = KDTree(points)
    neighbors = tree.query_radius(points, r=threshold)
    is_lonely = np.array([len(n) <= 1 for n in neighbors])
    return points[~is_lonely]


def remove_lonely_clusters(points: Cloud, threshold: float = 0.1) -> Cloud:
    """
    Removes clusters that are not connected to the main body of points.

    Parameters:
    points (ndarray): A 2D NumPy array of shape (N, D) representing
        N points in D dimensions.
    threshold (float): The minimum distance between points required for
        a point to be considered "close" to another point.

    Returns:
    ndarray: A 2D NumPy array of shape (M, D) representing the
        remaining M points in D dimensions.
    """
    # Perform clustering using DBSCAN
    clustering = DBSCAN(eps=threshold, min_samples=2).fit(points)
    labels = clustering.labels_

    # Count the number of points in each cluster
    unique_labels, counts = np.unique(labels, return_counts=True)

    # Find the label of the largest cluster (ignoring noise points with label
    # -1)
    largest_cluster_label = unique_labels[np.argmax(counts)]

    # Keep only the points that belong to the largest cluster
    return points[labels == largest_cluster_label]


def kernel_smoothing_z(
    points: np.ndarray, bandwidth: float = 0.1
) -> np.ndarray:
    """
    Applies kernel smoothing to the z-coordinates of a set of points.

    Args:
        points: A numpy array of shape (n, 3) representing the points.
        bandwidth: The bandwidth parameter of the kernel function.

    Returns:
        A numpy array of shape (n, 3)
        representing the smoothed points with only the z-coordinates changed.
    """
    if len(points) < 1:
        return points

    # Separate x, y, and z coordinates
    x, y, z = points.T

    # Define a Gaussian kernel function
    def kernel(d):
        return np.exp(-0.5 * d ** 2)

    # Smooth the z-coordinates of the points using kernel smoothing
    z_smoothed = []
    for i in range(len(points)):
        dist = np.sqrt((x - x[i]) ** 2 + (y - y[i]) ** 2)
        w = kernel(dist / bandwidth)
        z_smoothed.append(np.sum(w * z) / np.sum(w))
    z_smoothed = np.array(z_smoothed)

    # Return smoothed points with only the z-coordinates changed
    return np.column_stack([x, y, z_smoothed])


def translate(pcd: o3d.geometry.PointCloud, translation: np.ndarray) -> None:
    """
    Transforms the position of a point cloud by a given translation vector.

    Args:
        pcd (open3d.geometry.PointCloud): The point cloud to be transformed.
        translation (numpy.ndarray): The 3D translation vector.

    Returns:
        open3d.geometry.PointCloud: The transformed point cloud.
    """
    pcd.points = Vector3V(np.asarray(pcd.points) + translation)


def translate_all(
    pcds: List[o3d.geometry.PointCloud], translation: np.ndarray
) -> None:
    for maybe_pcd in pcds:
        if isinstance(maybe_pcd, o3d.geometry.PointCloud):
            translate(maybe_pcd, translation)
        elif isinstance(maybe_pcd, np.ndarray):
            maybe_pcd += translation


def cubic_curve(points: Cloud, num_samples: int = 500) -> Cloud:
    """
    Interpolates a cloud of 3D points with a cubic spline curve.

    Args:
        points: A numpy array of shape (n, 3) representing the cloud of points.
        num_samples: The number of points in the interpolated curve.

    Returns:
        A numpy array of shape (num_samples, 3)
        representing the interpolated curve.
    """
    if len(points) < 1:
        return points

    # Smooth the z-coordinates of the points
    points = kernel_smoothing_z(points, 0.2)

    # Interpolate a curve through the points using cubic spline interpolation
    indices = np.arange(0, len(points))

    x_coef = CubicSpline(indices, points[:, 0])
    y_coef = CubicSpline(indices, points[:, 1])
    z_coef = CubicSpline(indices, points[:, 2])

    samples = np.linspace(0, len(points), num=num_samples)
    x_interp = x_coef(samples)
    y_interp = y_coef(samples)
    z_interp = z_coef(samples)

    # Stack the x, y, and z coordinates into a (num_samples, 3) array
    return np.column_stack([x_interp, y_interp, z_interp])


def smoothed(
    points: Cloud, num_samples: int = 500, smoothing: int = 0
) -> Cloud:
    """
    Interpolates a cloud of 3D points with a spline curve.

    Args:
        points: A numpy array of shape (n, 3) representing the cloud of points.
        num_samples: The number of points in the interpolated curve.
        smoothing: The smoothing factor to use for spline interpolation.

    Returns:
        A numpy array of shape (num_samples, 3)
            representing the interpolated curve.
    """
    if len(points) < 10:
        return points

    # Transpose the input array for use with the spline interpolation function
    data = points.T

    # Use the spline interpolation function to interpolate the curve
    tck, _ = splprep(data, s=smoothing, per=True)

    # Evaluate the spline at evenly spaced points along the curve
    t = np.linspace(0, 1, num_samples)
    new = np.array(splev(t, tck)).T

    # Return the interpolated curve
    return new


def remove_radial_outliers(
    points: Cloud, idx_offset: int = 3, z_offset: float = 0.4
) -> Cloud:
    """
    Removes radial outliers from a point cloud.

    Args:
        points (np.ndarray): The point cloud as a N x 3
            array of (x, y, z) coordinates.
        offset (int): The offset to use for computing neighbors. Default is 3.

    Returns:
        np.ndarray: The filtered point cloud with radial outliers removed.
    """
    prev = np.roll(points[:, 2], +idx_offset)
    next = np.roll(points[:, 2], -idx_offset)

    outliers = points[:, 2] > np.maximum(prev, next) + z_offset
    filtered_points = points[~outliers]

    return filtered_points


