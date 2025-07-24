import scipy
import numpy as np
import open3d as o3d

from concave_hull import concave_hull
from shapely.geometry import Polygon
from shapely.vectorized import contains
from sklearn.neighbors import NearestNeighbors

from internutopia.core.task.metric import BaseMetric

from ..config.task_config import ManipulationSuccessMetricCfg, ManipulationTaskCfg


XY_DISTANCE_CLOSE_THRESHOLD = 0.15
MAX_TO_BE_TOUCHING_DISTANCE = 0.1
MIN_ABOVE_BELOW_DISTANCE = 0.05
MAX_TO_BE_SUPPORTING_AREA_RATIO = 1.5
MIN_TO_BE_SUPPORTED_AREA_RATIO = 0.7
MIN_TO_BE_ABOVE_BELOW_AREA_RATIO = 0.1
INSIDE_PROPORTION_THRESH = 0.5
ANGLE_THRESHOLD = 45


@BaseMetric.register('ManipulationSuccessMetric')
class ManipulationSuccessMetric(BaseMetric):
    """
    Calculate the success of this episode
    """

    def __init__(self, config: ManipulationSuccessMetricCfg, task_config: ManipulationTaskCfg):
        super().__init__(config, task_config)
        
        self.step = 0
        self.episode_sr = 0
        self.first_success_step = -1
        self.max_vertices_num = 10000

        self.object_prim_paths = set()
        for goal in self.task_config.target:
            for subgoal in goal:
                if "another_obj2_uid" in subgoal:
                    self.object_prim_paths.add(subgoal["another_obj2_uid"])
                self.object_prim_paths.add(subgoal["obj1_uid"])
                self.object_prim_paths.add(subgoal["obj2_uid"])

    def reset(self):
        self.step = 0
        self.episode_sr = 0
        self.first_success_step = -1
    
    def recursive_get_mesh(self, prim, coord_prim) -> np.ndarray:
        vertices = []

        try:
            from omni.isaac.core.utils.mesh import get_mesh_vertices_relative_to

            vertices.append(get_mesh_vertices_relative_to(prim, coord_prim))
        except Exception as e:
            pass

        children = prim.GetChildren()
        for child in children:
            child_vertices = self.recursive_get_mesh(child, coord_prim)
            if child_vertices is not None:
                vertices.append(child_vertices)

        if len(vertices) > 0:
            return np.concatenate(vertices, axis=0)
        else:
            return None

    def calc_episode_sr(self):
        """
        This function is called at each world step.
        """
        from omni.isaac.core.utils.stage import get_current_stage

        stage = get_current_stage()
        coord_prim = stage.GetPrimAtPath(f"/World/env_{self.env_id}/scene")
        
        point_cloud_list = {}
        for uid in self.object_prim_paths:
            mesh_prim = stage.GetPrimAtPath(f"/World/env_{self.env_id}/scene/obj_{uid}")
            vertices = self.recursive_get_mesh(mesh_prim, coord_prim)
            if len(vertices) > self.max_vertices_num:
                step = len(vertices) // self.max_vertices_num + 1
                vertices = vertices[::step]
            point_cloud_list[uid] = vertices

        try:
            self.episode_sr = check_finished(self.task_config.target, point_cloud_list)
        except Exception as e:
            self.episode_sr = 0

        if self.episode_sr==1 and self.first_success_step < 0:
            self.first_success_step = self.step

        return {
            'task_name': self.task_config.task_name,
            'episode_name': self.task_config.episode_name,
            'episode_sr': self.episode_sr,
            'first_success_step': self.first_success_step,
            'episode_step': self.step,
        }

    def update(self, task_obs: dict):
        self.step += 1

    def get_episode_sr(self):
        return self.episode_sr

    def calc(self):
        """
        This function is called to calculate the metrics when the episode is terminated.
        """
        return {
            'task_name': self.task_config.task_name,
            'episode_name': self.task_config.episode_name,
            'episode_sr': self.episode_sr,
            'first_success_step': self.first_success_step,
            'episode_step': self.step,
        }


def check_finished(goals, pclist):
    max_sr = 0
    for goal in goals:
        sr = 0
        for subgoal in goal:
            if "another_obj2_uid" in subgoal:
                pcd3 = pclist[subgoal["another_obj2_uid"]]
            else:
                pcd3 = None
            if check_subgoal_finished_rigid(
                subgoal, pclist[subgoal["obj1_uid"]], pclist[subgoal["obj2_uid"]], pcd3
            ):
                sr += 1 / len(goal)
        max_sr = max(max_sr, sr)
    return max_sr


def check_subgoal_finished_rigid(subgoal, pcd1, pcd2, pcd3=None):
    relation_list = get_related_position(pcd1, pcd2, pcd3)
    if subgoal["position"] == "top" or subgoal["position"] == "on":
        croped_pcd2 = crop_pcd(pcd1, pcd2)
        if len(croped_pcd2) > 0:
            relation_list_2 = get_related_position(pcd1, croped_pcd2)
            if "on" in relation_list_2:
                return True
    if subgoal["position"] == "top" or subgoal["position"] == "on":
        if "on" not in relation_list and "in" not in relation_list:
            return False
    else:
        if subgoal["position"] not in relation_list:
            return False
    return True


def get_related_position(pcd1, pcd2, pcd3=None):
    max_pcd1 = np.max(pcd1, axis=0)
    min_pcd1 = np.min(pcd1, axis=0)
    max_pcd2 = np.max(pcd2, axis=0)
    min_pcd2 = np.min(pcd2, axis=0)
    return infer_spatial_relationship(
        pcd1, pcd2, min_pcd1, max_pcd1, min_pcd2, max_pcd2, pcd3
    )


def infer_spatial_relationship(
    point_cloud_a,
    point_cloud_b,
    min_points_a,
    max_points_a,
    min_points_b,
    max_points_b,
    point_cloud_c=None,
    error_margin_percentage=0.01,
):
    relation_list = []
    if point_cloud_c is None:
        xy_dist = calculate_xy_distance_between_two_point_clouds(
            point_cloud_a, point_cloud_b
        )
        if xy_dist > XY_DISTANCE_CLOSE_THRESHOLD * (1 + error_margin_percentage):
            return []
        dist = calculate_distance_between_two_point_clouds(point_cloud_a, point_cloud_b)
        a_bottom_b_top_dist = min_points_b[2] - max_points_a[2]
        a_top_b_bottom_dist = min_points_a[2] - max_points_b[2]
        if dist < MAX_TO_BE_TOUCHING_DISTANCE * (1 + error_margin_percentage):
            if is_inside(
                src_pts=point_cloud_a,
                target_pts=point_cloud_b,
                thresh=INSIDE_PROPORTION_THRESH,
            ):
                relation_list.append("in")
            elif is_inside(
                src_pts=point_cloud_b,
                target_pts=point_cloud_a,
                thresh=INSIDE_PROPORTION_THRESH,
            ):
                relation_list.append("out of")
            # on, below
            iou_2d, i_ratios, a_ratios = iou_2d_via_boundaries(
                min_points_a, max_points_a, min_points_b, max_points_b
            )
            i_target_ratio, i_anchor_ratio = i_ratios
            target_anchor_area_ratio, anchor_target_area_ratio = a_ratios
            # Target/a supported-by the anchor/b
            a_supported_by_b = False
            if (
                i_target_ratio > MIN_TO_BE_SUPPORTED_AREA_RATIO
                and abs(a_top_b_bottom_dist)
                <= MAX_TO_BE_TOUCHING_DISTANCE * (1 + error_margin_percentage)
                and target_anchor_area_ratio < MAX_TO_BE_SUPPORTING_AREA_RATIO
            ):
                a_supported_by_b = True
            # Target/a supporting the anchor/b
            a_supporting_b = False
            if (
                i_anchor_ratio > MIN_TO_BE_SUPPORTED_AREA_RATIO
                and abs(a_bottom_b_top_dist)
                <= MAX_TO_BE_TOUCHING_DISTANCE * (1 + error_margin_percentage)
                and anchor_target_area_ratio < MAX_TO_BE_SUPPORTING_AREA_RATIO
            ):
                a_supporting_b = True
            if a_supported_by_b:
                relation_list.append("on")
            elif a_supporting_b:
                relation_list.append("below")
            else:
                relation_list.append("near")

        if xy_dist <= XY_DISTANCE_CLOSE_THRESHOLD * (1 + error_margin_percentage):
            x_overlap = (
                (min_points_a[0] <= max_points_b[0] <= max_points_a[0])
                or (min_points_a[0] <= min_points_b[0] <= max_points_a[0])
                or (min_points_b[0] <= min_points_a[0] <= max_points_b[0])
                or (min_points_b[0] <= max_points_a[0] <= max_points_b[0])
            )
            y_overlap = (
                (min_points_a[1] <= max_points_b[1] <= max_points_a[1])
                or (min_points_a[1] <= min_points_b[1] <= max_points_a[1])
                or (min_points_b[1] <= min_points_a[1] <= max_points_b[1])
                or (min_points_b[1] <= max_points_a[1] <= max_points_b[1])
            )
            if x_overlap and y_overlap:
                # If there is overlap on both X and Y axes, classify as "near"
                if "near" not in relation_list:
                    relation_list.append("near")
            elif x_overlap:
                # Objects are close in the X axis; determine Left-Right relationship
                if max_points_a[1] < min_points_b[1]:
                    relation_list.append("left")
                elif max_points_b[1] < min_points_a[1]:
                    relation_list.append("right")
            elif y_overlap:
                # Objects are close in the Y axis; determine Front-Back relationship
                if max_points_a[0] < min_points_b[0]:
                    relation_list.append("front")
                elif max_points_b[0] < min_points_a[0]:
                    relation_list.append("back")
    else:

        def compute_centroid(point_cloud):
            return np.mean(point_cloud, axis=0)

        anchor1_center = compute_centroid(point_cloud_b)
        anchor2_center = compute_centroid(point_cloud_c)
        target_center = compute_centroid(point_cloud_a)
        vector1 = target_center - anchor1_center
        vector2 = anchor2_center - target_center
        vector1_norm = vector1 / np.linalg.norm(vector1)
        vector2_norm = vector2 / np.linalg.norm(vector2)
        cosine_angle = np.dot(vector1_norm, vector2_norm)
        angle = np.degrees(np.arccos(cosine_angle))
        if angle < ANGLE_THRESHOLD:
            relation_list.append("between")
    return relation_list


def calculate_xy_distance_between_two_point_clouds(point_cloud_a, point_cloud_b):
    point_cloud_a = point_cloud_a[:, :2]
    point_cloud_b = point_cloud_b[:, :2]
    nn = NearestNeighbors(n_neighbors=1).fit(point_cloud_a)
    distances, _ = nn.kneighbors(point_cloud_b)
    res = np.min(distances)
    return res


def calculate_distance_between_two_point_clouds(point_cloud_a, point_cloud_b):
    nn = NearestNeighbors(n_neighbors=1).fit(point_cloud_a)
    distances, _ = nn.kneighbors(point_cloud_b)
    res = np.min(distances)
    return res


def is_inside(src_pts, target_pts, thresh=0.5):
    try:
        hull = scipy.spatial.ConvexHull(target_pts)
    except:
        return False
    # print("vertices of hull: ", np.array(hull.vertices).shape)
    hull_vertices = np.array([[0, 0, 0]])
    for v in hull.vertices:
        try:
            hull_vertices = np.vstack(
                (
                    hull_vertices,
                    np.array([target_pts[v, 0], target_pts[v, 1], target_pts[v, 2]]),
                )
            )
        except:
            import pdb

            pdb.set_trace()
    hull_vertices = hull_vertices[1:]

    num_src_pts = len(src_pts)
    # Don't want threshold to be too large (specially with more objects, like 4, 0.9*thresh becomes too large)
    thresh_obj_particles = thresh * num_src_pts
    src_points_in_hull = is_point_in_hull(src_pts, hull_vertices)
    # print("src pts in target, thresh: ", src_points_in_hull.sum(), thresh_obj_particles)
    if src_points_in_hull.sum() > thresh_obj_particles:
        return True
    else:
        return False


def is_point_in_hull(p, hull):
    """
    Test if points in `p` are in `hull`

    `p` should be a `NxK` coordinates of `N` points in `K` dimensions
    `hull` is either a scipy.spatial.Delaunay object or the `MxK` array of the
    coordinates of `M` points in `K`dimensions for which Delaunay triangulation
    will be computed
    """
    if not isinstance(hull, scipy.spatial.Delaunay):
        hull = scipy.spatial.Delaunay(hull)

    return hull.find_simplex(p) >= 0


def crop_pcd(pcd1, pcd2):
    contour1 = get_xy_contour(pcd1, contour_type="concave_hull").buffer(0.05)
    xy_points = pcd2[:, :2]
    mask = contains(contour1, xy_points[:, 0], xy_points[:, 1])
    return pcd2[mask]


def get_xy_contour(points, contour_type="convex_hull"):
    if type(points) == o3d.geometry.PointCloud:
        points = np.asarray(points.points)
    if points.shape[1] == 3:
        points = points[:, :2]
    if contour_type == "convex_hull":
        xy_points = points
        hull = scipy.spatial.ConvexHull(xy_points)
        hull_points = xy_points[hull.vertices]
        sorted_points = sort_points_clockwise(hull_points)
        polygon = Polygon(sorted_points)
    elif contour_type == "concave_hull":
        xy_points = points
        concave_hull_points = concave_hull(xy_points)
        polygon = Polygon(concave_hull_points)
    return polygon


def iou_2d_via_boundaries(min_points_a, max_points_a, min_points_b, max_points_b):
    a_xmin, a_xmax, a_ymin, a_ymax = (
        min_points_a[0],
        max_points_a[0],
        min_points_a[1],
        max_points_a[1],
    )
    b_xmin, b_xmax, b_ymin, b_ymax = (
        min_points_b[0],
        max_points_b[0],
        min_points_b[1],
        max_points_b[1],
    )

    box_a = [a_xmin, a_ymin, a_xmax, a_ymax]
    box_b = [b_xmin, b_ymin, b_xmax, b_ymax]
    xA = max(box_a[0], box_b[0])
    yA = max(box_a[1], box_b[1])
    xB = min(box_a[2], box_b[2])
    yB = min(box_a[3], box_b[3])

    # compute the area of intersection rectangle
    inter_area = max(0, xB - xA) * max(0, yB - yA)
    box_a_area = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    box_b_area = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    if box_a_area + box_b_area - inter_area == 0:
        iou = 0
    else:
        iou = inter_area / float(box_a_area + box_b_area - inter_area)
    if box_a_area == 0 or box_b_area == 0:
        i_ratios = [0, 0]
        a_ratios = [0, 0]
    else:
        i_ratios = [inter_area / float(box_a_area), inter_area / float(box_b_area)]
        a_ratios = [box_a_area / box_b_area, box_b_area / box_a_area]

    return iou, i_ratios, a_ratios


def sort_points_clockwise(points):
    center = np.mean(points, axis=0)
    angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
    sorted_indices = np.argsort(angles)
    return points[sorted_indices]
