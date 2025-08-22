import numpy as np
import open3d as o3d
import scipy
from concave_hull import concave_hull
from scipy.spatial.transform import Rotation as R
import copy

from internutopia.core.task.metric import BaseMetric
from shapely.geometry import Polygon
from shapely.vectorized import contains
from sklearn.neighbors import NearestNeighbors

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
                if 'another_obj2_uid' in subgoal:
                    self.object_prim_paths.add(subgoal['another_obj2_uid'])
                self.object_prim_paths.add(subgoal['obj1_uid'])
                self.object_prim_paths.add(subgoal['obj2_uid'])

        self.mesh_cache = {}

    def reset(self):
        self.step = 0
        self.episode_sr = 0
        self.first_success_step = -1

    def calc_episode_sr(self):
        """
        This function is called at each world step.
        """
        from isaacsim.core.prims import SingleXFormPrim

        point_cloud_list = {}
        for uid in self.object_prim_paths:
            if uid not in self.mesh_cache:
                self.mesh_cache[uid] = get_mesh_info(SingleXFormPrim(f'/World/env_{self.env_id}/scene/obj_{uid}'))
            o3d_pcd = get_pcd_from_mesh(
                get_current_mesh(
                    SingleXFormPrim(f'/World/env_{self.env_id}/scene/obj_{uid}'),
                    self.mesh_cache[uid]
                ),
                num_points=self.max_vertices_num
            )
            vertices = np.asarray(o3d_pcd.points)
            point_cloud_list[uid] = vertices

        try:
            if self.step % 10 ==0:
                self.episode_sr = check_finished(self.task_config.target, point_cloud_list)
        except Exception:
            self.episode_sr = 0

        if self.episode_sr == 1 and self.first_success_step < 0:
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
            if 'another_obj2_uid' in subgoal:
                pcd3 = pclist[subgoal['another_obj2_uid']]
            else:
                pcd3 = None
            if check_subgoal_finished_rigid(
                subgoal, pclist[subgoal['obj1_uid']], pclist[subgoal['obj2_uid']], pcd3
            ):
                sr += 1 / len(goal)
        max_sr = max(max_sr, sr)
    return max_sr


def check_subgoal_finished_rigid(subgoal, pcd1, pcd2, pcd3=None):
    relation_list = get_related_position(pcd1, pcd2, pcd3)
    if subgoal['position'] == 'top' or subgoal['position'] == 'on':
        croped_pcd2 = crop_pcd(pcd1, pcd2)
        if len(croped_pcd2) > 0:
            relation_list_2 = get_related_position(pcd1, croped_pcd2)
            if 'on' in relation_list_2:
                return True
    if subgoal['position'] == 'top' or subgoal['position'] == 'on':
        if 'on' not in relation_list and 'in' not in relation_list:
            return False
    else:
        if subgoal['position'] not in relation_list:
            return False
    return True


def get_related_position(pcd1, pcd2, pcd3=None):
    max_pcd1 = np.max(pcd1, axis=0)
    min_pcd1 = np.min(pcd1, axis=0)
    max_pcd2 = np.max(pcd2, axis=0)
    min_pcd2 = np.min(pcd2, axis=0)
    return infer_spatial_relationship(pcd1, pcd2, min_pcd1, max_pcd1, min_pcd2, max_pcd2, pcd3)


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
        xy_dist = calculate_xy_distance_between_two_point_clouds(point_cloud_a, point_cloud_b)
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
                relation_list.append('in')
            elif is_inside(
                src_pts=point_cloud_b,
                target_pts=point_cloud_a,
                thresh=INSIDE_PROPORTION_THRESH,
            ):
                relation_list.append('out of')
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
                relation_list.append('on')
            elif a_supporting_b:
                relation_list.append('below')
            else:
                relation_list.append('near')

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
                if 'near' not in relation_list:
                    relation_list.append('near')
            elif x_overlap:
                # Objects are close in the X axis; determine Left-Right relationship
                if max_points_a[1] < min_points_b[1]:
                    relation_list.append('left')
                elif max_points_b[1] < min_points_a[1]:
                    relation_list.append('right')
            elif y_overlap:
                # Objects are close in the Y axis; determine Front-Back relationship
                if max_points_a[0] < min_points_b[0]:
                    relation_list.append('front')
                elif max_points_b[0] < min_points_a[0]:
                    relation_list.append('back')
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
            relation_list.append('between')
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


def is_point_in_convex_hull_fast(points, hull_obj, tolerance=1e-12):
    """
    Faster method using ConvexHull equations directly

    `points` should be a `NxK` coordinates of `N` points in `K` dimensions
    `hull_obj` should be a scipy.spatial.ConvexHull object
    """
    return np.all(np.add(np.dot(points, hull_obj.equations[:, :-1].T),
                         hull_obj.equations[:, -1]) <= tolerance, axis=1)


def is_inside(src_pts, target_pts, thresh=0.5, use_fast_method=True):
    try:
        hull = scipy.spatial.ConvexHull(target_pts)
    except:
        return False
    num_src_pts = len(src_pts)
    thresh_obj_particles = thresh * num_src_pts
    if use_fast_method:
        src_points_in_hull = is_point_in_convex_hull_fast(src_pts, hull)
    else:
        hull_vertices = target_pts[hull.vertices]
        src_points_in_hull = is_point_in_hull(src_pts, hull_vertices)
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
    contour1 = get_xy_contour(pcd1, contour_type='concave_hull').buffer(0.05)
    xy_points = pcd2[:, :2]
    mask = contains(contour1, xy_points[:, 0], xy_points[:, 1])
    return pcd2[mask]


def get_xy_contour(points, contour_type='convex_hull'):
    if type(points) is o3d.geometry.PointCloud:
        points = np.asarray(points.points)
    if points.shape[1] == 3:
        points = points[:, :2]
    if contour_type == 'convex_hull':
        xy_points = points
        hull = scipy.spatial.ConvexHull(xy_points)
        hull_points = xy_points[hull.vertices]
        sorted_points = sort_points_clockwise(hull_points)
        polygon = Polygon(sorted_points)
    elif contour_type == 'concave_hull':
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


def get_mesh_from_points_and_faces(points, faceVertexCounts, faceVertexIndices):
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(points)
    triangles = []
    idx = 0
    for count in faceVertexCounts:
        if count == 3:
            triangles.append(faceVertexIndices[idx : idx + 3])
        elif count == 4:
            face_indices = faceVertexIndices[idx : idx + 4]
            triangles.append([face_indices[0], face_indices[1], face_indices[2]])
            triangles.append([face_indices[0], face_indices[2], face_indices[3]])
        elif count > 4:
            face_indices = faceVertexIndices[idx : idx + count]
            for i in range(1, count - 1):
                triangles.append(
                    [face_indices[0], face_indices[i], face_indices[i + 1]]
                )
        idx += count
    mesh.triangles = o3d.utility.Vector3iVector(triangles)
    mesh.compute_vertex_normals()
    return mesh


def recursive_parse(prim):
    from pxr import UsdGeom  # type: ignore

    translation = prim.GetAttribute('xformOp:translate').Get()
    if translation is None:
        translation = np.zeros(3)
    else:
        translation = np.array(translation)
    scale = prim.GetAttribute('xformOp:scale').Get()
    if scale is None:
        scale = np.ones(3)
    else:
        scale = np.array(scale)
    orient = prim.GetAttribute('xformOp:orient').Get()
    if orient is None:
        orient = np.zeros([4, 1])
        orient[0] = 1.0
    else:
        r = orient.GetReal()
        i, j, k = orient.GetImaginary()
        orient = np.array([r, i, j, k]).reshape(4, 1)
    rotation_matrix = o3d.geometry.get_rotation_matrix_from_quaternion(orient)
    points_total = []
    faceuv_total = []
    normals_total = []
    faceVertexCounts_total = []
    faceVertexIndices_total = []
    mesh_total = []
    if prim.IsA(UsdGeom.Mesh):
        mesh_path = str(prim.GetPath()).split('/')[-1]
        if not mesh_path == 'SM_Dummy':
            mesh_total.append(mesh_path)
            points = prim.GetAttribute('points').Get()
            normals = prim.GetAttribute('normals').Get()
            faceVertexCounts = prim.GetAttribute('faceVertexCounts').Get()
            faceVertexIndices = prim.GetAttribute('faceVertexIndices').Get()
            faceuv = prim.GetAttribute('primvars:st').Get()
            if points is None:
                points = []
            if normals is None:
                normals = []
            if faceVertexCounts is None:
                faceVertexCounts = []
            if faceVertexIndices is None:
                faceVertexIndices = []
            if faceuv is None:
                faceuv = []
            normals = [_ for _ in normals]
            faceVertexCounts = [_ for _ in faceVertexCounts]
            faceVertexIndices = [_ for _ in faceVertexIndices]
            faceuv = [_ for _ in faceuv]
            ps = []
            for p in points:
                x, y, z = p
                p = np.array((x, y, z))
                ps.append(p)
            points = ps
            base_num = len(points_total)
            for idx in faceVertexIndices:
                faceVertexIndices_total.append(base_num + idx)
            faceVertexCounts_total += faceVertexCounts
            faceuv_total += faceuv
            normals_total += normals
            points_total += points
    else:
        children = prim.GetChildren()
        for child in children:
            points, faceuv, normals, faceVertexCounts, faceVertexIndices, mesh_list = (
                recursive_parse(child)
            )
            base_num = len(points_total)
            for idx in faceVertexIndices:
                faceVertexIndices_total.append(base_num + idx)
            faceVertexCounts_total += faceVertexCounts
            faceuv_total += faceuv
            normals_total += normals
            points_total += points
            mesh_total += mesh_list
    new_points = []
    for i, p in enumerate(points_total):
        pn = np.array(p)
        pn *= scale
        pn = np.matmul(rotation_matrix, pn)
        pn += translation
        new_points.append(pn)
    return (
        new_points,
        faceuv_total,
        normals_total,
        faceVertexCounts_total,
        faceVertexIndices_total,
        mesh_total,
    )


def get_mesh_from_prim(prim):
    points, faceuv, normals, faceVertexCounts, faceVertexIndices, mesh_total = (
        recursive_parse(prim)
    )
    mesh = get_mesh_from_points_and_faces(points, faceVertexCounts, faceVertexIndices)
    return mesh


def get_pcd_from_mesh(mesh, num_points=1000):
    pcd = mesh.sample_points_uniformly(number_of_points=num_points)
    return pcd


def inverse_transform_mesh(mesh, scale_factors, quaternion, translation_vector):
    vertices = mesh.vertices
    vertices = vertices - translation_vector
    rotation = R.from_quat(quaternion)
    vertices = rotation.inv().apply(vertices)
    vertices = vertices / np.array(scale_factors)
    transformed_mesh = copy.deepcopy(mesh)
    transformed_mesh.vertices = o3d.utility.Vector3dVector(vertices)
    return transformed_mesh


def forward_transform_mesh(mesh, scale_factors, quaternion, translation_vector):
    vertices = mesh.vertices
    vertices = vertices * np.array(scale_factors)
    rotation = R.from_quat(quaternion)
    vertices = rotation.apply(vertices)
    vertices = vertices + translation_vector
    transformed_mesh = copy.deepcopy(mesh)
    transformed_mesh.vertices = o3d.utility.Vector3dVector(vertices)
    return transformed_mesh


def get_world_mesh(mesh, prim_path):
    from omni.isaac.core.utils.prims import get_prim_at_path, get_prim_parent, get_prim_path  # type: ignore
    prim = get_prim_at_path(prim_path)
    mesh = copy.deepcopy(mesh)
    while get_prim_path(prim) != '/':
        scale = prim.GetAttribute('xformOp:scale').Get()
        if scale is None:
            scale = np.ones(3)
        else:
            scale = np.array(scale)
        trans = prim.GetAttribute('xformOp:translate').Get()
        if trans is None:
            trans = np.zeros(3)
        else:
            trans = np.array(trans)
        quat = prim.GetAttribute('xformOp:orient').Get()
        if quat is not None:
            r = quat.GetReal()
            i, j, k = quat.GetImaginary()
            quat = np.array([i, j, k, r])
        else:
            quat = np.array([0, 0, 0, 1])
        mesh = forward_transform_mesh(mesh, scale, quat, trans)
        prim = get_prim_parent(prim)
    return mesh


def get_mesh_info(object):
    try:
        mesh = get_mesh_from_prim(object.prim)
    except:
        return None
    scale = object.get_local_scale()
    trans, quat = object.get_local_pose()
    quat = quat[[1, 2, 3, 0]]
    mesh = inverse_transform_mesh(mesh, scale, quat, trans)
    mesh_info = {}
    mesh_info['mesh'] = get_world_mesh(mesh, object.prim_path)
    mesh_info['trans'], mesh_info['quat'] = object.get_world_pose()
    mesh_info['quat'] = mesh_info['quat'][[1, 2, 3, 0]]
    mesh_info['scale'] = np.array([1, 1, 1])
    return mesh_info


def transform_between_meshes(
    mesh_A, scale_A, quat_A, trans_A, scale_B, quat_B, trans_B
):
    mesh_in_world_frame = inverse_transform_mesh(mesh_A, scale_A, quat_A, trans_A)
    transformed_mesh_B = forward_transform_mesh(
        mesh_in_world_frame, scale_B, quat_B, trans_B
    )
    return transformed_mesh_B


def get_current_mesh(object, mesh_dict):
    scale = np.array([1, 1, 1])
    trans, quat = object.get_world_pose()
    quat = quat[[1, 2, 3, 0]]
    return transform_between_meshes(
        mesh_dict['mesh'],
        mesh_dict['scale'],
        mesh_dict['quat'],
        mesh_dict['trans'],
        scale,
        quat,
        trans,
    )
