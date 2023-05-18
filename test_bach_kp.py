import imageio
import numpy as np
import cv2
import struct
import os

import scipy
from scipy.spatial import ConvexHull
from tqdm import tqdm

from scipy.spatial.transform import Rotation


def CameraProjPoint3(vertices, pose, intrinsic):
    """
    Performs the Orthogonal projection on a batch of 3D vertices.

    Parameters:
    vertices (numpy array): A (batch_size x n x 3) numpy array containing the 3D vertices for each batch.
    mvp (numpy array): A (batch_size x 4 x 4) numpy array representing the MVP matrix for each batch.

    Returns:
    numpy array: A (batch_size x n x 2) numpy array containing the projected vertices for each batch.
    """
    # Add homogeneous coordinate of 1 to the vertices
    ones = np.ones((vertices.shape[0], vertices.shape[1], 1))
    vertices = np.concatenate((vertices, ones), axis=-1)

    # Perform MVP projection on the vertices using einsum
    projected_vertices = np.einsum('bij,bkj->bik', vertices, pose)

    # print('| projected_vertices3D: ',projected_vertices.min((0,1)), projected_vertices.max((0,1)))
    # projected_vertices = (projected_vertices + 1.5) / 3
    projected_vertices = np.einsum('bij,kj->bik', projected_vertices[:, :, :3], intrinsic)
    projected_vertices = projected_vertices[:, :, :2] / projected_vertices[:, :, 2:]
    print('| projected_vertices2D: ',projected_vertices)

    # Return the projected vertices without the homogeneous coordinate
    return projected_vertices


def get_video_infos(video_path):
    vid_cap = cv2.VideoCapture(video_path)
    height = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fps = vid_cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(vid_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    return {'height': height, 'width': width, 'fps': fps, 'total_frames': total_frames}


def point_in_hull(points, hull, tolerance=1e-12):
    return np.all(np.add(np.dot(points, hull.equations[:, :-1].T), hull.equations[:, -1]) <= tolerance, axis=1)


def get_hull_mask(w, h, points):
    hull = scipy.spatial.ConvexHull(points)
    deln = scipy.spatial.Delaunay(points[hull.vertices])

    # Instead of allocating a giant array for all indices in the volume,
    # just iterate over the slices one at a time.
    idx_2d = np.indices([w, h], np.int16)
    idx_2d = idx_2d.transpose(1, 2, 0)
    mask = np.zeros([w, h])
    s = deln.find_simplex(idx_2d)
    mask[(s != -1)] = 1
    mask = mask.transpose(1, 0)
    return mask


def draw_landmarks2d(image, landmarks2d, color):
    for lid, landmark in landmarks2d:
        cv2.circle(image, (int(landmark[0]), int(landmark[1])), 2, color, -1)
        cv2.putText(
            image, str(lid), (int(landmark[0]), int(landmark[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    return image


def main(vid_path, mvp_dat_path, model_dat_path, vertexes_dat_path, out_path, vertexes_type='bach'):
    template_infos = get_video_infos(vid_path)
    height = template_infos['height']
    width = template_infos['width']
    fps = template_infos['fps']

    reader = imageio.get_reader(vid_path)
    writer = imageio.get_writer(out_path, fps=fps)
    # mvp = np.fromfile(mvp_dat_path, dtype=np.float32).reshape(-1, 4, 4)
    # mvp = np.transpose(mvp, axes=(0, 2, 1))
    world2cam = np.fromfile(model_dat_path, dtype=np.float32).reshape(-1, 4, 4)
    world2cam = np.transpose(world2cam, axes=(0, 2, 1))
    world2cam[:, 1:3] = -world2cam[:, 1:3]
    if vertexes_type == "full":
        number_vertexes = 1651
    elif vertexes_type == "bach":
        number_vertexes = 221
    else:
        assert vertexes_type == "ibug"
        number_vertexes = 68
    ldm_metric = np.fromfile(vertexes_dat_path, dtype=np.float32).reshape(-1, number_vertexes, 3)
    ldm_metric[..., :] = ldm_metric[..., :] / 10
    world2cam[:, :3, 3] = world2cam[:, :3, 3] / 10

    focal = 1
    w = 1
    h = 1

    # Camera intrinsic matrix
    intrinsic = np.array([
        [focal * height / width, 0.0, w / 2.0],
        [0.0, focal, h / 2.0],
        [0.0, 0.0, 1.0]
    ])

    np.set_printoptions(suppress=True)
    # P_mat = mvp @ np.linalg.inv(world2cam)
    # print('proj_mat: ', P_mat)

    # landmarks3d_all_perspective = PerspectiveMultiplyPoint3(video_vertexes, mvp)
    # landmarks3d_all_perspective[..., 1] = -landmarks3d_all_perspective[..., 1]
    # print('intrinsic: ', intrinsic[0])
    ldm = CameraProjPoint3(ldm_metric, world2cam, intrinsic)

    print('extrinsic: ', world2cam[0])
    cam2world = np.linalg.inv(world2cam)
    print('cam_pose: ', cam2world[0])

    cam_xyx = cam2world[..., :-1, -1]
    dis = ldm_metric - cam_xyx[:, None, :]

    dis = np.sqrt((dis * dis).sum(-1))
    print(cam_xyx.shape, ldm_metric.shape, dis.min(), dis.max())

    r = Rotation.from_matrix(world2cam[:, :3, :3])
    euler_angles = r.as_euler('xyz', degrees=True)
    print('extrinsic euler_angles: ', euler_angles[:3])

    r = Rotation.from_matrix(cam2world[:, :3, :3])
    euler_angles = r.as_euler('xyz', degrees=True)
    print('cam pose euler_angles: ', euler_angles[:3])

    print("landmarks3d_all max, min:", ldm.max(1).mean(0), ldm.min(1).mean(0))
    print("landmarks_metric max, min:", ldm_metric.max(1).mean(0), ldm_metric.min(1).mean(0))
    print(ldm.min((0, 1)), ldm.max((0, 1)))

    # ldm[..., 0] = width - ldm[..., 0] * width
    ldm[..., 0] = ldm[..., 0] * width
    ldm[..., 1] = ldm[..., 1] * height
    print(ldm.min((0, 1)), ldm.max((0, 1)))

    x_min, y_min = 0, 0
    x_max, y_max = width, height
    # x_min, y_min = np.min(landmarks3d_all, axis=(0, 1))[:2]
    # x_min = max(np.round(x_min).astype(int) - 50, 0)
    # y_min = max(np.round(y_min).astype(int) - 50, 0)
    # x_max, y_max = np.max(landmarks3d_all, axis=(0, 1))[:2]
    # x_max = np.round(x_max).astype(int) + 50
    # y_max = np.round(y_max).astype(int) + 50
    # landmarks3d_all[..., 0] = landmarks3d_all[..., 0] - x_min
    # landmarks3d_all[..., 1] = landmarks3d_all[..., 1] - y_min
    # print("xmin, ymin, xmax, ymax: ", x_min, y_min, x_max, y_max)

    # ldm_2d = (ldm_metric[..., :2] + 1) * width / 2
    ldm_2d = ldm[..., :2]
    face_hull = ConvexHull(ldm_2d[0])
    face_ct = np.sort(face_hull.vertices)
    # print("facect_points: ", list(face_ct))

    downface_contour = [5, 215, 218] + list(range(147, 219)) + list(range(72, 83))
    exclude = [187, 188, 189, 190]
    downface_contour = set(downface_contour) - set(exclude)
    downface_contour = np.array(sorted(downface_contour))
    df_hull = ConvexHull(ldm_2d[0, downface_contour])
    dfct = np.sort(downface_contour[df_hull.vertices])
    # print("dfct_kp: ", list(dfct))
    downface = np.argwhere(point_in_hull(ldm_2d[0], df_hull) > 0)[:, 0]
    # print("df_kp: ", list(downface))
    upface = set(np.arange(0, ldm_2d.shape[1])) - set(downface)
    upface = np.array(sorted(upface))
    lip_kp = np.array(list(range(41, 77)))
    # print("uf_kp: ", list(upface))
    # print("lip_kp: ", list(lip_kp))

    vis_kp = np.arange(0, ldm_2d.shape[1])
    # vis_kp = np.array(sorted(
    #     set(list(range(0, landmarks2d_all.shape[1]))) - set(list(range(41, 77)))
    # ))
    ldm_2d = [zip(vis_kp, x[vis_kp]) for x in ldm_2d]
    vertexes_frames = min(ldm_metric.shape[0], 1000)
    for idx in tqdm(range(vertexes_frames), desc='Processing'):
        if out_path is not None:
            frame = reader.get_next_data()
            annotated_image = frame[y_min:y_max, x_min:x_max, ::-1].copy()
        if out_path is not None:
            draw_landmarks2d(annotated_image, ldm_2d[idx], (125, 0, 200))
            writer.append_data(annotated_image[:, :, ::-1])
    if out_path is not None:
        reader.close()
        writer.close()


if __name__ == '__main__':
    # video_path = 'data/raw/sa_2davatar/training_data/Bailuyao/segments/video_segments/bailuyao_dialog_000002.mp4'
    # base_path = "/".join(video_path.split("/")[:-3])
    # item_name = video_path.split("/")[-1]
    # vertexes_dat_path = f'{base_path}/bach_ldm3/{item_name}_bach.dat'
    # mvp_dat_path = f'{base_path}/bach_ldm3/{item_name}_mvp.dat'
    # model_dat_path = f'{base_path}/bach_ldm3/{item_name}_model.dat'

    video_path = 'tmp/bailuyao_dialog_000008.mp4'
    # video_path = '/mnt/bn/ailabrenyi/projects/avatar_toolkit/May.mp4'
    item_name = video_path.split("/")[-1]
    base_path = "/".join(video_path.split("/")[:-1])
    vertexes_dat_path = f'{base_path}/{item_name}_bach.dat'
    mvp_dat_path = f'{base_path}/{item_name}_mvp.dat'
    model_dat_path = f'{base_path}/{item_name}_model.dat'

    out_path = 'tmp/a.mp4'
    main(video_path, mvp_dat_path, model_dat_path, vertexes_dat_path, out_path)
