from plyfile import PlyData
import numpy as np
from torch.utils.data import DataLoader,Dataset,random_split
import os
import pandas as pd


labels = ((255, 255, 255), (255, 0, 0), (255, 128, 0), (128, 128, 0), (128, 255, 0), (255, 255, 0), (0, 255, 0), (0, 255, 255), (0, 128, 255), (0, 255, 128), (0, 0, 255), (128, 0, 255), (255, 0, 128), (255, 0, 255), (128, 255, 255), (255, 128, 255), (255, 255, 128)) #, (128, 0, 0), (128, 64, 0), (64, 128, 0), (64, 128, 0), (64, 64, 0), (0, 64, 0), (0, 64, 128), (0, 128, 64), (0, 64, 128), (0, 0, 64), (128, 0, 64), (64, 0, 128), (64, 0, 64), (128, 64, 64), (64, 128, 64), (64, 64, 128))



def get_data(path=""):
    row_data = PlyData.read(path)  # read ply file
    points = np.array(pd.DataFrame(row_data.elements[0].data))
    faces = np.array(pd.DataFrame(row_data.elements[1].data))
    n_face = faces.shape[0]  # number of faces
    xyz = points[:, :3] # coordinate of vertex shape=[N, 3]
    normal = points[:, 3:]  # normal of vertex shape=[N, 3]
    label_face = np.zeros([n_face,1]).astype('int32')
    label_face_onehot = np.zeros([n_face,17]).astype(('int32'))
    """ index of faces shape=[N, 3] """
    index_face = np.concatenate((faces[:, 0]), axis=0).reshape(n_face, 3)
    """ RGB of faces shape=[N, 3] """
    RGB_face = faces[:, 1:4]
    """ coordinate of 3 vertexes  shape=[N, 9] """
    xyz_face = np.concatenate((xyz[index_face[:, 0], :], xyz[index_face[:, 1], :],xyz[index_face[:, 2], :]), axis=1)
    """  normal of 3 vertexes  shape=[N, 9] """
    normal_vertex = np.concatenate((normal[index_face[:, 0], :], normal[index_face[:, 1], :],normal[index_face[:, 2], :]), axis=1)

    normal_face = faces[:, 5:]
    x1, y1, z1 = xyz_face[:, 0], xyz_face[:, 1], xyz_face[:, 2]
    x2, y2, z2 = xyz_face[:, 3], xyz_face[:, 4], xyz_face[:, 5]
    x3, y3, z3 = xyz_face[:, 6], xyz_face[:, 7], xyz_face[:, 8]
    x_centre = (x1 + x2 + x3) / 3
    y_centre = (y1 + y2 + y3) / 3
    z_centre = (z1 + z2 + z3) / 3
    centre_face = np.concatenate((x_centre.reshape(n_face,1),y_centre.reshape(n_face,1),z_centre.reshape(n_face,1)), axis=1)
    """ get points of each face, concat all of above, shape=[N, 24]"""
    points_face = np.concatenate((xyz_face, centre_face, normal_vertex, normal_face), axis=1).astype('float32')
    """ get label of each face """
    for i, label in enumerate(labels):
        label_face[(RGB_face == label).all(axis=1)] = i
        label_face_onehot[(RGB_face == label).all(axis=1), i] = 1
    return index_face, points_face, label_face, label_face_onehot, points


def generate_plyfile(index_face, point_face, label_face, path= " "):
    """
    Input:
        index_face: index of points in a face [N, 3]
        points_face: 3 points coordinate and 3 normals [N, 24]
        label_face: label of face [N, 1]
        path: path to save new generated ply file
    Return:
    """
    unique_index = np.unique(index_face.flatten())  # get unique points index
    flag = np.zeros([unique_index.max()+1, 2]).astype('uint64')
    order = 0
    with open(path, "w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write("comment VCGLIB generated\n")
        f.write("element vertex " + str(unique_index.shape[0]) + "\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property float nx\n")
        f.write("property float ny\n")
        f.write("property float nz\n")
        f.write("element face " + str(index_face.shape[0]) + "\n")
        f.write("property list uchar int vertex_indices\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("property uchar alpha\n")
        f.write("property float nx\n")
        f.write("property float ny\n")
        f.write("property float nz\n")
        f.write("end_header\n")
        for i, index in enumerate(index_face):
            for j, data in enumerate(index):
                if flag[data, 0] == 0:  # if this point has not been wrote
                    xyz = point_face[i, 3*j:3*(j+1)]  # Get coordinate
                    xyz_nor = point_face[i, 3*(j+3):3*(j+4)]
                    f.write(str(xyz[0]) + " " + str(xyz[1]) + " " + str(xyz[2]) + " " + str(xyz_nor[0]) + " "
                            + str(xyz_nor[1]) + " " + str(xyz_nor[2]) + "\n")
                    flag[data, 0] = 1  # this point has been wrote
                    flag[data, 1] = order  # give point a new index
                    order = order + 1  # index add 1 for next point
        normal_face = (point_face[:, 9:12] + point_face[:, 12:15] + point_face[:, 15:18])/3.
        for i, data in enumerate(index_face):  # write new point index for every face
            RGB = labels[label_face[i, 0]]  # Get RGB value according to face label
            f.write(str(3) + " " + str(int(flag[data[0], 1])) + " " + str(int(flag[data[1], 1])) + " "
                    + str(int(flag[data[2], 1])) + " " + str(RGB[0]) + " " + str(RGB[1]) + " "
                    + str(RGB[2]) + " " + str(255) + " " + str(normal_face[i,0]) + " " + str(normal_face[i, 1]) + " " + str(normal_face[i, 2]) +"\n")
        f.close()


class plydataset(Dataset):

    def __init__(self, path="data/train"):
        self.root_path = path
        self.file_list = os.listdir(path)
    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, item):
        read_path = os.path.join(self.root_path, self.file_list[item])
        index_face, points_face, label_face, label_face_onehot, points = get_data(path=read_path)
        raw_points_face = points_face.copy()

        # move all mesh to origin
        centre = points_face[:, 9:12].mean(axis=0)
        points_face[:, 0:3] -= centre
        points_face[:, 3:6] -= centre
        points_face[:, 6:9] -= centre
        points_face[:, 9:12] = (points_face[:, 0:3] + points_face[:, 3:6] + points_face[:, 6:9]) / 3
        points[:, :3] -= centre
        max = points.max()
        points_face[:, :12] = points_face[:, :12] / max

        # normalized data
        maxs = points[:, :3].max(axis=0)
        mins = points[:, :3].min(axis=0)
        means = points[:, :3].mean(axis=0)
        stds = points[:, :3].std(axis=0)
        nmeans = points[:, 3:].mean(axis=0)
        nstds = points[:, 3:].std(axis=0)
        nmeans_f = points_face[:, 21:].mean(axis=0)
        nstds_f = points_face[:, 21:].std(axis=0)
        for i in range(3):
            #normalize coordinate
            points_face[:, i] = (points_face[:, i] - means[i]) / stds[i]  # point 1
            points_face[:, i + 3] = (points_face[:, i + 3] - means[i]) / stds[i]  # point 2
            points_face[:, i + 6] = (points_face[:, i + 6] - means[i]) / stds[i]  # point 3
            points_face[:, i + 9] = (points_face[:, i + 9] - mins[i]) / (maxs[i] - mins[i])  # centre
            #normalize normal vector
            points_face[:, i + 12] = (points_face[:, i + 12] - nmeans[i]) / nstds[i]  # normal1
            points_face[:, i + 15] = (points_face[:, i + 15] - nmeans[i]) / nstds[i]  # normal2
            points_face[:, i + 18] = (points_face[:, i + 18] - nmeans[i]) / nstds[i]  # normal3
            points_face[:, i + 21] = (points_face[:, i + 21] - nmeans_f[i]) / nstds_f[i]  # face normal

        # SY 12 -> 3: with our current mesh density, mesh faces are quite small,
        # so face center and face normal as a whole is a good representation for
        # the face already; So we do not need positions and normals of the
        # three vertices (i.e. point 1 2 3).
        # We can add the following line to return only face center and face normal:
        # points_face = points_face[:, (9,10,11,21,22,23)]
        # I have done very preliminary test which looked good, but I did not get
        # enough time to pursue this effort; need to test/compare/verify 10+ cases.
        # nonetheless, this change won't reduce size of TSGCNet model, its size
        # in both memory and resulting model files, is decided by the layers in
        # TSGCNet neural network. I have not checked difference in training and
        # inference speed though.
        return index_face, points_face, label_face, label_face_onehot, self.file_list[item], raw_points_face


if __name__ == "__main__":
    print(" ")








