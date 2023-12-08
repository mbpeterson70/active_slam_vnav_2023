import open3d as o3d
import numpy as np
import open3d.visualization.gui as gui

def get_cov_ellipsoid(cov, mu, nstd=1):
    assert cov.shape==(3,3)

    # Find and sort eigenvalues to correspond to the covariance matrix
    eigvals, eigvecs = np.linalg.eigh(cov)
    idx = np.sum(cov,axis=0).argsort()
    eigvals_temp = eigvals[idx]
    idx = eigvals_temp.argsort()
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:,idx]

    # Set of all spherical angles to draw our ellipsoid
    n_points = 25
    theta = np.linspace(0, 2*np.pi, n_points)
    phi = np.linspace(0, np.pi, n_points)

    # Width, height and depth of ellipsoid
    rx, ry, rz = nstd * np.sqrt(eigvals)

    # Get the xyz points for plotting
    # Cartesian coordinates that correspond to the spherical angles:
    X = rx * np.outer(np.cos(theta), np.sin(phi))
    Y = ry * np.outer(np.sin(theta), np.sin(phi))
    Z = rz * np.outer(np.ones_like(theta), np.cos(phi))

    # Rotate ellipsoid for off axis alignment
    old_shape = X.shape
    # Flatten to vectorise rotation
    X,Y,Z = X.flatten(), Y.flatten(), Z.flatten()
    X,Y,Z = np.matmul(eigvecs, np.array([X,Y,Z]))
    X,Y,Z = X.reshape(old_shape), Y.reshape(old_shape), Z.reshape(old_shape)
   
    # Add in offsets for the mean
    X = X + mu[0]
    Y = Y + mu[1]
    Z = Z + mu[2]
    
    return X,Y,Z

def generateRandomCovarianceMatrix(dimension,scale):

    A = (np.random.random((dimension,dimension))-0.5)
    cov = A@A.T*scale

    return cov

class Open3dVisualizer():
    def __init__(self):
        self.points = []
        self.point_covariances = []
        self.point_names = []
        self.point_colors = []

        self.poses = []
        self.pose_covariances = []
        self.pose_names = []
        self.pose_colors = []

        return

    def addPoint(self,xyz,cov,name,color):
        self.points.append(xyz)
        self.point_covariances.append(cov)
        self.point_names.append(name)
        self.point_colors.append(color)
    
    def addPose(self,T,cov,name):
        self.poses.append(T)
        self.pose_covariances.append(cov)
        self.pose_names.append(name)

    def render(self, eye=None, lookAt=None):

        allCoords = []

        app = gui.Application.instance
        app.initialize()
        vis = o3d.visualization.O3DVisualizer("Visualization", 1024, 768)
        vis.show_settings = False
        vis.enable_raw_mode(True)

        numPoints = len(self.points)

        pointsAsArray = np.zeros((numPoints,3))
        colorsAsArray = np.zeros((numPoints,3))
        
        # Create poses
        for pose_idx, (T, pose_covariance, pose_name) in enumerate(zip(self.poses,self.pose_covariances,self.pose_names)):
            pose_mesh = o3d.geometry.TriangleMesh.create_coordinate_frame().transform(T)
            vis.add_geometry(pose_name,pose_mesh)
            vis.add_3d_label(T[0:3,3],pose_name)
            allCoords.append(T[0:3,3])

            if (pose_covariance is not None):
                cov_translation = pose_covariance[3:6,3:6]
                xs, ys, zs = get_cov_ellipsoid(cov_translation, T[0:3,3])

                poseCovPointCloudCoords = np.vstack((xs.flatten(),ys.flatten(),zs.flatten())).T

                numCovPoints, _ = poseCovPointCloudCoords.shape

                cov_lines = o3d.geometry.LineSet()
                lines = [[x, x + 1] for x in range(numCovPoints - 1)]
                colors = np.tile(np.array([0.1,0.1,0.1]), (len(lines), 1))
                cov_lines.points = o3d.utility.Vector3dVector(poseCovPointCloudCoords)
                cov_lines.lines = o3d.utility.Vector2iVector(lines)
                cov_lines.colors = o3d.utility.Vector3dVector(colors)
                vis.add_geometry(f"{pose_name} pose cov lines",cov_lines)


        # Create line elements showing covariance ellipses
        for point_idx, (point, cov, color, name) in enumerate(zip(self.points, self.point_covariances, self.point_colors, self.point_names)):

            x = point[0]
            y = point[1]
            z = point[2]

            pointsAsArray[point_idx,:] = point
            colorsAsArray[point_idx,:] = color

            if (cov is not None):

                xs, ys, zs = get_cov_ellipsoid(cov, point)

                covPointCloudCoords = np.vstack((xs.flatten(),ys.flatten(),zs.flatten())).T

                numCovPoints, _ = covPointCloudCoords.shape

                cov_lines = o3d.geometry.LineSet()
                lines = [[x, x + 1] for x in range(numCovPoints - 1)]
                colors = np.tile(np.array(color), (len(lines), 1))
                cov_lines.points = o3d.utility.Vector3dVector(covPointCloudCoords)
                cov_lines.lines = o3d.utility.Vector2iVector(lines)
                cov_lines.colors = o3d.utility.Vector3dVector(colors)
                vis.add_geometry(f"{name} point cov lines",cov_lines)

            vis.add_3d_label(point,name)

            allCoords.append(point)

        meanVisPoints = o3d.geometry.PointCloud()
        meanVisPoints.points = o3d.utility.Vector3dVector(pointsAsArray)
        meanVisPoints.colors = o3d.utility.Vector3dVector(colorsAsArray)
        vis.add_geometry("mean points",meanVisPoints)

        allCoords = np.vstack(allCoords)


        if (lookAt is None):
            allCoordsMean = np.mean(allCoords,axis=0)
            lookAt = allCoordsMean

        if (eye is None):
            eye = allCoordsMean + [0,0,10]

        # Find an "up" direction that is parallel to look direction
        lookDir = lookAt-eye
        lookDir = lookDir/np.linalg.norm(lookDir)
        up = np.array([0,0,1])
        side = np.cross(up,lookDir)
        up = np.cross(lookDir,side)

        vis.setup_camera(45,lookAt,eye,up)

        vis.set_background(np.array([1.0,1.0,1.0,1.0]),None)

        vis.show_skybox(False)

        vis.show_ground = False
        vis.show_axes = False

        app.add_window(vis)
        app.run()



if __name__=="__main__":
    from scipy.spatial.transform import Rotation as ScipyRot

    N = 10

    vis = Open3dVisualizer()

    color = [0,0,0]

    for n in range(N):
        xyz = np.random.uniform(size=(3))*10
        cov = generateRandomCovarianceMatrix(3,1.0)
        vis.addPoint(xyz,cov,f"{n}",color)

        print("cov shape=")
        print(cov.shape)

        T = np.eye(4)
        T[0,3] = xyz[0]

        cov_pose = generateRandomCovarianceMatrix(6,1.0)

        vis.addPose(T,cov_pose,str(n))
    
    vis.render()