import numpy as np
from matplotlib import pyplot as plt
import cv2, os, time
import shutil # for folders
import plotly.graph_objects as go


class Trackers:
    def __init__(self, src, sz = 50, n=8):
        self.src = src
        self.sz = sz
        self.Im = []
        self.cIm = []
        self.trackers = []
        self.n = n

    def load_frames(self):
        self.Im = []
        files = (os.listdir(self.src))

        for f in sorted(files):
            # self.cIm.append(cv2.imread(f"{self.src}/{f}", cv2.IMREAD_COLOR))
            # self.Im.append(cv2.imread(f"{self.src}/{f}", cv2.IMREAD_GRAYSCALE))

            img = cv2.imread(f"{self.src}/{f}", cv2.IMREAD_COLOR)
            g_img = cv2.imread(f"{self.src}/{f}", cv2.IMREAD_GRAYSCALE)
            
            if img is None:
                continue

            self.cIm.append(img)
            self.Im.append(g_img)

        return self.Im
    
    def select_point(self, frame0):
        # plt.imshow(frame0, cmap="gray")
        plt.imshow(frame0[:,:,::-1])
        plt.title(f"select {self.n} points")
        points = plt.ginput(self.n, timeout=0)
        plt.close()

        self.tracks = [[] for _ in range(self.n)]
        
        for i, pt in enumerate(points):
            bbox = (np.int64(pt[0] - self.sz//2), np.int64(pt[1] - self.sz//2), np.int64(self.sz), np.int64(self.sz))
            self.tracks[i].append(bbox)
            
            # self.trackers.append(Tracker(self.src, pt, self.sz, self.Im))
            # tracker = cv2.TrackerKCF_create()
            # tracker.init(frame0, bbox)
            # tracker.init(frame0, np.int64(bbox))
            self.trackers.append(cv2.TrackerKCF_create())
            self.trackers[-1].init(frame0, bbox)

        self.dim = [frame0.shape[1], frame0.shape[0]]

    def load_video(self):
        self.Im = []
        
        cap = cv2.VideoCapture(self.src)
        
        while True:
            ret, img = cap.read()
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            if not ret:
                print("failed")
                break
            
            self.cIm.append(img)
            self.Im.append(img_gray)
        
        cv2.destroyAllWindows()
        cap.release()
        

    def setup(self):
        self.select_point(self.cIm[0])
    
    def run_trackers(self):
        X = np.arange(0, self.dim[0], dtype=np.float32)
        Y = np.arange(0, self.dim[1], dtype=np.float32)
        Xq, Yq = np.meshgrid(X, Y)

        for frame in self.cIm[1:]:
            
            for i in range(len(self.trackers)):
                frame_mapped = cv2.remap(frame, Xq, Yq, cv2.INTER_LINEAR)
                
                ret, bbox = self.trackers[i].update(frame_mapped)
                
                
                if ret:
                    self.tracks[i].append(bbox)
                else:
                    self.tracks[i].append(self.tracks[i][-1])

        return self.tracks

    

    def plot_trackers(self, out=""):
        tracks = np.array(self.tracks)
        dim = [self.Im[0].shape[1], self.Im[0].shape[0]]
        xs = np.arange(0, dim[0], dtype=np.float64)
        # ys = np.arange(0, dim[1], dtype=np.float64)
        # print(tracks.shape)
        if out:
            self.create_folder(out)
        
        for i in range(len(self.Im)):
            cur = tracks[:, i]
            # xs = np.array(cur[:, 0] + cur[:, 2])
            # ys = np.array(cur[:, 1] + cur[:, 3])
            xs = np.array(cur[:, 0] + cur[:, 2]//2)
            ys = np.array(cur[:, 1] + cur[:, 3]//2)
            labels = [f"x{j+1}" for j in range(self.n)]
            
            # print(xs, ys)
            
            plt.imshow(self.cIm[i][:,:,::-1])
            plt.scatter(xs, ys, s=10)
            for j, s in enumerate(labels):
                plt.annotate(s, (xs[j], ys[j]))
            
            plt.pause(0.01)
            if out:
                plt.savefig(f"{out}/{i:05}.png", format="png")
            plt.clf()
        
        # cv2.destroyAllWindows()


    def W(self):
        tracks = np.array(self.tracks)
        # 2m x n (m = # of imgs, n = # of pts)
        self.W = np.array([], dtype=np.float64).reshape(-1, self.n)

        for i in range(len(self.Im)):
            cur = tracks[:, i]
            xs = np.array(cur[:, 0] + cur[:, 2]/2, dtype=np.float64).reshape(1,-1)
            ys = np.array(cur[:, 1] + cur[:, 3]/2, dtype=np.float64).reshape(1,-1)

            self.W = np.vstack((self.W, xs, ys))
            # x y x y x y


        return self.W


        



    def create_folder(self, out: str):
        if out == "":
            return
        try:
            shutil.rmtree(out)
            print("deleted")
        except:
            print("folder(s) dne")

        os.mkdir(out)
        print("folders created")
        
    def output(self, out: str, frame, i, name=""):
        if out == "":
            return

        cv2.imwrite(f"{out}/{name}{i:05d}.png", frame)
    

def SFM(W):
    centroids = W.mean(axis=1).reshape(-1,1)
    W = W - centroids
    U, S, Vh = np.linalg.svd(W)

    M = np.hstack((
    (S[0] * U[:,0]).reshape(-1,1),
    (S[1] * U[:,1]).reshape(-1,1),
    (S[2] * U[:,2]).reshape(-1,1)
    ))

    X = Vh.T[:,:3].T


    return M, X

def plot3d(x,y,z):
    # fig = plt.figure()
    # ax = fig.add_subplot(projection='3d')

    # ax.scatter(x,y,z)
    # ax.set_xlabel("x")
    # ax.set_ylabel("y")
    # ax.set_zlabel("z")
    # plt.show()

    fig = go.Figure(data=[go.Scatter3d(
        x=x,y=y,z=z, 
        mode='markers'
        )])
    fig.show()



if __name__ == "__main__":
    # trackers = Trackers("house", sz=25, n=8)
    trackers = Trackers("7_1b_imgs_4", sz=25, n=8)
    trackers.load_frames()
    trackers.setup()
    t = trackers.run_trackers()
    trackers.plot_trackers("temp_out")
    plt.close()

    W = trackers.W()
    
    M, X = SFM(W)
    print(X.shape)
    np.save("X.npy", X)
    plot3d(X[0,:], X[1,:], X[2,:])
    # trackers.plot_rect_trackers()


