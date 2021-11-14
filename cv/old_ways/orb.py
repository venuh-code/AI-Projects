import cv2
from matplotlib import pyplot as plt
import numpy as np
from sklearn.cluster import KMeans

MIN_MATCH_COUNT = 10
img1 = cv2.imread(r"E:\task\data\cvs.jpg", 0)
img1 = cv2.resize(img1, (120,120))
img2 = cv2.imread(r"E:\task\data\cv.jpg", 0) #train

orb = cv2.ORB_create(10000,1.2,nlevels=12,edgeThreshold=3)
kp1,des1 = orb.detectAndCompute(img1,None)
kp2,des2 = orb.detectAndCompute(img2,None)

#kmeans
x = np.array([kp2[0].pt])
for i in range(len(kp2)):
    x = np.append(x, [kp2[i].pt], axis=0)
x = x[1 : len(x)]
clf = KMeans(n_clusters=3)
clf.fit(x)
labels = clf.labels_
cluster_centers = clf.cluster_centers_
labels_unique = np.unique(labels)
n_clusters_ = len(labels_unique)
print("number of estimated clusters : %d" % n_clusters_)

#add list
s = [None] * n_clusters_
for i in range(n_clusters_):
    l = clf.labels_
    d, = np.where(l == i)
    print(d.__len__())
    s[i] = list(kp2[xx]  for xx in d)

#======
des2_ = des2
for i in range(n_clusters_):
    kp2 = s[i]
    l = clf.labels_
    d, = np.where(l == i)
    des2 = des2_[d,]

    #ding yi pi pei suanfa
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    des1 = np.float32(des1)
    des2 = np.float32(des2)

    #star to match
    matches = flann.knnMatch(des1, des2, 2)
    good = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)
    if len(good) > 3:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,2)
        if M is None:
            print("NO homography")
        else:
            matchesMask = mask.ravel().tolist()
            h,w = img1.shape
            pts = np.float32([[0,0],[0,h-1],[w-1,h-1],[w-1,0]]).reshape(-1,1,2)
            dst = cv2.perspectiveTransform(pts,M)
            img2 = cv2.polylines(img2,[np.int32(dst)], True, 255, 3, cv2.LINE_AA)
            draw_params = dict(
                matchColor=(0,255,0),
                singlePointColor=None,
                matchesMask = matchesMask,
                flags = 2,
            )

            img3 = cv2.drawMatches(img1,kp1,img2,kp2,good,None, **draw_params)
            plt.imshow(img3, "gray")
            plt.show()
    else:
        print("Not enough matches are found-%d/%d" %(len(good), MIN_MATCH_COUNT))
        matchesMask = None