"""
 ----------------------------------------------------
  @Author: tsukasa
  @Affiliation: Waseda University
  @Email: rinsa@suou.waseda.jp
  @Date: 2019-07-12 17:42:11
  @Last Modified by:   tsukasa
  @Last Modified time: 2019-07-12 20:04:39
 ----------------------------------------------------

"""

from collections import deque
import numpy as np


class AABB():
    def __init__(self, botleft, topright):
        '''
        Class for Axis-Aligned Bounding Box
        '''

        self.botleft = botleft
        self.topright = topright


class BVHNode():
    def __init__(self, aabb, triangles, vernormals, triindices, child1, child2):
        '''
        aabb                              : see AABB class avobe
        triangles, vernormals, triindeces : geometry info from input 3d gepmetory
        child1, 2                         : child node
        '''

        self.aabb = aabb
        self.triangles = triangles
        self.vernormals = vernormals
        self.triindices = triindices
        self.child1 = child1
        self.child2 = child2


def buildBVH(triangles, vernormals, triindices):
    '''
    function to create bvh structure
    '''

    ## compute aabb
    aabb = AABB(triangles.min(axis=1).min(axis=0), triangles.max(axis=1).max(axis=0))

    if (triangles.shape[0] <= 64):
        return BVHNode(aabb, triangles, vernormals, triindices, None, None)

    else:
        ## set variavle
        mincost = np.inf
        minaxis = -1
        minsplit = -1
        centroids = np.mean(triangles, axis=1)

        for axis in range(3):
            ## sort triangles with respect centroids
            sortedtris = triangles[np.argsort(centroids[:, axis])]

            ## calc left
            leftaabb = AABB(np.ones((3,), dtype=np.float32) * np.inf,
                            np.ones((3,), dtype=np.float32) * -np.inf)
            leftcost = np.zeros((len(sortedtris),), dtype=np.float32)
            for i, tri in enumerate(sortedtris):
                leftaabb.botleft = np.minimum(leftaabb.botleft, tri.min(axis=0))
                leftaabb.topright = np.maximum(leftaabb.topright, tri.max(axis=0))
                diag = np.abs(leftaabb.topright - leftaabb.botleft)
                left_index = i
                leftcost[left_index] = (diag[0] * diag[1] + diag[1] * diag[2] + diag[2] * diag[
                    0]) * i  ## why * i? not 2?

            ## calc left
            rightaabb = AABB(np.ones((3,), dtype=np.float32) * np.inf,
                             np.ones((3,), dtype=np.float32) * -np.inf)
            rightcost = np.zeros((len(sortedtris),), dtype=np.float32)
            for i, tri in enumerate(sortedtris[::-1]):
                rightaabb.botleft = np.minimum(rightaabb.botleft, tri.min(axis=0))
                rightaabb.topright = np.maximum(rightaabb.topright, tri.max(axis=0))
                diag = np.abs(rightaabb.topright - rightaabb.botleft)
                right_index = len(sortedtris) - 1 - i
                rightcost[right_index] = (diag[0] * diag[1] + diag[1] * diag[2] + diag[2] * diag[0]) * i

                ## update minimum cost, axis, index
            if ((leftcost + rightcost).min() < mincost):
                mincost = (leftcost + rightcost).min()
                minaxis = axis
                minsplit = np.argmin(leftcost + rightcost)

        sortedindices = np.argsort(centroids[:, minaxis])
        sortedtriindices = triindices[sortedindices]
        sortedtris = triangles[sortedindices]
        sortedvern = vernormals[sortedindices]
        node1 = buildBVH(sortedtris[:minsplit], sortedvern[:minsplit], sortedtriindices[:minsplit])
        node2 = buildBVH(sortedtris[minsplit:], sortedvern[minsplit:], sortedtriindices[minsplit:])

        return BVHNode(aabb,
                       np.zeros((0, 3, 3), dtype=np.float32),
                       np.zeros((0, 3, 3), dtype=np.float32),
                       np.zeros((0,), dtype=np.int32),
                       node1,
                       node2)


class SerialBVHNode():
    def __init__(self, aabb, tristart, ntris):
        '''
        aabb      : see AABB class avobe
        tristart  : first triangle index of this node
        ntris     : number of triangles of this node
        child1, 2 : child node index, [Note]-1 means there is no child
        '''

        self.aabb = aabb
        self.tristart = tristart
        self.ntris = ntris
        self.child1 = -1
        self.child2 = -1


def BVHserializer(root):
    ## turn hierarchy into list using Breadth First Search
    queue = deque()
    nodelist = []

    queue.append(root)

    while (len(queue) > 0):
        front = queue.popleft()
        nodelist.append(front)

        if (front.child1 is not None):
            queue.append(front.child1)
        if (front.child2 is not None):
            queue.append(front.child2)

    for node in nodelist:
        node.parent = None

        ## generate serial node lists and triangle list
    serialnodelist = []
    serialtrilist = np.zeros((0, 3, 3), dtype=np.float32)
    serialnormallist = np.zeros((0, 3, 3), dtype=np.float32)
    serialtriindlist = np.zeros((0,), dtype=np.int32)
    for i, node in enumerate(nodelist):
        serialnodelist.append(SerialBVHNode(node.aabb,
                                            serialtrilist.shape[0],
                                            node.triangles.shape[0]))

        if (node.child1 is not None):
            node.child1.parent = serialnodelist[-1]
        if (node.child2 is not None):
            node.child2.parent = serialnodelist[-1]

        serialtrilist = np.vstack((serialtrilist, node.triangles))
        serialnormallist = np.vstack((serialnormallist, node.vernormals))
        serialtriindlist = np.hstack((serialtriindlist, node.triindices))

        ## find children by looking up parent
    for i, node, snode in zip(range(len(nodelist)), nodelist, serialnodelist):
        if (node.parent is not None):

            if (node.parent.child1 == -1):
                node.parent.child1 = i
                continue

            if (node.parent.child2 == -1):
                node.parent.child2 = i
                continue

    return serialnodelist, serialtrilist, serialnormallist, serialtriindlist