import numpy as np
import math as m
from proj1_3.code.graph_search import graph_search
import matplotlib.pyplot as plt
import pdb


class WorldTraj(object):
    """

    """
    # def __init__(self, points):
    #     """
    #     This is the constructor for the Trajectory object. A fresh trajectory
    #     object will be constructed before each mission. For a waypoint
    #     trajectory, the input argument is an array of 3D destination
    #     coordinates. You are free to choose the times of arrival and the path
    #     taken between the points in any way you like.

    #     You should initialize parameters and pre-compute values such as
    #     polynomial coefficients here.

    #     Inputs:
    #         points, (N, 3) array of N waypoint coordinates in 3D
    #     """

    #     # STUDENT CODE HERE
    #     self.points = points
    def __init__(self, world, start, goal):
        """
        This is the constructor for the trajectory object. A fresh trajectory
        object will be constructed before each mission. For a world trajectory,
        the input arguments are start and end positions and a world object. You
        are free to choose the path taken in any way you like.

        You should initialize parameters and pre-compute values such as
        polynomial coefficients here.

        Parameters:
            world, World object representing the environment obstacles
            start, xyz position in meters, shape=(3,)
            goal,  xyz position in meters, shape=(3,)

        """

        # You must choose resolution and margin parameters to use for path
        # planning. In the previous project these were provided to you; now you
        # must chose them for yourself. Your may try these default values, but
        # you should experiment with them!
        self.resolution = np.array([0.25, 0.25, 0.25])
        self.margin = 0.25

        # You must store the dense path returned from your Dijkstra or AStar
        # graph search algorithm as an object member. You will need it for
        # debugging, it will be used when plotting results.
        self.graphpath = graph_search(world, self.resolution, self.margin, start, goal, astar=True)

        # You must generate a sparse set of waypoints to fly between. Your
        # original Dijkstra or AStar path probably has too many points that are
        # too close together. Store these waypoints as a class member; you will
        # need it for debugging and it will be used when plotting results.
        # self.points = np.zeros((1,3)) # shape=(n_pts,3)
        self.Start = start
        self.Goal = goal
        self.points = self.graphpath
        self.path = self.graphpath
        self.path_length = float(round(np.sum(np.linalg.norm(np.diff(self.path, axis=0),axis=1)),3))
        self.v_av = 2.0
        self.Time_list = self.get_time_list(self.v_av,self.path,self.path_length)
        self.K = self.get_coeff_5order(self.Time_list,self.path)
        
        

    def update(self, t):
        """
        Given the present time, return the desired flat output and derivatives.

        Inputs
            t, time, s
        Outputs
            flat_output, a dict describing the present desired flat outputs with keys
                x,        position, m
                x_dot,    velocity, m/s
                x_ddot,   acceleration, m/s**2
                x_dddot,  jerk, m/s**3
                x_ddddot, snap, m/s**4
                yaw,      yaw angle, rad
                yaw_dot,  yaw rate, rad/s
        """
        path = self.graphpath
        Time_list = self.Time_list


        x        = np.zeros((3,))
        x_dot    = np.zeros((3,))
        x_ddot   = np.zeros((3,))
        x_dddot  = np.zeros((3,))
        x_ddddot = np.zeros((3,))
        yaw = 0
        yaw_dot = 0

        # STUDENT CODE HERE
        # print('t',t)
        
        
        
        # K = self.get_coeff(Time_list,path)
        K = self.K
        K = np.flip(K,axis=1)

        for i in range(len(Time_list)-1):          
            if t == np.inf:
                x = path[-1]
            elif Time_list[i] < t < Time_list[i+1]:
                t = t-Time_list[i]
                c = K[:,4*i:4*(i+1),:]
                c0 = c[:,0,:]
                c1 = c[:,1,:]
                c2 = c[:,2,:]
                c3 = c[:,3,:]
                x_ddot = np.concatenate(6*c3*t + 2*c2)
                x_dot  = np.concatenate(3*c3*t**2 + 2*c2*t + c1)
                x = np.concatenate(c3*t**3 + c2*t**2 + c1*t + c0)
            elif Time_list[-1] <= t:
                x = path[-1]
            else:
                pass


        flat_output = { 'x':x, 'x_dot':x_dot, 'x_ddot':x_ddot, 'x_dddot':x_dddot, 'x_ddddot':x_ddddot,
                        'yaw':yaw, 'yaw_dot':yaw_dot}
        return flat_output

    def get_coeff(self,time,path):

        waypoint = np.zeros((3,len(path),1))
        waypoint[0,:,:] = path[:,0].reshape(len(path),1)
        waypoint[1,:,:] = path[:,1].reshape(len(path),1)
        waypoint[2,:,:] = path[:,2].reshape(len(path),1)

        td = np.diff(time)
        td = td.reshape(len(td),1)
        
        i = np.array([0,1]).reshape(1,2)
        seg_time = td*i
        seg_time = seg_time.reshape(2*len(seg_time),1)

        # Position Constraints
        
        blks_pos = np.hstack((seg_time**3,seg_time**2,seg_time**1,seg_time**0))
        pos_blks = np.zeros((3,blks_pos.shape[0],blks_pos.shape[1]))
        pos_blks[0,:,:] = blks_pos
        pos_blks[1,:,:] = blks_pos
        pos_blks[2,:,:] = blks_pos
        m = len(path)-1
        Apos = np.zeros((3,2*m,4*m))
        for idx in range(m):
            i = 2*idx
            if idx ==0:
                Apos[:,i:i+2,-4*(idx+1):] = pos_blks[:,i:i+2,:]
            elif idx == m-1:
                Apos[:,i:,-4*(idx+1):-4*(idx+1)+4] = pos_blks[:,i:,:]
            else:
                Apos[:,i:i+2,-4*(idx+1):-4*(idx+1)+4] = pos_blks[:,i:i+2,:]

        # Velocity Constraints
        
        blks_vel = np.hstack((3*seg_time[:-1]**2,2*seg_time[:-1]**1,seg_time[:-1]**0,np.zeros(seg_time[:-1].shape)))
        vel_stack = np.zeros((3,blks_vel.shape[0],blks_vel.shape[1]))
        vel_stack[0,:,:] = blks_vel
        vel_stack[1,:,:] = blks_vel
        vel_stack[2,:,:] = blks_vel

        m = len(path)-1
        Avel = np.zeros((3,2*m,4*m))

        for idx in range(len(path)-2):
            i = 2*idx
            if idx ==0:
                Avel[:,i:i+2,-4*(idx+1):] = vel_stack[:,i:i+2,:]
            elif idx == len(path)-1:
                Avel[:,i:,-4*(idx+1):-4*(idx+1)+4] = vel_stack[:,i:,:]
            else:
                Avel[:,i:i+2,-4*(idx+1):-4*(idx+1)+4] = vel_stack[:,i:i+2,:] 

        Avel = Avel[:,1:,:]
        Avel = Avel[:,1::2,:] - Avel[:,0:-1:2,:]
        Avel[:,-1,2] = 1 

        Avel_end = np.zeros((3,1,Avel.shape[2]))
        Avel_end[:,0,0] = 3*seg_time[-1]**2
        Avel_end[:,0,1] = 2*seg_time[-1]
        Avel_end[:,0,2] = 1
        Avel_start = np.zeros((3,1,Avel.shape[2]))
        Avel_start[:,0,-2] = 1



        # Accleration Constraints
        blks_acc = np.hstack((6*seg_time[:-1]**1,2*seg_time[:-1]**0,np.zeros(seg_time[:-1].shape),np.zeros(seg_time[:-1].shape)))
        acc_stack = np.zeros((3,blks_acc.shape[0],blks_acc.shape[1]))
        acc_stack[0,:,:] = blks_acc
        acc_stack[1,:,:] = blks_acc
        acc_stack[2,:,:] = blks_acc
        m = len(path)-1
        Aacc = np.zeros((3,2*m,4*m))
        for idx in range(len(path)-2):
            i = 2*idx
            if idx ==0:
                Aacc[:,i:i+2,-4*(idx+1):] = acc_stack[:,i:i+2,:]
            elif idx == len(path)-1:
                Aacc[:,i:,-4*(idx+1):-4*(idx+1)+4] = acc_stack[:,i:,:]
            else:
                Aacc[:,i:i+2,-4*(idx+1):-4*(idx+1)+4] = acc_stack[:,i:i+2,:] 

        Aacc = Aacc[:,1:,:]
        Aacc = Aacc[:,1::2,:] - Aacc[:,0:-1:2,:]
        # Aacc[:,-1,5] = 2

        # Aacc_end = np.zeros((3,1,Aacc.shape[2]))
        # Aacc_end[:,0,0] = 6*seg_time[-1]
        # Aacc_end[:,0,1] = 2
        # Aacc_start = np.zeros((3,1,Aacc.shape[2]))
        # Aacc_start[:,0,-3] = 2

        # A = np.zeros((3,4*len(path),4*len(path)))

        A = np.concatenate((Apos,Avel,Aacc,Avel_end,Avel_start),axis=1)
        # print(np.linalg.matrix_rank(A))
        B = np.zeros((3,4*m,1))
        B[:,:2*(len(path)-1):2,0] =  waypoint[:,:-1,0]
        B[:,1:2*(len(path))-1:2,0]  =  waypoint[:,1:,0]
        K = np.matmul(np.linalg.inv(A),B)

        # p0 = plt.figure(0)
        # plt.imshow(K[0,:,:])
        # # plt.imshow(A[0,:,:])
        # p1 = plt.figure(1)
        # plt.imshow(K[1,:,:])
        # # plt.imshow(A[1,:,:])
        # p2 = plt.figure(2)
        # plt.imshow(K[2,:,:])
        # # plt.imshow(A[2,:,:])
        # plt.show()

        return K

    def get_coeff_5order(self,time,path):

        waypoint = np.zeros((3,len(path),1))
        waypoint[0,:,:] = path[:,0].reshape(len(path),1)
        waypoint[1,:,:] = path[:,1].reshape(len(path),1)
        waypoint[2,:,:] = path[:,2].reshape(len(path),1)

        td = np.diff(time)
        td = td.reshape(len(td),1)
        
        i = np.array([0,1]).reshape(1,2)
        seg_time = td*i
        seg_time = seg_time.reshape(2*len(seg_time),1)

        # Position Constraints --> End points      (2m) 
        blks_pos = np.hstack((seg_time**5,seg_time**4,seg_time**3,seg_time**2,seg_time**1,seg_time**0))
        pos_blks = np.zeros((3,blks_pos.shape[0],blks_pos.shape[1]))
        pos_blks[0,:,:] = blks_pos
        pos_blks[1,:,:] = blks_pos
        pos_blks[2,:,:] = blks_pos
        m = len(path)-1
        Apos = np.zeros((3,2*m,6*m))
        for idx in range(m):
            i = 2*idx
            if idx ==0:
                Apos[:,i:i+2,-6*(idx+1):] = pos_blks[:,i:i+2,:]
            elif idx == m-1:
                Apos[:,i:,-6*(idx+1):-6*(idx+1)+6] = pos_blks[:,i:,:]
            else:
                Apos[:,i:i+2,-6*(idx+1):-6*(idx+1)+6] = pos_blks[:,i:i+2,:]

        # Velocity Constraints --> Continuity (m-1)
              
        blks_vel = np.hstack((5*seg_time[:-1]**4, 4*seg_time[:-1]**3, 3*seg_time[:-1]**2, 2*seg_time[:-1]**1, seg_time[:-1]**0, np.zeros(seg_time[:-1].shape)))
        vel_stack = np.zeros((3,blks_vel.shape[0],blks_vel.shape[1]))
        vel_stack[0,:,:] = blks_vel
        vel_stack[1,:,:] = blks_vel
        vel_stack[2,:,:] = blks_vel

        m = len(path)-1
        Avel = np.zeros((3,2*m,6*m))

        for idx in range(len(path)-2):
            i = 2*idx
            if idx ==0:
                Avel[:,i:i+2,-6*(idx+1):] = vel_stack[:,i:i+2,:]
            elif idx == len(path)-1:
                Avel[:,i:,-6*(idx+1):-6*(idx+1)+6] = vel_stack[:,i:,:]
            else:
                Avel[:,i:i+2,-6*(idx+1):-6*(idx+1)+6] = vel_stack[:,i:i+2,:] 

        Avel = Avel[:,1:,:]
        Avel = Avel[:,1::2,:] - Avel[:,0:-1:2,:]
        Avel[:,-1,2] = 1 

        # Velocity constraints at start and end points are zero (2)
        Avel_end = np.zeros((3,1,Avel.shape[2]))
        Avel_end[:,0,0] = 5*seg_time[-1]**4
        Avel_end[:,0,1] = 4*seg_time[-1]**3
        Avel_end[:,0,2] = 3*seg_time[-1]**2
        Avel_end[:,0,3] = 2*seg_time[-1]**1
        Avel_end[:,0,4] = 1

        Avel_start = np.zeros((3,1,Avel.shape[2]))
        Avel_start[:,0,-2] = 1



        # Accleration Constraints --> Continuity (m-1)
        blks_acc = np.hstack((20*seg_time[:-1]**3,12*seg_time[:-1]**2,6*seg_time[:-1]**1,2*seg_time[:-1]**0,np.zeros(seg_time[:-1].shape),np.zeros(seg_time[:-1].shape)))
        acc_stack = np.zeros((3,blks_acc.shape[0],blks_acc.shape[1]))
        acc_stack[0,:,:] = blks_acc
        acc_stack[1,:,:] = blks_acc
        acc_stack[2,:,:] = blks_acc
        m = len(path)-1
        Aacc = np.zeros((3,2*m,6*m))
        for idx in range(len(path)-2):
            i = 2*idx
            if idx ==0:
                Aacc[:,i:i+2,-6*(idx+1):] = acc_stack[:,i:i+2,:]
            elif idx == len(path)-1:
                Aacc[:,i:,-6*(idx+1):-6*(idx+1)+6] = acc_stack[:,i:,:]
            else:
                Aacc[:,i:i+2,-6*(idx+1):-6*(idx+1)+6] = acc_stack[:,i:i+2,:] 

        Aacc = Aacc[:,1:,:]
        Aacc = Aacc[:,1::2,:] - Aacc[:,0:-1:2,:]

        # Acceleration constraints at Start and End are zero (2)
        Aacc_start = np.zeros((3,1,Avel.shape[2]))
        Aacc_start[:,0,-3] = 2
        
        Aacc_end = np.zeros((3,1,Avel.shape[2]))
        Aacc_end[:,0,0] = 20*seg_time[-1]**3
        Aacc_end[:,0,1] = 12*seg_time[-1]**2
        Aacc_end[:,0,2] = 6*seg_time[-1]
        Aacc_end[:,0,3] = 2


        # Jerk Constraints --> Continuity (m-1)
        blks_jerk = np.hstack((60*seg_time[:-1]**2,24*seg_time[:-1]**1,6*seg_time[:-1]**0,np.zeros(seg_time[:-1].shape),np.zeros(seg_time[:-1].shape),np.zeros(seg_time[:-1].shape)))
        jerk_stack = np.zeros((3,blks_acc.shape[0],blks_acc.shape[1]))
        jerk_stack[0,:,:] = blks_jerk
        jerk_stack[1,:,:] = blks_jerk
        jerk_stack[2,:,:] = blks_jerk
        m = len(path)-1
        Ajerk = np.zeros((3,2*m,6*m))
        for idx in range(len(path)-2):
            i = 2*idx
            if idx ==0:
                Ajerk[:,i:i+2,-6*(idx+1):] = jerk_stack[:,i:i+2,:]
            elif idx == len(path)-1:
                Ajerk[:,i:,-6*(idx+1):-6*(idx+1)+6] = jerk_stack[:,i:,:]
            else:
                Ajerk[:,i:i+2,-6*(idx+1):-6*(idx+1)+6] = jerk_stack[:,i:i+2,:] 

        Ajerk = Ajerk[:,1:,:]
        Ajerk = Ajerk[:,1::2,:] - Ajerk[:,0:-1:2,:]

        # Snap Constraints --> Continuity (m-1)
        blks_snap = np.hstack((120*seg_time[:-1]**1,24*seg_time[:-1]**0,np.zeros(seg_time[:-1].shape),np.zeros(seg_time[:-1].shape),np.zeros(seg_time[:-1].shape),np.zeros(seg_time[:-1].shape)))
        snap_stack = np.zeros((3,blks_snap.shape[0],blks_snap.shape[1]))
        snap_stack[0,:,:] = blks_snap
        snap_stack[1,:,:] = blks_snap
        snap_stack[2,:,:] = blks_snap
        m = len(path)-1
        Asnap = np.zeros((3,2*m,6*m))
        for idx in range(len(path)-2):
            i = 2*idx
            if idx ==0:
                Asnap[:,i:i+2,-6*(idx+1):] = snap_stack[:,i:i+2,:]
            elif idx == len(path)-1:
                Asnap[:,i:,-6*(idx+1):-6*(idx+1)+6] = snap_stack[:,i:,:]
            else:
                Asnap[:,i:i+2,-6*(idx+1):-6*(idx+1)+6] = snap_stack[:,i:i+2,:] 

        Asnap = Asnap[:,1:,:]
        Asnap = Asnap[:,1::2,:] - Asnap[:,0:-1:2,:]
        

        # Total Constraints 2m + 4(m-1) + 4 = 6m 
        A = np.concatenate((Apos,Avel,Aacc,Ajerk,Asnap,Avel_start,Aacc_end,Avel_end,Aacc_start),axis=1)
        # print(np.linalg.matrix_rank(Aacc_end))
        for i in range(A[0,:,:].shape[1]):
            print(i,np.linalg.matrix_rank(A[0,:i+1,:]))
        # print(A[0,:i+1,:])
        # print(np.linalg.matrix_rank(A),A[0,:,:])
        B = np.zeros((3,6*m,1))
        B[:,:2*(len(path)-1):2,0]   =  waypoint[:,:-1,0]
        B[:,1:2*(len(path))-1:2,0]  =  waypoint[:,1:,0]
        K = np.matmul(np.linalg.inv(A),B)

        # p0 = plt.figure(0)
        # # plt.imshow(K[0,:,:])
        # plt.imshow(A[0,:,:])
        # p1 = plt.figure(1)
        # # plt.imshow(K[1,:,:])
        # plt.imshow(A[1,:,:])
        # p2 = plt.figure(2)
        # # plt.imshow(K[2,:,:])
        # plt.imshow(A[2,:,:])
        # plt.show()

        return K

    def get_time_list(self,v_av,path,path_length):
        Time_list = np.zeros((len(path)-1,1))
        for i in range(len(path)-1):
            d = path[i+1] - path[i]
            del_t = abs(d/v_av)
            del_t = np.max(del_t)
            # if np.linalg.norm(d) < path_length/len(path):
            #     del_t = del_t + 0.075
            Time_list[i] = del_t
            # print('!')

        Time_list = np.append(0,Time_list)
        Time_list = np.cumsum(Time_list)

        return Time_list



    