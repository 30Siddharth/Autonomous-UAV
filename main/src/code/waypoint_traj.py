import numpy as np
import math as m

class WorldTraj(object):
    """

    """
    def __init__(self, points):
        """
        This is the constructor for the Trajectory object. A fresh trajectory
        object will be constructed before each mission. For a waypoint
        trajectory, the input argument is an array of 3D destination
        coordinates. You are free to choose the times of arrival and the path
        taken between the points in any way you like.

        You should initialize parameters and pre-compute values such as
        polynomial coefficients here.

        Inputs:
            points, (N, 3) array of N waypoint coordinates in 3D
        """

        # STUDENT CODE HERE
        self.points = points
        

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
        x        = np.zeros((3,))
        x_dot    = np.zeros((3,))
        x_ddot   = np.zeros((3,))
        x_dddot  = np.zeros((3,))
        x_ddddot = np.zeros((3,))
        yaw = 0
        yaw_dot = 0

        # STUDENT CODE HERE
        print('t',t)

        '''
        I first store all the points in an array. 
        I make another array of time stamps, each spaced at some predefined $\delta t$. 
        Then I slice this new vector of time stamps and choose only the points corresponding to the points. This way each trajectory between any two adjacent point is fixed.
        Then we compute the velocity vector between any two adjacent point.
        We use linear interpolation to travel between these points.

        Advantages: 
        -- The quadrotor will always be facing in the right direction. 
        -- The farther the points, the higher the speed

        Disadvantages:
        -- If the points are close enough then this might be slow
        '''
        

        l = len(self.points)
        # del_t = 2
        # Time = np.arange(0,60,del_t)
        # Time = Time[:l]
        

        Speed = 2
        Dis_diff = np.diff(self.points, axis=0)
        # Sign = np.sign(Dis_diff)
        # # Velocity = np.array([np.multiply(Speed,Sign[i]) for i in range(len(Sign))])
        normx = np.array([np.linalg.norm(Dis_diff[i]) for i,_ in enumerate(Dis_diff)])
        l = 1/normx
        dir_cos = (Dis_diff.T*l).T
        Velocity = dir_cos*Speed
        normv = np.linalg.norm(Velocity, axis = 1)
        delTime = normx/normv
        Time = np.array([np.sum(delTime[:i]) for i in range(len(delTime)+1)])
        # print('TIme Differences',delTime)
        # print('Time', Time)
        # print('Velocity', Velocity)
        # print('Points', self.points)
        # print('Displacemnt', Dis_diff)

        pos = 0

        for i in  range(len(Time)-1):
            if t == np.inf:
                x = self.points[-1]
            elif Time[i] < t <= Time[i+1]:
                pos = i
                x_dot = Velocity[pos]
                x = x_dot*(t-Time[i]) + self.points[pos]
                print('X--> ',x)
                print('V--> ',x_dot)           
            elif Time[-1] <= t:
                x = self.points[-1]
            else:
                pass
        print('X--> ',x)
        # print('V--> ',x_dot)

        flat_output = { 'x':x, 'x_dot':x_dot, 'x_ddot':x_ddot, 'x_dddot':x_dddot, 'x_ddddot':x_ddddot,
                        'yaw':yaw, 'yaw_dot':yaw_dot}
        return flat_output

        #  l = np.array([np.sum(time[:i]) for i in range(len(time)+1)])
        # sp = np.array([np.multiply(speed,sign[i]) for i in range(len(sign))])




    # def line(self,t1,t2,z1,z2):
    #     a = (z2-z1)/(t2-t1)
    #     b = z1 - t1*a

    #     return a,b
                

        # t0 = 0.1
        # if t < t0:
        #     x = self.points[0]
        #     if l > 1:
        #         if self.points[-1]==self.points[0] and self.points[0]==self.points[1]:
        #             x = self.points[0]
        #         else:
        #             x = self.points[0]

        # elif t0 <= t < t0 + 3:
        #     if self.points[0]==self.points[1] and self.points[1]==self.points[2]:
        #             x = self.points[1]
        #     else:
        #         a,b = self.line(t0,t0+3,self.points[0], self.points[1])
        #         x_dot = a
        #         x = a*t + b
        #         print('X', x)
        #         print('vel', x_dot)

        # elif t0 + 3 <= t < t0 + 6:
        #     if self.points[1]==self.points[2] and self.points[2]==self.points[3]:
        #             x = self.points[1]
        #     else:
        #         a,b = self.line(t0,t0+3,self.points[0], self.points[1])
        #         x_dot = a
        #         x = a*t + b
        #         print('X', x)
        #         print('vel', x_dot)


        # elif t0 + 6 <= t < t0 + 9:
        #     if self.points[2]==self.points[3] and self.points[3]==self.points[4]:
        #             x = self.points[1]
        #     else:
        #         a,b = self.line(t0,t0+3,self.points[0], self.points[1])
        #         x_dot = a
        #         x = a*t + b
        #         print('X', x)
        #         print('vel', x_dot)













        # elif t0 + 3 <= t < t0 + 6:
        #     a,b = self.line(t0+3,t0+6,self.points[1], self.points[2])
        #     x_dot = a
        #     x = a*t + b
        #     print('X', x)
        #     print('vel', x_dot)
        # elif t0+6 <= t < t0+9:
        #     a,b = self.line(t0+6,t0+9,self.points[2], self.points[3])
        #     x_dot = a
        #     x = a*t + b
        #     print('X', x)
        #     print('vel', x_dot)
        # else:
        #     x = self.points[-1]
        #     print('X', x)
        #     print('vel', x_dot)

        # points = self.points
        # l = len(points)
        # i = -l
        
        # if i < -1:
        #     print(points[i])
        #     i = i+1  


