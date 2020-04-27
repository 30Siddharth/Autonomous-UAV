from heapq import heappush, heappop, heapify # Recommended.
import numpy as np
import time
from flightsim.world import World
from flightsim.axes3ds import Axes3Ds
import matplotlib.pyplot as plt
# from proj1_3.code.occupancy_map import OccupancyMap # Recommended.
from proj1_3.code.occupancy_map import OccupancyMap # Recommended.

def graph_search(world, resolution, margin, start, goal, astar):
	"""
	Parameters:
		world,      World object representing the environment obstacles
		resolution, xyz resolution in meters for an occupancy map, shape=(3,)
		margin,     minimum allowed distance in meters from path to obstacles.
		start,      xyz position in meters, shape=(3,)
		goal,       xyz position in meters, shape=(3,)
		astar,      if True use A*, else use Dijkstra
	Output:
		path,       xyz position coordinates along the path in meters with
					shape=(N,3). These are typically the centers of visited
					voxels of an occupancy map. The first point must be the
					start and the last point must be the goal. If no path
					exists, return None.
	"""

	# While not required, we have provided an occupancy map you may use or modify.
	occ_map = OccupancyMap(world, resolution, margin)
	# Retrieve the index in the occupancy grid matrix corresponding to a position in space.
	start_index = tuple(occ_map.metric_to_index(start))
	goal_index = tuple(occ_map.metric_to_index(goal))
	print('Start:', start_index)
	print('Goal:', goal_index)

	Cost_dic = {}
	Parent_dic = dict()
	# start_time = time.time()
	for i,_ in enumerate(occ_map.map[:]):
		for j,_ in enumerate(occ_map.map[i][:]):
			for k,_ in enumerate(occ_map.map[i][j][:]):
				if occ_map.map[i][j][k] == False:   # Only consider points where there are no obstacles
					parent = {tuple(np.array([i,j,k])):tuple(np.array([0,0,0]))}
					Parent_dic.update(parent)   # Dictionary of Parents
					if tuple(np.array([i,j,k])) != start_index:
						cost = {tuple(np.array([i,j,k])): np.inf}
					else:
						cost = {tuple(np.array([i,j,k])): 0}
					Cost_dic.update(cost)       # Dictionary of costs

	heap = []
	

	node = 0
	
	heappush(heap,[Cost_dic[start_index],start_index])
	min_cost = heappop(heap)  #<---------------<<<<< Reuse this
	
	# ---------- Done with intialization ------------------
	goal_flag = 0
	impossible = 0
	flag = True
	# while heap and min_cost[0] < np.inf:
	

	if goal_index in Cost_dic.keys():
		while goal_flag == 0:
		# while heap:
			U = min_cost[1]
			neigh = get_neighbour(U)
			neigh_idx = 0
			no_neigbour = 0
			g = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]  # <-------<< List to store cost of all neigbours
			h = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]  # <-------<< Heuristics
			f = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]  # <-------<< Varibale 
			while neigh and neigh_idx < len(neigh):
				no_neigbour += 1
				if neigh[neigh_idx] in Cost_dic.keys():
					# A* in action                
					if astar == True:   #<----------------------<< Checking for A* flag
				
						if 0<= neigh_idx <18:
							g[neigh_idx] = Cost_dic[U] + 1.414
							h[neigh_idx] = np.linalg.norm(np.asarray(goal_index)- np.asarray(neigh[neigh_idx]))
							f[neigh_idx] = g[neigh_idx] + h[neigh_idx]
						elif neigh_idx in [19,20,22,23]:
							g[neigh_idx] = Cost_dic[U] + 1.414
							h[neigh_idx] = np.linalg.norm(np.asarray(goal_index)- np.asarray(neigh[neigh_idx]))
							f[neigh_idx] = g[neigh_idx] + h[neigh_idx] 
						else:
							g[neigh_idx] = Cost_dic[U] + 1
							h[neigh_idx] = np.linalg.norm(np.asarray(goal_index)- np.asarray(neigh[neigh_idx]))
							f[neigh_idx] = g[neigh_idx] + h[neigh_idx]
						
						if neigh[neigh_idx] == goal_index:
							print('Reached Goal')
							goal_flag = 1
						if g[neigh_idx] < Cost_dic[neigh[neigh_idx]]:
							Cost_dic[neigh[neigh_idx]] = g[neigh_idx]
							heappush(heap,[f[neigh_idx],neigh[neigh_idx]])
							Parent_dic[neigh[neigh_idx]] = U
              						
					# Dijkstra in action
					else:

						if 0 <= neigh_idx <18:
							g[neigh_idx] = Cost_dic[U] + 5
						elif neigh_idx in [19,20,22,23]:
							g[neigh_idx] = Cost_dic[U] + 1.5
						else:
							g[neigh_idx] = Cost_dic[U] + 1   

						if neigh[neigh_idx] == goal_index:
							print('Reached Goal')
							goal_flag = 1
						if g[neigh_idx] < Cost_dic[neigh[neigh_idx]]:   
							Cost_dic[neigh[neigh_idx]] = g[neigh_idx]
							heappush(heap,[Cost_dic[neigh[neigh_idx]],neigh[neigh_idx]])
							Parent_dic[neigh[neigh_idx]] = U

				neigh_idx += 1  # <-------------<< Goto next Neighbour
				
			if impossible != 1 and heap ==[]:
				impossible = 1
				flag = False
				break

			min_cost = heappop(heap)  #<---------------<<<<< Reuse this
			node += 1
			# print('Cost to go: ', min_cost[0])
			

		print('Nodes: ', node)
		path = []   
		point = goal_index

		while flag is True:   # <-------------------<< Create the path list
			mtpoint = occ_map.index_to_metric_center(Parent_dic[point])
			# mtpoint = occ_map.index_to_metric_negative_corner(Parent_dic[point])
			path.append(mtpoint)
			point = Parent_dic[point]
			if point == start_index:
				flag = False
		if impossible == 0:      
			path.append(start)  
			path.reverse()
			path.append(goal)
			# k = path.sort()
			path = np.asarray(path)            
			length = float(round(np.sum(np.linalg.norm(np.diff(path, axis=0),axis=1)),3))
			print('Path Length: ', length)
		elif impossible == 1:
			path = None

		sparse_path = get_path(path,goal,length)
		# sparse_path = path
		return sparse_path

	else:
		goal_flag == 1
		path = []
		path = [start_index]
		path = np.asarray(path)
		print('Goal Not Reachable')

		return  None


def get_neighbour(U):
	up1     = tuple(np.array([U[0]+1,U[1]+1,U[2]+1]))
	up2     = tuple(np.array([U[0]+1,U[1],U[2]+1]))
	up3     = tuple(np.array([U[0]+1,U[1]-1,U[2]+1]))
	up4     = tuple(np.array([U[0]-1,U[1]+1,U[2]+1]))
	up5     = tuple(np.array([U[0]-1,U[1],U[2]+1]))
	up6     = tuple(np.array([U[0]-1,U[1]-1,U[2]+1]))
	up7     = tuple(np.array([U[0],U[1]+1,U[2]+1]))
	up8     = tuple(np.array([U[0],U[1],U[2]+1]))
	up9     = tuple(np.array([U[0],U[1]-1,U[2]+1]))

	down1   = tuple(np.array([U[0]+1,U[1]+1,U[2]-1]))
	down2   = tuple(np.array([U[0]+1,U[1],U[2]-1]))
	down3   = tuple(np.array([U[0]+1,U[1]-1,U[2]-1]))
	down4   = tuple(np.array([U[0]-1,U[1]+1,U[2]-1]))
	down5   = tuple(np.array([U[0]-1,U[1],U[2]-1]))
	down6   = tuple(np.array([U[0]-1,U[1]-1,U[2]-1]))
	down7   = tuple(np.array([U[0],U[1]+1,U[2]-1]))
	down8   = tuple(np.array([U[0],U[1],U[2]-1]))
	down9   = tuple(np.array([U[0],U[1]-1,U[2]-1]))    

	
	left     = tuple(np.array([U[0]+1,U[1],U[2]]))
	leftd1   = tuple(np.array([U[0]+1,U[1]+1,U[2]]))
	leftd2   = tuple(np.array([U[0]+1,U[1]-1,U[2]]))
	right    = tuple(np.array([U[0]-1,U[1],U[2]]))
	rightd1  = tuple(np.array([U[0]-1,U[1]+1,U[2]]))
	rightd2  = tuple(np.array([U[0]-1,U[1]-1,U[2]]))
	front    = tuple(np.array([U[0],U[1]+1,U[2]]))
	back     = tuple(np.array([U[0],U[1]-1,U[2]]))

	neigh = [up1,up2,up3, up4,up5,up6,up7,up8,up9,down1,down2,down3,down4,down5,down6,down7,down8,down9,left,leftd1,leftd2,right,rightd1,rightd2,front,back]
	return neigh


def get_path(points,goal,path_length):
		# points = self.points
		path = points[0].reshape(1,3)
		# path = np.append(path[:,:],goal.reshape(1,3),axis=0)
		goal_flag = False 
		j = 0   
		while goal_flag is False:
			for i in range(len(points)-2):
				if all(points[i-j+2] == goal):   
					goal_flag = True
					l1 = (path[-1] - points[-2].reshape(1,3))
					length1 = float(round(np.sum(np.linalg.norm(l1)),3))
					if length1 > path_length/10:
						path = np.append(path,(path[-1] + points[-2].reshape(1,3))/2,axis=0)
						path = np.append(path[:,:],goal.reshape(1,3),axis=0)
							# print('Long Path',points[i-j+1,:].reshape(1,3),length)
					elif length1 < path_length/25:
							# path = np.append(path,points[i-j+1,:].reshape(1,3),axis=0)
						print('Very Short',points[i-j+1,:].reshape(1,3),length)
						path = np.append(path[:,:],goal.reshape(1,3),axis=0)
					else:
						# path = np.append(path,points[i-j+1,:].reshape(1,3),axis=0)
						path = np.append(path,(path[-1] + points[-2].reshape(1,3))/2,axis=0)
						path = np.append(path[:,:],goal.reshape(1,3),axis=0)
				else:
					m1 = points[i-j+1] - points[i-j]
					m2 = points[i-j+2] - points[i-j+1]
					if all(m1==m2) is False:
						l = (path[-1] - points[i-j+1,:].reshape(1,3))
						# print(path[-1])  
						# print (points[i-j+1,:].reshape(1,3))
						# print(l)
						length = float(round(np.sum(np.linalg.norm(l)),3))
						if length > path_length/10:
							path = np.append(path,(path[-1] + points[i-j+1,:].reshape(1,3))/2,axis=0)
							path = np.append(path,(path[-1] + points[i-j+2,:].reshape(1,3))/2,axis=0)
							path = np.append(path,points[i-j+1,:].reshape(1,3),axis=0)
							# print('Long Path',points[i-j+1,:].reshape(1,3),length)
						elif length < path_length/25:
							path = np.append(path,points[i-j+1,:].reshape(1,3),axis=0)
							print('Very Short',points[i-j+1,:].reshape(1,3),length)
						else:
							path = np.append(path,points[i-j+1,:].reshape(1,3),axis=0)
							# print('Short Path',points[i-j+1,:].reshape(1,3),length)
						points = points[i-j+1:,:]
						j = i+1

				
		return path
	





