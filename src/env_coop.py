#!/usr/bin/env python

from threading import Lock
import rospy
import numpy as np
import math
from math import  pi
from geometry_msgs.msg import Twist, Pose
from gazebo_msgs.msg import  ModelState
from gazebo_msgs.srv import SetModelState
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from std_srvs.srv import Empty, EmptyRequest
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from respawnGoal import Respawn
import copy, random
from mpc_filter import filter_mpc
target_not_movable = False



class Env():
    def __init__(self, act_state_dim):
        self.goal_x = 0
        self.goal_y = 0
        self.heading_rob_goal = [0,0,0]
        self.initGoal = True
        self.get_goalbox = False
        self.goal_distance = 0
        self.past_position = [0,0,0]
        self.past_action = np.zeros([3,2])
        self.cols = 0
        self.First_step = True
        self.DataFill = False
        self.DataFill2 = False
        self.odom = [0,0,0]
        self.goals_cnt  =0
        
        self.act_state_dim = act_state_dim
        self.time_odom0_new = 0
        self.scan0 = None
        self.mutex = Lock()
        self.mutex2 = Lock()
        self.mutex1 = Lock()
        self.mutex3 = Lock()
        self.pub_cmd_vel = [0, 0, 0]
        self.sub_odom = [0, 0, 0]
        self.sub_scan = [0, 0, 0]
        self.min_scan = 0
        self.time_scan0  = 0
        self.time_odom = 0
        self.reward_dists = 0
        self.reward_goal_dist = 0   
        self.reward_mpc = 0

        self.past_distance = 20
        self.stopped = [0, 0, 0]
        self.deviate = [0, 0, 0]
        self.dist0 = 0
        self.dist1 = 0
        self.dist_goal = 0
        self.rel_ang_robs1 = 0
        self.rel_ang_robs2 = 0
        self.obst_dist = np.zeros(6)
        self.goal_cor = np.zeros(2)
        self.dist_robots = 1.5 # chanage or randomize in reset
        self.action = []
        self.state_actor_scan = [[],[],[]]
        self.position = [Pose().position, Pose().position,Pose().position]
        self.yaw = [0,0,0]
        for i in range(3):
            self.pub_cmd_vel[i] = rospy.Publisher('/tb3_'+str(i)+'/cmd_vel', Twist, queue_size=1)
            self.sub_odom[i] = rospy.Subscriber('/tb3_'+str(i)+'/odom', Odometry, self.getOdometry,i )
        self.sub_scan[0] =  rospy.Subscriber('/tb3_0/scan', LaserScan, self.getData1)
        self.sub_scan[1] =  rospy.Subscriber('/tb3_1/scan', LaserScan, self.getData2)
        self.sub_scan[2] =  rospy.Subscriber('/tb3_2/scan', LaserScan, self.getData3)
        
        self.reset_proxy = rospy.ServiceProxy('gazebo/reset_simulation', Empty)
        self.unpause_proxy = rospy.ServiceProxy('gazebo/unpause_physics', Empty)
        self.pause_proxy = rospy.ServiceProxy('gazebo/pause_physics', Empty)
        self.set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)

        self.respawn_goal = Respawn()
        self.dist_01 = 0
        self.dist_02 = 0
        self.dist_12 = 0
        self.dist_cntr0 = 0
        self.dist_cntr1 = 0
        self.dist_cntr2 = 0
        self.centroid_pos_x = 0
        self.centroid_pos_y = 0
        self.cntr_rewards = 0
        self.prev_robot_dist = [0,0,0]
        
        self.ret_act = []
        self.filter_mpc = [filter_mpc(), filter_mpc(),filter_mpc()]
        #Keys CTRL + c will stop script
        rospy.on_shutdown(self.shutdown)
    
    def RstSubs(self):
        for i in range(3):
            self.pub_cmd_vel[i].unregister()
            self.sub_odom[i].unregister()
            self.sub_scan[i].unregister()
        for i in range(3):
            self.pub_cmd_vel[i] = rospy.Publisher('/tb3_'+str(i)+'/cmd_vel', Twist, queue_size=1)
            self.sub_odom[i] = rospy.Subscriber('/tb3_'+str(i)+'/odom', Odometry, self.getOdometry,i )

        self.sub_scan[0] =  rospy.Subscriber('/tb3_0/scan', LaserScan, self.getData1)
        self.sub_scan[1] =  rospy.Subscriber('/tb3_1/scan', LaserScan, self.getData2)
        self.sub_scan[2] =  rospy.Subscriber('/tb3_2/scan', LaserScan, self.getData3)
    
    
    def shutdown(self):
        #you can stop turtlebot by publishing an empty Twist
        rospy.loginfo("Stopping TurtleBot")
        for i in range(3):
            self.pub_cmd_vel[i].publish(Twist())
            
        rospy.sleep(1)

    def getGoalDistace(self):
        self.past_distance = self.goal_distance
        self.goal_distance = round(math.hypot(self.goal_x - self.centroid_pos_x, self.goal_y - self.centroid_pos_y), 2)

        return self.goal_distance


    def getOdometry(self, odom, ind):
        self.mutex2.acquire()
        self.odom[ind] = odom
        self.mutex2.release()
        if ind ==2:
            self.DataFill2 = True
            self.time_odom = odom.header.stamp

    def getData1(self, scan):
        self.mutex.acquire()
        self.scan0 = scan

        self.mutex.release()
    def getData2(self, scan):
        self.mutex1.acquire()
        self.scan1 = scan
        self.mutex1.release()
    def getData3(self, scan):
        self.mutex3.acquire()
        self.scan2 = scan
        self.mutex3.release()
        self.DataFill = True

    def critic_getState(self):
        self.mutex2.acquire()
        odom = self.odom
        self.mutex2.release()
        for ind in range(3):
            self.past_position[ind] = copy.deepcopy(self.position[ind])
            self.position[ind] = odom[ind].pose.pose.position
            orientation = odom[ind].pose.pose.orientation
            orientation_list = [orientation.x, orientation.y, orientation.z, orientation.w]
            _, _, yaw = euler_from_quaternion(orientation_list)
            self.yaw[ind] = yaw
            robot_goal_angle = math.atan2(self.goal_y - self.position[ind].y, self.goal_x - self.position[ind].x)
            heading_r = robot_goal_angle - self.yaw[ind]
            if heading_r > pi:
                heading_r -= 2 * pi

            elif heading_r < -pi:
                heading_r += 2 * pi

            self.heading_rob_goal[ind] = round(heading_r, 2)
        xs = 0
        ys = 0
        for i in range(3):
            xs += self.position[i].x
            ys += self.position[i].y
        self.centroid_pos_x = xs/3
        self.centroid_pos_y = ys/3

        self.dist_01 = round(math.hypot(self.position[0].x- self.position[1].x, self.position[0].y - self.position[1].y),2)
        self.dist_02 = round(math.hypot(self.position[0].x- self.position[2].x, self.position[0].y - self.position[2].y),2)
        self.dist_12 = round(math.hypot(self.position[1].x- self.position[2].x, self.position[1].y - self.position[2].y),2)
        self.getGoalDistace()
        
        self.mutex.acquire()
        scan0 = self.scan0
        self.mutex.release()
        self.mutex1.acquire()
        scan1 = self.scan1
        self.mutex1.release()
        self.mutex3.acquire()
        scan2 = self.scan2
        self.mutex3.release()
        if self.First_step:
            self.scan_angle_inc = scan0.angle_increment
            self.ang_min = scan0.angle_min
            self.First_step = False
        scan = [scan0.ranges, scan1.ranges, scan2.ranges]
        # print(scan)
        scan_np = np.array(scan)
        scan_np [np.isinf(scan_np)] = 5
        scan_np [np.isnan(scan_np)] = 0
        # print(scan_np)
        self.state_actor_scan[0], self.state_actor_scan[1], self.state_actor_scan[2]= np.split(scan_np,3)
        self.state_actor_scan[0], self.state_actor_scan[1], self.state_actor_scan[2] = \
            self.state_actor_scan[0][0], self.state_actor_scan[1][0], self.state_actor_scan[2][0]           
        
    def getState(self,ind):
        state_actor = []
        heading_r = self.heading_rob_goal[ind]
        min_range = 0.15 # radius of robot
        done = False
        
        if ind == 0:
            self.dist_goal = round(math.hypot(self.position[0].x- self.goal_x, self.position[0].y - self.goal_y),2)

            rel_01 = math.atan2(self.position[1].y - self.position[0].y, self.position[1].x - self.position[0].x)
            rel_01 = rel_01 - self.yaw[ind]
            if rel_01 > pi:
                rel_01 -= 2 * pi
            elif rel_01 < -pi:
                rel_01 += 2 * pi
            self.rel_ang_robs1 = rel_01
            rel_02 = math.atan2(self.position[2].y - self.position[0].y, self.position[2].x - self.position[0].x)
            rel_02 = rel_02 - self.yaw[ind]
            if rel_02 > pi:
                rel_02 -= 2 * pi
            elif rel_02 < -pi:
                rel_02 += 2 * pi
            self.rel_ang_robs2 = rel_02

            self.dist0 = self.dist_01
            self.dist1 = self.dist_02

        elif ind == 1:
            self.dist_goal = round(math.hypot(self.position[1].x- self.goal_x, self.position[1].y - self.goal_y),2)

            rel_10 = math.atan2(self.position[0].y - self.position[1].y, self.position[0].x - self.position[1].x)
            rel_10 = rel_10 - self.yaw[ind]
            if rel_10 >  pi:
                rel_10 -= 2 * pi
            elif rel_10 < -pi:
                rel_10 += 2 * pi
            self.rel_ang_robs1 = rel_10
            rel_12 = math.atan2(self.position[2].y - self.position[1].y, self.position[2].x - self.position[1].x)
            rel_12 = rel_12 - self.yaw[ind]
            if rel_12 > pi:
                rel_12 -= 2 * pi
            elif rel_12 < - pi:
                rel_12 += 2 * pi
            self.rel_ang_robs2 = rel_12

            self.dist0 = self.dist_01
            self.dist1 = self.dist_12
            
        elif ind == 2:
            self.dist_goal = round(math.hypot(self.position[2].x- self.goal_x, self.position[2].y - self.goal_y),2)

            rel_20 = math.atan2(self.position[0].y - self.position[2].y, self.position[0].x - self.position[2].x)
            rel_20 = rel_20 - self.yaw[ind]
            if rel_20 > pi:
                rel_20 -= 2 * pi
            elif rel_20 < - pi:
                rel_20 += 2 * pi
            self.rel_ang_robs1 = rel_20
            rel_21 = math.atan2(self.position[1].y - self.position[2].y, self.position[1].x - self.position[2].x)
            rel_21 = rel_21 - self.yaw[ind]
            if rel_21 > pi:
                rel_21 -= 2 * pi
            elif rel_21 < - pi:
                rel_21 += 2 * pi
            self.rel_ang_robs2 = rel_21

            self.dist0 = self.dist_02
            self.dist1 = self.dist_12

        # Check for collisions ************************************************
        if min_range > min(self.state_actor_scan[ind]) > 0 :
            self.cols += 1
            rospy.loginfo('Robot '+ str(ind)+" Collision!!")
            # print("collisions: ",self.cols)
            done = True


        #  Check for stuck condition *********************************************
        a, b, c, d = round((self.position[ind].x),2), round((self.past_position[ind].x),2), round((self.position[ind].y),2), round((self.past_position[ind].y),2)
        
        if a == b and c == d :
            self.stopped[ind] += 1
            if self.stopped[ind] == 30:
                # rospy.loginfo('Robot '+ str(ind)+ ' stuck')
                self.stopped[ind] = 0
                done = True
        else:
            self.stopped[ind] = 0

        
        #***********************************************************************
        id, self.obst_dist[ind*2], self.obst_dist[ind*2 +1] = self.Closest_obstacle(self.state_actor_scan[ind], ind)
        # distance to nearest obstacle
        min_scan = self.state_actor_scan[ind][id]
        rad = self.scan_angle_inc * id
        if rad > pi:
            rad -= 2 * pi
        elif rad < -pi:
            rad += 2 * pi
        self.min_scan = min(self.state_actor_scan[ind])
        #****** nearest obstacle
        state_actor.append(min_scan)
        state_actor.append(rad)
        #****** neighbour robot 1
        state_actor.append(self.dist0)
        state_actor.append(self.rel_ang_robs1)
        #****** neighbour robot 2
        state_actor.append(self.dist1)
        state_actor.append(self.rel_ang_robs2)
        #****** distance and angle to goal
        state_actor.append(self.dist_goal)
        state_actor.append(heading_r)
        #****** past action
        past_v_norm , past_w_norm = self.Normalize_act(self.past_action[ind][0], self.past_action[ind][1])
        state_actor.append(past_v_norm)
        state_actor.append(past_w_norm)
        state_actor.append(self.goal_distance)
        state_actor.append(self.dist_robots)


        # self.obst_dist[ind*2] = self.position[ind].x + min_scan* math.cos(rad) # x of nearest obstacle
        # self.obst_dist[ind*2+1] = self.position[ind].y + min_scan* math.sin(rad)# y of nearest obstacle

        if self.goal_distance < 0.2:
            rospy.loginfo("Goal!!")
            self.get_goalbox = True

        return state_actor, done

    def Closest_obstacle(self, scan, ind):
        agents = list(range(3))
        agents.remove(ind)
        sorted_scan = np.sort(scan, kind='mergesort')
        rot_mat = np.array([[math.cos(self.yaw[ind]), -math.sin(self.yaw[ind]), self.position[ind].x],
                           [math.sin(self.yaw[ind]), math.cos(self.yaw[ind]), self.position[ind].y],
                           [0, 0, 1]])
        for id, value in enumerate(sorted_scan):
            indx = scan.tolist().index(value)
            rad = self.scan_angle_inc * indx
            vec_loc = np.array([value * math.cos(rad), value * math.sin(rad), 1])
            vec_rot = rot_mat @ vec_loc
            obst_dist_x = vec_rot[0]
            obst_dist_y = vec_rot[1]
            dist_ibst_n1 = round(math.hypot(obst_dist_x - self.position[agents[0]].x, obst_dist_y - self.position[agents[0]].y), 2)
            dist_ibst_n2 = round(math.hypot(obst_dist_x - self.position[agents[1]].x, obst_dist_y - self.position[agents[1]].y), 2)
            if dist_ibst_n1 > 0.2 and dist_ibst_n2 > 0.2:      
                break
        return indx, obst_dist_x, obst_dist_y

    def Normalize_act(self, vel, w, max1 = 0.75, min1 = 0, max2 = 0.75, min2 = -0.75):
        return (vel-min1)/(max1-min1), (w-min2)/(max2-min2)

    def setReward(self, done, ind):
        # *******************dist to neighbours **********************
        if self.dist0 <= (self.dist_robots-0.05) or self.dist0 >= (self.dist_robots+0.05):
            r1 = - 2.0* abs(self.dist0 - self.dist_robots)
        else:
            r1 = 0
        if self.dist1 <= (self.dist_robots-0.05) or self.dist1 >= (self.dist_robots+0.05):
            r2 = -  2.0* abs(self.dist1 - self.dist_robots)
        else:
            r2 = 0
      
        r3 =  -3.5* self.goal_distance 
        
        
        r4 = -5 * abs(self.action[ind][0] - self.ret_act[ind][0])
        r5 = -5 * abs(self.action[ind][1] - self.ret_act[ind][1])


        reward =  r1 + r2 + r3 + r4 + r5

        self.reward_dists = r1 + r2
        self.reward_goal_dist = r3  
        self.reward_mpc = r4 + r5
#********************************************************

        if done:
            done = True
            reward = -7000
            for i in range(3):
                self.pub_cmd_vel[i].publish(Twist())

        if self.get_goalbox:
            reward = 300
            for i in range(3):
                self.pub_cmd_vel[i].publish(Twist())
            self.cntr_rewards+=1
            if self.cntr_rewards==3:
                self.get_goalbox = False
                self.cntr_rewards = 0
                self.goals_cnt += 1
                self.goal_x, self.goal_y = self.respawn_goal.getPosition( delete=True)            


        return reward, done

    def step(self, action, past_action):
        states = np.zeros([3,self.act_state_dim])
        dones = [0,0,0]
        rewards = [0,0,0]
        self.ret_act = []
        self.past_action = past_action
        self.action = action
        pos1x = 0
        pos1y = 0
        pos2x = 0
        pos2y = 0
        for i in range(3):
            if i == 0:
                pos1x = self.position[1].x
                pos1y = self.position[1].y
                pos2x = self.position[2].x
                pos2y = self.position[2].y
            if i == 1:
                pos1x = self.position[0].x
                pos1y = self.position[0].y
                pos2x = self.position[2].x
                pos2y = self.position[2].y
            if i == 2:
                pos1x = self.position[0].x
                pos1y = self.position[0].y
                pos2x = self.position[1].x
                pos2y = self.position[1].y
            curr_state_np = np.array([round(self.position[i].x,2), round(self.position[i].y,2), round(self.yaw[i], 2)])
            filter_mpc = self.filter_mpc[i].run_mpc(action[i], curr_state_np, self.obst_dist[2*i],
                                self.obst_dist[2*i+1], pos1x, pos1y, pos2x, pos2y, i)
            self.action_mpc = [filter_mpc[0], filter_mpc[1]]
            self.ret_act.append(np.asarray(self.action_mpc))

        for i in range(3):
            vel_cmd = Twist()
            vel_cmd.linear.x = self.ret_act[i][0]
            vel_cmd.angular.z =  self.ret_act[i][1]
            self.pub_cmd_vel[i].publish(vel_cmd)
            
        self.critic_getState()
        
        norm_scans = []
        rl_act_norm = self.ret_act ## just for dimensions

        for i in range(3):
            state_act, done = self.getState(i)
            reward, done = self.setReward(done,i)
            states[i] = np.asarray(state_act)
            dones[i] = done
            rewards[i] = reward
            norm_scans.append(self.state_actor_scan[i]/5.0)
            rl_act_norm[i][0] , rl_act_norm[i][1] = self.Normalize_act(self.action[i][0],self.action[i][1])
        dones = np.asarray(dones)
        state_critic = np.reshape(states, (1,-1))
        form_error = abs(self.dist_01 - self.dist_robots) + abs(self.dist_02 - self.dist_robots) + abs(self.dist_12 - self.dist_robots)
        v_error = sum(abs(self.ret_act[0] - self.action[0]))/3.0
        w_error = sum(abs(self.ret_act[1] - self.action[1]))/3.0
        return state_critic, states, rewards, dones, norm_scans, form_error, self.ret_act, rl_act_norm, \
            v_error, w_error, self.cols, self.goals_cnt, self.reward_dists, self.reward_goal_dist, self.reward_mpc

    def reset(self):
        rospy.wait_for_service('gazebo/reset_simulation')
        try:
            self.reset_proxy()
        except (rospy.ServiceException) as e:
            print("gazebo/reset_simulation service call failed")

        self.reward_dists = 0
        self.reward_goal_dist = 0
        self.reward_mpc = 0

        self.goals_cnt = 0
        self.dist_robots = random.choice([1.0, 1.5, 1.25])
        limits = [0.5,4.0,0.5,4.0 , 0.5,4.0,-0.5,-4.0 , -0.5,-4.0,-0.5,-4.0]
        thetas = [0, pi/2, -pi/2, pi]
        self.pause_proxy(EmptyRequest())    
        for i in range(3):
            state = ModelState()
            state.model_name = "tb3_"+str(i)
            state.reference_frame = "world"
            state.pose.position.x = np.random.uniform(limits[0+i*4],limits[1+i*4])
            state.pose.position.y = np.random.uniform(limits[2+i*4],limits[3+i*4])
            state.pose.position.z = 0
            quaternion = quaternion_from_euler(0, 0, thetas[np.random.randint(0,3)])
            state.pose.orientation.x = quaternion[0]
            state.pose.orientation.y = quaternion[1]
            state.pose.orientation.z = quaternion[2]
            state.pose.orientation.w = quaternion[3]
            rospy.wait_for_service('/gazebo/set_model_state')
            try:
                set_state = self.set_state
                result = set_state(state)
                assert result.success is True
            except rospy.ServiceException:
                print("/gazebo/get_model_state service call failed") 
        self.unpause_proxy(EmptyRequest()) 

        self.RstSubs()
        state_act = np.zeros([3,self.act_state_dim])
        self.stopped = [0, 0, 0]
        self.deviate = [0, 0, 0]
        self.action = np.zeros([3,2])
        self.ret_act = np.zeros([3,2])
        if self.initGoal:
            self.goal_x, self.goal_y = self.respawn_goal.getPosition()
            self.initGoal = False
        else:
            self.goal_x, self.goal_y = self.respawn_goal.getPosition(delete=True)
        self.goal_cor[0] = self.goal_x
        self.goal_cor[1] = self.goal_y
        self.past_action = np.zeros([3,2])
       
        while not self.DataFill:
           rospy.sleep(0.1)
        while not self.DataFill2:
           rospy.sleep(0.1)
        self.critic_getState()
        norm_scans = []

        for ind in range(3):
            state_act_single, _ = self.getState(ind)
            state_act[ind] = np.asarray(state_act_single)
            norm_scans.append(self.state_actor_scan[ind]/5.0)

       
        state_critic = np.reshape(state_act, (1,-1))
        return [state_critic, state_act, norm_scans]
