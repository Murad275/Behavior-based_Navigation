#!/usr/bin/env python
import rospy
import random
import time
import os
from gazebo_msgs.srv import SpawnModel, DeleteModel, SetModelState
from gazebo_msgs.msg import ModelStates, ModelState
from geometry_msgs.msg import Pose

class Respawn():
    def __init__(self):
        self.modelPath = os.path.dirname(os.path.realpath(__file__))
        self.modelPath = self.modelPath.replace('formationcontrol/src',
                                                'turtlebot3_simulations/turtlebot3_gazebo/models/tb_circle/circle/model.sdf')
        self.f = open(self.modelPath, 'r')
        self.model = self.f.read()
        self.goal_position = Pose()
        self.init_goal_x = 1.05
        self.init_goal_y = 1.2
        self.goal_position.position.x = self.init_goal_x
        self.goal_position.position.y = self.init_goal_y
        self.modelName = 'goal'
        self.last_index = 0
        self.sub_model = rospy.Subscriber('gazebo/model_states', ModelStates, self.checkModel)
        self.check_model = False
        self.index = 0
        self.indx = 0



    def checkModel(self, model):
        self.check_model = False
        for i in range(len(model.name)):
            if model.name[i] == "goal":
                self.check_model = True

    def respawnModel(self):
        while True:
            if not self.check_model:
                rospy.wait_for_service('gazebo/spawn_sdf_model')
                spawn_model_prox = rospy.ServiceProxy('gazebo/spawn_sdf_model', SpawnModel)
                spawn_model_prox(self.modelName, self.model, 'robotos_name_space', self.goal_position, "world")

                break
            else:
                pass

    def deleteModel(self):
        while True:
            if self.check_model:
                rospy.wait_for_service('gazebo/delete_model')
                del_model_prox = rospy.ServiceProxy('gazebo/delete_model', DeleteModel)
                del_model_prox(self.modelName)
                break
            else:
                pass
    

    def getPosition(self, delete=False):
        
        if delete:
            self.deleteModel()
        else:
            if self.check_model:
                rospy.wait_for_service('gazebo/delete_model')
                del_model_prox = rospy.ServiceProxy('gazebo/delete_model', DeleteModel)
                del_model_prox(self.modelName)
        
     
        goal_x = random.randrange(-35, 35) / 10.0 # To be tuned based on the environment
        goal_y = random.randrange(-35, 35) / 10.0
        self.goal_position.position.x = goal_x
        self.goal_position.position.y = goal_y
        
        time.sleep(0.5)
        self.respawnModel()

        return self.goal_position.position.x, self.goal_position.position.y
