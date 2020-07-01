#!/usr/bin/env python

import sys
import copy
import rospy
import threading
import matplotlib.pyplot as plt
import numpy as np

import moveit_commander
import moveit_msgs.msg
import geometry_msgs.msg
from sensor_msgs.msg import JointState
from math import pi
from std_msgs.msg import String
from moveit_commander.conversions import pose_to_list


def all_close(goal, actual, tolerance):
  """
  Convenience method for testing if a list of values are within a tolerance of their counterparts in another list
  @param: goal       A list of floats, a Pose or a PoseStamped
  @param: actual     A list of floats, a Pose or a PoseStamped
  @param: tolerance  A float
  @returns: bool
  """
  all_equal = True
  if type(goal) is list:
    for index in range(len(goal)):
      if abs(actual[index] - goal[index]) > tolerance:
        return False

  elif type(goal) is geometry_msgs.msg.PoseStamped:
    return all_close(goal.pose, actual.pose, tolerance)

  elif type(goal) is geometry_msgs.msg.Pose:
    return all_close(pose_to_list(goal), pose_to_list(actual), tolerance)

  return True


class MoveGroupPythonIntefaceTutorial(object):
  """MoveGroupPythonIntefaceTutorial"""
  def __init__(self):
    super(MoveGroupPythonIntefaceTutorial, self).__init__()
    moveit_commander.roscpp_initialize(sys.argv)
    rospy.init_node('move_group_python_interface_tutorial', anonymous=True)
    robot = moveit_commander.RobotCommander()
    scene = moveit_commander.PlanningSceneInterface()
    group_name = "panda_arm"
    move_group = moveit_commander.MoveGroupCommander(group_name)
    display_trajectory_publisher = rospy.Publisher('/move_group/display_planned_path',
                                                   moveit_msgs.msg.DisplayTrajectory,
                                                   queue_size=20)

    planning_frame = move_group.get_planning_frame()
    eef_link = move_group.get_end_effector_link()
    group_names = robot.get_group_names()

    # Misc variables
    self.box_name = ''
    self.robot = robot
    self.scene = scene
    self.move_group = move_group
    self.display_trajectory_publisher = display_trajectory_publisher
    self.planning_frame = planning_frame
    self.eef_link = eef_link
    self.group_names = group_names


  def go_to_pose_goal(self):
    move_group = self.move_group
    pose_goal = geometry_msgs.msg.Pose()

    # Pose goals:
    pose_goal.orientation.w = 1.0
    pose_goal.position.x = 0.5
    pose_goal.position.y = 0.3
    pose_goal.position.z = 0.4

    move_group.set_pose_target(pose_goal)
    plan = move_group.go(wait=True)
    move_group.stop()
    move_group.clear_pose_targets()
    current_pose = self.move_group.get_current_pose().pose
    return all_close(pose_goal, current_pose, 0.01)


def storingData(tutorial):

    print("hey:")
    print(tutorial.move_group.get_current_pose().pose)


def joint():
    rospy.Subscriber("/joint_states", JointState, jointCall)


def jointCall(message):

    global q
    q1 = message.position[0]
    q2 = message.position[1]
    q3 = message.position[2]
    q4 = message.position[3]
    q5 = message.position[4]
    q6 = message.position[5]
    q7 = message.position[6]
    q = np.array([q1,q2,q3,q4,q5,q6,q7])
    print(q)

    qd1 = message.velocity[0]
    qd2 = message.velocity[1]
    # qd = np.array([qd1, qd2])


def main():
  global q
  q = np.array([0,0,0,0,0,0,0])
  try:
    tutorial = MoveGroupPythonIntefaceTutorial()
    t1 = threading.Thread(target=storingData, args=(tutorial,))
    print("Example run joint setpoint:")
    raw_input()


    tutorial.go_to_pose_goal()
    print("Task complete!")
    rospy.spin()


  except rospy.ROSInterruptException:
    return
  except KeyboardInterrupt:
    return


if __name__ == '__main__':
  main()
