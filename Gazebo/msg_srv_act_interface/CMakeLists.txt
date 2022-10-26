################################################################################
# Set minimum required version of cmake, project name and compile options
################################################################################
cmake_minimum_required(VERSION 3.5)
project(msg_srv_act_interface)

# if(NOT CMAKE_C_STANDARD)
#   set(CMAKE_C_STANDARD 99)
# endif()

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic")
  # add_compile_options(-Wall -Wextra -Wpedantic)
endif()

################################################################################
# Find and load build settings from external packages
################################################################################
find_package(ament_cmake REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(rosidl_default_generators REQUIRED)

################################################################################
# Declare ROS messages, services and actions
################################################################################
set(msg_files
  "msg/Test.msg"
  "msg/CAToOffboard.msg"
  "msg/PFGuidToPFAtt.msg"
  "msg/PFGprToPFGuid.msg"
  "msg/PFAttToPFGuid.msg"
  "msg/OffboardToPFGpr.msg"
  "msg/PFAttToOffboard.msg"
)

set(srv_files
  "srv/MapInit.srv"
  "srv/PathPlanningSetpoint.srv"
  "srv/PathFollowingSetpoint.srv"
  "srv/CollisionAvoidanceSetpoint.srv"
  "srv/PathFollowingGuid.srv"
  "srv/PathFollowingGPR.srv"
  "srv/PPToPF.srv"
)

set(action_files
  # "action/ArithmeticChecker.action"
)
rosidl_generate_interfaces(${PROJECT_NAME}
  ${msg_files}
  ${srv_files}
  ${action_files}
  DEPENDENCIES builtin_interfaces
)

################################################################################
# Macro for ament package
################################################################################
ament_export_dependencies(rosidl_default_runtime)
ament_package()
