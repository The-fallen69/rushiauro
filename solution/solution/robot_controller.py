import sys
import math
from typing import Optional

import rclpy
from rclpy. node import Node
from rclpy.signals import SignalHandlerOptions
from rclpy.executors import ExternalShutdownException

from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry

from assessment_interfaces.msg import BarrelList, ZoneList, RadiationList
from auro_interfaces.srv import ItemRequest


class RobotController(Node):
    def __init__(self):
        super().__init__('robot_controller')

        self.declare_parameter('robot_id', 'robot1')
        self.declare_parameter('x', 0.0)
        self.declare_parameter('y', 0.0)
        self.declare_parameter('yaw', 0.0)

        self.declare_parameter('linear_speed', 0.18)
        self.declare_parameter('angular_speed', 0.9)
        self.declare_parameter('obstacle_distance', 0.35)
        self.declare_parameter('pickup_size', 0.40)
        self.declare_parameter('offload_size', 0.30)
        self.declare_parameter('radiation_threshold', 45)

        self.declare_parameter('overshoot_linear', 0.12)
        self.declare_parameter('overshoot_ticks', 12)

        self.declare_parameter('exit_decon_linear', 0.12)
        self.declare_parameter('exit_decon_angular', 0.18)

        self.declare_parameter('search_linear', 0.15)
        self.declare_parameter('search_angular', 0.25)

        self.robot_id = self.get_parameter('robot_id').value
        self.initial_x = self.get_parameter('x').value
        self. initial_y = self.get_parameter('y').value
        self.initial_yaw = self. get_parameter('yaw').value

        self.first_time = True
        self.holding_item = False
        self.latest_barrels:  Optional[BarrelList] = None
        self.latest_zones: Optional[ZoneList] = None
        self.latest_radiation: Optional[RadiationList] = None
        self.latest_scan: Optional[LaserScan] = None
        self.latest_odom: Optional[Odometry] = None

        self.pending_action = None
        self.pending_future = None

        self.overshooting_pickup = False
        self.overshoot_ticks_left = 0

        self.last_green_zone = None
        self.green_zone_memory_ticks = 0
        self.declare_parameter('zone_memory_ticks', 50)

        self.search_ticks = 0
        self. search_turn_direction = 1

        self.cmd_pub = self.create_publisher(Twist, '/robot1/cmd_vel', 10)
        self.create_subscription(BarrelList, '/robot1/barrels', self.barrels_cb, 10)
        self.create_subscription(ZoneList, '/robot1/zones', self. zones_cb, 10)
        self.create_subscription(RadiationList, '/radiation_levels', self.radiation_cb, 10)
        self.create_subscription(LaserScan, '/robot1/scan', self.scan_cb, 10)
        self.create_subscription(Odometry, '/robot1/odom', self.odom_cb, 10)

        self.pickup_client = self.create_client(ItemRequest, '/pick_up_item')
        self.offload_client = self. create_client(ItemRequest, '/offload_item')
        self.decon_client = self.create_client(ItemRequest, '/decontaminate')

        self.timer_period = 0.1
        self.timer = self.create_timer(self.timer_period, self.control_loop)

    def barrels_cb(self, msg: BarrelList):
        self.latest_barrels = msg

    def zones_cb(self, msg: ZoneList):
        self.latest_zones = msg
        for z in msg.data:
            if z.zone == 1:
                self.last_green_zone = z
                self.green_zone_memory_ticks = int(self.get_parameter('zone_memory_ticks').value)
                self.get_logger().info(f'Green zone detected:  x={z.x}, size={z.size}')

    def radiation_cb(self, msg: RadiationList):
        self.latest_radiation = msg

    def scan_cb(self, msg: LaserScan):
        self.latest_scan = msg

    def odom_cb(self, msg: Odometry):
        self.latest_odom = msg

    def get_radiation_level(self) -> int:
        if self. latest_radiation is None:
            return 0
        for entry in self.latest_radiation. data:
            if entry.robot_id == self.robot_id:
                return entry.level
        return 0

    def pick_best_barrel(self):
        if not self.latest_barrels or len(self.latest_barrels. data) == 0:
            return None
        return max(self.latest_barrels.data, key=lambda b: b.size)

    def pick_zone(self, zone_type: int):
        if not self.latest_zones or len(self.latest_zones.data) == 0:
            return None
        zones = [z for z in self. latest_zones.data if z. zone == zone_type]
        if not zones:
            return None
        return max(zones, key=lambda z: z.size)

    def obstacle_ahead(self) -> bool:
        if self.latest_scan is None or not self.latest_scan.ranges:
            return False
        ranges = list(self.latest_scan.ranges)
        front = ranges[-10: ] + ranges[:10]
        front = [r for r in front if not math.isinf(r) and not math.isnan(r)]
        if not front:
            return False
        return min(front) < self.get_parameter('obstacle_distance').value

    def make_twist(self, linear: float, angular: float) -> Twist:
        msg = Twist()
        msg.linear.x = float(linear)
        msg.angular.z = float(angular)
        return msg

    def approach_by_image(self, x: float, size: float, target_size: float):
        error = max(min(x / 320.0, 1.0), -1.0)
        angular = -error * self.get_parameter('angular_speed').value
        linear = self.get_parameter('linear_speed').value

        if size >= target_size:
            linear = 0.0
        return self.make_twist(linear, angular)

    def search_for_zone(self) -> Twist:
        """Move FORWARD while turning to find the green zone - NOT just spinning"""
        self.search_ticks += 1
        
        linear = self.get_parameter('search_linear').value
        angular = self.get_parameter('search_angular').value * self.search_turn_direction

        if self.search_ticks % 60 == 0:
            self. search_turn_direction *= -1
            self.get_logger().info('Changing search direction.. .')

        if self.obstacle_ahead():
            linear = 0.0
            angular = 0.6 * self.search_turn_direction
            self.get_logger().info('Obstacle detected, turning...')

        return self.make_twist(linear, angular)

    def call_item_service(self, client, action_name:  str):
        if self.pending_future is not None:
            return
        if not client.wait_for_service(timeout_sec=0.2):
            self.get_logger().warn(f'{action_name} service not available')
            return
        req = ItemRequest. Request()
        req.robot_id = self.robot_id
        self.pending_action = action_name
        self. pending_future = client.call_async(req)

    def handle_pending_action(self):
        if self.pending_future is None:
            return
        if not self.pending_future.done():
            return
        result = self.pending_future.result()
        if result is None:
            self.get_logger().warn(f'{self.pending_action} failed (no result)')
        else:
            self.get_logger().info(f'{self.pending_action}:  {result.success} - {result.message}')
            if self.pending_action == 'pick_up' and result.success:
                self.holding_item = True
                self. search_ticks = 0
                self. get_logger().info('=== BARREL PICKED UP!  Now searching for GREEN ZONE ===')
            if self.pending_action == 'offload' and result.success:
                self.holding_item = False
                self.get_logger().info('=== BARREL DEPOSITED! Ready for next barrel ===')
            if self. pending_action == 'decontaminate' and result.success:
                self.get_logger().info('=== DECONTAMINATION COMPLETE! ===')
        self.pending_future = None
        self.pending_action = None

    def control_loop(self):
        if self.first_time:
            self.get_logger().info(
                f"Initial pose - x:  {self.initial_x}, y: {self.initial_y}, yaw: {self.initial_yaw}.  Ready to go."
            )
            self.first_time = False

        self.handle_pending_action()

        if self.green_zone_memory_ticks > 0:
            self.green_zone_memory_ticks -= 1
        else:
            self.last_green_zone = None

        radiation_level = self.get_radiation_level()
        need_decon = radiation_level >= self.get_parameter('radiation_threshold').value

        decon_zone = self.pick_zone(0)
        green_zone = self.pick_zone(1)
        barrel = self.pick_best_barrel()

        twist = self.make_twist(0.0, 0.0)

        if self.overshooting_pickup:
            twist = self.make_twist(self.get_parameter('overshoot_linear').value, 0.0)
            self.overshoot_ticks_left -= 1
            if self.overshoot_ticks_left <= 0:
                self.overshooting_pickup = False
                if not self.holding_item and self.pending_future is None:
                    self. call_item_service(self.pickup_client, 'pick_up')

        elif need_decon and decon_zone is not None and not self.holding_item:
            twist = self.approach_by_image(decon_zone.x, decon_zone.size,
                                           self.get_parameter('offload_size').value)
            if decon_zone.size >= self.get_parameter('offload_size').value:
                self.call_item_service(self.decon_client, 'decontaminate')

        elif self.holding_item:
            target_green = green_zone or self.last_green_zone
            
            if target_green is not None:
                self.get_logger().info(f'Approaching green zone: x={target_green.x}, size={target_green.size}')
                twist = self.approach_by_image(target_green.x, target_green.size,
                                               self.get_parameter('offload_size').value)
                if target_green.size >= self.get_parameter('offload_size').value:
                    self.get_logger().info('At green zone, depositing barrel.. .')
                    self.call_item_service(self.offload_client, 'offload')
            else:
                self.get_logger().info('Searching for green zone (moving forward)...')
                twist = self.search_for_zone()

        else:
            if barrel is not None:
                twist = self.approach_by_image(barrel.x, barrel.size,
                                               self.get_parameter('pickup_size').value)
                if barrel.size >= self.get_parameter('pickup_size').value:
                    if not self.holding_item and self.pending_future is None:
                        self.overshooting_pickup = True
                        self.overshoot_ticks_left = int(self.get_parameter('overshoot_ticks').value)
            else:
                twist = self.make_twist(0.12, 0.35)

        if self.obstacle_ahead() and twist.linear.x > 0:
            if self.holding_item:
                twist = self.make_twist(0.05, 0.7 * self.search_turn_direction)
            else:
                twist = self.make_twist(0.0, 0.8)

        self.cmd_pub.publish(twist)

    def destroy_node(self):
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions. NO)
    node = RobotController()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except ExternalShutdownException: 
        sys.exit(1)
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
