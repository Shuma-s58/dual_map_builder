#!/usr/bin/env python3
import math
import os

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy

import tf2_ros
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import OccupancyGrid
from sensor_msgs.msg import LaserScan

UNKNOWN  = -1
FREE     =  0
OCCUPIED = 100


def quat_to_yaw(q) -> float:
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def transform_points(pts, tx, ty, yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    R = np.array([[c, -s], [s, c]])
    return (R @ pts.T).T + np.array([tx, ty])


def scan_to_points(scan):
    ranges = np.asarray(scan.ranges, dtype=np.float32)
    angles = scan.angle_min + np.arange(len(ranges)) * scan.angle_increment
    valid  = np.isfinite(ranges) & (ranges >= scan.range_min) & (ranges <= scan.range_max)
    r, a = ranges[valid], angles[valid]
    return np.column_stack([r * np.cos(a), r * np.sin(a)])


def bresenham_batch(rx, ry, ex_arr, ey_arr, width, height):
    free_cells = set()
    for ex, ey in zip(ex_arr, ey_arr):
        x0, y0, x1, y1 = rx, ry, int(ex), int(ey)
        dx, dy = abs(x1 - x0), abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy
        while not (x0 == x1 and y0 == y1):
            if 0 <= x0 < width and 0 <= y0 < height:
                free_cells.add((x0, y0))
            e2 = 2 * err
            if e2 > -dy: err -= dy; x0 += sx
            if e2 <  dx: err += dx; y0 += sy
    return free_cells


class DynamicOccupancyMap:
    """
    slam_toolboxの/mapに合わせてwidth/height/originを動的に追従するマップ。
    /map更新のたびにresize()を呼ぶことで既存データを保持しつつ拡張する。
    """

    def __init__(self, resolution):
        self.resolution = resolution
        self.width    = 0
        self.height   = 0
        self.origin_x = 0.0
        self.origin_y = 0.0
        self.data     = np.zeros((0, 0), dtype=np.int8)

    def is_initialized(self):
        return self.width > 0 and self.height > 0

    def resize(self, new_width, new_height, new_origin_x, new_origin_y):
        """
        slam_toolboxの/mapサイズ・originに合わせてグリッドを拡張する。
        既存データは新グリッド上の対応する位置にコピーする。
        """
        if (new_width == self.width and new_height == self.height and
                abs(new_origin_x - self.origin_x) < 1e-6 and
                abs(new_origin_y - self.origin_y) < 1e-6):
            return  # 変化なし

        new_data = np.full((new_height, new_width), UNKNOWN, dtype=np.int8)

        if self.is_initialized():
            # 既存データを新グリッドの対応位置にコピー
            # 旧originが新グリッド上で何セル目にあるか
            offset_x = int(round((self.origin_x - new_origin_x) / self.resolution))
            offset_y = int(round((self.origin_y - new_origin_y) / self.resolution))

            # コピー範囲を計算（はみ出し防止）
            src_x0 = max(0, -offset_x)
            src_y0 = max(0, -offset_y)
            src_x1 = min(self.width,  new_width  - offset_x)
            src_y1 = min(self.height, new_height - offset_y)

            dst_x0 = src_x0 + offset_x
            dst_y0 = src_y0 + offset_y
            dst_x1 = src_x1 + offset_x
            dst_y1 = src_y1 + offset_y

            if src_x1 > src_x0 and src_y1 > src_y0:
                new_data[dst_y0:dst_y1, dst_x0:dst_x1] = \
                    self.data[src_y0:src_y1, src_x0:src_x1]

        self.data     = new_data
        self.width    = new_width
        self.height   = new_height
        self.origin_x = new_origin_x
        self.origin_y = new_origin_y

    def world_to_cell(self, wx, wy):
        cx = int((wx - self.origin_x) / self.resolution)
        cy = int((wy - self.origin_y) / self.resolution)
        return cx, cy

    def in_bounds(self, cx, cy):
        return 0 <= cx < self.width and 0 <= cy < self.height

    def update(self, pts_world, robot_wx, robot_wy):
        if not self.is_initialized():
            return

        rx, ry = self.world_to_cell(robot_wx, robot_wy)

        ex_arr = ((pts_world[:, 0] - self.origin_x) / self.resolution).astype(int)
        ey_arr = ((pts_world[:, 1] - self.origin_y) / self.resolution).astype(int)

        mask = (
            (ex_arr >= 0) & (ex_arr < self.width) &
            (ey_arr >= 0) & (ey_arr < self.height)
        )
        ex_arr = ex_arr[mask]
        ey_arr = ey_arr[mask]

        if len(ex_arr) == 0:
            return

        free_cells = bresenham_batch(rx, ry, ex_arr, ey_arr, self.width, self.height)
        for fx, fy in free_cells:
            if self.data[fy, fx] == UNKNOWN:
                self.data[fy, fx] = FREE

        self.data[ey_arr, ex_arr] = OCCUPIED

    def to_occupancy_grid_msg(self, frame_id, stamp):
        msg = OccupancyGrid()
        msg.header.stamp              = stamp
        msg.header.frame_id           = frame_id
        msg.info.resolution           = self.resolution
        msg.info.width                = self.width
        msg.info.height               = self.height
        msg.info.origin.position.x    = self.origin_x
        msg.info.origin.position.y    = self.origin_y
        msg.info.origin.orientation.w = 1.0
        msg.data = self.data.flatten().tolist()
        return msg

    def save_pgm_yaml(self, base_path):
        pgm_path  = base_path + '.pgm'
        yaml_path = base_path + '.yaml'
        pgm = np.full((self.height, self.width), 205, dtype=np.uint8)
        pgm[self.data == FREE]     = 254
        pgm[self.data == OCCUPIED] = 0
        pgm_flip = np.flipud(pgm)
        with open(pgm_path, 'wb') as f:
            f.write(f'P5\n{self.width} {self.height}\n255\n'.encode())
            f.write(pgm_flip.tobytes())
        with open(yaml_path, 'w') as f:
            f.write(f'image: {os.path.basename(pgm_path)}\n')
            f.write(f'resolution: {self.resolution}\n')
            f.write(f'origin: [{self.origin_x:.4f}, {self.origin_y:.4f}, 0.0]\n')
            f.write('negate: 0\noccupied_thresh: 0.65\nfree_thresh: 0.196\n')
        return pgm_path, yaml_path


class DualMapNode(Node):

    def __init__(self):
        super().__init__('dual_map_node')

        self.declare_parameter('scan_top_topic', '/scan_top')
        self.declare_parameter('map_frame',      'map')
        self.declare_parameter('base_frame',     'base_link')
        self.declare_parameter('map_resolution',  0.05)
        self.declare_parameter('save_map',        True)
        self.declare_parameter('save_dir',        '/tmp/dual_maps')

        p = self.get_parameter
        self.scan_top_topic = p('scan_top_topic').value
        self.map_frame      = p('map_frame').value
        self.base_frame     = p('base_frame').value
        self.map_resolution = p('map_resolution').value
        self.save_map       = p('save_map').value
        self.save_dir       = p('save_dir').value

        self.tf_buffer   = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # 動的サイズマップ
        self.map_top = DynamicOccupancyMap(self.map_resolution)

        map_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        scan_qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
        )

        self.pub_map_bottom = self.create_publisher(OccupancyGrid, '/map_bottom', map_qos)
        self.pub_map_top    = self.create_publisher(OccupancyGrid, '/map_top',    map_qos)

        self.create_subscription(OccupancyGrid, '/map',              self.cb_map, map_qos)
        self.create_subscription(LaserScan,     self.scan_top_topic, self.cb_top, scan_qos)

        self.create_timer(1.0, self.publish_map_top)

        self.get_logger().info(
            f'\n=== DualMapNode 起動 ===\n'
            f'  /map        → /map_bottom (リレー、動的サイズ追従)\n'
            f'  top scan    : {self.scan_top_topic} → /map_top\n'
            f'  base frame  : {self.base_frame}'
        )

    def cb_map(self, msg: OccupancyGrid):
        """
        slam_toolboxの/map受信:
          1. /map_bottomにリレー
          2. map_topのサイズ・originをslam_toolboxに追従させる
        """
        self.pub_map_bottom.publish(msg)

        new_w  = msg.info.width
        new_h  = msg.info.height
        new_ox = msg.info.origin.position.x
        new_oy = msg.info.origin.position.y

        prev_w = self.map_top.width
        self.map_top.resize(new_w, new_h, new_ox, new_oy)

        if prev_w == 0:
            self.get_logger().info(
                f'/map_top 初期化: {new_w}x{new_h} cells, '
                f'origin=({new_ox:.2f}, {new_oy:.2f})'
            )
        elif self.map_top.width != prev_w:
            self.get_logger().info(
                f'/map_top リサイズ: {new_w}x{new_h} cells, '
                f'origin=({new_ox:.2f}, {new_oy:.2f})'
            )

    def lookup_pose(self):
        try:
            tf: TransformStamped = self.tf_buffer.lookup_transform(
                self.map_frame,
                self.base_frame,
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.1),
            )
            return (
                tf.transform.translation.x,
                tf.transform.translation.y,
                quat_to_yaw(tf.transform.rotation),
            )
        except Exception as e:
            self.get_logger().warn(
                f'TF取得失敗: {e}', throttle_duration_sec=3.0)
            return None

    def lookup_sensor_offset(self, sensor_frame):
        try:
            tf: TransformStamped = self.tf_buffer.lookup_transform(
                self.base_frame, sensor_frame,
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.5),
            )
            return (
                tf.transform.translation.x,
                tf.transform.translation.y,
                quat_to_yaw(tf.transform.rotation),
            )
        except Exception:
            return 0.0, 0.0, 0.0

    def cb_top(self, scan: LaserScan):
        if not self.map_top.is_initialized():
            self.get_logger().info(
                '/map未受信のためスキップ中...',
                throttle_duration_sec=3.0,
            )
            return

        pose = self.lookup_pose()
        if pose is None:
            return

        tx, ty, yaw = pose
        sx, sy, syaw = self.lookup_sensor_offset(scan.header.frame_id)

        pts = scan_to_points(scan)
        if len(pts) == 0:
            return

        pts_base = transform_points(pts, sx, sy, syaw)
        pts_map  = transform_points(pts_base, tx, ty, yaw)
        self.map_top.update(pts_map, tx, ty)

    def publish_map_top(self):
        if not self.map_top.is_initialized():
            return
        stamp = self.get_clock().now().to_msg()
        self.pub_map_top.publish(
            self.map_top.to_occupancy_grid_msg(self.map_frame, stamp)
        )

    def save_maps(self):
        if not self.save_map:
            return
        if not self.map_top.is_initialized():
            self.get_logger().warn('map_top 未初期化のため保存スキップ')
            return
        os.makedirs(self.save_dir, exist_ok=True)
        t_pgm, t_yaml = self.map_top.save_pgm_yaml(
            os.path.join(self.save_dir, 'map_top'))
        self.get_logger().info(
            f'\n=== マップ保存完了 ===\n'
            f'  {t_pgm}\n  {t_yaml}\n'
            f'  map_bottom の保存:\n'
            f'  ros2 run nav2_map_server map_saver_cli -f {self.save_dir}/map_bottom'
        )


def main(args=None):
    rclpy.init(args=args)
    node = DualMapNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.save_maps()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
