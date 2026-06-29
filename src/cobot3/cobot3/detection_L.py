import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from std_msgs.msg import Int32
from geometry_msgs.msg import Point
from tf2_msgs.msg import TFMessage

import numpy as np
import cv2
from ultralytics import YOLO
from scipy.spatial.transform import Rotation as R


TF_TOPIC = "/tf"
PARENT_FRAME = "World"
CHILD_FRAME = "dual_suction_mount"

T_MOUNT_CAMERA = np.array([
    [ 3.44617717e-08, -9.32350347e-03, -9.99956535e-01,  4.00000004e-02],
    [ 2.48545910e-08, -9.99956535e-01,  9.32350347e-03,  2.85827362e-10],
    [-1.00000000e+00, -2.51748152e-08, -3.42285421e-08, -1.15000000e-02],
    [ 0.0,             0.0,             0.0,             1.0]
])
class RGBYOLOViewer(Node):
    def __init__(self):
        super().__init__('yolo_image_window_L')

        self.create_subscription(Image, '/rgb_L', self.image_callback, 10)
        self.create_subscription(Image, '/depth_L', self.depth_callback, 10)
        self.create_subscription(TFMessage, TF_TOPIC, self.tf_callback, 10)

        self.box_exist_pub = self.create_publisher(Int32, '/box_exist_L', 10)
        self.barcode_exist_pub = self.create_publisher(Int32, '/barcode_exist_L', 10)

        self.box_coordinate_left_pub = self.create_publisher(Point, '/box_coordinate_left_L', 10)
        self.box_coordinate_center_pub = self.create_publisher(Point, '/box_coordinate_center_L', 10)
        self.box_coordinate_right_pub = self.create_publisher(Point, '/box_coordinate_right_L', 10)

        self.model = YOLO('/home/rokey/cobot3_ws/src/cobot3/cobot3/best.pt')

        self.depth_image = None
        self.T_world_mount = None

        self.fx = 317.0431199837856
        self.fy = 317.0431199837856
        self.cx = 320.0
        self.cy = 320.0

        self.get_logger().info('YOLO RIGHT dual_suction_mount camera node started')
        self.get_logger().info(f'Subscribed TF topic: {TF_TOPIC}')

        # Create resizable OpenCV window and set to 320x320
        cv2.namedWindow('YOLO Detection L', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('YOLO Detection L', 320, 320)
    def tf_callback(self, msg):
        for tf in msg.transforms:
            if tf.header.frame_id != PARENT_FRAME:
                continue
            if tf.child_frame_id != CHILD_FRAME:
                continue

            t = tf.transform.translation
            q = tf.transform.rotation

            T = np.eye(4)
            T[:3, :3] = R.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
            T[:3, 3] = [t.x, t.y, t.z]

            self.T_world_mount = T

    def depth_callback(self, msg):
        try:
            if msg.encoding == '32FC1':
                depth = np.frombuffer(msg.data, dtype=np.float32)
                self.depth_image = depth.reshape((msg.height, msg.width))

            elif msg.encoding == '16UC1':
                depth = np.frombuffer(msg.data, dtype=np.uint16)
                depth = depth.reshape((msg.height, msg.width))
                self.depth_image = depth.astype(np.float32) / 1000.0

            else:
                self.get_logger().error(f'Unsupported depth encoding: {msg.encoding}')

        except Exception as e:
            self.get_logger().error(f'Depth callback failed: {e}')

    def image_msg_to_bgr(self, msg):
        img = np.frombuffer(msg.data, dtype=np.uint8)

        if msg.encoding == 'rgb8':
            img = img.reshape((msg.height, msg.width, 3))
            return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        if msg.encoding == 'bgr8':
            return img.reshape((msg.height, msg.width, 3))

        if msg.encoding == 'rgba8':
            img = img.reshape((msg.height, msg.width, 4))
            return cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)

        if msg.encoding == 'bgra8':
            img = img.reshape((msg.height, msg.width, 4))
            return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        raise ValueError(f'Unsupported RGB encoding: {msg.encoding}')

    def pixel_depth_to_camera(self, u, v, depth):
        Xc = (u - self.cx) * depth / self.fx
        Yc = -(v - self.cy) * depth / self.fy
        Zc = -depth
        return np.array([Xc, Yc, Zc, 1.0])
    def camera_to_world(self, u, v, depth):
        if self.T_world_mount is None:
            self.get_logger().warn('No dual_suction_mount TF received yet')
            return None

        p_camera = self.pixel_depth_to_camera(u, v, depth)

        T_world_camera = self.T_world_mount @ T_MOUNT_CAMERA
        p_world_h = T_world_camera @ p_camera

        return p_world_h[:3]

    def get_depth_at_pixel(self, u, v):
        if self.depth_image is None:
            return None

        h, w = self.depth_image.shape

        u = int(np.clip(u, 0, w - 1))
        v = int(np.clip(v, 0, h - 1))

        depth = float(self.depth_image[v, u])

        if depth <= 0.0 or np.isnan(depth) or np.isinf(depth):
            return None

        return depth

    def publish_exist(self, box_exist, barcode_exist):
        msg = Int32()
        msg.data = int(box_exist)
        self.box_exist_pub.publish(msg)

        msg = Int32()
        msg.data = int(barcode_exist)
        self.barcode_exist_pub.publish(msg)

    def publish_point(self, pub, p_world):
        msg = Point()
        msg.x = float(p_world[0])
        msg.y = float(p_world[1])
        msg.z = float(p_world[2])
        pub.publish(msg)

    def image_callback(self, msg):
        try:
            frame = self.image_msg_to_bgr(msg)
            results = self.model(frame, verbose=False, conf=0.70)

            box_detected = False
            barcode_detected = False

            best_box = None
            best_conf = 0.0
            barcode_boxes = []

            for result in results:
                for box in result.boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])

                    if conf < 0.5:
                        continue

                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    if cls_id == 0:
                        box_detected = True
                        if conf > best_conf:
                            best_conf = conf
                            best_box = (x1, y1, x2, y2, conf)

                    elif cls_id == 1:
                        barcode_detected = True
                        barcode_boxes.append((x1, y1, x2, y2, conf))

            self.publish_exist(
                box_exist=1 if box_detected else 0,
                barcode_exist=1 if barcode_detected else 0
            )

            for bx1, by1, bx2, by2, bconf in barcode_boxes:
                cv2.rectangle(frame, (bx1, by1), (bx2, by2), (0, 0, 255), 3)
                cv2.putText(
                    frame,
                    f'BARCODE {bconf:.2f}',
                    (bx1, max(by1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA
                )

            if best_box is not None:
                x1, y1, x2, y2, conf = best_box

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    f'BEST BOX {conf:.2f}',
                    (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA
                )

                v_center = int((y1 + y2) / 2)

                margin = 0
                u_left = int(x1 + margin)
                u_center = int((x1 + x2) / 2)
                u_right = int(x2 - margin)

                center_depth = self.get_depth_at_pixel(u_center, v_center)

                if center_depth is not None:
                    points = {
                        'left': (u_left, v_center, self.box_coordinate_left_pub),
                        'center': (u_center, v_center, self.box_coordinate_center_pub),
                        'right': (u_right, v_center, self.box_coordinate_right_pub),
                    }

                    world_points = {}

                    for name, (u, v, pub) in points.items():
                        p_world = self.camera_to_world(u, v, center_depth)

                        if p_world is None:
                            continue

                        self.publish_point(pub, p_world)
                        world_points[name] = p_world

                        if name == 'left':
                            color = (255, 0, 0)
                        elif name == 'center':
                            color = (255, 0, 255)
                        else:
                            color = (0, 0, 255)

                        cv2.circle(frame, (u, v), 7, color, -1)

                    if 'center' in world_points:
                        p = world_points['center']
                        cv2.putText(
                            frame,
                            f'C: {p[0]:.2f}, {p[1]:.2f}, {p[2]:.2f}',
                            (x1, min(y2 + 25, frame.shape[0] - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (255, 0, 255),
                            2,
                            cv2.LINE_AA
                        )

                    if len(world_points) == 3:
                        self.get_logger().info(
                            f'center_depth={center_depth:.3f} / '
                            f'left=({world_points["left"][0]:.3f}, {world_points["left"][1]:.3f}, {world_points["left"][2]:.3f}) / '
                            f'center=({world_points["center"][0]:.3f}, {world_points["center"][1]:.3f}, {world_points["center"][2]:.3f}) / '
                            f'right=({world_points["right"][0]:.3f}, {world_points["right"][1]:.3f}, {world_points["right"][2]:.3f})'
                        )

                else:
                    self.get_logger().warn(
                        f'Invalid CENTER depth at pixel ({u_center}, {v_center})'
                    )

            else:
                self.get_logger().info('/box_exist_R=0, box coordinates not published')

            cv2.putText(
                frame,
                f'box_exist={1 if box_detected else 0}, barcode_exist={1 if barcode_detected else 0}',
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )

            cv2.imshow('YOLO Detection L', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                rclpy.shutdown()

        except Exception as e:
            self.get_logger().error(f'YOLO detection failed: {e}')


def main(args=None):
    rclpy.init(args=args)
    node = RGBYOLOViewer()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        cv2.destroyAllWindows()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()