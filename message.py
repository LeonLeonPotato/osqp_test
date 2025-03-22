import numpy as np
import math
import time
import pygame as pg
import pymunk

class Pose:
    def __init__(self, x=0.0, y=0.0, theta=0.0):
        """
        A simple class to represent a robot's 2D pose.
        
        :param x: The x-coordinate (float)
        :param y: The y-coordinate (float)
        :param theta: The orientation in radians (float)
        """
        self.x = x
        self.y = y
        self.theta = theta

    def __str__(self):
        """
        Returns a string representation of the pose.
        """
        return f"Pose(x={self.x}, y={self.y}, theta={self.theta})"

    def as_tuple(self):
        """
        Returns the pose as a tuple (x, y, theta).
        """
        return (self.x, self.y, self.theta)

    def set_pose(self, x, y, theta):
        """
        Sets new values for the pose attributes.
        """
        self.x = x
        self.y = y
        self.theta = theta

    def translate(self, dx, dy):
        """
        Translates the pose by dx in the x direction and dy in the y direction,
        maintaining the same orientation.
        """
        self.x += dx
        self.y += dy

    def rotate(self, dtheta):
        """
        Rotates the pose by dtheta (in radians), about its current location.
        """
        self.theta += dtheta
        # Optionally, you can normalize theta to keep it in [0, 2π) or (-π, π)
        # self.theta %= (2.0 * math.pi)
        
    def distance_to(self, other):
        """
        Computes the Euclidean distance between this pose and another RobotPose.
        """
        return math.hypot(other.x - self.x, other.y - self.y)

class Drivetrain:
    def __init__(self, gain, tc, kf):
        self.gain = gain
        self.tc = tc
        self.kf = kf

        self.position = 0
        self.velocity = 0
        self._accel = 0
    
    def update(self, u, dt):
        vd = 0 if (abs(u) < self.kf) else (self.gain * (u - self.kf * np.sign(u)))
        self._accel = (vd - self.velocity) / self.tc
        self.position += 0.5 * self._accel * dt ** 2 + self.velocity * dt
        self.velocity += self._accel * dt
        return self.position, self.velocity, self._accel
    
    def get_state(self):
        return np.array([self.position, self.velocity])
    
    def get_accel(self):
        return self._accel
    
    def get_velocity(self):
        return self.velocity
    
    def get_position(self):
        return self.position
    
    def get_rpm(self):
        return self.velocity * 60 / (2 * math.pi)
    
    def set_velocity(self, velocity):
        self.velocity = velocity

class Robot:
    def __init__(self, space, length, width, 
                 mass, moment, 
                 gain, tc, kf,
                 wx=0.0, wy=0.0, wtheta=0.0, dt=-1.0):
        self.space = space
        self.width = width
        self.length = length
        self.gain = gain
        self.tc   = tc
        self.kf   = kf
        self.left = Drivetrain(gain, tc, kf)
        self.right = Drivetrain(gain, tc, kf)
        self.dt = dt
        self.last_time = -1

        # Robot body
        self.body = pymunk.Body(mass, moment, body_type=pymunk.Body.DYNAMIC)
        self.body.position = (wx, wy)
        self.body.angle = wtheta
        self.start_pose = Pose(wx, wy, wtheta)
        
        hw = width / 2.0
        hl = length / 2.0
        corners = [(-hw, -hl), (-hw, hl), (hw, hl), (hw, -hl)]
        self.shape = pymunk.Poly(self.body, corners)
        self.shape.friction = 0.1  # Example friction
        self.shape.elasticity = 0.0

        self.space.add(self.body, self.shape)

        self.lw = 0
        self.rw = 0

    def update(self, u):
        if self.dt < 0:
            if (self.last_time < 0):
                dt = 0
            else:
                dt = time.time() - self.last_time
            self.last_time = time.time()
        else:
            dt = self.dt

        mass = self.body.mass
        rotation = np.array([
            [math.cos(self.body.angle), -math.sin(self.body.angle)],
            [math.sin(self.body.angle), math.cos(self.body.angle)]
        ])

        r = 0.04125 * 0.75
        I = 0.5 * 0.05 * r ** 2
        mu = 0.1

        max_fric = mu * mass * 9.81 / 2

        _, lw, la = self.left.update(u[0], dt)
        _, rw, lr = self.right.update(u[1], dt)

        lv = np.array(self.body.velocity_at_local_point((self.width/2, 0))) @ rotation
        rv = np.array(self.body.velocity_at_local_point((-self.width/2, 0))) @ rotation
        l_friction = np.array((0, -lw * r)) + lv
        l_slipping = (np.linalg.norm(l_friction) > 1e-2) or (abs(la * r) > max_fric)
        if (np.linalg.norm(l_friction) > 1e-2): # Kinetic friction
            l_friction = -mu * mass * 9.81 * 0.5 * l_friction / (np.linalg.norm(l_friction) + 1e-6)
        else:
            mag = np.clip(-la * I / r, -max_fric, max_fric)
            l_friction = np.array([0, mag])
        self.body.apply_force_at_local_point(tuple(l_friction), (self.width/2, 0))

        r_friction = np.array((0, -rw * r)) + rv
        r_slipping = (np.linalg.norm(r_friction) > 1e-2) or (abs(la * r) > max_fric)
        if (np.linalg.norm(r_friction) > 1e-2): # Kinetic friction
            r_friction = -mu * mass * 9.81 * 0.5 * r_friction / (np.linalg.norm(r_friction) + 1e-6)
        else:
            mag = np.clip(-lr * I / r, -max_fric, max_fric)
            r_friction = np.array([0, mag])
        self.body.apply_force_at_local_point(tuple(r_friction), (-self.width/2, 0))

        self.space.step(dt)

        print(l_slipping, r_slipping)

        if not l_slipping: # sync velocities
            lv = np.array(self.body.velocity_at_local_point((self.width/2, 0))) @ rotation
            self.left.set_velocity(lv[1] / r)

        if not r_slipping:
            rv = np.array(self.body.velocity_at_local_point((-self.width/2, 0))) @ rotation
            self.right.set_velocity(rv[1] / r)


    def get_local_state(self):
        """
        Return the locally observed state of the robot in [x, y, theta].
        """
        x, y = self.body.position
        theta = self.body.angle
        return np.array([x - self.start_pose.x, y - self.start_pose.y, theta - self.start_pose.theta])
    
    def get_state(self):
        """
        Return the global state of the robot in [x, y, theta].
        """
        x, y = self.body.position
        theta = self.body.angle
        return np.array([x, y, theta])
    
class Renderer:
    FIELD_IMAGE = pg.image.load("assets/field.jpg")

    def __init__(self, robot, fps=60):
        self.robot = robot

        pg.init()
        self.screen = pg.display.set_mode((700, 700))
        self.clock = pg.time.Clock()
        self.scale = Renderer.FIELD_IMAGE.get_size()[0] / 3.6 # px / m
        self.robot_image = pg.image.load("assets/robot.png")
        self.robot_image = pg.transform.scale(self.robot_image, (self.robot.width * self.scale, self.robot.length * self.scale))
        # self.robot_image = pg.transform.rotate(self.robot_image, 180)

    def draw_robot(self):
        for event in pg.event.get():
            if event.type == pg.QUIT:
                pg.quit()
                return

        x, y, theta = self.robot.get_state()
        self.screen.blit(self.FIELD_IMAGE, (0, 0))
        x *= self.scale
        y *= self.scale

        robot_image = pg.transform.rotate(self.robot_image, math.degrees(-theta))
        self.screen.blit(robot_image, robot_image.get_rect(center=(x, y)).topleft)

        pg.draw.circle(self.screen, (255, 0, 0), (int(x), int(y)), 5)

if __name__ == "__main__":

    def add_walls(space):
        spacing = 0.15
        walls = [pymunk.Segment(space.static_body, (-spacing, -spacing), (3.6 + spacing, -spacing), 0.2),
                 pymunk.Segment(space.static_body, (-spacing, -spacing), (-spacing, 3.6 + spacing), 0.2),
                 pymunk.Segment(space.static_body, (3.6 + spacing, -spacing), (3.6 + spacing, 3.6 + spacing), 0.2),
                 pymunk.Segment(space.static_body, (-spacing, 3.6 + spacing), (3.6 + spacing, 3.6 + spacing), 0.2)]
        for wall in walls:
            wall.friction = 0.1
            wall.elasticity = 0
            space.add(wall)

    def __test():
        space = pymunk.Space()
        add_walls(space)
        robot = Robot(
            space, 
            length=0.4054,
            width=0.3029, 
            mass=5, 
            moment=0.5, 
            gain=4.79965544298, 
            tc=0.02, 
            kf=0.8, 
            wx=1.8, 
            wy=1.8,
            wtheta=0
        )
        renderer = Renderer(robot)

        t = time.time()
        while True:
            t = time.time()
            robot.update([12 * np.sin(t), 6])

            renderer.draw_robot()
            pg.display.flip()

    __test()