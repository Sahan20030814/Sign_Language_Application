"""
renderer3d.py
---------------
A minimal, dependency-free 3D software renderer.

Why this exists: real 3D engines (Three.js, pyrender/OpenGL) either can't be
embedded inside a Tkinter window (pywebview opens its own separate window)
or need heavy/fragile dependencies (PyOpenGL, GPU drivers, licensed body
models like SMPL-X) that were not reliably installable/testable here.

This module instead does the whole pipeline itself in plain Python +
numpy + Pillow (both already dependencies of this project):
  1. Build simple primitive meshes (spheres, cylinders) as numpy vertex
     arrays + triangle-index lists.
  2. Transform + project them with a basic perspective camera.
  3. Flat-shade each triangle from one directional light.
  4. Depth-sort (painter's algorithm) and rasterize with
     PIL.ImageDraw.polygon.

It is genuinely 3D (real perspective + depth), just deliberately simple
(low-poly) so it renders fast enough in pure Python without a GPU.
"""

import numpy as np
from PIL import Image, ImageDraw

# ----------------------------------------------------------------------------
# Mesh builders
# ----------------------------------------------------------------------------
def make_sphere(radius, center, n_lat=8, n_lon=10):
    verts = []
    for i in range(n_lat + 1):
        theta = np.pi * i / n_lat
        for j in range(n_lon):
            phi = 2 * np.pi * j / n_lon
            x = radius * np.sin(theta) * np.cos(phi)
            y = radius * np.cos(theta)
            z = radius * np.sin(theta) * np.sin(phi)
            verts.append([x, y, z])
    verts = np.array(verts) + np.array(center, dtype=float)
    faces = []
    for i in range(n_lat):
        for j in range(n_lon):
            a = i * n_lon + j
            b = i * n_lon + (j + 1) % n_lon
            c = (i + 1) * n_lon + (j + 1) % n_lon
            d = (i + 1) * n_lon + j
            faces.append((a, b, c))
            faces.append((a, c, d))
    return verts, faces


def make_cylinder(radius, p0, p1, n_lon=6):
    p0, p1 = np.array(p0, dtype=float), np.array(p1, dtype=float)
    axis = p1 - p0
    length = np.linalg.norm(axis)
    axis_n = axis / length if length > 1e-6 else np.array([0.0, 0.0, 1.0])
    tmp = np.array([1.0, 0.0, 0.0]) if abs(axis_n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(axis_n, tmp)
    u /= np.linalg.norm(u)
    v = np.cross(axis_n, u)
    verts = []
    for k in (0, 1):
        c = p0 if k == 0 else p1
        for j in range(n_lon):
            phi = 2 * np.pi * j / n_lon
            offset = radius * (np.cos(phi) * u + np.sin(phi) * v)
            verts.append(c + offset)
    verts = np.array(verts)
    faces = []
    for j in range(n_lon):
        a, b = j, (j + 1) % n_lon
        c, d = n_lon + (j + 1) % n_lon, n_lon + j
        faces.append((a, b, c))
        faces.append((a, c, d))
    return verts, faces


# ----------------------------------------------------------------------------
# Camera / renderer
# ----------------------------------------------------------------------------
class Renderer3D:
    def __init__(self, width, height, focal=None, light_dir=(-0.4, 0.6, 1.0)):
        self.width = max(int(width), 10)
        self.height = max(int(height), 10)
        self.focal = focal if focal is not None else self.height * 0.9
        light = np.array(light_dir, dtype=float)
        self.light_dir = light / np.linalg.norm(light)

    def _project(self, p):
        x, y, z = p
        if z <= 0.05:
            return None
        sx = self.width / 2 + self.focal * x / z
        sy = self.height / 2 - self.focal * y / z
        return sx, sy

    def render(self, mesh_groups, cam_distance=4.5, bg=(0, 0, 0), rotation_y=0.0):
        """
        mesh_groups: list of (verts[Nx3], faces[list of (i,j,k)], color(r,g,b))
        Returns a PIL.Image.
        """
        img = Image.new("RGB", (self.width, self.height), bg)
        draw = ImageDraw.Draw(img)
        triangles = []

        cos_a, sin_a = np.cos(rotation_y), np.sin(rotation_y)
        rot = np.array([[cos_a, 0, sin_a], [0, 1, 0], [-sin_a, 0, cos_a]])

        for verts, faces, color in mesh_groups:
            v = verts @ rot.T if rotation_y else verts
            v = v.copy()
            v[:, 2] += cam_distance
            for f in faces:
                p0, p1, p2 = v[f[0]], v[f[1]], v[f[2]]
                normal = np.cross(p1 - p0, p2 - p0)
                nlen = np.linalg.norm(normal)
                if nlen < 1e-9:
                    continue
                normal = normal / nlen
                if normal[2] > 0:
                    continue  # basic backface culling
                shade = max(0.28, float(np.dot(normal, self.light_dir)))
                shaded_color = tuple(min(255, int(c * shade)) for c in color)
                s0, s1, s2 = self._project(p0), self._project(p1), self._project(p2)
                if None in (s0, s1, s2):
                    continue
                avg_z = (p0[2] + p1[2] + p2[2]) / 3
                triangles.append((avg_z, [s0, s1, s2], shaded_color))

        triangles.sort(key=lambda t: -t[0])  # farthest first (painter's algorithm)
        for _, pts, color in triangles:
            draw.polygon(pts, fill=color)
        return img
