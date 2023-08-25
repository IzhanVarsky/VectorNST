import random

import numpy as np
import numpy.random as npr
import pydiffvg
import torch


def get_bezier_circle(radius=1, segments=4, bias=None):
    points = []
    if bias is None:
        bias = (random.random(), random.random())
    avg_degree = 360 / (segments * 3)
    for i in range(0, segments * 3):
        point = (np.cos(np.deg2rad(i * avg_degree)),
                 np.sin(np.deg2rad(i * avg_degree)))
        points.append(point)
    points = torch.tensor(points)
    points = points * radius + torch.tensor(bias).unsqueeze(dim=0)
    points = points.type(torch.FloatTensor)
    return points


class random_coord_init():
    def __init__(self, canvas_width, canvas_height):
        self.canvas_width = canvas_width
        self.canvas_height = canvas_height

    def __call__(self):
        return [npr.uniform(0, 1) * self.canvas_width,
                npr.uniform(0, 1) * self.canvas_height]


def init_shapes_LIVE(num_paths,
                     canvas_width, canvas_height,
                     rand_opacity=False,
                     init_type='random',
                     min_num_segments=1, max_num_segments=4,
                     radius=5,
                     stroke_width=0.0):
    shapes = []
    shape_groups = []

    pos_init_method = random_coord_init(canvas_width, canvas_height)

    for i in range(num_paths):
        num_segments = random.randint(min_num_segments, max_num_segments)
        num_control_points = torch.zeros(num_segments, dtype=torch.int32) + 2

        if init_type == "random":
            points = []
            p0 = pos_init_method()
            points.append(p0)
            for j in range(num_segments):
                p1 = (p0[0] + radius * npr.uniform(-0.5, 0.5),
                      p0[1] + radius * npr.uniform(-0.5, 0.5))
                p2 = (p1[0] + radius * npr.uniform(-0.5, 0.5),
                      p1[1] + radius * npr.uniform(-0.5, 0.5))
                p3 = (p2[0] + radius * npr.uniform(-0.5, 0.5),
                      p2[1] + radius * npr.uniform(-0.5, 0.5))
                points.append(p1)
                points.append(p2)
                if j < num_segments - 1:
                    points.append(p3)
                    p0 = p3
            points = torch.FloatTensor(points)
        elif init_type == "circle":
            if radius is None:
                radius = npr.uniform(0.5, 1)
            center = pos_init_method()
            points = get_bezier_circle(
                radius=radius, segments=num_segments,
                bias=center)

        path = pydiffvg.Path(num_control_points=num_control_points,
                             points=points,
                             stroke_width=torch.tensor(stroke_width),
                             is_closed=True)
        shapes.append(path)

        opacity = random.random() if rand_opacity else 1.0
        fill_color_init = torch.tensor([random.random(), random.random(),
                                        random.random(), opacity])
        opacity = random.random() if rand_opacity else 1.0
        stroke_color_init = torch.tensor([random.random(), random.random(),
                                          random.random(), opacity])

        path_group = pydiffvg.ShapeGroup(
            shape_ids=torch.tensor([len(shapes) - 1]),
            fill_color=fill_color_init,
            stroke_color=stroke_color_init,
        )
        shape_groups.append(path_group)

    return shapes, shape_groups


def init_with_rand_curves(num_paths, canvas_width, canvas_height,
                          rand_opacity=False,
                          min_num_segments=1, max_num_segments=4,
                          radius=0.1,
                          stroke_width=1.0):
    shapes = []
    shape_groups = []
    for i in range(num_paths):
        num_segments = random.randint(min_num_segments, max_num_segments)
        num_control_points = torch.zeros(num_segments, dtype=torch.int32) + 2
        points = []
        p0 = (random.random(), random.random())
        points.append(p0)
        for j in range(num_segments):
            p1 = (p0[0] + radius * (random.random() - 0.5),
                  p0[1] + radius * (random.random() - 0.5))
            p2 = (p1[0] + radius * (random.random() - 0.5),
                  p1[1] + radius * (random.random() - 0.5))
            p3 = (p2[0] + radius * (random.random() - 0.5),
                  p2[1] + radius * (random.random() - 0.5))
            points.append(p1)
            points.append(p2)
            points.append(p3)
            p0 = p3
        points = torch.tensor(points)
        points[:, 0] *= canvas_width
        points[:, 1] *= canvas_height
        path = pydiffvg.Path(num_control_points=num_control_points,
                             points=points, stroke_width=torch.tensor(stroke_width),
                             is_closed=False)
        shapes.append(path)
        opacity = random.random() if rand_opacity else 1.0
        path_group = pydiffvg.ShapeGroup(shape_ids=torch.tensor([len(shapes) - 1]),
                                         fill_color=None,
                                         stroke_color=torch.tensor([random.random(), random.random(),
                                                                    random.random(), opacity]))
        shape_groups.append(path_group)
    return shapes, shape_groups
