import copy
import os
import random
import time

import numpy as np
import pydiffvg
import torch
import torch.optim as optim
import torchvision.transforms as transforms
from PIL import Image

from common import get_style_model_and_losses
from contour_loss import ContourLoss
from init import init_with_rand_curves, init_shapes_LIVE
from lpips_loss import LPIPS
from utils import logging, get_model_name, svg_render, save_png_svg, save_tensor_img
from xing_loss import xing_loss

pydiffvg.set_use_gpu(torch.cuda.is_available())
# pydiffvg.set_use_gpu(False)
device = pydiffvg.get_device()

default_logger = lambda s="": print(s, flush=True)


def image_loader(image_path, canvas_width, canvas_height):
    loader = transforms.ToTensor()
    image = Image.open(image_path)
    image = image.resize((canvas_width, canvas_height))
    image = loader(image)
    return image.to(device)


def alpha_blending(img, alpha_blending_coef=1.0, method=1):
    if img.shape[-1] == 3:
        return img
    if method == 0:
        img = img[:, :, :3] * img[:, :, 3:] + \
              alpha_blending_coef * torch.ones([img.shape[0], img.shape[1], 3]).to(device) * \
              (1 - img[:, :, 3:])
        return img
    if method == 1:
        return img[:, :, :3]
    raise f"Unknown alpha blending method `{method}`"


def get_contours(canvas_width, canvas_height, shapes, shape_groups):
    shape_groups_new = copy.deepcopy(shape_groups)
    with torch.no_grad():
        make_contoured_shape_groups(shape_groups_new)
        img = svg_render(canvas_width, canvas_height, shapes, shape_groups_new)
    return alpha_blending(img).permute(2, 0, 1).unsqueeze(0)


def change_stroke_width(shapes, canvas_width, canvas_height, width_value=0.5):
    for shape in shapes:
        shape.stroke_width = torch.tensor(width_value)


def make_contoured_shape_groups(shape_groups, fill_color=0.):
    fill_color = float(fill_color)
    stroke_color = 1. - fill_color
    for shape_group in shape_groups:
        if isinstance(shape_group.fill_color, torch.Tensor):
            shape_group.fill_color = torch.tensor([fill_color, fill_color, fill_color, 1.])
            shape_group.stroke_color = torch.tensor([stroke_color, stroke_color, stroke_color, 1.])
        elif isinstance(shape_group.fill_color, pydiffvg.LinearGradient):
            # Supports only fill_color=0
            shape_group.fill_color.stop_colors *= 0
            shape_group.fill_color.stop_colors += 1
            # TODO: some problems with bag.svg was
            if shape_group.stroke_color is not None:
                shape_group.stroke_color *= 0
                shape_group.stroke_color += 1


def run_style_transfer(cur_logger,
                       folder_to_save_results,
                       name,
                       content_img, style_img,
                       canvas_width, canvas_height,
                       content_shapes, content_shape_groups,
                       num_steps, point_rate, color_rate, width_rate,
                       alpha_coef, alpha_method,
                       perception_weight=1.0,
                       contour_weight=100.0,
                       style_weight=0.0,
                       content_weight=0.0,
                       xing_weight=0.0,
                       optimize_opacity=False,
                       init_type='with_content',
                       init_num_paths=128,
                       init_min_num_segments=1,
                       init_max_num_segments=4,
                       init_radius=0.1,
                       init_stroke_width=1.0,
                       max_stroke_width=1.0,
                       saving_verbose_lvl=0):
    """Run the style transfer."""
    cur_logger('Running params:')
    cur_logger(f'device={device}')
    cur_logger(f'saving_verbose_lvl={saving_verbose_lvl}')
    cur_logger(f'canvas_width={canvas_width}')
    cur_logger(f'canvas_height={canvas_height}')
    cur_logger(f'num_steps={num_steps}')
    cur_logger(f'point_rate={point_rate}')
    cur_logger(f'color_rate={color_rate}')
    cur_logger(f'width_rate={width_rate}')
    cur_logger(f'perception_weight={perception_weight}')
    cur_logger(f'contour_weight={contour_weight}')
    cur_logger(f'style_weight={style_weight}')
    cur_logger(f'content_weight={content_weight}')
    cur_logger(f'xing_weight={xing_weight}')
    cur_logger(f'optimize_opacity={optimize_opacity}')
    cur_logger(f'init_type={init_type}')
    cur_logger(f'init_num_paths={init_num_paths}')
    cur_logger(f'init_min_num_segments={init_min_num_segments}')
    cur_logger(f'init_max_num_segments={init_max_num_segments}')
    cur_logger(f'init_radius={init_radius}')
    cur_logger(f'init_stroke_width={init_stroke_width}')
    cur_logger(f'max_stroke_width={max_stroke_width}')
    cur_logger(f'content_shapes_len={len(content_shapes)}')
    cur_logger('=' * 20)
    cur_logger('Building the style transfer model..')

    content_style_loss_vgg_model, style_losses, content_losses = \
        get_style_model_and_losses(style_img, content_img, device)
    content_style_loss_vgg_model.requires_grad_(False)

    target_contour = get_contours(canvas_width, canvas_height, content_shapes, content_shape_groups)

    if saving_verbose_lvl < 3:
        pydiffvg.imwrite(target_contour.squeeze(0).permute(1, 2, 0).cpu(),
                         f'{folder_to_save_results}/contour.png', gamma=1.0)

    perception_loss = LPIPS(device=device).to(device)
    contour_loss = ContourLoss(target_contour)

    point_params = []
    color_params = []
    orig_opacities = []
    stroke_width_params = []

    if init_type == 'with_content':
        shapes, shape_groups = content_shapes, content_shape_groups
    elif init_type == 'with_rand_clipdraw':
        shapes, shape_groups = init_with_rand_curves(init_num_paths, canvas_width, canvas_height,
                                                     rand_opacity=optimize_opacity,
                                                     min_num_segments=init_min_num_segments,
                                                     max_num_segments=init_max_num_segments,
                                                     radius=init_radius,
                                                     stroke_width=init_stroke_width, )
    elif init_type == 'with_rand_LIVE':
        shapes, shape_groups = init_shapes_LIVE(init_num_paths, canvas_width, canvas_height,
                                                rand_opacity=optimize_opacity,
                                                init_type='random',
                                                min_num_segments=init_min_num_segments,
                                                max_num_segments=init_max_num_segments,
                                                radius=init_radius,
                                                stroke_width=init_stroke_width, )
    elif init_type == 'with_circles_LIVE':
        shapes, shape_groups = init_shapes_LIVE(init_num_paths, canvas_width, canvas_height,
                                                rand_opacity=optimize_opacity,
                                                init_type='circle',
                                                min_num_segments=init_min_num_segments,
                                                max_num_segments=init_max_num_segments,
                                                radius=init_radius,
                                                stroke_width=init_stroke_width, )
    else:
        print(f"Unknown init_type: `{init_type}`! Stopping...")
        return

    for shape in shapes:
        if isinstance(shape, pydiffvg.Path) or isinstance(shape, pydiffvg.Polygon):
            point_params.append(shape.points.requires_grad_())
            stroke_width_params.append(shape.stroke_width.requires_grad_())
    for shape_group in shape_groups:
        if isinstance(shape_group.fill_color, torch.Tensor):
            color_params.append(shape_group.fill_color.requires_grad_())
            orig_opacities.append(shape_group.fill_color[-1].detach().clone())
        elif isinstance(shape_group.fill_color, pydiffvg.LinearGradient):
            point_params.append(shape_group.fill_color.begin.requires_grad_())
            point_params.append(shape_group.fill_color.end.requires_grad_())
            color_params.append(shape_group.fill_color.stop_colors.requires_grad_())
            orig_opacities.append(shape_group.fill_color.stop_colors[-1].detach().clone())
        if isinstance(shape_group.stroke_color, torch.Tensor):
            color_params.append(shape_group.stroke_color.requires_grad_())
            orig_opacities.append(shape_group.stroke_color[-1].detach().clone())
        elif isinstance(shape_group.stroke_color, pydiffvg.LinearGradient):
            point_params.append(shape_group.stroke_color.begin.requires_grad_())
            point_params.append(shape_group.stroke_color.end.requires_grad_())
            color_params.append(shape_group.stroke_color.stop_colors.requires_grad_())
            orig_opacities.append(shape_group.stroke_color.stop_colors[-1].detach().clone())

    point_optimizer = optim.Adam(point_params, lr=point_rate)
    color_optimizer = optim.Adam(color_params, lr=color_rate)
    stroke_width_optimizers = optim.Adam(stroke_width_params, lr=width_rate)

    total_params_count = 0
    for p in [point_params, color_params, stroke_width_params]:
        total_params_count += sum(list(map(lambda arr: np.prod(arr.shape), p)))
    cur_logger(f'total_params_count={total_params_count}')

    cur_logger('Optimizing..')
    run = [0]
    losses = []
    best_loss = {'step': 0, 'loss': None}
    while run[0] <= num_steps:
        for sl in style_losses:
            sl.loss = 0
        for cl in content_losses:
            cl.loss = 0

        point_optimizer.zero_grad()
        color_optimizer.zero_grad()
        stroke_width_optimizers.zero_grad()

        img = svg_render(canvas_width, canvas_height, shapes, shape_groups)

        contour = get_contours(canvas_width, canvas_height, shapes, shape_groups).detach()

        if saving_verbose_lvl < 1:
            save_tensor_img(img.permute(2, 0, 1),
                            f'{folder_to_save_results}/{name}_step_{run[0]}_before_alpha.png')
            pydiffvg.imwrite(contour.squeeze(0).permute(1, 2, 0).cpu(),
                             f'{folder_to_save_results}/{name}_step_{run[0]}_contour.png', gamma=1.0)

        img = alpha_blending(img, alpha_coef, alpha_method).permute(2, 0, 1).unsqueeze(0).to(device)

        if saving_verbose_lvl < 2:
            save_png_svg(canvas_width, canvas_height, shapes, shape_groups,
                         f'{folder_to_save_results}/{name}_step_{run[0]}.png')
            save_tensor_img(img[0], f'{folder_to_save_results}/{name}_step_{run[0]}_alpha_blended.png')

        content_style_loss_vgg_model(img)
        style_score = 0
        content_score = 0
        for sl in style_losses:
            style_score += sl.loss
        for cl in content_losses:
            content_score += cl.loss
        weighted_style_score = style_score * style_weight
        weighted_content_score = content_score * content_weight

        perc_loss = perception_loss(img, style_img)
        cont_loss = contour_loss(contour)
        weighted_perc_score = perc_loss * perception_weight
        weighted_contour_score = cont_loss * contour_weight

        if xing_weight != 0:
            xing_scale = 0.01
            xing_score = xing_loss(point_params, xing_scale)
        else:
            xing_score = torch.tensor(0)
        weighted_xing = xing_score * xing_weight

        loss = weighted_perc_score + weighted_contour_score + \
               weighted_style_score + weighted_content_score + weighted_xing
        loss.backward()

        losses.append({'step': run[0], 'loss': loss.item()})
        if best_loss['loss'] is None or loss.item() < best_loss['loss']:
            best_loss['step'] = run[0]
            best_loss['loss'] = loss.item()

        if run[0] % 1 == 0:
            cur_logger("run {} {}:".format(name, run))
            cur_logger('Style Loss: {:4f} Content Loss: {:4f}'.format(
                style_score.item(), content_score.item()))
            cur_logger('Weighted Style Loss: {:4f} Weighted Content Loss: {:4f}'.format(
                weighted_style_score.item(), weighted_content_score.item()))
            cur_logger('Perception Loss: {:4f} Contour Loss: {:4f}'.format(
                perc_loss.item(), cont_loss.item()))
            cur_logger('Weighted Perception Loss: {:4f} Weighted Contour Loss: {:4f}'.format(
                weighted_perc_score.item(), weighted_contour_score.item()))
            cur_logger('Xing Loss: {:4f}'.format(xing_score.item()))
            cur_logger('Weighted Xing Loss: {:4f}'.format(weighted_xing.item()))
            cur_logger(f'Best Total Loss: {best_loss["loss"]} on step {best_loss["step"]}')
            cur_logger()

        point_optimizer.step()
        if not optimize_opacity:
            for color_param in color_params:
                color_param.grad[-1] = 0
        color_optimizer.step()
        stroke_width_optimizers.step()

        for color in color_params:
            color.data.clamp_(0, 1)
        for w in stroke_width_params:
            w.data.clamp_(0.0, max_stroke_width)
        run[0] += 1
    cur_logger("=" * 30)
    cur_logger("LOSSES:")
    cur_logger(losses)
    cur_logger("BEST LOSS:")
    cur_logger(best_loss)
    return shapes, shape_groups


def run_vector_nst(content_img_path, style_img_path,
                   num_iter=200,
                   saving_verbose_lvl=3,
                   results_folder="results"):
    name = get_model_name(content_img_path, style_img_path)
    os.makedirs(results_folder, exist_ok=True)
    folder_to_save_results = f"{results_folder}/vector_nst_{name}"
    os.makedirs(folder_to_save_results, exist_ok=True)
    log_file_path = f'{folder_to_save_results}/log.txt'
    cur_logger = lambda s="": logging(s, log_file_path)

    canvas_width, canvas_height, shapes, shape_groups = pydiffvg.svg_to_scene(content_img_path)

    max_sz = max(canvas_width, canvas_height)
    new_max_sz = 512
    scale = new_max_sz / max_sz
    canvas_width, canvas_height = pydiffvg.rescale_points(shapes, shape_groups, scale, canvas_width, canvas_height)
    import math
    canvas_width, canvas_height = math.ceil(canvas_width), math.ceil(canvas_height)

    save_png_svg(canvas_width, canvas_height, shapes, shape_groups, f'{folder_to_save_results}/init.png')

    # rates
    if len(shapes) < 300:
        point_rate = 0.2
        color_rate = 0.01
        width_rate = 0.1
    elif len(shapes) < 1000:
        point_rate = 0.3
        color_rate = 0.01
        width_rate = 0.1
    elif len(shapes) < 1600:
        point_rate = 0.4
        color_rate = 0.01
        width_rate = 0.1
    else:
        point_rate = 0.8
        color_rate = 0.01
        width_rate = 0.1

    content_img = svg_render(canvas_width, canvas_height, shapes, shape_groups)
    style_img = image_loader(style_img_path, canvas_width, canvas_height)
    style_img = alpha_blending(style_img.permute(1, 2, 0)).permute(2, 0, 1).unsqueeze(0).to(device)
    content_img = alpha_blending(content_img).permute(2, 0, 1).unsqueeze(0).to(device)

    save_tensor_img(content_img[0], f'{folder_to_save_results}/init_content_blended.png')
    save_tensor_img(style_img[0], f'{folder_to_save_results}/init_style_image.png')

    assert style_img.size() == content_img.size(), \
        "we need to import style and content images of the same size"

    tic = time.perf_counter()

    alpha_coef = 1.0
    alpha_method = 1
    shapes, shape_groups = run_style_transfer(
        cur_logger, folder_to_save_results, name,
        content_img, style_img,
        canvas_width, canvas_height,
        shapes, shape_groups, num_iter, point_rate, color_rate, width_rate,
        saving_verbose_lvl=saving_verbose_lvl, alpha_coef=alpha_coef, alpha_method=alpha_method)

    save_png_svg(canvas_width, canvas_height, shapes, shape_groups, f'{folder_to_save_results}/output_{name}.png')
    toc = time.perf_counter()
    cur_logger("+" * 50)
    cur_logger(f"{name} is finished!")
    cur_logger(f"Total Time: {toc - tic:0.4f} seconds")
    cur_logger("+" * 50)


def run_vector_nst_multi_params(content_svg, style, style_mime_type,
                                iter_num, is_lr_default,
                                lr_point, lr_color, lr_stroke_width,
                                contour_loss_weight, perception_loss_weight,
                                style_loss_weight, content_loss_weight,
                                xing_loss_weight,
                                optimize_opacity,
                                init_type,
                                init_num_paths,
                                init_min_num_segments,
                                init_max_num_segments,
                                init_radius,
                                init_stroke_width,
                                max_stroke_width,
                                alpha_blending_method, alpha_blending_coef):
    alpha_method = 1 if alpha_blending_method == "method1" else 0
    alpha_coef = float(alpha_blending_coef)
    name = "no_name"

    canvas_width, canvas_height, shapes, shape_groups = pydiffvg.svg_str_to_scene(content_svg)

    max_sz = max(canvas_width, canvas_height)
    new_max_sz = 512
    scale = new_max_sz / max_sz
    canvas_width, canvas_height = pydiffvg.rescale_points(shapes, shape_groups, scale, canvas_width, canvas_height)
    import math
    canvas_width, canvas_height = math.ceil(canvas_width), math.ceil(canvas_height)

    if style_mime_type == "image/svg+xml":
        img_style_tensor = svg_render(*pydiffvg.svg_str_to_scene(style), quality=4)
        style_pil_img = transforms.ToPILImage()(img_style_tensor.permute(2, 0, 1))
        image = style_pil_img.resize((canvas_width, canvas_height))
        style_tensor = transforms.ToTensor()(image)
    else:
        import re
        from io import BytesIO
        import base64
        image_data = re.sub('^data:image/.+;base64,', '', style)
        image_data = base64.b64decode(image_data)
        style_tensor = image_loader(BytesIO(image_data), canvas_width, canvas_height)
    style_tensor = style_tensor.to(device)

    content_img = svg_render(canvas_width, canvas_height, shapes, shape_groups)
    style_img = alpha_blending(style_tensor.permute(1, 2, 0), alpha_coef, alpha_method)
    style_img = style_img.permute(2, 0, 1).unsqueeze(0).to(device)
    content_img = alpha_blending(content_img, alpha_coef, alpha_method)
    content_img = content_img.permute(2, 0, 1).unsqueeze(0).to(device)

    assert style_img.size() == content_img.size(), \
        "we need to import style and content images of the same size"

    # rates
    if is_lr_default.lower() == "true":
        if len(shapes) < 300:
            point_rate = 0.2
            color_rate = 0.01
            width_rate = 0.1
        elif len(shapes) < 1000:
            point_rate = 0.3
            color_rate = 0.01
            width_rate = 0.1
        elif len(shapes) < 1600:
            point_rate = 0.4
            color_rate = 0.01
            width_rate = 0.1
        else:
            point_rate = 0.8
            color_rate = 0.01
            width_rate = 0.1
    else:
        point_rate = float(lr_point)
        color_rate = float(lr_color)
        width_rate = float(lr_stroke_width)

    optimize_opacity = True if optimize_opacity.lower() == "true" else False

    tic = time.perf_counter()
    shapes, shape_groups = \
        run_style_transfer(default_logger, "", name, content_img,
                           style_img, canvas_width, canvas_height, shapes,
                           shape_groups, int(iter_num), point_rate, color_rate, width_rate,
                           alpha_coef,
                           alpha_method,
                           perception_weight=float(perception_loss_weight),
                           contour_weight=float(contour_loss_weight),
                           style_weight=float(style_loss_weight),
                           content_weight=float(content_loss_weight),
                           xing_weight=float(xing_loss_weight),
                           optimize_opacity=optimize_opacity,
                           init_type=init_type,
                           init_num_paths=int(init_num_paths),
                           init_min_num_segments=int(init_min_num_segments),
                           init_max_num_segments=int(init_max_num_segments),
                           init_radius=float(init_radius),
                           init_stroke_width=float(init_stroke_width),
                           max_stroke_width=float(max_stroke_width),
                           saving_verbose_lvl=5)

    toc = time.perf_counter()
    default_logger("+" * 50)
    default_logger(f"{name} is finished!")
    default_logger(f"Total Time: {toc - tic:0.4f} seconds")
    default_logger("+" * 50)

    return pydiffvg.svg_to_str(canvas_width, canvas_height, shapes, shape_groups)
