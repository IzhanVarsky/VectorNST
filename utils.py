from PIL import Image
from torchvision import transforms
import pydiffvg


def svg_render(canvas_width, canvas_height, shapes, shape_groups, quality=2):
    scene_args = pydiffvg.RenderFunction.serialize_scene(canvas_width, canvas_height, shapes, shape_groups)
    render = pydiffvg.RenderFunction.apply
    img = render(canvas_width,  # width
                 canvas_height,  # height
                 quality,  # num_samples_x
                 quality,  # num_samples_y
                 0,  # seed
                 None,
                 *scene_args)
    return img


def save_tensor_img(img, path):
    unloader = transforms.ToPILImage()
    save_pil_img(unloader(img), path)


def save_png_svg(canvas_width, canvas_height, shapes, shape_groups, f, quality=2):
    img = svg_render(canvas_width, canvas_height, shapes, shape_groups, quality)
    save_tensor_img(img.permute(2, 0, 1), f)
    # pydiffvg.imwrite(img.cpu(), f, gamma=1.0)
    pydiffvg.save_svg(f.replace(".png", ".svg"), canvas_width, canvas_height, shapes, shape_groups)


def pil_convert_RGBA_to_RGB(img, back_color=255):
    png = img.convert('RGBA')
    background = Image.new('RGBA', png.size, (back_color, back_color, back_color))
    alpha_composite = Image.alpha_composite(background, png)
    image = alpha_composite.convert("RGB")
    return image


def save_pil_img(img, path):
    img.save(path, quality=100, subsampling=0)
    # pil_convert_RGBA_to_RGB(img).save(path.replace(".png", ".jpg"), format='JPEG', quality=100, subsampling=0)


def get_model_name(content_img_path, style_img_path):
    s_name = style_img_path.split('/')[-1].split('.')[0]
    c_name = content_img_path.split('/')[-1].split('.')[0]
    return f"{c_name}_stylized_{s_name}"


def logging(msg, log_file=None):
    print(msg)
    if log_file is not None:
        with open(log_file, 'a+') as f:
            f.write(str(msg) + '\n')
