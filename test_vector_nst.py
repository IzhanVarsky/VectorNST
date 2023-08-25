import argparse

from vector_nst import run_vector_nst

parser = argparse.ArgumentParser(description='VectorNST runner')
parser.add_argument('--content_img', required=True,
                    type=str, help='Input content image in .svg format.')
parser.add_argument('--style_img', required=True,
                    type=str,
                    help='Input style image in bitmap format. '
                         'If you image is in .svg format you should first rasterize it.')
parser.add_argument('--num_iter', required=False, default=100,
                    type=int, help='Number of optimizing iterations to perform.')
parser.add_argument('--output_dir', required=False, default="results",
                    type=str, help='Folder to save the results.')
parser.add_argument('--saving_verbose_lvl', required=False, default=1,
                    type=int, help='If 0 or less then all intermediate files will be saved. '
                                   'If 1 then images before alpha blending will not be saved. '
                                   'If 2 then the images in the intermediate steps will not be saved. '
                                   'If 3 or more then only init content, style, '
                                   'contour and output images will be saved.')
args = parser.parse_args()

content_img_path = args.content_img
style_img_path = args.style_img
num_iter = args.num_iter
saving_verbose_lvl = args.saving_verbose_lvl
results_folder = args.output_dir

run_vector_nst(content_img_path, style_img_path, num_iter, saving_verbose_lvl, results_folder)
