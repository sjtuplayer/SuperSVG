import pydiffvg
import os
import glob
import torch

svg_dir = "/home/huteng/SVG/AttnPainter/compared-results/vector_magic_detail/art/"
output_dir = "./Vector Magic/"
os.makedirs(output_dir,exist_ok=True)

for svg_path in glob.glob(svg_dir + '*.svg'):

    svg_name = svg_path.split('/')[-1].split('.')[0]
    print(svg_name)
    save_path = os.path.join(output_dir, svg_name + '.png')

    canvas_width, canvas_height, shapes, shape_groups = pydiffvg.svg_to_scene(svg_path)

    scene_args = pydiffvg.RenderFunction.serialize_scene(canvas_width, canvas_height, shapes, shape_groups)

    render = pydiffvg.RenderFunction.apply

    img = render(canvas_width, canvas_height, 2, 2, 0, None, *scene_args)

    para_bg = torch.tensor([1., 1., 1.], requires_grad=False, device=img.device)
    img = img[:, :, 3:4] * img[:, :, :3] + para_bg * (1 - img[:, :, 3:4])

    img = img.detach().cpu()

    pydiffvg.imwrite(img, save_path, gamma=1.0)