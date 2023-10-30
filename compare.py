import numpy as np
import pandas as pd
import glob
import pydiffvg

SAMVG_dirs = {
    'VM':'/home/huteng/SVG/AttnPainter/compared-results/vector_magic_detail/art/'
}

df = pd.DataFrame(columns=['type'] + list(SAMVG_dirs.keys()))

avg_params = {'type': 'avg_params'}
avg_paths = {'type': 'avg_paths'}

for key in SAMVG_dirs.keys():
    SVG_dir = SAMVG_dirs[key]
    total_params = 0
    total_paths = 0
    count = 0
    for svg in glob.glob(SVG_dir + '*.svg'):
        print(svg)
        name = svg.split('/')[-1].split('.')[0]
        category = name.split('_')[0]

        print(name)
        if name=='1':
            continue
        canvas_width, canvas_height, shapes, shape_groups = pydiffvg.svg_to_scene(svg)

        num_paths = len(shapes)
        # if num_paths >= 128:
        #     continue
        total_paths += len(shapes)
        count += 1

        for shape in shapes:
            total_params += shape.points.shape[0] * 2  # points
            # fill color
            if key == 'SAMVG_alpha':
                total_params += 1
            total_params += 3
        print(total_paths,total_params)
    avg_paths[key] = total_paths / count
    avg_params[key] = total_params / count

print(avg_paths)
print(avg_params)
df = df.append(avg_paths, ignore_index=True)
df = df.append(avg_params, ignore_index=True)

df.to_csv('./stats/avg_params_v2.csv')
print(df)


