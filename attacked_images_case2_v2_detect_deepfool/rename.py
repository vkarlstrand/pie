import os
filelist = os.listdir()
for filename in filelist:
    if filename.endswith('.png'):
        if filename.startswith('ori_'):
            os.rename(filename, filename.strip('ori_').strip('.png') + '_img.png')
        elif filename.startswith('pert_'):
            os.rename(filename, filename.strip('pert_').strip('.png') + '_att.png')