import matplotlib.pyplot as plt

def plot_dataset(dataset, rows, cols, image_width):
    fig, axs = plt.subplots(rows, cols, figsize=(cols*image_width, rows*image_width))
    for row in range(rows):
        for col in range(cols):
            img = dataset[row*cols+col][0]['image']
            lbl = dataset[row*cols+col][1]
            axs[row,col].imshow(img.permute(1,2,0))
            axs[row,col].axis('off')
            axs[row,col].set_title('Class '+str(lbl), fontsize=20)
    return fig, axs