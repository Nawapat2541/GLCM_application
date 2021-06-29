import matplotlib.pyplot as plt

from skimage.feature import greycoprops, greycomatrix
from skimage import data

PATCH_SIZE = 33

# open the camera image
image = data.camera()

# select some patches from grassy areas of the image
grass_locations = [(280, 454), (342, 223), (444, 192), (455, 455)]
grass_patches = []
for loc in grass_locations:
    grass_patches.append(image[loc[0]:loc[0] + PATCH_SIZE,
                         loc[1]:loc[1] + PATCH_SIZE])

# select some patches from sky areas of the image
sky_locations = [(38, 34), (139, 12), (37, 437), (145, 379)]
sky_patches = []
for loc in sky_locations:
    sky_patches.append(image[loc[0]:loc[0] + PATCH_SIZE,
                       loc[1]:loc[1] + PATCH_SIZE])

# select some patches from shirt areas of the image
shirt_locations = [(300, 76), (146, 104), (175, 65), (256, 23)]
shirt_patches = []
for loc in shirt_locations:
    shirt_patches.append(image[loc[0]:loc[0] + PATCH_SIZE,
                         loc[1]:loc[1] + PATCH_SIZE])


# compute some GLCM properties each patch
xs = []
ys = []
for patch in (grass_patches + sky_patches + shirt_patches):
    glcm = greycomatrix(patch, distances=[5], angles=[0], levels=256,
                        symmetric=True, normed=True)
    xs.append(greycoprops(glcm, 'dissimilarity')[0, 0])
    ys.append(greycoprops(glcm, 'correlation')[0, 0])

# create the figure
fig = plt.figure(figsize=(8, 8))

# display original image with locations of patches
ax = fig.add_subplot(5, 2, 1)
ax.imshow(image, cmap=plt.set_cmap('Greys'),
          vmin=0, vmax=255)
for (y, x) in grass_locations:
    ax.plot(x + PATCH_SIZE / 3, y + PATCH_SIZE / 3, 'gs')
for (y, x) in sky_locations:
    ax.plot(x + PATCH_SIZE / 3, y + PATCH_SIZE / 3, 'bs')
for (y, x) in shirt_locations:
    ax.plot(x + PATCH_SIZE / 3, y + PATCH_SIZE / 3, 'rs')
ax.set_xlabel('Original Image')
ax.set_xticks([])
ax.set_yticks([])
ax.axis('image')

# for each patch, plot (dissimilarity, correlation)
ax = fig.add_subplot(5, 2, 2)
ax.plot(xs[:len(grass_patches)], ys[:len(grass_patches)], 'go',
        label='Grass')
ax.plot(xs[len(sky_patches):], ys[len(sky_patches):], 'bo',
        label='Sky')
ax.plot(xs[len(shirt_patches):], ys[len(shirt_patches):], 'ro',
        label='Shirt')
ax.set_xlabel('GLCM Dissimilarity')
ax.set_ylabel('GLCM Correlation')
ax.legend()

# display the image patches
for i, patch in enumerate(grass_patches):
    ax = fig.add_subplot(5, len(grass_patches), len(grass_patches)*1 + i + 1)
    ax.imshow(patch, cmap=plt.set_cmap('Greys'),
              vmin=0, vmax=255)
    ax.set_xlabel('Grass %d' % (i + 1))

for i, patch in enumerate(sky_patches):
    ax = fig.add_subplot(5, len(sky_patches), len(sky_patches)*2 + i + 1)
    ax.imshow(patch, cmap=plt.set_cmap('Greys'),
              vmin=0, vmax=255)
    ax.set_xlabel('Sky %d' % (i + 1))

for i, patch in enumerate(shirt_patches):
    ax = fig.add_subplot(5, len(shirt_patches), len(shirt_patches)*3 + i + 1)
    ax.imshow(patch, cmap=plt.set_cmap('Greys'),
              vmin=0, vmax=255)
    ax.set_xlabel('Shirt %d' % (i + 1))

# display the patches and plot
fig.suptitle('Grey level co-occurrence matrix features', fontsize=14, y=1.05)
plt.tight_layout()
plt.show()
