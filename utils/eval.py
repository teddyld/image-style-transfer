from skimage import feature
import matplotlib.pyplot as plt

WIKIART_STYLE_MAP = {
    'Baroque': 1,
    'Cubism': 2,
    'Early_Renaissance': 3,
    'Pointillism': 4,
    'Ukiyo_e': 5,
}


from skimage.metrics import structural_similarity

def compute_ssim(content_images, stylised_images):
    """Returns the average structural similarity (SSIM) between the content and stylised images"""
    ssim_sum = 0
    for content, stylised in zip(content_images, stylised_images):
        
        # Convert to numpy array
        content_img = content.detach().cpu().numpy()
        stylised_img = stylised.detach().cpu().numpy()
        
        # Convert image to grayscale
        gray_content = content_img[0, :, :]
        gray_stylised = stylised_img[0, :, :]
        
        # Compute SSIM
        score = structural_similarity(gray_content, gray_stylised, data_range=1.0)
        ssim_sum += score
    
    return ssim_sum

def plot_results(content_images, style_images, style_labels, stylised_images, nrows=5, model_name=""):
    """Plot the stylisation results
    
    Args:
        content_images (tensor -> shape(B, C, H, W)): Defines the content images
        style_images (tensor -> shape(B, C, H, W)): Defines the style images
        style_labels (tensor -> shape(B)): Defines the labels of the style images
        stylised_images (tensor -> shape(B, C, H, W)): Defines the output stylised images
        nrows (int): Defines the number of samples to plot
        model_name (string): Defines the model name to print on the plot
    """
    fig, axes = plt.subplots(nrows=nrows, ncols=5, figsize=(20, 20), subplot_kw={'xticks': [], 'yticks': []})
    for i in range(nrows):
        content_img = content_images[i].cpu()
        style_img = style_images[i].cpu()
        stylised_img = stylised_images[i].cpu()
        label = style_labels[i].cpu()
        
        # Plot content, style and stylised images
        axes[i][0].imshow(content_img.permute(1, 2, 0))
        axes[i][1].imshow(style_img.permute(1, 2, 0))
        axes[i][2].imshow(stylised_img.permute(1, 2, 0))
        
        label = list(WIKIART_STYLE_MAP.keys())[list(WIKIART_STYLE_MAP.values()).index(label)]
        axes[i][1].set_title(label, size=12)
        
        # Plot canny edges of content and stylised image
        content_img = content_img.permute(1, 2, 0).numpy()[:, :, 0]
        stylised_img = stylised_img.permute(1, 2, 0).numpy()[:, :, 0]
        axes[i][3].imshow(feature.canny(content_img, sigma=1), cmap="copper")
        axes[i][4].imshow(feature.canny(stylised_img, sigma=1), cmap="copper")
        
        # Set subplot titles
        if i == nrows - 1:
            axes[i][0].text(0.5, -0.07, "Content", size=16, ha="center", transform=axes[i][0].transAxes)
            axes[i][1].text(0.5, -0.07, "Style", size=16, ha="center", transform=axes[i][1].transAxes)
            axes[i][2].text(0.5, -0.07, f"{model_name}", size=16, ha="center", transform=axes[i][2].transAxes)
            axes[i][3].text(0.5, -0.07, f"Content Canny Edges", size=16, ha="center", transform=axes[i][3].transAxes)
            axes[i][4].text(0.5, -0.07, f"{model_name} Canny Edges", size=16, ha="center", transform=axes[i][4].transAxes)

    fig.suptitle(f'{model_name} Results', size=21, y=0.99)
    plt.tight_layout()
    plt.show()