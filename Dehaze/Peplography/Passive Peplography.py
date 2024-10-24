import numpy as np
import matplotlib.pyplot as plt
import cv2

# Photon counting detection model
def PCmodel(img, Np):
    """
    image Matrix, Number of Photons
    :param img: 2D or 3D image(grayscale or color image)
    :param Np: Number of Photons
    :return: Normalize image (Max = 1, Min = 0)
    """
    Planck = 6.62607015e-34
    img = np.array(img)

    width, height = img.shape[0], img.shape[1]

    if img.ndim == 2:
        NormImg = img / np.sum(img)
        # Poisson distrbution in Numpy can't calculate about 2 or multi dimensions.    
        linearImg = NormImg.copy().flatten()
        photonDetection = np.random.poisson(Np*linearImg)

        addPhoton = linearImg + photonDetection

        # if the value is over the 1, that pixels are saturated pixels
        reconImg = np.clip(addPhoton, 0, 1)
        return reconImg.reshape(width, height)
    else:
        gamma = [1.4497, 1.1270, 1] # R, G, B's gamma values, respectively
        colorImg = np.zeros_like(img)
        for channel in enumerate(gamma):
            NormImg = img[:, :, channel[0]] / np.sum(img[:, :, channel[0]])
            linearImg = NormImg.copy().flatten()
            photonDetection = np.random.poisson(Np*linearImg*channel[1])

            addPhoton = photonDetection
            reconImg = np.clip(addPhoton, 0, 1)
            colorImg[:, :, channel[0]] += reconImg.reshape(width, height)
        return np.array(colorImg)

# Passive Peplography
def passivePeplography(haze, Np=100000, Wx=1000, Wy=1000):
    haze = np.array(haze)
    
    width, height = haze.shape[0], haze.shape[1]
    estimationWindows = np.ones((Wx, Wy)) / (Wx * Wy)

    scatteringMedia = cv2.filter2D(tarRGB, -1, estimationWindows)

    Ip = haze - scatteringMedia
    
    Ip = (Ip - Ip.min())/(Ip.max() - Ip.min())
    
    photonCounting = PCmodel(Ip, Np)
    
    return photonCounting

if __name__ == "__main__":
    imgPath = "haze image path"
    bgr = cv2.imread(imgPath, cv2.IMREAD_COLOR)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    haze = np.array(rgb)

    dehaze = passivePeplography(haze)

    plt.figure("compare", figsize=[10, 4])
    plt.suptitle("dehaze")

    plt.subplot(1, 2, 1)
    plt.title("Original image")
    plt.imshow(haze)
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title("Reconstructed image")
    plt.imshow(dehaze)
    plt.axis("off")

    plt.tightlayout()
    plt.show()
