# UAS_MACHINE_VISION_ALY MUHAMMAD GANY

import numpy as np
import cv2 as cv

def map_to_8bit(img_in):
    img_out = None
    img_out = ((img_in - img_in.min()) * 255/(img_in.max() - img_in.min())).astype(np.uint8)
    return img_out

# Lengkapi fungsi berikut ini agar menghasilkan luaran berupa hasil convolution dengan parameter sebagai berikut
# stride = 1
# same padding
# kernel : merupakan filter yang digunakan, ukuran dapat bervariasi (3x3, 5x5, 7x7, etc)
# pada fungsi di bawah ini, lakukan konversi img_in dari BGR ke Grayscale terlebih dahulu, kemudian kalkulasi hasil correlation
def convolve(img_in, kernel):
    img_out = None
    # Tuliskan source code anda disini
    img_gray = cv.cvtColor(img_in, cv.COLOR_BGR2GRAY)
    
    stride = 1
    pad = kernel.shape[0] -1 //2
    filter_size = kernel.shape[0]
    k = filter_size //2
    
    w_out = int((img_gray.shape[1]+2*pad-filter_size)/stride)+1
    h_out = int((img_gray.shape[0]+2*pad-filter_size)/stride)+1
    img_out = np.zeros((h_out, w_out))
    
    img_samepad = cv.copyMakeBorder(img_gray,pad,pad,pad,pad,cv.BORDER_CONSTANT)
    
    for i in range(0, img_out.shape[0]):
        for j in range(0, img_out.shape[1]):
            val = 0
            for u in range(-1*k, k+1):
                 for v in range(-1*k, k+1):
                      val += kernel[u+k,v+k] * img_samepad[i-u+k,j-v+k]
            img_out[i,j] = val
            
    # akhir code
    return img_out

def main():
    bgr_image = cv.imread("../images/image.png")
    # program untuk menguji coba convolution filter, edge detection dengan sobel filter
    sobel_x = np.array([[1, 0, -1],
                        [2, 0, -2],
                        [1, 0, -1]])

    sobel_y = np.array([[1, 2, 1],
                        [0, 0, 0],
                        [-1, -2, -1]])
    output_x = convolve(bgr_image, sobel_x)
    output_y = convolve(bgr_image, sobel_y)
    sobel_output = np.sqrt(output_x**2 + output_y**2)
    # cv.imshow("Sobel X", map_to_8bit(output_x))
    # cv.imshow("Sobel Y", map_to_8bit(output_y))
    cv.imshow("Sobel Output", map_to_8bit(sobel_output))
    cv.waitKey(0)

if __name__ == "__main__":
    main()
