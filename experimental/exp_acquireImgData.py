try:
    import cv2
    import matplotlib.pyplot as plt
    import numpy as np
except:
    print("\n---\nThis examples requires cv2 (python3-opencv)!\n---\n")
    raise
import sys
import time
import visiontransfer

MAX_IMAGES = 1
if __name__=='__main__':
    metaData = {}
    device_enum = visiontransfer.DeviceEnumeration()
    devices = device_enum.discover_devices()
    if len(devices) < 1:
        print('No devices found')
        sys.exit(1)

    print('Found these devices:')
    for i, info in enumerate(devices):
        print(f'  {i+1}: {info}')
    selected_device = 0 if len(devices)==1 else (int(input('Device to open: ') or '1')-1)
    print(f'Selected device #{selected_device+1}')
    device = devices[selected_device]

    print('Ask parameter server to set stereo mode ...')
    params = visiontransfer.DeviceParameters(device)
    params.set_operation_mode(visiontransfer.OperationMode.STEREO_MATCHING)

    print('Starting acquisition ...')
    transfer = visiontransfer.AsyncTransfer(device)

    i = 0
    while True:
        if i>=MAX_IMAGES: break
        #retrive image set (left and right image)
        image_set = transfer.collect_received_image_set()
    
        # get image data from left camera
        pixeldata_left = image_set.get_pixel_data(
            visiontransfer.ImageType.IMAGE_LEFT, force8bit=True, do_copy=True
        )
        # get disparity data
        # retrieve disparity range: see Nerian Scarlet Documentation (chapter 7.2 Disparity Maps, 24.04.2023) for more information
        disprange = image_set.get_disparity_range()[1]
        if disprange == 511:
            factor = 8  # factor to retrieve the disparity values from neriam image set
        elif disprange == 255:
            factor = 16
        else:
            raise ValueError
        pixeldata_disparity = (
            image_set.get_pixel_data(
                visiontransfer.ImageType.IMAGE_DISPARITY, do_copy=True
            )
            / factor
        )
        disparityConversionFactor = 1/np.max(pixeldata_disparity)*255 # factor to convert disparity value in 0-255 image range

        # aggregate metadata information
        metaData["width"] = image_set.get_height()
        metaData["height"] = image_set.get_width()
        metaData["qmatrix"] = image_set.get_qmatrix()
        metaData["cx"] = -metaData["qmatrix"][0, 3] # According to https://stackoverflow.com/questions/27374970/q-matrix-for-the-reprojectimageto3d-function-in-opencv
        metaData["cy"] = -metaData["qmatrix"][1, 3]
        metaData["fx"] = metaData["qmatrix"][2, 3]
        metaData["fy"] = metaData["qmatrix"][2, 3]
        metaData["Tx"] = -1/metaData["qmatrix"][3, 2]
        metaData["disparityConversionFactor"] = disparityConversionFactor
        
        #show color image
        fig = plt.figure()
        if len(pixeldata_left.shape)==3:
            # Color image: Nerian API uses RGB, OpenCV uses BGR
            img_left_rbg = cv2.cvtColor(pixeldata_left, cv2.COLOR_RGB2BGR)
            #cv2.imwrite(f'channel{ch}_'+('%02d'%i)+'.png', cv2.cvtColor(imgdata, cv2.COLOR_RGB2BGR))
        else:
            img_left_rbg = np.stack(pixeldata_left,pixeldata_left,pixeldata_left)
            plt.imshow(pixeldata_left)
            plt.show(block=True)
        plt.imshow(img_left_rbg)
        plt.show(block=False)
            #cv2.imwrite(f'channel{ch}_'+('%02d'%i)+'.png', imgdata)

        # show disparity image
        fig = plt.figure()
        img_disparity = (pixeldata_disparity * disparityConversionFactor).astype(np.uint8)
        if len(pixeldata_disparity.shape)==3:
            print("Obtained 3 dimensions for each pixel. Expected to obtain only one dimension")
            raise
        else:
            plt.imshow(img_disparity, cmap="gray")
            plt.show(block=True)
            #cv2.imwrite(f'channel{ch}_'+('%02d'%i)+'.png', imgdata)

        # save images
        i += 1
    
    # save meta data