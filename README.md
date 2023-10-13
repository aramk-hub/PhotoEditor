# Mini Photo Editor

## How It Works
This photo editor has three different editing options as of now, with more to come. The options are denoising, HDR imaging via exposure stacking, and focus stacking. As of now, the editor uses three different scripts, but the goal is to create a small application for learning purposes. All output images will be placed in the `./Outputs` directory.

### Denoising
The editor can denoise an image similar to below, with a noisy image on the left and denoised on the right: 
<img src="noisy_image.png" alt="drawing" width="300"></img>

To do so, add an image to the `./Denoise` directory, and run the program with `python denoiser.py` or `python3 denoiser.py`. 

### HDR Imaging 
HDR imaging allows for a brighter pictures with more contrast and realistic lighting, rather than the standard dynamic range most cameras take photos with. To accomplish this, computers need several photos, from exposures above and below that of the reference image. Once we have this, we can "stack" the images together. To use this section, upload your pictures to the `./HDR` directory and run the script with `python mergeExposure.py` or `python3 mergeExposure.py`. 

### Focus Stacking - COMING SOON
Focus stacking allows for unique photo composition that we otherwise find difficult to accomplish without the help of computer vision. Essentially, we can use multiple photos of a scene with different focus points, and "stack" them again in order to create a photo with multiple parts in focus. To use this option, upload your images to the `./FocusStacking` directory and run the script with `python focusStacker.py` or `python3 focusStacker.py`. 

### Remarks
This editor utilizes OpenCV to edit the photos and many functions within it. The main idea I struggled with was image registration, and correctly aligning the images in order to match all the properties correctly. This was done to learn more about computer vision as an entry point into a Master's program, and I wish to keep this project going as I work through the field.
