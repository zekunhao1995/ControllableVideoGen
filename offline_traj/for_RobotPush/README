# NOTES ON USAGE
For generating trajectories from video (Tuned for Robot Push dataset).
As a part of code for "Controllable Video Generation with Sparse Trajectories", CVPR'18.
- **batch_process_dataset.py**: Generate trajectories. To set up search for comments containing `[EDIT ME!]`. Train/test split hard-coded inside.
- **view_traj.py**: Visualize generated trajectories. Detailed instructions inside the file.
- **\*.cpp** & **\*.h**: Code for *Dense Trajectories* algorithm. Slightly modified. 

**Warning: The code is provided in its original form without any cleanup.**

# NOTES ON MODIFICATIONS
Code originated from:
http://lear.inrialpes.fr/people/wang/dense_trajectories
```
@inproceedings{wang:2011:inria-00583818:1,
  AUTHOR = {Heng Wang and Alexander Kl{\"a}ser and Cordelia Schmid and Cheng-Lin Liu},
  TITLE = {{Action Recognition by Dense Trajectories}},
  BOOKTITLE = {IEEE Conference on Computer Vision \& Pattern Recognition},
  YEAR = {2011},
  MONTH = Jun,
  PAGES = {3169-3176},
  ADDRESS = {Colorado Springs, United States},
  URL = {http://hal.inria.fr/inria-00583818/en}
}
```
- Modified to support more modern version of OpenCV
- Need OpenCV >= 3.0 with "Contrib" add-in for SURF and SIFT feature extraction.
- Converted stand-alone excutable to dynamic library for Python CFFI calling


# Followings are the original README for Dense Trajectories


### Compiling ###

In order to compile the improved trajectories code, you need to have the following libraries installed in your system:
* OpenCV library (tested with OpenCV-2.4.2)
* ffmpeg library (tested with ffmpeg-0.11.1)

Currently, the libraries are the latest versions. In case they will be out of date, you can also find them on our website: http://lear.inrialpes.fr/people/wang/improved_trajectories

If these libraries are installed correctly, simply type 'make' to compile the code. The executable will be in the directory './release/'.

### test video decoding  ###

The most complicated part of compiling is to install opencv and ffmpeg. To make sure your video is decoded properly, we have a simple code (named 'Video.cpp') for visualization:

./release/Video your_video.avi

If your video plays smoothly, congratulations! You are just one step before getting the features.

If there is a bug and the video can't be decoded, you need first fix your bug. You can find plenty of instructions about how to install opencv and ffmpeg on the web.

### compute features on a test video ###

Once you are able to decode the video, computing our features is simple:

./release/DenseTrackStab ./test_sequences/person01_boxing_d1_uncomp.avi | gzip > out.features.gz

Now you want to compare your file out.features.gz with the file that we have computed to verify that everything is working correctly. To do so, type:

vimdiff out.features.gz ./test_sequences/person01_boxing_d1.gz 

Note that due to different versions of codecs, your features may be slightly different with ours. But the major part should be the same.

Due to the randomness of RANSAC, you may get different features for some videos. But for the example "person01_boxing_d1_uncomp.avi", I don't observe any randomness. 

There are more explanations about our features on the website, and also a list of FAQ.

### History ###

* October 2013: improved_trajectory_release.tar.gz
                The code is an extension of dense_trajectory_release_v1.2.tar.gz

### Bugs and extensions ###

If you find bugs, etc., feel free to drop me a line. Also if you developed some extension to the program, let me know and I can include it in the code. You can find my contact data on my webpage, as well.

http://lear.inrialpes.fr/people/wang/

### LICENSE CONDITIONS ###

Copyright (C) 2011 Heng Wang 

This program is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License
as published by the Free Software Foundation; either version 2
of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, write to the Free Software
Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.

