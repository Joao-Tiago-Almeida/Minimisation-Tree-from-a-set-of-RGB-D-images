# Minimisation Tree from a set of RGB-D Images

3D scene reconstruction is an important technique in the computer vision field. Inspired by the structure from motion systems, we propose a tree that reduces the cumulative error of the rigid transformation to the world reference image, using a set of images which are gained by a commonly used camera. There are many key techniques in 3D reconstruction from image sequences, including feature matching, RANSAC, rigid transformation, projective reconstruction, etc.


The system tracks all the images and tries to associate to another image that is already known to be connected to the world reference. The algorithm keeps checking all the neighbours with an incremental distance and evaluates the percentage of inliers detected, therefore it allows to connect non sequential images if the rigid transformation is valid. This method constructs an almost optimal tree must faster than comparing every image to all the other data set images.


The effectiveness of our algorithms is evaluated in the experiments with real image sequences.

Report: [file](./Minimisation_Tree_from_a_set_of_RGB_D_images.pdf)

Final Grade: 17/20
