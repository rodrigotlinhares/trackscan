main.o: main.cc /usr/local/include/opencv/cv.h \
 /usr/local/include/opencv/highgui.h mcv.h \
 planar_pattern_detector_builder.h planar_pattern_detector.h keypoint.h \
 pyr_yape06.h fine_gaussian_pyramid.h homography_estimator.h \
 homography06.h affine_image_generator06.h affine_transformation_range.h \
 fern_based_point_classifier.h ferns.h template_matching_based_tracker.h
mcv.o: mcv.cc /usr/local/include/opencv/cv.h \
 /usr/local/include/opencv/highgui.h mcv.h
planar_pattern_detector.o: planar_pattern_detector.cc mcv.h \
 /usr/local/include/opencv/cv.h planar_pattern_detector.h keypoint.h \
 pyr_yape06.h fine_gaussian_pyramid.h homography_estimator.h \
 homography06.h affine_image_generator06.h affine_transformation_range.h \
 fern_based_point_classifier.h ferns.h buffer_management.h
planar_pattern_detector_builder.o: planar_pattern_detector_builder.cc \
 mcv.h /usr/local/include/opencv/cv.h planar_pattern_detector_builder.h \
 planar_pattern_detector.h keypoint.h pyr_yape06.h \
 fine_gaussian_pyramid.h homography_estimator.h homography06.h \
 affine_image_generator06.h affine_transformation_range.h \
 fern_based_point_classifier.h ferns.h
affine_transformation_range.o: affine_transformation_range.cc general.h \
 affine_transformation_range.h
buffer_management.o: buffer_management.cc mcv.h \
 /usr/local/include/opencv/cv.h buffer_management.h
pyr_yape06.o: pyr_yape06.cc mcv.h /usr/local/include/opencv/cv.h \
 buffer_management.h pyr_yape06.h keypoint.h fine_gaussian_pyramid.h
homography06.o: homography06.cc homography06.h \
 /usr/local/include/opencv/cv.h
homography_estimator.o: homography_estimator.cc buffer_management.h \
 /usr/local/include/opencv/cv.h homography_estimator.h homography06.h \
 cmphomo.h
fine_gaussian_pyramid.o: fine_gaussian_pyramid.cc mcv.h \
 /usr/local/include/opencv/cv.h mcvGaussianSmoothing.h \
 fine_gaussian_pyramid.h
mcvGaussianSmoothing.o: mcvGaussianSmoothing.cc buffer_management.h \
 /usr/local/include/opencv/cv.h mcvGaussianSmoothing.h mcv.h
affine_image_generator06.o: affine_image_generator06.cc general.h mcv.h \
 /usr/local/include/opencv/cv.h affine_image_generator06.h \
 affine_transformation_range.h
fern_based_point_classifier.o: fern_based_point_classifier.cc mcv.h \
 /usr/local/include/opencv/cv.h fern_based_point_classifier.h keypoint.h \
 ferns.h fine_gaussian_pyramid.h affine_image_generator06.h \
 affine_transformation_range.h
ferns.o: ferns.cc ferns.h fine_gaussian_pyramid.h \
 /usr/local/include/opencv/cv.h
template_matching_based_tracker.o: template_matching_based_tracker.cc \
 homography_estimator.h homography06.h /usr/local/include/opencv/cv.h \
 template_matching_based_tracker.h mcv.h
cmphomo.o: cmphomo.cc
