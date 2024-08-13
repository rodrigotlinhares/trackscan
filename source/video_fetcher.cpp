#include "video_fetcher.h"
#include <boost/thread.hpp>
#include <boost/thread/mutex.hpp>

VideoFetcher::VideoFetcher()
: copyToLast(0)
{
}

VideoFetcher::~VideoFetcher() {
}

void VideoFetcher::copyTo(cv::Mat* dest, unsigned& lastFrame) {
	while( state() == lastFrame ) {
		boost::this_thread::yield();
	}

	{
		std::pair<cv::Mat*,boost::mutex*> image = lockImage();
		boost::lock_guard<boost::mutex> image_lock(*image.second,boost::adopt_lock_t());
        //image.first->copyTo(*dest);

        cv::resize(*(image.first), *dest, dest->size(), 0, 0, cv::INTER_NEAREST);

//        cv::imshow("teste1", *dest);
//        cv::waitKey(0);


        lastFrame = state();
	}	
}
