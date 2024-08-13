#ifndef _VIDEO_FETCHER_HPP_321089401984190_
#define _VIDEO_FETCHER_HPP_321089401984190_

#include <opencv/cv.h>
#include <opencv/highgui.h>

namespace boost {
class mutex;
}

class VideoFetcher {
public:
	VideoFetcher();
	virtual ~VideoFetcher();
	virtual unsigned state() const=0;
	virtual std::pair<unsigned,unsigned> size() const=0;
	virtual std::pair<cv::Mat*,boost::mutex*> lockImage()=0;

	/**
	 * Keeps locked until a new frame is available. The frame is copied to @param dest
	 */
	inline void copyTo(cv::Mat* dest) {
		copyTo(dest,copyToLast);
	}

	/**
	 * Keeps locked until a new frame is available. The frame is copied to @param dest, 
	 * @param in/out lastFrame, must contain the last copied frame number and will be written with the just copied frame number,
	 * i.e. just give the VideoFetcher anything that it will keep track of the frame count
	 */
	void copyTo(cv::Mat* dest, unsigned& lastFrame);
private:
	unsigned copyToLast;
};

#endif
