#ifndef _BLACK_MAGIC_THREAD_H_8127u48912412_
#define _BLACK_MAGIC_THREAD_H_8127u48912412_

#ifdef USE_DECKLINK

#include <opencv/cv.h>
#include "video_fetcher.h"

class BlackmagicThread : public VideoFetcher {
public:
	BlackmagicThread();
	virtual ~BlackmagicThread();

	unsigned state() const;
	std::pair<unsigned,unsigned> size() const;

	std::pair<cv::Mat*,boost::mutex*> lockImage();

	struct Private;
private:
	BlackmagicThread(const BlackmagicThread&);
	void operator=(const BlackmagicThread&);

	Private* _this;
};

#endif//USE_DECKLINK

#endif
