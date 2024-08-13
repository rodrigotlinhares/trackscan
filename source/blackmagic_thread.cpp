#ifdef USE_DECKLINK

#include "blackmagic_thread.h"
#include "DeckLinkAPI.h"
#include <cassert>
#ifdef USE_ATOMIC
#include <atomic>
#endif
#include <boost/thread.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>

#if 0 //Handycam
#define RESX 1920
#define RESY 1080
#define MODE "HD 1080i 59.94"
#else
#define RESX 1280
#define RESY 720
#define MODE "HD 720p 60"
#endif

namespace {
class Capturer;
}
struct BlackmagicThread::Private {
	Private() : state(0) {
        nextImage.create(RESX,RESY,CV_8UC3);
//        image_big_rotated.create(RESX,RESY,CV_8UC3);
//        rot_mat.create(2,3,CV_32FC1);
//        rot_mat.at<float>(0,0) = 0;
//        rot_mat.at<float>(0,1) = -1;
//        rot_mat.at<float>(0,2) = image_big_rotated.cols;
//        rot_mat.at<float>(1,0) = 1;
//        rot_mat.at<float>(1,1) = 0;
//        rot_mat.at<float>(1,2) = 0;
    }
	IDeckLinkInput *deckLinkInput;
	Capturer* capturer;

	cv::Mat nextImage;
	boost::mutex imageMutex;
	unsigned state;

//    cv::Mat image_big_rotated;
//    cv::Mat rot_mat;

};
namespace {
struct yuv_word {
	unsigned char cb0,y0,cr0,y1;
};

struct rgb {
	unsigned char b,g,r;
};
struct rgb_word {
	rgb rgb0, rgb1;

	void from_yuv(const yuv_word& yuv) {
		static const int C1 = 1.164*512.0;
		static const int C2 = 1.596*512.0;
		static const int C3 = 0.813*512.0;
		static const int C4 = 0.392*512.0;
		static const int C5 = 2.017*512.0;
#define color_clamp_and_normalize(a) (    (a < 0) ? 0 : ( (a >= (256<<9)) ? 255 : ((a)>>9) )    )
		//From http://en.wikipedia.org/wiki/YCbCr
		const int Y1 = C1*(yuv.y0);
		const int Y2 = C1*(yuv.y1);
		const int CR0 = (yuv.cr0-128);
		const int CB0 = (yuv.cb0-128);
		rgb0.r = color_clamp_and_normalize(Y1 + C2*CR0		);
		rgb0.g = color_clamp_and_normalize(Y1 - C3*CR0 - C4*CB0	);
		rgb0.b = color_clamp_and_normalize(Y1	      + C5*CB0	);
		rgb1.r = color_clamp_and_normalize(Y2 + C2*CR0		);
		rgb1.g = color_clamp_and_normalize(Y2 - C3*CR0 - C4*CB0	);
		rgb1.b = color_clamp_and_normalize(Y2	      + C5*CB0	);
	}
};

class Capturer : public IDeckLinkInputCallback {
public:
	Capturer(BlackmagicThread::Private* _this)
	: refCount(0), _this(_this)
	{
	}
	~Capturer() {
	}

	virtual HRESULT STDMETHODCALLTYPE QueryInterface(REFIID iid, LPVOID *ppv) { return E_NOINTERFACE; }
	virtual ULONG STDMETHODCALLTYPE AddRef(void) {
		return ++refCount;
	}
	virtual ULONG STDMETHODCALLTYPE  Release(void) {
		--refCount;
		if( refCount == 0 ) {
			delete this;
			return 0;
		}
		return refCount;
	}
	virtual HRESULT STDMETHODCALLTYPE VideoInputFormatChanged(BMDVideoInputFormatChangedEvents, IDeckLinkDisplayMode*, BMDDetectedVideoInputFormatFlags) {
		assert(false && "Format changed!");
		return S_OK;
	}
	virtual HRESULT STDMETHODCALLTYPE VideoInputFrameArrived(IDeckLinkVideoInputFrame* videoFrame, IDeckLinkAudioInputPacket*) {
		if (videoFrame->GetFlags() & bmdFrameHasNoInputSource ) {
			std::cerr << "No frame" << std::endl;
            if( _this->nextImage.empty() ) {
				_this->nextImage.create(RESY,RESX,CV_8UC3);
			}
		} else {
//			boost::posix_time::ptime initTime = boost::posix_time::microsec_clock::universal_time();
			assert( videoFrame->GetRowBytes() * videoFrame->GetHeight() == RESX*RESY*2 );
			void* frameBytes;
			videoFrame->GetBytes(&frameBytes);
			boost::lock_guard<boost::mutex> locker(_this->imageMutex);
            _this->nextImage.create(RESX,RESY,CV_8UC3);

			static const int width=RESX;
			static const int height=RESY;

			const yuv_word* in = reinterpret_cast<yuv_word*>(frameBytes);
            rgb* output = reinterpret_cast<rgb*>(_this->nextImage.data);

            //			for( unsigned int idx=0; idx< (width*height)/2; ++idx) {
            //				output[idx].from_yuv(in[idx]);
            //			}

            // RESX = 4, RESY = 3
            // y    x  idx     idx_out0    idx_out1
            // 0    0   0       0
            // 0    2   1
            // 1    0   2
            // 1    2   3
            // 2    0   4
            // 2    2   5
#define INDEX(x,y,LARGURA) ((y)*LARGURA + x)
            for(unsigned int y=0; y<RESY; y++)
                for(unsigned int x=0; x<RESX; x+=2)
                {
                    unsigned idx_in = INDEX(x/2,y,RESX/2);
//                    unsigned idx_out0 = INDEX(x,RESY-1-y,RESX);
//                    unsigned idx_out1 = INDEX(x+1,RESY-1-y,RESX); // transposto NAO rotacionado
                    unsigned idx_out0 = INDEX(RESY-1-y,x,RESY);
                    unsigned idx_out1 = INDEX(RESY-1-y,x+1,RESY);

                    rgb_word _rgb;
                    _rgb.from_yuv(in[idx_in]);

//                    if( ((x/20) + (y/20))%2 == 0 ) {
//                        _rgb.rgb0.b *= 0.25;
//                        _rgb.rgb0.g *= 0.25;
//                        _rgb.rgb0.r *= 0.25;
//                        _rgb.rgb1.b *= 0.25;
//                        _rgb.rgb1.g *= 0.25;
//                        _rgb.rgb1.r *= 0.25;
//                    } else {
////                        _rgb.rgb0.b = 255;
////                        _rgb.rgb0.g = 255;
////                        _rgb.rgb0.r = 255;
////                        _rgb.rgb1.b = 255;
////                        _rgb.rgb1.g = 255;
////                        _rgb.rgb1.r = 255;
//                    }

                    output[idx_out0] = _rgb.rgb0;
                    output[idx_out1] = _rgb.rgb1;

                }
//            for( unsigned int idx=0; idx< (width*height)/2; ++idx) {
//                output[idx].from_yuv(in[idx]);
//            }

//            cv::warpAffine(_this->nextImage, _this->image_big_rotated, _this->rot_mat, _this->image_big_rotated.size(), cv::INTER_NEAREST);

			_this->state++;
//			boost::posix_time::ptime now = boost::posix_time::microsec_clock::universal_time();
//			std::cout << "Took " << ((now-initTime).total_microseconds()/1000.0) << "ms to convert the image." << std::endl;
		}
		return S_OK; 
	}

	inline unsigned getRefCount() const {
		return refCount;
	}
private:
#ifdef USE_ATOMIC
	std::atomic<unsigned long> refCount;
#else
	unsigned long refCount;
#endif
	BlackmagicThread::Private* _this;
};

};
//namespace

BlackmagicThread::BlackmagicThread()
: _this(new Private)
{
	IDeckLinkIterator *deckLinkIterator = CreateDeckLinkIteratorInstance();
	assert(deckLinkIterator);

	IDeckLink* board = NULL;
	while( true ) {
		IDeckLink* lastboard = board;
		bool ok = deckLinkIterator->Next(&board)==S_OK;
		if( !ok ) {
			board = lastboard;
			break;
		}

		const char* name;
		board->GetModelName(&name);
		std::cout << "Found board " << name << std::endl;
		delete[] name;
	}
	deckLinkIterator->Release();

	_this->deckLinkInput = NULL;
	bool ok = board->QueryInterface(IID_IDeckLinkInput, (void**)(&_this->deckLinkInput))==S_OK;
	assert(ok);
	board->Release();

	IDeckLinkDisplayModeIterator *displayModeIterator = NULL;
	ok = _this->deckLinkInput->GetDisplayModeIterator(&displayModeIterator)==S_OK;
	assert(ok);

	bool found=false;
	BMDDisplayMode selectedDisplayMode;
	IDeckLinkDisplayMode *displayMode;
	std::cout << "Video Modes:" << std::endl;
	while (displayModeIterator->Next(&displayMode) == S_OK) {
		const char *displayModeName;
		displayMode->GetName(&displayModeName);
		if( std::string(displayModeName) == MODE) {
			std::cout << "* ";
			found = true;
			selectedDisplayMode = displayMode->GetDisplayMode();
		}
		std::cout << displayModeName << std::endl;
		delete[] displayModeName;
	}
	assert(found);
	displayModeIterator->Release();

	_this->capturer = new Capturer(_this);
	_this->capturer->AddRef();
	_this->deckLinkInput->SetCallback(_this->capturer);
	BMDPixelFormat pixelFormat = bmdFormat8BitYUV;
	ok = _this->deckLinkInput->EnableVideoInput(selectedDisplayMode,pixelFormat,0) == S_OK;
	assert(ok);
	ok = _this->deckLinkInput->StartStreams() == S_OK;
	assert(ok);
}

BlackmagicThread::~BlackmagicThread() {
	bool ok = _this->deckLinkInput->StopStreams() == S_OK;
	_this->deckLinkInput->Release();
	assert(ok);
	while( _this->capturer->getRefCount() > 1 ) {
		boost::this_thread::yield();
	}
	_this->capturer->Release();
	delete _this;
}

std::pair<unsigned,unsigned> BlackmagicThread::size() const {
    return std::make_pair(RESY,RESX);
}

unsigned BlackmagicThread::state() const {
	return _this->state;
}

std::pair<cv::Mat*,boost::mutex*> BlackmagicThread::lockImage() {
	_this->imageMutex.lock();
	while( _this->nextImage.empty() ) {
		_this->imageMutex.unlock();
		boost::this_thread::yield();
		_this->imageMutex.lock();
    }
    return std::make_pair(&_this->nextImage,&_this->imageMutex);
    //return std::make_pair(&_this->image_big_rotated,&_this->imageMutex);
}


#endif //USEDECKLINK
