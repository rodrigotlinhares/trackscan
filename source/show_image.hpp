#ifndef SHOW_IMAGE_HPP_489127498127489124_
#define SHOW_IMAGE_HPP_489127498127489124_

#include <tr1/memory>

namespace cv {
class Mat;
};

struct Texture {
	unsigned id;
	unsigned w, h;
};

class ShowImage {
public:
	class Event;
	class KeyboardEvent;
	class MouseEvent;
public:
	ShowImage(unsigned w=1024, unsigned h=768);
	virtual ~ShowImage();

	Texture allocateTexture(unsigned w, unsigned h);

	/**
	 * @param mat The cv::Mat to be shown. RGB matrix is expected
	 * @param x0,y0,w,h the region on the screen that will be used to draw the image
	 * @param texture an already allocated texture to hold mat
	 *
	 * If w or h is equals 0 then the image is drawn with its original size
	 *
	 * @precondition texture and mat have matching sizes
	 * @remark Call draw() to finish the frame
	 */
	void show(const cv::Mat* mat, const Texture& texture, float x0=0, float y0=0, float w=1, float h=1);

	void start();
	void draw();

	Event* getEvent();
private:
	ShowImage(const ShowImage& copy);
	void operator=(const ShowImage& assign);

	struct SetupGl;
	SetupGl* _gl;

	Event* last_event;
};

// cut here --------------------------------------------------------------------------------- >8

class ShowImage::Event {
public:
	enum Type {
			KEYBOARD,QUIT,MOUSE
	} type;
	Event(Type type) : type(type) {
	}
	virtual ~Event() {
	}
};

class ShowImage::KeyboardEvent : public ShowImage::Event {
public:
	enum Key {
		KEY_BACKSPACE, KEY_TAB, KEY_CLEAR, KEY_RETURN, KEY_PAUSE, KEY_ESCAPE, KEY_SPACE, KEY_EXCLAIM, KEY_QUOTEDBL, KEY_HASH, KEY_DOLLAR, KEY_AMPERSAND, 
		KEY_QUOTE, KEY_LEFTPAREN, KEY_RIGHTPAREN, KEY_ASTERISK, KEY_PLUS, KEY_COMMA, KEY_MINUS, KEY_PERIOD, KEY_SLASH, KEY_0, KEY_1, KEY_2, KEY_3, KEY_4, 
		KEY_5, KEY_6, KEY_7, KEY_8, KEY_9, KEY_COLON, KEY_SEMICOLON, KEY_LESS, KEY_EQUALS, KEY_GREATER, KEY_QUESTION, KEY_AT, KEY_LEFTBRACKET, KEY_BACKSLASH, 
		KEY_RIGHTBRACKET, KEY_CARET, KEY_UNDERSCORE, KEY_BACKQUOTE, KEY_a, KEY_b, KEY_c, KEY_d, KEY_e, KEY_f, KEY_g, KEY_h, KEY_i, KEY_j, KEY_k, KEY_l, 
		KEY_m, KEY_n, KEY_o, KEY_p, KEY_q, KEY_r, KEY_s, KEY_t, KEY_u, KEY_v, KEY_w, KEY_x, KEY_y, KEY_z, KEY_DELETE, KEY_KP0, KEY_KP1, KEY_KP2, KEY_KP3, 
		KEY_KP4, KEY_KP5, KEY_KP6, KEY_KP7, KEY_KP8, KEY_KP9, KEY_KP_PERIOD, KEY_KP_DIVIDE, KEY_KP_MULTIPLY, KEY_KP_MINUS, KEY_KP_PLUS, KEY_KP_ENTER, 
		KEY_KP_EQUALS, KEY_UP , KEY_DOWN, KEY_RIGHT, KEY_LEFT, KEY_INSERT, KEY_HOME, KEY_END, KEY_PAGEUP, KEY_PAGEDOWN, KEY_F1 , KEY_F2 , KEY_F3 , KEY_F4 , 
		KEY_F5 , KEY_F6 , KEY_F7 , KEY_F8 , KEY_F9 , KEY_F10, KEY_F11, KEY_F12, KEY_F13, KEY_F14, KEY_F15, KEY_NUMLOCK, KEY_CAPSLOCK, KEY_SCROLLOCK, KEY_RSHIFT, 
		KEY_LSHIFT, KEY_RCTRL, KEY_LCTRL, KEY_RALT, KEY_LALT, KEY_RMETA, KEY_LMETA, KEY_LSUPER, KEY_RSUPER, KEY_MODE, KEY_HELP, KEY_PRINT, KEY_SYSREQ, 
		KEY_BREAK, KEY_MENU, KEY_POWER, KEY_EURO
	};

	KeyboardEvent(bool pressed, Key key) : Event(KEYBOARD), pressed(pressed), key(key) {}
	virtual ~KeyboardEvent() {
	}

	bool pressed;
	Key key;
};

class ShowImage::MouseEvent : public ShowImage::Event {
public:
	enum Button {
		LEFT=1,RIGHT=2,MIDDLE=4
	};
	MouseEvent(Button button, float x, float y, bool pressed, bool dragging)
	: Event(MOUSE), button(button), x(x), y(y), pressed(pressed), dragging(dragging)
	{
	}
	virtual ~MouseEvent() {
	}

	std::pair<float,float> get(float x0=0, float y0=0, float w=1, float h=1) {
		return std::make_pair(
			(x-x0)/w,
			(y-y0)/h
		);
	}

	Button button;
	float x,y;
	bool pressed;
	bool dragging;
};

#endif
