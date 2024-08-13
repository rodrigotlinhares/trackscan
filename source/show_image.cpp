#include "show_image.hpp"
#include <GL/glew.h>
#include <GL/gl.h>
#include <SDL/SDL.h>
#include <tr1/memory>
#include <stdexcept>
#include <map>

#include <opencv2/opencv.hpp>

#include <iostream>

namespace {

char vertexShader[] = "attribute vec2 vertex;"
"varying vec2 textureCoord;"
"void main(void) {"
"	textureCoord = ( vertex + vec2(1.0,1.0) ) * 0.5;"
"	textureCoord.y = 1.0-textureCoord.y;"
"       gl_Position = vec4(vertex,0,1);"
"}";

char fragShader[] = "uniform sampler2D texture;"
"varying vec2 textureCoord;"
"void main(void) {"
/*Input actually was bgra, fixing:*/
"	int coordX = int(gl_FragCoord.x/10.0);"
"	int coordY = int(gl_FragCoord.y/10.0);"
"	float grey = 0.6;"
"	if( mod(coordX+coordY,2) == 1.0 ) {"
"		grey = 0.3;"
"	}"
"	vec4 bgra = texture2D(texture,textureCoord);"
"	float alpha = bgra[3];"
"	float alphaI = 1.0-alpha;"
"	gl_FragColor.r = bgra[2]*alpha + alphaI*grey;"
"	gl_FragColor.g = bgra[1]*alpha + alphaI*grey;"
"	gl_FragColor.b = bgra[0]*alpha + alphaI*grey;"
"}";

GLuint createShader(const char* source, GLenum shaderType) {
        GLuint shader = glCreateShader(shaderType);
        glShaderSource(shader,1,&source,NULL);
        glCompileShader(shader);
 
        GLint compileStatus;
        glGetShaderiv(shader, GL_COMPILE_STATUS, &compileStatus);
        if( compileStatus == GL_FALSE ) {
		int msgLength;
		glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &msgLength);

		char* msg = new char[msgLength];
		glGetShaderInfoLog(shader, msgLength, &msgLength, msg);

		std::runtime_error except(std::string("Error on compiling shader:\n")+msg);
		delete[] msg;

		throw except;
        }
 
        return shader;
}

GLuint createProgram(GLuint shaderOne, GLuint shaderTwo) {
        GLuint program = glCreateProgram();
        glAttachShader(program,shaderOne);
        glAttachShader(program,shaderTwo);
        glLinkProgram(program);
 
        GLint linkStatus;
        glGetProgramiv(program, GL_LINK_STATUS, &linkStatus);
        if( linkStatus == GL_FALSE ) {
                throw std::runtime_error("Error linking program");
        }
 
 
        return program;
}

//static const unsigned int videoFlags = SDL_OPENGL | SDL_GL_DOUBLEBUFFER | SDL_HWSURFACE | SDL_HWACCEL | SDL_RESIZABLE;
static const unsigned int videoFlags = SDL_OPENGL | SDL_GL_DOUBLEBUFFER | SDL_HWSURFACE | SDL_HWACCEL ;

}

struct ShowImage::SetupGl {
	SetupGl(unsigned w, unsigned h) {
		if( SDL_Init( SDL_INIT_VIDEO) < 0 ) {
                	throw __LINE__;
	        }
        	surface = SDL_SetVideoMode(w, h,  24,
				videoFlags
        	);
	        if( surface == NULL ) throw std::runtime_error("Couldnt create surface");
 
		if( glewInit() != GLEW_OK ) throw std::runtime_error("Couldnt init glew");
	        glViewport(0,0,w,h);
		glEnable( GL_TEXTURE_2D );

		if( glGetError() != GL_NO_ERROR ) throw std::runtime_error("Error initializing gl");

		GLuint vert = createShader(vertexShader,GL_VERTEX_SHADER);
		GLuint frag = createShader(fragShader,GL_FRAGMENT_SHADER);
		program = createProgram(vert,frag);

		glGenBuffers(1, &shape);
		float square[6*2] = {-1,-1,  1,-1,  1,1,  1,1, -1,1, -1,-1};
		glBindBuffer(GL_ARRAY_BUFFER,shape);
		glBufferData(GL_ARRAY_BUFFER,6*2*sizeof(float),square,GL_STATIC_DRAW);
		vertexLocation = glGetAttribLocation(program,"vertex");
		textureLocation = glGetUniformLocation(program,"texture");
		if( vertexLocation < 0 || textureLocation < 0 ) throw std::runtime_error("Error initializing gl shader");
		glEnableVertexAttribArray(vertexLocation);
		glClearColor(0,0,0,0);

        glPixelStorei(GL_PACK_ALIGNMENT,1);
        glPixelStorei(GL_UNPACK_ALIGNMENT,1);

		if( glGetError() != GL_NO_ERROR ) throw std::runtime_error("Error initializing gl shader");

		_w = w;
		_h = h;
	}
	virtual ~SetupGl() {
		SDL_Quit();
	}

	Texture allocateTexture(unsigned w, unsigned h) {
		Texture result;
		result.w = w;
		result.h = h;

		glGenTextures(1,&result.id);
		glBindTexture(GL_TEXTURE_2D,result.id);
		glTexImage2D(GL_TEXTURE_2D,0,GL_RGBA,w,h,0,GL_RGB,GL_UNSIGNED_BYTE,NULL);
		glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_LINEAR);
		if( glGetError() != GL_NO_ERROR ) throw std::runtime_error("Error initializing gl texture");

		return result;
	}

	void retexture(void* data, const Texture& texture) {
		glBindTexture(GL_TEXTURE_2D,texture.id);
		glTexSubImage2D(GL_TEXTURE_2D,0,0,0,texture.w,texture.h,GL_RGB,GL_UNSIGNED_BYTE,data);
		if( glGetError() != GL_NO_ERROR ) throw std::runtime_error("Error update gl texture");
	};

	void show(unsigned texture) {
		glUseProgram(program);

		glBindBuffer(GL_ARRAY_BUFFER,shape);
		glVertexAttribPointer( vertexLocation,2,GL_FLOAT,false,0,0);

		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, texture);
		glUniform1i(textureLocation,0);

		glDrawArrays(GL_TRIANGLES,0,6);
	}

	void start() {
		glClear(GL_COLOR_BUFFER_BIT);
	}

	void draw() {
		SDL_GL_SwapBuffers();
		glViewport(0,0,_w,_h);
	}

	void viewport(float x0, float y0, float w, float h) {
		glViewport(x0*_w,y0*_h, w*_w, h*_h );
	};

	void resize(SDL_Event event) {
		_w = event.resize.w;
		_h = event.resize.h;
		surface = SDL_SetVideoMode(_w,_h,0,videoFlags);
		if( surface == 0 ) throw 1;
	}

	SDL_Surface* surface;
	GLuint program;
	GLuint shape;
	GLint vertexLocation;
	GLint textureLocation;
	unsigned _w;
	unsigned _h;
};

ShowImage::ShowImage(unsigned w, unsigned h) 
: _gl(new SetupGl(w,h)), last_event(NULL)
{
}
ShowImage::~ShowImage() {
	delete _gl;
	delete last_event;
}

Texture ShowImage::allocateTexture(unsigned w, unsigned h) {
	return _gl->allocateTexture(w,h);
}

void ShowImage::show(const cv::Mat* mat, const Texture& texture, float x0, float y0, float w, float h) {
	assert( mat->size().width == texture.w && mat->size().height == texture.h );

	_gl->retexture(mat->data,texture);
	_gl->viewport(x0,y0,w,h);
	_gl->show(texture.id);
}

void ShowImage::start() {
	_gl->start();
}

void ShowImage::draw() {
	_gl->draw();
};

ShowImage::KeyboardEvent::Key translate_key(const SDLKey& key);
ShowImage::MouseEvent::Button convert_mouse_button(unsigned char button);

ShowImage::Event* ShowImage::getEvent() {
	delete last_event;
	last_event = NULL;
	SDL_Event event;
	if( !SDL_PollEvent(&event) ) {
		return NULL;
	}
	
	switch( event.type ) {
		case SDL_QUIT:
			last_event = new Event(Event::QUIT);
			return last_event;
		break;
		case SDL_VIDEORESIZE:
			_gl->resize(event);
		break;
		case SDL_MOUSEMOTION:
		{
			float fx = event.motion.x/float(_gl->_w);
			float fy = event.motion.y/float(_gl->_h);
			int mouse_button = SDL_GetMouseState(NULL,NULL);
			for( unsigned int i=0;i<3;++i ) {
				if( mouse_button&SDL_BUTTON(i) ) {
					last_event = new MouseEvent(convert_mouse_button(SDL_BUTTON(i)),fx,fy,event.type==SDL_MOUSEBUTTONDOWN,true);
					return last_event;
				}
			}
		}
		break;
		case SDL_MOUSEBUTTONDOWN:
		case SDL_MOUSEBUTTONUP:
		{
			int x, y;
			SDL_GetMouseState(&x,&y);
			float fx = x/float(_gl->_w);
			float fy = y/float(_gl->_h);
			last_event = new MouseEvent(convert_mouse_button(event.button.button),fx,fy,event.type==SDL_MOUSEBUTTONDOWN,false);
			return last_event;
		}
		break;
		case SDL_KEYDOWN:
		case SDL_KEYUP:
			last_event = new KeyboardEvent(event.type==SDL_KEYDOWN,translate_key(event.key.keysym.sym));
			return last_event;
		break;		
	}
	return NULL;
}

// cut here ---------------------------------------------------- >8
ShowImage::MouseEvent::Button convert_mouse_button(unsigned char button) {
	switch( button ) {
		case SDL_BUTTON_LEFT:
			return ShowImage::MouseEvent::LEFT;
			break;
		case SDL_BUTTON_RIGHT:
			return ShowImage::MouseEvent::RIGHT;
			break;
		case SDL_BUTTON_MIDDLE:
			return ShowImage::MouseEvent::MIDDLE;
			break;
		default:
			assert(false);
			break;
	};
	return ShowImage::MouseEvent::LEFT;
}

namespace {
std::map<SDLKey,ShowImage::KeyboardEvent::Key> init_keymap() {
	std::map<SDLKey,ShowImage::KeyboardEvent::Key> keyMapping;
	keyMapping[SDLK_BACKSPACE] = ShowImage::KeyboardEvent::KEY_BACKSPACE;
	keyMapping[SDLK_TAB] = ShowImage::KeyboardEvent::KEY_TAB;
	keyMapping[SDLK_CLEAR] = ShowImage::KeyboardEvent::KEY_CLEAR;
	keyMapping[SDLK_RETURN] = ShowImage::KeyboardEvent::KEY_RETURN;
	keyMapping[SDLK_PAUSE] = ShowImage::KeyboardEvent::KEY_PAUSE;
	keyMapping[SDLK_ESCAPE] = ShowImage::KeyboardEvent::KEY_ESCAPE;
	keyMapping[SDLK_SPACE] = ShowImage::KeyboardEvent::KEY_SPACE;
	keyMapping[SDLK_EXCLAIM] = ShowImage::KeyboardEvent::KEY_EXCLAIM;
	keyMapping[SDLK_QUOTEDBL] = ShowImage::KeyboardEvent::KEY_QUOTEDBL;
	keyMapping[SDLK_HASH] = ShowImage::KeyboardEvent::KEY_HASH;
	keyMapping[SDLK_DOLLAR] = ShowImage::KeyboardEvent::KEY_DOLLAR;
	keyMapping[SDLK_AMPERSAND] = ShowImage::KeyboardEvent::KEY_AMPERSAND;
	keyMapping[SDLK_QUOTE] = ShowImage::KeyboardEvent::KEY_QUOTE;
	keyMapping[SDLK_LEFTPAREN] = ShowImage::KeyboardEvent::KEY_LEFTPAREN;
	keyMapping[SDLK_RIGHTPAREN] = ShowImage::KeyboardEvent::KEY_RIGHTPAREN;
	keyMapping[SDLK_ASTERISK] = ShowImage::KeyboardEvent::KEY_ASTERISK;
	keyMapping[SDLK_PLUS] = ShowImage::KeyboardEvent::KEY_PLUS;
	keyMapping[SDLK_COMMA] = ShowImage::KeyboardEvent::KEY_COMMA;
	keyMapping[SDLK_MINUS] = ShowImage::KeyboardEvent::KEY_MINUS;
	keyMapping[SDLK_PERIOD] = ShowImage::KeyboardEvent::KEY_PERIOD;
	keyMapping[SDLK_SLASH] = ShowImage::KeyboardEvent::KEY_SLASH;
	keyMapping[SDLK_0] = ShowImage::KeyboardEvent::KEY_0;
	keyMapping[SDLK_1] = ShowImage::KeyboardEvent::KEY_1;
	keyMapping[SDLK_2] = ShowImage::KeyboardEvent::KEY_2;
	keyMapping[SDLK_3] = ShowImage::KeyboardEvent::KEY_3;
	keyMapping[SDLK_4] = ShowImage::KeyboardEvent::KEY_4;
	keyMapping[SDLK_5] = ShowImage::KeyboardEvent::KEY_5;
	keyMapping[SDLK_6] = ShowImage::KeyboardEvent::KEY_6;
	keyMapping[SDLK_7] = ShowImage::KeyboardEvent::KEY_7;
	keyMapping[SDLK_8] = ShowImage::KeyboardEvent::KEY_8;
	keyMapping[SDLK_9] = ShowImage::KeyboardEvent::KEY_9;
	keyMapping[SDLK_COLON] = ShowImage::KeyboardEvent::KEY_COLON;
	keyMapping[SDLK_SEMICOLON] = ShowImage::KeyboardEvent::KEY_SEMICOLON;
	keyMapping[SDLK_LESS] = ShowImage::KeyboardEvent::KEY_LESS;
	keyMapping[SDLK_EQUALS] = ShowImage::KeyboardEvent::KEY_EQUALS;
	keyMapping[SDLK_GREATER] = ShowImage::KeyboardEvent::KEY_GREATER;
	keyMapping[SDLK_QUESTION] = ShowImage::KeyboardEvent::KEY_QUESTION;
	keyMapping[SDLK_AT] = ShowImage::KeyboardEvent::KEY_AT;
	keyMapping[SDLK_LEFTBRACKET] = ShowImage::KeyboardEvent::KEY_LEFTBRACKET;
	keyMapping[SDLK_BACKSLASH] = ShowImage::KeyboardEvent::KEY_BACKSLASH;
	keyMapping[SDLK_RIGHTBRACKET] = ShowImage::KeyboardEvent::KEY_RIGHTBRACKET;
	keyMapping[SDLK_CARET] = ShowImage::KeyboardEvent::KEY_CARET;
	keyMapping[SDLK_UNDERSCORE] = ShowImage::KeyboardEvent::KEY_UNDERSCORE;
	keyMapping[SDLK_BACKQUOTE] = ShowImage::KeyboardEvent::KEY_BACKQUOTE;
	keyMapping[SDLK_a] = ShowImage::KeyboardEvent::KEY_a;
	keyMapping[SDLK_b] = ShowImage::KeyboardEvent::KEY_b;
	keyMapping[SDLK_c] = ShowImage::KeyboardEvent::KEY_c;
	keyMapping[SDLK_d] = ShowImage::KeyboardEvent::KEY_d;
	keyMapping[SDLK_e] = ShowImage::KeyboardEvent::KEY_e;
	keyMapping[SDLK_f] = ShowImage::KeyboardEvent::KEY_f;
	keyMapping[SDLK_g] = ShowImage::KeyboardEvent::KEY_g;
	keyMapping[SDLK_h] = ShowImage::KeyboardEvent::KEY_h;
	keyMapping[SDLK_i] = ShowImage::KeyboardEvent::KEY_i;
	keyMapping[SDLK_j] = ShowImage::KeyboardEvent::KEY_j;
	keyMapping[SDLK_k] = ShowImage::KeyboardEvent::KEY_k;
	keyMapping[SDLK_l] = ShowImage::KeyboardEvent::KEY_l;
	keyMapping[SDLK_m] = ShowImage::KeyboardEvent::KEY_m;
	keyMapping[SDLK_n] = ShowImage::KeyboardEvent::KEY_n;
	keyMapping[SDLK_o] = ShowImage::KeyboardEvent::KEY_o;
	keyMapping[SDLK_p] = ShowImage::KeyboardEvent::KEY_p;
	keyMapping[SDLK_q] = ShowImage::KeyboardEvent::KEY_q;
	keyMapping[SDLK_r] = ShowImage::KeyboardEvent::KEY_r;
	keyMapping[SDLK_s] = ShowImage::KeyboardEvent::KEY_s;
	keyMapping[SDLK_t] = ShowImage::KeyboardEvent::KEY_t;
	keyMapping[SDLK_u] = ShowImage::KeyboardEvent::KEY_u;
	keyMapping[SDLK_v] = ShowImage::KeyboardEvent::KEY_v;
	keyMapping[SDLK_w] = ShowImage::KeyboardEvent::KEY_w;
	keyMapping[SDLK_x] = ShowImage::KeyboardEvent::KEY_x;
	keyMapping[SDLK_y] = ShowImage::KeyboardEvent::KEY_y;
	keyMapping[SDLK_z] = ShowImage::KeyboardEvent::KEY_z;
	keyMapping[SDLK_DELETE] = ShowImage::KeyboardEvent::KEY_DELETE;
	keyMapping[SDLK_KP0] = ShowImage::KeyboardEvent::KEY_KP0;
	keyMapping[SDLK_KP1] = ShowImage::KeyboardEvent::KEY_KP1;
	keyMapping[SDLK_KP2] = ShowImage::KeyboardEvent::KEY_KP2;
	keyMapping[SDLK_KP3] = ShowImage::KeyboardEvent::KEY_KP3;
	keyMapping[SDLK_KP4] = ShowImage::KeyboardEvent::KEY_KP4;
	keyMapping[SDLK_KP5] = ShowImage::KeyboardEvent::KEY_KP5;
	keyMapping[SDLK_KP6] = ShowImage::KeyboardEvent::KEY_KP6;
	keyMapping[SDLK_KP7] = ShowImage::KeyboardEvent::KEY_KP7;
	keyMapping[SDLK_KP8] = ShowImage::KeyboardEvent::KEY_KP8;
	keyMapping[SDLK_KP9] = ShowImage::KeyboardEvent::KEY_KP9;
	keyMapping[SDLK_KP_PERIOD] = ShowImage::KeyboardEvent::KEY_KP_PERIOD;
	keyMapping[SDLK_KP_DIVIDE] = ShowImage::KeyboardEvent::KEY_KP_DIVIDE;
	keyMapping[SDLK_KP_MULTIPLY] = ShowImage::KeyboardEvent::KEY_KP_MULTIPLY;
	keyMapping[SDLK_KP_MINUS] = ShowImage::KeyboardEvent::KEY_KP_MINUS;
	keyMapping[SDLK_KP_PLUS] = ShowImage::KeyboardEvent::KEY_KP_PLUS;
	keyMapping[SDLK_KP_ENTER] = ShowImage::KeyboardEvent::KEY_KP_ENTER;
	keyMapping[SDLK_KP_EQUALS] = ShowImage::KeyboardEvent::KEY_KP_EQUALS;
	keyMapping[SDLK_UP] = ShowImage::KeyboardEvent::KEY_UP;
	keyMapping[SDLK_DOWN] = ShowImage::KeyboardEvent::KEY_DOWN;
	keyMapping[SDLK_RIGHT] = ShowImage::KeyboardEvent::KEY_RIGHT;
	keyMapping[SDLK_LEFT] = ShowImage::KeyboardEvent::KEY_LEFT;
	keyMapping[SDLK_INSERT] = ShowImage::KeyboardEvent::KEY_INSERT;
	keyMapping[SDLK_HOME] = ShowImage::KeyboardEvent::KEY_HOME;
	keyMapping[SDLK_END] = ShowImage::KeyboardEvent::KEY_END;
	keyMapping[SDLK_PAGEUP] = ShowImage::KeyboardEvent::KEY_PAGEUP;
	keyMapping[SDLK_PAGEDOWN] = ShowImage::KeyboardEvent::KEY_PAGEDOWN;
	keyMapping[SDLK_F1] = ShowImage::KeyboardEvent::KEY_F1;
	keyMapping[SDLK_F2] = ShowImage::KeyboardEvent::KEY_F2;
	keyMapping[SDLK_F3] = ShowImage::KeyboardEvent::KEY_F3;
	keyMapping[SDLK_F4] = ShowImage::KeyboardEvent::KEY_F4;
	keyMapping[SDLK_F5] = ShowImage::KeyboardEvent::KEY_F5;
	keyMapping[SDLK_F6] = ShowImage::KeyboardEvent::KEY_F6;
	keyMapping[SDLK_F7] = ShowImage::KeyboardEvent::KEY_F7;
	keyMapping[SDLK_F8] = ShowImage::KeyboardEvent::KEY_F8;
	keyMapping[SDLK_F9] = ShowImage::KeyboardEvent::KEY_F9;
	keyMapping[SDLK_F10] = ShowImage::KeyboardEvent::KEY_F10;
	keyMapping[SDLK_F11] = ShowImage::KeyboardEvent::KEY_F11;
	keyMapping[SDLK_F12] = ShowImage::KeyboardEvent::KEY_F12;
	keyMapping[SDLK_F13] = ShowImage::KeyboardEvent::KEY_F13;
	keyMapping[SDLK_F14] = ShowImage::KeyboardEvent::KEY_F14;
	keyMapping[SDLK_F15] = ShowImage::KeyboardEvent::KEY_F15;
	keyMapping[SDLK_NUMLOCK] = ShowImage::KeyboardEvent::KEY_NUMLOCK;
	keyMapping[SDLK_CAPSLOCK] = ShowImage::KeyboardEvent::KEY_CAPSLOCK;
	keyMapping[SDLK_SCROLLOCK] = ShowImage::KeyboardEvent::KEY_SCROLLOCK;
	keyMapping[SDLK_RSHIFT] = ShowImage::KeyboardEvent::KEY_RSHIFT;
	keyMapping[SDLK_LSHIFT] = ShowImage::KeyboardEvent::KEY_LSHIFT;
	keyMapping[SDLK_RCTRL] = ShowImage::KeyboardEvent::KEY_RCTRL;
	keyMapping[SDLK_LCTRL] = ShowImage::KeyboardEvent::KEY_LCTRL;
	keyMapping[SDLK_RALT] = ShowImage::KeyboardEvent::KEY_RALT;
	keyMapping[SDLK_LALT] = ShowImage::KeyboardEvent::KEY_LALT;
	keyMapping[SDLK_RMETA] = ShowImage::KeyboardEvent::KEY_RMETA;
	keyMapping[SDLK_LMETA] = ShowImage::KeyboardEvent::KEY_LMETA;
	keyMapping[SDLK_LSUPER] = ShowImage::KeyboardEvent::KEY_LSUPER;
	keyMapping[SDLK_RSUPER] = ShowImage::KeyboardEvent::KEY_RSUPER;
	keyMapping[SDLK_MODE] = ShowImage::KeyboardEvent::KEY_MODE;
	keyMapping[SDLK_HELP] = ShowImage::KeyboardEvent::KEY_HELP;
	keyMapping[SDLK_PRINT] = ShowImage::KeyboardEvent::KEY_PRINT;
	keyMapping[SDLK_SYSREQ] = ShowImage::KeyboardEvent::KEY_SYSREQ;
	keyMapping[SDLK_BREAK] = ShowImage::KeyboardEvent::KEY_BREAK;
	keyMapping[SDLK_MENU] = ShowImage::KeyboardEvent::KEY_MENU;
	keyMapping[SDLK_POWER] = ShowImage::KeyboardEvent::KEY_POWER;
	keyMapping[SDLK_EURO] = ShowImage::KeyboardEvent::KEY_EURO;
	return keyMapping;
}
}

ShowImage::KeyboardEvent::Key translate_key(const SDLKey& key) {
	static std::map<SDLKey,ShowImage::KeyboardEvent::Key> keyMapping = init_keymap();

	return keyMapping[key];
}
