#ifndef GUI_H
#define GUI_H

#ifdef _WIN32
  #ifdef GUI_EXPORTS
    #define GUI_API __declspec(dllexport)
  #else
    #define GUI_API __declspec(dllimport)
  #endif
#else
  #define GUI_API
#endif

enum AlignFlag
{
  AlignLeft = 0x0001,
  AlignRight = 0x0002,
  AlignHCenter = 0x0004,
  AlignJustify = 0x0008,

  AlignTop = 0x0020,
  AlignBottom = 0x0040,
  AlignVCenter = 0x0080,

  AlignCenter = AlignVCenter | AlignHCenter
};

enum SizePolicyFlag
{
  SizePolicyGrowFlag = 1,
  SizePolicyExpandFlag = 2,
  SizePolicyShrinkFlag = 4,
  SizePolicyIgnoreFlag = 8      
};
  
enum SizePolicy
{
  SizePolicyFixed = 0,
  SizePolicyMinimum = SizePolicyGrowFlag,
  SizePolicyMaximum = SizePolicyShrinkFlag,
  SizePolicyPreferred = SizePolicyGrowFlag | SizePolicyShrinkFlag,
  SizePolicyExpanding = SizePolicyGrowFlag | SizePolicyShrinkFlag | SizePolicyExpandFlag,
  SizePolicyMinimumExpanding = SizePolicyGrowFlag | SizePolicyExpandFlag,
  SizePolicyIgnored = SizePolicyShrinkFlag | SizePolicyGrowFlag | SizePolicyIgnoreFlag
};  

enum SizeConstraint
{
  SizeConstraintDefault = 0,
  SizeConstraintFixed = 3,
  SizeConstraintMinimum = 2,
  SizeConstraintMaximum = 4,
  SizeConstraintMinAndMax = 5,
  SizeConstraintOff = 1
};

enum FocusPolicy
{
  FocusPolicyTab = 0x1,
  FocusPolicyClick = 0x2,
  FocusPolicyStrong = FocusPolicyTab | FocusPolicyClick | 0x8,
  FocusPolicyWheel  = FocusPolicyStrong | 0x4,
  FocusPolicyNoFocus = 0
};

enum SliderTicks
{
  TicksOff = 0,
  TicksBothSides = 3,
  TicksAbove = 1,
  TicksBelow = 2,
  TicksLeft = TicksAbove,
  TicksRight = TicksBelow
};

enum FrameShadow
{
  ShadowPlain = 0x0010,
  ShadowRaised = 0x0020,
  ShadowSunken = 0x0030    
};
  
enum FrameShape
{
  ShapeNoFrame = 0,
  ShapeBox = 0x0001,
  ShapePanel = 0x0002,
  ShapeStyledPanel = 0x0006,
  ShapeHLine = 0x0004,
  ShapeVLine = 0x0005    
};

enum MouseButton
{
  ButtonLeft = 0x00000001,
  ButtonRight = 0x00000002,
  ButtonMiddle = 0x00000004
};

enum Key
{
  KeyEscape = 0x01000000,
  KeyTab = 0x01000001,
  KeyBacktab = 0x01000002,
  KeyBackspace = 0x01000003,
  KeyReturn = 0x01000004,
  KeyEnter = 0x01000005,
  KeyInsert = 0x01000006,
  KeyDelete = 0x01000007,
  KeyPause = 0x01000008,
  KeyPrint = 0x01000009,
  KeySysReq = 0x0100000a,
  KeyClear = 0x0100000b,
  KeyHome = 0x01000010,
  KeyEnd = 0x01000011,
  KeyLeft = 0x01000012,
  KeyUp = 0x01000013,
  KeyRight = 0x01000014,
  KeyDown = 0x01000015,
  KeyPageUp = 0x01000016,
  KeyPageDown = 0x01000017,
  KeyShift = 0x01000020,
  KeyControl = 0x01000021,
  KeyMeta = 0x01000022,
  KeyAlt = 0x01000023,
  KeyAltGr = 0x01001103,
  KeyCapsLock = 0x01000024,
  KeyNumLock = 0x01000025,
  KeyScrollLock = 0x01000026,
  KeyF1 = 0x01000030,
  KeyF2 = 0x01000031,
  KeyF3 = 0x01000032,
  KeyF4 = 0x01000033,
  KeyF5 = 0x01000034,
  KeyF6 = 0x01000035,
  KeyF7 = 0x01000036,
  KeyF8 = 0x01000037,
  KeyF9 = 0x01000038,
  KeyF10 = 0x01000039,
  KeyF11 = 0x0100003a,
  KeyF12 = 0x0100003b,
  KeyF13 = 0x0100003c,
  KeyF14 = 0x0100003d,
  KeyF15 = 0x0100003e,
  KeyF16 = 0x0100003f,
  KeyF17 = 0x01000040,
  KeyF18 = 0x01000041,
  KeyF19 = 0x01000042,
  KeyF20 = 0x01000043,
  KeyF21 = 0x01000044,
  KeyF22 = 0x01000045,
  KeyF23 = 0x01000046,
  KeyF24 = 0x01000047,
  KeyF25 = 0x01000048,
  KeyF26 = 0x01000049,
  KeyF27 = 0x0100004a,
  KeyF28 = 0x0100004b,
  KeyF29 = 0x0100004c,
  KeyF30 = 0x0100004d,
  KeyF31 = 0x0100004e,
  KeyF32 = 0x0100004f,
  KeyF33 = 0x01000050,
  KeyF34 = 0x01000051,
  KeyF35 = 0x01000052,
  KeySuper_L = 0x01000053,
  KeySuper_R = 0x01000054,
  KeyMenu = 0x01000055,
  KeyHyper_L = 0x01000056,
  KeyHyper_R = 0x01000057,
  KeyHelp = 0x01000058,
  KeyDirection_L = 0x01000059,
  KeyDirection_R = 0x01000060,
  KeySpace = 0x20,
  KeyAny = KeySpace,
  KeyExclam = 0x21,
  KeyQuoteDbl = 0x22,
  KeyNumberSign = 0x23,
  KeyDollar = 0x24,
  KeyPercent = 0x25,
  KeyAmpersand = 0x26,
  KeyApostrophe = 0x27,
  KeyParenLeft = 0x28,
  KeyParenRight = 0x29,
  KeyAsterisk = 0x2a,
  KeyPlus = 0x2b,
  KeyComma = 0x2c,
  KeyMinus = 0x2d,
  KeyPeriod = 0x2e,
  KeySlash = 0x2f,
  Key0 = 0x30,
  Key1 = 0x31,
  Key2 = 0x32,
  Key3 = 0x33,
  Key4 = 0x34,
  Key5 = 0x35,
  Key6 = 0x36,
  Key7 = 0x37,
  Key8 = 0x38,
  Key9 = 0x39,
  KeyColon = 0x3a,
  KeySemicolon = 0x3b,
  KeyLess = 0x3c,
  KeyEqual = 0x3d,
  KeyGreater = 0x3e,
  KeyQuestion = 0x3f,
  KeyAt = 0x40,
  KeyA = 0x41,
  KeyB = 0x42,
  KeyC = 0x43,
  KeyD = 0x44,
  KeyE = 0x45,
  KeyF = 0x46,
  KeyG = 0x47,
  KeyH = 0x48,
  KeyI = 0x49,
  KeyJ = 0x4a,
  KeyK = 0x4b,
  KeyL = 0x4c,
  KeyM = 0x4d,
  KeyN = 0x4e,
  KeyO = 0x4f,
  KeyP = 0x50,
  KeyQ = 0x51,
  KeyR = 0x52,
  KeyS = 0x53,
  KeyT = 0x54,
  KeyU = 0x55,
  KeyV = 0x56,
  KeyW = 0x57,
  KeyX = 0x58,
  KeyY = 0x59,
  KeyZ = 0x5a,
  KeyBracketLeft = 0x5b,
  KeyBackslash = 0x5c,
  KeyBracketRight = 0x5d,
  KeyAsciiCircum = 0x5e,
  KeyUnderscore = 0x5f,
  KeyQuoteLeft = 0x60,
  KeyBraceLeft = 0x7b,
  KeyBar = 0x7c,
  KeyBraceRight = 0x7d,
  KeyAsciiTilde = 0x7e
};

enum CursorShape
{
  CursorArrow = 0,
  CursorUpArrow = 1,
  CursorCross = 2,
  CursorWait = 3,
  CursorIBeam = 4,
  CursorSizeVer = 5,
  CursorSizeHor = 6,
  CursorSizeBDiag = 7,
  CursorSizeFDiag = 8,
  CursorSizeAll = 9,
  CursorBlank = 10,
  CursorSplitV = 11,
  CursorSplitH = 12,
  CursorPointingHand = 13,
  CursorForbidden = 14,
  CursorOpenHand = 17,
  CursorClosedHand = 18,
  CursorWhatsThis = 15,
  CursorBusy = 16
};

class OptsPrivate;

class Opts
{ 
public:  
GUI_API Opts();
GUI_API Opts(const Opts&);
GUI_API Opts(const Opts&,const Opts&);
GUI_API Opts(const Opts&,const Opts&,const Opts&);
GUI_API Opts(const Opts&,const Opts&,const Opts&,const Opts&);
GUI_API Opts(const Opts&,const Opts&,const Opts&,const Opts&,const Opts&);
GUI_API ~Opts();

GUI_API Opts& operator=(const Opts&);

GUI_API Opts& align(int alignFlags);
GUI_API Opts& stretch(int stretch);
GUI_API Opts& cell(int row,int column);  
GUI_API Opts& span(int rowSpan,int columnSpan);

GUI_API Opts& sizePolicy(SizePolicy horizontal,SizePolicy vertical);

GUI_API Opts& minimumWidth(int width);
GUI_API Opts& minimumHeight(int height);
GUI_API Opts& minimumSize(int width,int height);
  
GUI_API Opts& maximumWidth(int width); 
GUI_API Opts& maximumHeight(int height);
GUI_API Opts& maximumSize(int width,int height);
  
GUI_API Opts& fixedWidth(int width); 
GUI_API Opts& fixedHeight(int height);
GUI_API Opts& fixedSize(int width,int height);

GUI_API Opts& initialGeometry(int x,int y,int width,int height);

GUI_API Opts& margins(int left,int top,int right,int bottom);
 
GUI_API Opts& enabled(bool enabled);

GUI_API Opts& cursor(CursorShape cursor);

GUI_API Opts& focusPolicy(FocusPolicy policy);

GUI_API Opts& toolTip(char* text);

// Label, LineEdit, GroupBox
GUI_API Opts& alignText(int alignFlags);

// LineEdit, SpinBox
GUI_API Opts& readOnly(bool readOnly);

// Slider, ScrollBar, SpinBox
GUI_API Opts& singleStep(int step);
GUI_API Opts& singleStep(float step);

// Slider, ScrollBar
GUI_API Opts& pageStep(int step);
GUI_API Opts& pageStep(float step);
GUI_API Opts& tracking(bool tracking);

// Slider
GUI_API Opts& tickInterval(int interval);
GUI_API Opts& tickPosition(SliderTicks ticks);

// SpinBox
GUI_API Opts& keyboardTracking(bool tracking);
GUI_API Opts& decimals(int decimals);
  
// Frame, Label
GUI_API Opts& frameShape(FrameShape shape);
GUI_API Opts& frameShadow(FrameShadow shadow);
GUI_API Opts& frameLineWidth(int width);
GUI_API Opts& frameMidLineWidth(int width);

// Window
GUI_API Opts& modal(bool modal);
GUI_API Opts& showTitleBar(bool showTitleBar);
GUI_API Opts& showMinimizeButton(bool showMinimizeButton);
GUI_API Opts& showMaximizeButton(bool showMaximizeButton);
GUI_API Opts& showMinMaxButtons(bool showMinimizeButton,bool showMaximizeButton);
GUI_API Opts& showCloseButton(bool showCloseButton);
GUI_API Opts& showSystemMenu(bool showSystemMenu);
GUI_API Opts& showFrame(bool showFrame);
GUI_API Opts& showMaximized(bool showMaximized);
GUI_API Opts& stayOnTop(bool stayOnTop);

// Layout
GUI_API Opts& sizeConstraint(SizeConstraint constraint);
GUI_API Opts& spacing(int spacing);
GUI_API Opts& horizontalSpacing(int spacing);
GUI_API Opts& verticalSpacing(int spacing);
GUI_API Opts& spacing(int hspacing,int vspacing);
  
OptsPrivate* opts;
};

GUI_API void guiInit(int& argc,char** argv);
GUI_API void guiInit();
GUI_API void guiUpdate(bool wait=false);
GUI_API void guiUpdateAndWait();
GUI_API void guiCleanup();

GUI_API void Label(int id,const char* text,const Opts& opts = Opts());

GUI_API void HSeparator(int id,const Opts& opts = Opts());
GUI_API void VSeparator(int id,const Opts& opts = Opts());
  
GUI_API bool Button(int id,const char* text,const Opts& opts = Opts());
GUI_API bool Button(int id,const char* iconFileName,const char* text,const Opts& opts = Opts());
GUI_API bool Button(int id,const int iconWidth,const int iconHeight,void* iconData,const Opts& opts = Opts());

GUI_API bool RadioButton(int id,const char* text,int tag,int* value,const Opts& opts = Opts());

GUI_API bool ToggleButton(int id,const char* text,bool* state,const Opts& opts = Opts());
GUI_API bool ToggleButton(int id,const char* iconFileName,const char* text,bool* state,const Opts& opts = Opts());

GUI_API bool CheckBox(int id,const char* text,bool* state,const Opts& opts = Opts());

GUI_API bool ComboBox(int id,int count,char** texts,int* index,const Opts& opts = Opts());

GUI_API bool TabBar(int id,int count,char** texts,int* index,const Opts& opts = Opts());

GUI_API bool SpinBox(int id,int min,int max,int* value,const Opts& opts = Opts());
GUI_API bool SpinBox(int id,float min,float max,float* value,const Opts& opts = Opts());

GUI_API bool LineEdit(int id,int* value,const Opts& opts = Opts());
GUI_API bool LineEdit(int id,float* value,const Opts& opts = Opts());

GUI_API bool HSlider(int id,int min,int max,int* value,const Opts& opts = Opts());
GUI_API bool HSlider(int id,float min,float max,float* value,const Opts& opts = Opts());

GUI_API bool VSlider(int id,int min,int max,int* value,const Opts& opts = Opts());
GUI_API bool VSlider(int id,float min,float max,float* value,const Opts& opts = Opts());

GUI_API bool HScrollBar(int id,int min,int max,int* value,const Opts& opts = Opts());
GUI_API bool HScrollBar(int id,float min,float max,float* value,const Opts& opts = Opts());

GUI_API bool VScrollBar(int id,int min,int max,int* value,const Opts& opts = Opts());
GUI_API bool VScrollBar(int id,float min,float max,float* value,const Opts& opts = Opts());

GUI_API void Spacer(int id,const Opts& opts = Opts());

GUI_API void WindowBegin(int id,const char* title,const Opts& opts = Opts());
GUI_API void WindowBegin(int id,const char* iconFileName,const char* title,const Opts& opts = Opts());
GUI_API void WindowEnd();

GUI_API bool windowCloseRequest();

GUI_API void raise();
GUI_API void lower();

GUI_API void FrameBegin(int id,const Opts& opts = Opts());
GUI_API void FrameEnd();

GUI_API void GroupBoxBegin(int id,const char* text,const Opts& opts = Opts());
GUI_API void GroupBoxEnd();

GUI_API void PixmapBegin(int id,const Opts& opts = Opts());
GUI_API void PixmapEnd();

GUI_API void pixmapBlit(int width,int height,const unsigned char* data);

GUI_API void HBoxLayoutBegin(int id,const Opts& opts = Opts());
GUI_API void HBoxLayoutEnd();

GUI_API void VBoxLayoutBegin(int id,const Opts& opts = Opts());
GUI_API void VBoxLayoutEnd();

GUI_API void GridLayoutBegin(int id,const Opts& opts = Opts());
GUI_API void GridLayoutEnd();

GUI_API char* FileOpenDialog(const char* caption = 0,const char* dir = 0,const char* filter = 0);
GUI_API char* FileSaveDialog(const char* caption = 0,const char* dir = 0,const char* filter = 0);

GUI_API void MessageDialog(const char* text);

GUI_API int widgetWidth();
GUI_API int widgetHeight();
GUI_API bool widgetResized(int* width = 0,int* height = 0);
GUI_API void widgetGeometry(int* x,int* y,int* w,int* h);
GUI_API void widgetSetGeometry(int x,int y,int w,int h);

GUI_API bool mouseDown(MouseButton button);
GUI_API bool mousePressed(MouseButton button);
GUI_API bool mouseUp(MouseButton button);

GUI_API int mouseX();
GUI_API int mouseY();

GUI_API int mouseWheelDelta();

GUI_API bool mouseIsOver();

GUI_API bool keyDown(Key key);
GUI_API bool keyPressed(Key key);
GUI_API bool keyUp(Key key);

GUI_API bool widgetHasFocus();

class GLContextPrivate;

class GLContext
{
public:
GUI_API GLContext();
GUI_API ~GLContext();
 
GUI_API void makeCurrent();
GUI_API void doneCurrent();

GLContextPrivate* glContextPrivate;
};

GUI_API void GLWidgetBegin(int id,GLContext* ctx,const Opts& opts = Opts());
GUI_API void GLWidgetEnd();

#endif
