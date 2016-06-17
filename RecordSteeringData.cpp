//This code uses arduino serial communication code by Tod E. Kurt, tod@todbot.com http://todbot.com/blog/

//  Created by Sully Chen
//  Copyright © 2015 Sully Chen. All rights reserved.

#include <stdio.h>    /* Standard input/output definitions */
#include <stdlib.h>
#include <stdint.h>   /* Standard types */
#include <string.h>   /* String function definitions */
#include <unistd.h>   /* UNIX standard function definitions */
#include <fcntl.h>    /* File control definitions */
#include <errno.h>    /* Error number definitions */
#include <termios.h>  /* POSIX terminal control definitions */
#include <sys/ioctl.h>
#include <getopt.h>
#include "opencv2/opencv.hpp"
#include <fstream>
#include <thread>

using namespace std;
using namespace cv;

void usage(void);
int serialport_init(const char* serialport, int baud);
int serialport_writebyte(int fd, uint8_t b);
int serialport_write(int fd, const char* str);
int serialport_read_until(int fd, char* buf, char until);

void usage(void) {
    printf("Usage: arduino-serial -p <serialport> [OPTIONS]\n"
    "\n"
    "Options:\n"
    "  -h, --help                   Print this help message\n"
    "  -p, --port=serialport        Serial port Arduino is on\n"
    "  -b, --baud=baudrate          Baudrate (bps) of Arduino\n"
    "  -s, --send=data              Send data to Arduino\n"
    "  -r, --receive                Receive data from Arduino & print it out\n"
    "  -n  --num=num                Send a number as a single byte\n"
    "  -d  --delay=millis           Delay for specified milliseconds\n"
    "\n"
    "Note: Order is important. Set '-b' before doing '-p'. \n"
    "      Used to make series of actions:  '-d 2000 -s hello -d 100 -r' \n"
    "      means 'wait 2secs, send 'hello', wait 100msec, get reply'\n"
    "\n");
}

char buf[256]; //where the serial messages received are stored

void thread1() //messages are read in a separate thread to fix timing issues with OpenCV
{
  int fd = 0;
  char serialport[256];
  int baudrate = B115200;  // default
  fd = serialport_init("/dev/cu.usbmodem1421", baudrate); //open serial port of arduino
  if(fd == -1)
    std::cout << "Error opening port!" << std::endl;
  while (true)
    serialport_read_until(fd, buf, '\n');
}

int main(int argc, char *argv[])
{
  //generate data directories
  for (int i = -18; i <= 18; i++)
  {
    std::string s;
    if (i == 0)
      s = "mkdir 0";
    else if (i < 0)
      s = "mkdir neg" + std::to_string(-i * 5);
    else
      s = "mkdir pos" + std::to_string(i * 5);
    std::cout << s << std::endl;
    system(s.c_str());
  }

  //open camera
  VideoCapture cap(0);
  if(!cap.isOpened())
      return -1;
  Mat edges;
  namedWindow("frame",1);
  int i = 0;
  std::cout << "Input starting index: ";
  cin >> i;
  cout << "\n";
  std::thread t(thread1);
  while (true)
  {
    int key_press = waitKey(10);
    if (key_press != 's')
    {
      Mat frame;
      cap >> frame; // get a new frame from camera

      //crop and resize
      Rect myROI(280, 0, 720, 720);
      Mat croppedImage = frame(myROI);
      resize(croppedImage, frame, Size(), 256.0f/720.0f, 256.0f/720.0f, cv::INTER_LANCZOS4);

      //canny edge filtering
      Canny(frame, frame, 50, 200, 3);

      //Preprocessing to remove highly cluttered areas from the image
      Mat mask;
      GaussianBlur(frame, mask, Size(35, 35), 10, 10);
      threshold(mask, mask, 60, 255, THRESH_BINARY_INV);
      bitwise_and(frame, mask, frame);

      //get steering angle from the serial data
      int angle = (int)::atof(buf);

      if (angle <= 90 && angle >= -90)
      {
        std::string s;
        angle = (angle / 5) * 5;
        if (angle == 0)
          s = "0";
        else if (angle < 0)
          s = "neg" + std::to_string(-angle);
        else
          s = "pos" + std::to_string(angle);
          std::cout << s << std::endl;
          imwrite(s + "/" + std::to_string(i) + ".jpg", frame);
        if (key_press == 'q')
          break;
        i++;
      }
      imshow("frame", frame);
    } else while (waitKey() != 'c');
  }
  exit(EXIT_SUCCESS);
}

int serialport_writebyte( int fd, uint8_t b)
{
    int n = write(fd,&b,1);
    if( n!=1)
        return -1;
    return 0;
}

int serialport_write(int fd, const char* str)
{
    int len = strlen(str);
    int n = write(fd, str, len);
    if( n!=len )
        return -1;
    return 0;
}

int serialport_read_until(int fd, char* buf, char until)
{
    char b[1];
    int i=0;
    do {
        int n = read(fd, b, 1);  // read a char at a time
        if( n==-1) return -1;    // couldn't read
        if( n==0 ) {
            usleep( 10 * 1000 ); // wait 10 msec try again
            continue;
        }
        buf[i] = b[0]; i++;
    } while( b[0] != until );

    buf[i] = 0;  // null terminate the string
    return 0;
}

// takes the string name of the serial port (e.g. "/dev/tty.usbserial","COM1")
// and a baud rate (bps) and connects to that port at that speed and 8N1.
// opens the port in fully raw mode so you can send binary data.
// returns valid fd, or -1 on error
int serialport_init(const char* serialport, int baud)
{
    struct termios toptions;
    int fd;

    //fprintf(stderr,"init_serialport: opening port %s @ %d bps\n",
    //        serialport,baud);

    fd = open(serialport, O_RDWR | O_NOCTTY | O_NDELAY);
    if (fd == -1)  {
        perror("init_serialport: Unable to open port ");
        return -1;
    }

    if (tcgetattr(fd, &toptions) < 0) {
        perror("init_serialport: Couldn't get term attributes");
        return -1;
    }
    speed_t brate = baud; // let you override switch below if needed
    switch(baud) {
    case 4800:   brate=B4800;   break;
    case 9600:   brate=B9600;   break;
#ifdef B14400
    case 14400:  brate=B14400;  break;
#endif
    case 19200:  brate=B19200;  break;
#ifdef B28800
    case 28800:  brate=B28800;  break;
#endif
    case 38400:  brate=B38400;  break;
    case 57600:  brate=B57600;  break;
    case 115200: brate=B115200; break;
    }
    cfsetispeed(&toptions, brate);
    cfsetospeed(&toptions, brate);

    // 8N1
    toptions.c_cflag &= ~PARENB;
    toptions.c_cflag &= ~CSTOPB;
    toptions.c_cflag &= ~CSIZE;
    toptions.c_cflag |= CS8;
    // no flow control
    toptions.c_cflag &= ~CRTSCTS;

    toptions.c_cflag |= CREAD | CLOCAL;  // turn on READ & ignore ctrl lines
    toptions.c_iflag &= ~(IXON | IXOFF | IXANY); // turn off s/w flow ctrl

    toptions.c_lflag &= ~(ICANON | ECHO | ECHOE | ISIG); // make raw
    toptions.c_oflag &= ~OPOST; // make raw

    // see: http://unixwiz.net/techtips/termios-vmin-vtime.html
    toptions.c_cc[VMIN]  = 0;
    toptions.c_cc[VTIME] = 20;

    if( tcsetattr(fd, TCSANOW, &toptions) < 0) {
        perror("init_serialport: Couldn't set term attributes");
        return -1;
    }

    return fd;
}
