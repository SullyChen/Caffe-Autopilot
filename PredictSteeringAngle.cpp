//This is a modified version of BVLC Caffe's "classification.cpp"

//  Created by Sully Chen
//  Copyright © 2015 Sully Chen. All rights reserved.

#include <iostream>
#include <caffe/caffe.hpp>
#define USE_OPENCV = 1;
#ifdef USE_OPENCV
#include "opencv2/opencv.hpp"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#endif  // USE_OPENCV
#include <algorithm>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <thread>

#include <stdio.h>    /* Standard input/output definitions */
#include <stdint.h>   /* Standard types */
#include <unistd.h>   /* UNIX standard function definitions */
#include <fcntl.h>    /* File control definitions */
#include <errno.h>    /* Error number definitions */
#include <termios.h>  /* POSIX terminal control definitions */
#include <sys/ioctl.h>
#include <getopt.h>

#include "SFML/Graphics.hpp"

const int width = 240; //SFML window width
const int height = 240; //SFML window height

#ifdef USE_OPENCV
using namespace caffe;  // NOLINT(build/namespaces)
using std::string;

char buf[256]; //where the serial messages received are stored

void usage(void);
int serialport_init(const char* serialport, int baud);
int serialport_writebyte(int fd, uint8_t b);
int serialport_write(int fd, const char* str);
int serialport_read_until(int fd, char* buf, char until);

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

/* Pair (label, confidence) representing a prediction. */
typedef std::pair<string, float> Prediction;

class Classifier {
 public:
  Classifier(const string& model_file,
             const string& trained_file,
             const string& mean_file,
             const string& label_file);

  std::vector<Prediction> Classify(const cv::Mat& img, int N = 5);

 private:
  void SetMean(const string& mean_file);

  std::vector<float> Predict(const cv::Mat& img);

  void WrapInputLayer(std::vector<cv::Mat>* input_channels);

  void Preprocess(const cv::Mat& img,
                  std::vector<cv::Mat>* input_channels);

 private:
  shared_ptr<Net<float> > net_;
  cv::Size input_geometry_;
  int num_channels_;
  cv::Mat mean_;
  std::vector<string> labels_;
};

Classifier::Classifier(const string& model_file,
                       const string& trained_file,
                       const string& mean_file,
                       const string& label_file) {
#ifdef CPU_ONLY
  Caffe::set_mode(Caffe::CPU);
#else
  Caffe::set_mode(Caffe::GPU);
#endif

  /* Load the network. */
  net_.reset(new Net<float>(model_file, TEST));
  net_->CopyTrainedLayersFrom(trained_file);

  CHECK_EQ(net_->num_inputs(), 1) << "Network should have exactly one input.";
  CHECK_EQ(net_->num_outputs(), 1) << "Network should have exactly one output.";

  Blob<float>* input_layer = net_->input_blobs()[0];
  num_channels_ = input_layer->channels();
  CHECK(num_channels_ == 3 || num_channels_ == 1)
    << "Input layer should have 1 or 3 channels.";
  input_geometry_ = cv::Size(input_layer->width(), input_layer->height());

  /* Load the binaryproto mean file. */
  SetMean(mean_file);

  /* Load labels. */
  std::ifstream labels(label_file.c_str());
  CHECK(labels) << "Unable to open labels file " << label_file;
  string line;
  while (std::getline(labels, line))
    labels_.push_back(string(line));

  Blob<float>* output_layer = net_->output_blobs()[0];
  CHECK_EQ(labels_.size(), output_layer->channels())
    << "Number of labels is different from the output layer dimension.";
}

static bool PairCompare(const std::pair<float, int>& lhs,
                        const std::pair<float, int>& rhs) {
  return lhs.first > rhs.first;
}

/* Return the indices of the top N values of vector v. */
static std::vector<int> Argmax(const std::vector<float>& v, int N) {
  std::vector<std::pair<float, int> > pairs;
  for (size_t i = 0; i < v.size(); ++i)
    pairs.push_back(std::make_pair(v[i], i));
  std::partial_sort(pairs.begin(), pairs.begin() + N, pairs.end(), PairCompare);

  std::vector<int> result;
  for (int i = 0; i < N; ++i)
    result.push_back(pairs[i].second);
  return result;
}

/* Return the top N predictions. */
std::vector<Prediction> Classifier::Classify(const cv::Mat& img, int N) {
  std::vector<float> output = Predict(img);

  N = std::min<int>(labels_.size(), N);
  std::vector<int> maxN = Argmax(output, N);
  std::vector<Prediction> predictions;
  for (int i = 0; i < N; ++i) {
    int idx = maxN[i];
    predictions.push_back(std::make_pair(labels_[idx], output[idx]));
  }

  return predictions;
}

/* Load the mean file in binaryproto format. */
void Classifier::SetMean(const string& mean_file) {
  BlobProto blob_proto;
  ReadProtoFromBinaryFileOrDie(mean_file.c_str(), &blob_proto);

  /* Convert from BlobProto to Blob<float> */
  Blob<float> mean_blob;
  mean_blob.FromProto(blob_proto);
  CHECK_EQ(mean_blob.channels(), num_channels_)
    << "Number of channels of mean file doesn't match input layer.";

  /* The format of the mean file is planar 32-bit float BGR or grayscale. */
  std::vector<cv::Mat> channels;
  float* data = mean_blob.mutable_cpu_data();
  for (int i = 0; i < num_channels_; ++i) {
    /* Extract an individual channel. */
    cv::Mat channel(mean_blob.height(), mean_blob.width(), CV_32FC1, data);
    channels.push_back(channel);
    data += mean_blob.height() * mean_blob.width();
  }

  /* Merge the separate channels into a single image. */
  cv::Mat mean;
  cv::merge(channels, mean);

  /* Compute the global mean pixel value and create a mean image
   * filled with this value. */
  cv::Scalar channel_mean = cv::mean(mean);
  mean_ = cv::Mat(input_geometry_, mean.type(), channel_mean);
}

std::vector<float> Classifier::Predict(const cv::Mat& img) {
  Blob<float>* input_layer = net_->input_blobs()[0];
  input_layer->Reshape(1, num_channels_,
                       input_geometry_.height, input_geometry_.width);
  /* Forward dimension change to all layers. */
  net_->Reshape();

  std::vector<cv::Mat> input_channels;
  WrapInputLayer(&input_channels);

  Preprocess(img, &input_channels);

  net_->Forward();

  /* Copy the output layer to a std::vector */
  Blob<float>* output_layer = net_->output_blobs()[0];
  const float* begin = output_layer->cpu_data();
  const float* end = begin + output_layer->channels();
  return std::vector<float>(begin, end);
}

/* Wrap the input layer of the network in separate cv::Mat objects
 * (one per channel). This way we save one memcpy operation and we
 * don't need to rely on cudaMemcpy2D. The last preprocessing
 * operation will write the separate channels directly to the input
 * layer. */
void Classifier::WrapInputLayer(std::vector<cv::Mat>* input_channels) {
  Blob<float>* input_layer = net_->input_blobs()[0];

  int width = input_layer->width();
  int height = input_layer->height();
  float* input_data = input_layer->mutable_cpu_data();
  for (int i = 0; i < input_layer->channels(); ++i) {
    cv::Mat channel(height, width, CV_32FC1, input_data);
    input_channels->push_back(channel);
    input_data += width * height;
  }
}

void Classifier::Preprocess(const cv::Mat& img,
                            std::vector<cv::Mat>* input_channels) {
  /* Convert the input image to the input image format of the network. */
  cv::Mat sample;
  if (img.channels() == 3 && num_channels_ == 1)
    cv::cvtColor(img, sample, cv::COLOR_BGR2GRAY);
  else if (img.channels() == 4 && num_channels_ == 1)
    cv::cvtColor(img, sample, cv::COLOR_BGRA2GRAY);
  else if (img.channels() == 4 && num_channels_ == 3)
    cv::cvtColor(img, sample, cv::COLOR_BGRA2BGR);
  else if (img.channels() == 1 && num_channels_ == 3)
    cv::cvtColor(img, sample, cv::COLOR_GRAY2BGR);
  else
    sample = img;

  cv::Mat sample_resized;
  if (sample.size() != input_geometry_)
    cv::resize(sample, sample_resized, input_geometry_);
  else
    sample_resized = sample;

  cv::Mat sample_float;
  if (num_channels_ == 3)
    sample_resized.convertTo(sample_float, CV_32FC3);
  else
    sample_resized.convertTo(sample_float, CV_32FC1);

  cv::Mat sample_normalized;
  cv::subtract(sample_float, mean_, sample_normalized);

  /* This operation will write the separate BGR planes directly to the
   * input layer of the network because it is wrapped by the cv::Mat
   * objects in input_channels. */
  cv::split(sample_normalized, *input_channels);

  CHECK(reinterpret_cast<float*>(input_channels->at(0).data)
        == net_->input_blobs()[0]->cpu_data())
    << "Input channels are not wrapping the input layer of the network.";
}

int main(int argc, char** argv) {
  if (argc != 5) {
    std::cerr << "Usage: " << argv[0]
              << " deploy.prototxt network.caffemodel"
              << " mean.binaryproto labels.txt" << std::endl;
    return 1;
  }

  //SFML things to draw steering wheel
  sf::String title_string = "Predicted";
  sf::String title_string2 = "Actual";
  sf::RenderWindow window(sf::VideoMode(width, height), title_string);
  sf::RenderWindow window2(sf::VideoMode(width, height), title_string2);
  window.setFramerateLimit(30);
  window2.setFramerateLimit(30);

  ::google::InitGoogleLogging(argv[0]);

  string model_file   = argv[1];
  string trained_file = argv[2];
  string mean_file    = argv[3];
  string label_file   = argv[4];
  Classifier classifier(model_file, trained_file, mean_file, label_file);

  cv::namedWindow("img",1);

  //open camera
  cv::VideoCapture cap(0);
  if(!cap.isOpened())
    return -1;

  //load the steering wheel image
  sf::Texture steeringWheelTexture;
  steeringWheelTexture.loadFromFile("steering_wheel_image.jpg");

  //create steering wheel sprite
  sf::Sprite steeringWheelSprite;
  steeringWheelSprite.setTexture(steeringWheelTexture);
  steeringWheelSprite.setPosition(120, 120);
  steeringWheelSprite.setRotation(30.0f);
  steeringWheelSprite.setOrigin(120, 120);

  double smoothed_angle = 0.0f; //Used as the final steering output

  int actual_angle = 0;

  std::thread t(thread1);

  while (true)
  {
    int actual_angle = (int)::atof(buf);
    if (cv::waitKey(10) == 'q') break; // break the loop if the key "q" is pressed
    cv::Mat img;
    cap >> img; // get a new frame from camera

    //crop the camera image
    cv::Rect myROI(280, 0, 720, 720);
    cv::Mat croppedImage = img(myROI);

    resize(croppedImage, img, cv::Size(), 256.0f/720.0f, 256.0f/720.0f, cv::INTER_LANCZOS4); //resize to 256x256

    cv::imshow("img", img); //show the image

    //canny edge filtering
    Canny(img, img, 50, 200, 3);

    //Preprocessing to remove highly cluttered areas from the image
    cv::Mat mask;
    GaussianBlur(img, mask, cv::Size(35, 35), 10, 10);
    threshold(mask, mask, 60, 255, cv::THRESH_BINARY_INV);
    bitwise_and(img, mask, img);

    cv::imshow("filtered img", img); //show the image

    std::vector<Prediction> predictions = classifier.Classify(img); //Use BVLC Caffe to classify the image

    system("clear"); //clear the console for neatness

    std::vector<int> angle_categories; //This vector stores all of the possible angle outputs from the classifier
    std::vector<double> outputs; //This vector stores all of the corresponding prediction outputs from the classifier

    //process classifier outputs
    for (int i = 0; i < predictions.size(); i++)
    {
      Prediction p = predictions[i];
      int angle_category;

      //if the category is just "0", then the angle_category is 0
      if (p.first == "0")
        angle_category = 0;
      else //otherwise, do further processing
      {
        //categories are in the format "posXXX" or "negXXX", where XXX is an angle measure
        angle_category = std::stoi(p.first.substr(3, p.first.length() - 3)); //convert the string to an integer, ignoring the first three characters which are likely "pos" or "neg"

        //if the first three letters are "neg", multiply the category by -1 to make the category negative
        if (p.first.substr(0, 3) == "neg")
          angle_category *= -1;
      }
      angle_categories.push_back(angle_category); //store the possible categories in the categories vector
      outputs.push_back(p.second); //store the corresponding output in the outputs vector
    }

    double output_angle = 0.0f; //this is used for a processing step before the final angle

    //do a weighted sum of the classifier outputs
    for (int i = 0; i < angle_categories.size(); i++)
      output_angle += angle_categories[i] * outputs[i];

    //make smooth angle transitions by turning the steering wheel based on the difference of the current angle
    //and the predicted angle
    smoothed_angle += 1.0f * pow(std::abs((output_angle - smoothed_angle)), 2.0f / 3.0f)
                          * (output_angle - smoothed_angle) / std::abs(output_angle - smoothed_angle);

    //print the angle
    if (smoothed_angle < 0)
      std::cout << std::fixed << std::setprecision(2) << -1.0f * smoothed_angle << " degrees left" << std::endl;
    else if (smoothed_angle > 0)
      std::cout << std::fixed << std::setprecision(2) << smoothed_angle << " degrees right" << std::endl;
    else
      std::cout << "0 degrees" << std::endl;

      //update SFML things
      steeringWheelSprite.setRotation(smoothed_angle);

      window.clear(sf::Color::White);
      window2.clear(sf::Color::White);

      window.draw(steeringWheelSprite);

      steeringWheelSprite.setRotation(actual_angle);
      window2.draw(steeringWheelSprite);

      window.display();
      window2.display();
  }
}
#else
int main(int argc, char** argv) {
  LOG(FATAL) << "This example requires OpenCV; compile with USE_OPENCV.";
}
#endif  // USE_OPENCV

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
