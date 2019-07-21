#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main( int argc, char** argv )
{
    if(argc < 5){
        cout<<" Usage: rpp_test input_image width height channel"<<endl;
        return -1;
    }

    int ip_width = atoi(argv[2]);
    int ip_height = atoi(argv[3]);
    int ip_channel = atoi(argv[4]);
    // Image read from user and fill it in Unsigned char buffer
    int total_pixels = ip_width * ip_height * ip_channel;
    unsigned char *ip_image = new unsigned char[total_pixels];

    // Open CV image read and display
    Mat image;
    image = imread(argv[1], CV_LOAD_IMAGE_COLOR);   // Read the file

    if(! image.data )                              // Check for invalid input
    {
        cout <<  "Could not open or find the image" << std::endl ;
        return -1;
    }

    namedWindow( "Display window", WINDOW_AUTOSIZE );// Create a window for display.
    imshow( "Display window", image );                   // Show our image inside it.

    waitKey(0);                                          // Wait for a keystroke in the window
    return 0;
}