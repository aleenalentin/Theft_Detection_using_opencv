
#include "imagep1.h"

int opencv1::TheftDetection()
{

    VideoCapture cap("/home/aleena/Desktop/project06/thief.mp4");
    
   //Create trackbars in "Control" window 
    namedWindow("Control", 0);
    
 int history = 600;
 int threshold = 14; 
 int shadow = 0;
 int kernelsize = 5;
 


//History (50- 1000)
createTrackbar("history", "Control", &history, 1000);
 
//threshold (5- 50)
createTrackbar("threshold", "Control", &threshold, 50);

//shadow
createTrackbar("shadow", "Control", &shadow, 1);

// kernalsize
createTrackbar("kernelsize", "Control", &kernelsize, 10);   
    
    

    Mat3b frame;
    Mat1b fmask;
    Mat kernel = getStructuringElement(MORPH_RECT, Size(kernelsize,kernelsize));
    Ptr<BackgroundSubtractorMOG2> bg = createBackgroundSubtractorMOG2(history, threshold, shadow);
   
  // Ptr<BackgroundSubtractorKNN> bg = createBackgroundSubtractorKNN(500, 400.0, false);
    for (;;)
    {
        // Capture frame
        cap >> frame;

        // Background subtraction
        bg->apply(frame, fmask, -1);

        // Clean foreground from noise
        morphologyEx(fmask, fmask, MORPH_OPEN, kernel);
        
    //Perform dilation next
    Mat dilate_img;
    dilate(fmask,dilate_img, kernel);
    imshow("dilate_img",dilate_img);

        // Find contours
        vector<vector<Point>> contours;
        //findContours(fmask.clone(), contours, RETR_TREE, CHAIN_APPROX_SIMPLE);
        
        findContours(dilate_img.clone(), contours, RETR_TREE, CHAIN_APPROX_SIMPLE);

        if (!contours.empty())
        {
            // Get largest contour
            int idx_largest_contour = -1;
            double area_largest_contour = 0.0;

            for (int i = 0; i < contours.size(); ++i)
            {
                double area = contourArea(contours[i]);
                if (area_largest_contour < area)
                {
                    area_largest_contour = area;
                    cout<<area_largest_contour<<"\n";
                    idx_largest_contour = i;
                }
            }
// choose the contour between 300 to 30000 
            if (area_largest_contour >= 300 && area_largest_contour <= 30000)
            {
              
                Rect roi = boundingRect(contours[idx_largest_contour]);
                drawContours(frame, contours, idx_largest_contour, Scalar(0, 0, 255));
                rectangle(frame, roi, Scalar(0, 255, 0));
            }
        }

        imshow("frame", frame);
        //imshow("mask", fmask);
        if (cv::waitKey(30) >= 0) break;
    }
    return 0;
}
