// OpenCVApplication.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "common.h"
#include "limits.h"



void testOpenImage()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src;
		src = imread(fname);
		imshow("image",src);
		waitKey();
	}
}

void testOpenImagesFld()
{
	char folderName[MAX_PATH];
	if (openFolderDlg(folderName)==0)
		return;
	char fname[MAX_PATH];
	FileGetter fg(folderName,"bmp");
	while(fg.getNextAbsFile(fname))
	{
		Mat src;
		src = imread(fname);
		imshow(fg.getFoundFileName(),src);
		if (waitKey()==27) //ESC pressed
			break;
	}
}

void testImageOpenAndSave()
{
	Mat src, dst;

	src = imread("Images/Lena_24bits.bmp", CV_LOAD_IMAGE_COLOR);	// Read the image

	if (!src.data)	// Check for invalid input
	{
		printf("Could not open or find the image\n");
		return;
	}

	// Get the image resolution
	Size src_size = Size(src.cols, src.rows);

	// Display window
	const char* WIN_SRC = "Src"; //window for the source image
	namedWindow(WIN_SRC, CV_WINDOW_AUTOSIZE);
	cvMoveWindow(WIN_SRC, 0, 0);

	const char* WIN_DST = "Dst"; //window for the destination (processed) image
	namedWindow(WIN_DST, CV_WINDOW_AUTOSIZE);
	cvMoveWindow(WIN_DST, src_size.width + 10, 0);

	cvtColor(src, dst, CV_BGR2GRAY); //converts the source image to a grayscale one

	imwrite("Images/Lena_24bits_gray.bmp", dst); //writes the destination to file

	imshow(WIN_SRC, src);
	imshow(WIN_DST, dst);

	printf("Press any key to continue ...\n");
	waitKey(0);
}

void testNegativeImage()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		double t = (double)getTickCount(); // Get the current time [s]
		
		Mat src = imread(fname,CV_LOAD_IMAGE_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat dst = Mat(height,width,CV_8UC1);
		// Asa se acceseaaza pixelii individuali pt. o imagine cu 8 biti/pixel
		// Varianta ineficienta (lenta)
		for (int i=0; i<height; i++)
		{
			for (int j=0; j<width; j++)
			{
				uchar val = src.at<uchar>(i,j);
				uchar neg = 255 - val;
				dst.at<uchar>(i,j) = neg;
			}
		}

		// Get the current time again and compute the time difference [s]
		t = ((double)getTickCount() - t) / getTickFrequency();
		// Print (in the console window) the processing time in [ms] 
		printf("Time = %.3f [ms]\n", t * 1000);

		imshow("input image",src);
		imshow("negative image",dst);
		waitKey();
	}
}

void testParcurgereSimplaDiblookStyle()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat dst = src.clone();

		double t = (double)getTickCount(); // Get the current time [s]

		// the fastest approach using the “diblook style”
		uchar *lpSrc = src.data;
		uchar *lpDst = dst.data;
		int w = (int) src.step; // no dword alignment is done !!!
		for (int i = 0; i<height; i++)
			for (int j = 0; j < width; j++) {
				uchar val = lpSrc[i*w + j];
				lpDst[i*w + j] = 255 - val;
			}

		// Get the current time again and compute the time difference [s]
		t = ((double)getTickCount() - t) / getTickFrequency();
		// Print (in the console window) the processing time in [ms] 
		printf("Time = %.3f [ms]\n", t * 1000);

		imshow("input image",src);
		imshow("negative image",dst);
		waitKey();
	}
}

void testColor2Gray()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src = imread(fname);

		int height = src.rows;
		int width = src.cols;

		Mat dst = Mat(height,width,CV_8UC1);

		// Asa se acceseaaza pixelii individuali pt. o imagine RGB 24 biti/pixel
		// Varianta ineficienta (lenta)
		for (int i=0; i<height; i++)
		{
			for (int j=0; j<width; j++)
			{
				Vec3b v3 = src.at<Vec3b>(i,j);
				uchar b = v3[0];
				uchar g = v3[1];
				uchar r = v3[2];
				dst.at<uchar>(i,j) = (r+g+b)/3;
			}
		}
		
		imshow("input image",src);
		imshow("gray image",dst);
		waitKey();
	}
}

void testBGR2HSV()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname);
		int height = src.rows;
		int width = src.cols;

		// Componentele d eculoare ale modelului HSV
		Mat H = Mat(height, width, CV_8UC1);
		Mat S = Mat(height, width, CV_8UC1);
		Mat V = Mat(height, width, CV_8UC1);

		// definire pointeri la matricele (8 biti/pixeli) folosite la afisarea componentelor individuale H,S,V
		uchar* lpH = H.data;
		uchar* lpS = S.data;
		uchar* lpV = V.data;

		Mat hsvImg;
		cvtColor(src, hsvImg, CV_BGR2HSV);

		// definire pointer la matricea (24 biti/pixeli) a imaginii HSV
		uchar* hsvDataPtr = hsvImg.data;

		for (int i = 0; i<height; i++)
		{
			for (int j = 0; j<width; j++)
			{
				int hi = i*width * 3 + j * 3;
				int gi = i*width + j;

				lpH[gi] = hsvDataPtr[hi] * 510 / 360;		// lpH = 0 .. 255
				lpS[gi] = hsvDataPtr[hi + 1];			// lpS = 0 .. 255
				lpV[gi] = hsvDataPtr[hi + 2];			// lpV = 0 .. 255
			}
		}

		imshow("input image", src);
		imshow("H", H);
		imshow("S", S);
		imshow("V", V);

		waitKey();
	}
}

void testResize()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src;
		src = imread(fname);
		Mat dst1,dst2;
		//without interpolation
		resizeImg(src,dst1,320,false);
		//with interpolation
		resizeImg(src,dst2,320,true);
		imshow("input image",src);
		imshow("resized image (without interpolation)",dst1);
		imshow("resized image (with interpolation)",dst2);
		waitKey();
	}
}

void testCanny()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src,dst,gauss;
		src = imread(fname,CV_LOAD_IMAGE_GRAYSCALE);
		double k = 0.4;
		int pH = 50;
		int pL = (int) k*pH;
		GaussianBlur(src, gauss, Size(5, 5), 0.8, 0.8);
		Canny(gauss,dst,pL,pH,3);
		imshow("input image",src);
		imshow("canny",dst);
		waitKey();
	}
}

void testVideoSequence()
{
	VideoCapture cap("Videos/rubic.avi"); // off-line video from file
	//VideoCapture cap(0);	// live video from web cam
	if (!cap.isOpened()) {
		printf("Cannot open video capture device.\n");
		waitKey(0);
		return;
	}
		
	Mat edges;
	Mat frame;
	char c;

	while (cap.read(frame))
	{
		Mat grayFrame;
		cvtColor(frame, grayFrame, CV_BGR2GRAY);
		Canny(grayFrame,edges,40,100,3);
		imshow("source", frame);
		imshow("gray", grayFrame);
		imshow("edges", edges);
		c = cvWaitKey(0);  // waits a key press to advance to the next frame
		if (c == 27) {
			// press ESC to exit
			printf("ESC pressed - capture finished\n"); 
			break;  //ESC pressed
		};
	}
}


void testSnap()
{
	VideoCapture cap(0); // open the deafult camera (i.e. the built in web cam)
	if (!cap.isOpened()) // openenig the video device failed
	{
		printf("Cannot open video capture device.\n");
		return;
	}

	Mat frame;
	char numberStr[256];
	char fileName[256];
	
	// video resolution
	Size capS = Size((int)cap.get(CV_CAP_PROP_FRAME_WIDTH),
		(int)cap.get(CV_CAP_PROP_FRAME_HEIGHT));

	// Display window
	const char* WIN_SRC = "Src"; //window for the source frame
	namedWindow(WIN_SRC, CV_WINDOW_AUTOSIZE);
	cvMoveWindow(WIN_SRC, 0, 0);

	const char* WIN_DST = "Snapped"; //window for showing the snapped frame
	namedWindow(WIN_DST, CV_WINDOW_AUTOSIZE);
	cvMoveWindow(WIN_DST, capS.width + 10, 0);

	char c;
	int frameNum = -1;
	int frameCount = 0;

	for (;;)
	{
		cap >> frame; // get a new frame from camera
		if (frame.empty())
		{
			printf("End of the video file\n");
			break;
		}

		++frameNum;
		
		imshow(WIN_SRC, frame);

		c = cvWaitKey(10);  // waits a key press to advance to the next frame
		if (c == 27) {
			// press ESC to exit
			printf("ESC pressed - capture finished");
			break;  //ESC pressed
		}
		if (c == 115){ //'s' pressed - snapp the image to a file
			frameCount++;
			fileName[0] = NULL;
			sprintf(numberStr, "%d", frameCount);
			strcat(fileName, "Images/A");
			strcat(fileName, numberStr);
			strcat(fileName, ".bmp");
			bool bSuccess = imwrite(fileName, frame);
			if (!bSuccess) 
			{
				printf("Error writing the snapped image\n");
			}
			else
				imshow(WIN_DST, frame);
		}
	}

}

void MyCallBackFunc(int event, int x, int y, int flags, void* param)
{
	//More examples: http://opencvexamples.blogspot.com/2014/01/detect-mouse-clicks-and-moves-on-image.html
	Mat* src = (Mat*)param;
	if (event == CV_EVENT_LBUTTONDOWN)
		{
			printf("Pos(x,y): %d,%d  Color(RGB): %d,%d,%d\n",
				x, y,
				(int)(*src).at<Vec3b>(y, x)[2],
				(int)(*src).at<Vec3b>(y, x)[1],
				(int)(*src).at<Vec3b>(y, x)[0]);
		}
}

void testMouseClick()
{
	Mat src;
	// Read image from file 
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		src = imread(fname);
		//Create a window
		namedWindow("My Window", 1);

		//set the callback function for any mouse event
		setMouseCallback("My Window", MyCallBackFunc, &src);

		//show the image
		imshow("My Window", src);

		// Wait until user press some key
		waitKey(0);
	}
}

/* Histogram display function - display a histogram using bars (simlilar to L3 / PI)
Input:
name - destination (output) window name
hist - pointer to the vector containing the histogram values
hist_cols - no. of bins (elements) in the histogram = histogram image width
hist_height - height of the histogram image
Call example:
showHistogram ("MyHist", hist_dir, 255, 200);
*/
void showHistogram(const std::string& name, int* hist, const int  hist_cols, const int hist_height)
{
	Mat imgHist(hist_height, hist_cols, CV_8UC3, CV_RGB(255, 255, 255)); // constructs a white image

	//computes histogram maximum
	int max_hist = 0;
	for (int i = 0; i<hist_cols; i++)
	if (hist[i] > max_hist)
		max_hist = hist[i];
	double scale = 1.0;
	scale = (double)hist_height / max_hist;
	int baseline = hist_height - 1;

	for (int x = 0; x < hist_cols; x++) {
		Point p1 = Point(x, baseline);
		Point p2 = Point(x, baseline - cvRound(hist[x] * scale));
		line(imgHist, p1, p2, CV_RGB(255, 0, 255)); // histogram bins colored in magenta
	}

	imshow(name, imgHist);
}


void add_factor(int additive)
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		double time = (double)getTickCount(); 

		Mat src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		int linii = src.rows;
		int col = src.cols;
		Mat dest = Mat(linii, col, CV_8UC1);

		for (int i = 0; i < linii; i++)
		{
			for (int j = 0; j < col; j++)
			{
				uchar val_current = src.at<uchar>(i, j);
				int check = 9999999;
				check = val_current + additive;
				if (check > 255)
				{
					check = 255;
				}
				else if (check < 0)
				{
					check = 0;
				}
				uchar gri = check;
				dest.at<uchar>(i, j) = gri;

			}
		}


		// Get the current time again and compute the time difference [s]
		time = ((double)getTickCount() - time) / getTickFrequency();
		// Print (in the console window) the processing time in [ms] 
		printf("Time = %.3f [ms]\n", time * 1000);

		imshow("input image", src);
		imshow("negative image", dest);
		waitKey();
	}

}


void mul_factor(int mul)
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		double time = (double)getTickCount();

		Mat src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		int linii = src.rows;
		int col = src.cols;
		Mat dest = Mat(linii, col, CV_8UC1);

		for (int i = 0; i < linii; i++)
		{
			for (int j = 0; j < col; j++)
			{
				uchar val_current = src.at<uchar>(i, j);
				int check = val_current * mul;
				if (check > 255)
				{
					check = 255;
				}
				else if (check < 0)
				{
					check = 0;
				}
				uchar gri = check;
				dest.at<uchar>(i, j) = gri;

			}
		}


		// Get the current time again and compute the time difference [s]
		time = ((double)getTickCount() - time) / getTickFrequency();
		// Print (in the console window) the processing time in [ms] 
		printf("Time = %.3f [ms]\n", time * 1000);

		imshow("input image", src);
		imshow("negative image", dest);
		waitKey();
	}
}

void create_image()
{
	Mat image(256, 256, CV_8UC3, Scalar(255, 255, 255));


	int W = image.cols / 2;
	int H = image.rows / 2;

	
	image(Rect(0, 0, W, H)) = Scalar(255, 255, 255); 

	image(Rect(W, 0, W, H)) = Scalar(0, 0, 255); 

	image(Rect(W, H, W, H)) = Scalar(0, 255, 255);

	image(Rect(0, H, W, H)) = Scalar(0, 2552, 0); 

	

	imshow("Colored", image);
	waitKey(0);
}



void print_mat(const Mat& matrix)
{
	for (int i = 0; i < 3; i++)
	{
		for (int j = 0; j < 3; j++)
		{
			printf("%f ", matrix.at<float>(i, j));
		}
		printf("\n");
	}
	printf("\n");
}



Mat matrix_inverse(Mat init) {
	Mat transpuse = Mat(init.cols, init.rows, CV_32F); 
	for (int i = 0; i < init.rows; ++i) {
		for (int j = 0; j < init.cols; ++j) {
			transpuse.at<float>(j, i) = init.at<float>(i, j);
		}
	}

	double determinant = cv::determinant(init);
	if (determinant == 0)
	{
		printf("Nu se poate calcula => inf");
		exit(-1);
	}
	//printf("det: %f\n", determinant);
	transpuse /= determinant;
	return transpuse;
	

}


void demo_inverse()
{
	float vals[9] = { 1, 2, 3, 4, 5, 6, 7, 8, 8 };
	Mat matrix(3, 3, CV_32FC1, vals);


	Mat inversa = matrix_inverse(matrix);

	print_mat(inversa);
	

}



void afisare_RGB()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		double time = (double)getTickCount();

		Mat src = imread(fname);


		int linii = src.rows;
		int col = src.cols;

		Mat red(linii, col, CV_8UC3);
		Mat green(linii, col, CV_8UC3);
		Mat blue(linii, col, CV_8UC3);

		for (int i = 0; i < linii; i++)
		{
			for (int j = 0; j < col; j++)
			{
				Vec3b pixel = src.at<Vec3b>(i, j);
				blue.at<Vec3b>(i, j) = Vec3b(pixel[0], 0, 0);
				green.at<Vec3b>(i, j) = Vec3b(0, pixel[1], 0);
				red.at<Vec3b>(i, j) = Vec3b(0, 0, pixel[2]);
			}
		}

		imshow("Red", red);
		imshow("Green", green);
		imshow("Blue", blue);

		time = ((double)getTickCount() - time) / getTickFrequency();
		printf("Time = %.3f [ms]\n", time * 1000);

		imshow("Original Image", src);
		waitKey();
	}
}


void convert_to_grayscale()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		double time = (double)getTickCount();

		Mat src = imread(fname);

		if (src.empty())
		{
			printf("Could not open or read the image.\n");
			return;
		}

		int rows = src.rows;
		int cols = src.cols;

		Mat gray_scale(rows, cols, CV_8UC1);

		for (int i = 0; i < rows; i++)
		{
			for (int j = 0; j < cols; j++)
			{
				Vec3b pixel = src.at<Vec3b>(i, j);

				uchar intensity = (pixel[2] + pixel[1] + pixel[0]) / 3;

				gray_scale.at<uchar>(i, j) = intensity;
			}
		}

		imshow("Gray", gray_scale);

		time = ((double)getTickCount() - time) / getTickFrequency();
		printf("Time = %.3f [ms]\n", time * 1000);

		imshow("Original Image", src);
		waitKey();
	}
}



void show_binary(int threshold)
{
	char fname[MAX_PATH];
	while (openFileDlg(fname)) {
		Mat src = imread(fname, CV_8UC1);
		int height = src.rows;
		int width = src.cols;
		Mat gray = Mat(height, width, CV_8UC1);
		for (int i = 0; i < height; i++)
			for (int j = 0; j < width; j++) {
				uchar v = src.at<uchar>(i, j);
				int val = v;
				if (val < threshold)
					gray.at<uchar>(i, j) = 0;
				else
					gray.at<uchar>(i, j) = 255;

			}
		imshow("imagine", src);
		imshow("gray", gray);
		waitKey();
	}
}


void convert_to_HSV()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		double time = (double)getTickCount();

		Mat src = imread(fname);

		Mat hsv(src.rows, src.cols, CV_8UC3);

		for (int y = 0; y < src.rows; y++)
		{
			for (int x = 0; x < src.cols; x++)
			{
				float r = src.at<Vec3b>(y, x)[2] / 255.0;
				float g = src.at<Vec3b>(y, x)[1] / 255.0;
				float b = src.at<Vec3b>(y, x)[0] / 255.0;

				float M = max( max(r, g), b);
				float m = min(min(r, g), b);
				float C = M - m;

				float V = M;

				float S = (V != 0) ? C / V : 0;
				float H = 0;
				if (C != 0)
				{
					if (M == r)
						H = 60 * (g - b) / C;
					else if (M == g)
						H = 120 + 60 * (b - r) / C;
					else if (M == b)
						H = 240 + 60 * (r - g) / C;
				}

				H = (H < 0) ? H + 360 : H;
				uchar H_norm = static_cast<uchar>(H * 255 / 360);
				uchar S_norm = static_cast<uchar>(S * 255);
				uchar V_norm = static_cast<uchar>(V * 255);

				hsv.at<Vec3b>(y, x) = Vec3b(H_norm, S_norm, V_norm);
			}
		}

		std::vector<cv::Mat> channels;
		split(hsv, channels);

		imshow("Hue", channels[0]);
		imshow("Saturation", channels[1]);
		imshow("Value", channels[2]);

		time = ((double)getTickCount() - time) / getTickFrequency();
		printf("Time = %.3f [ms]\n", time * 1000);

		cv::imshow("Original Image", src);
		cv::waitKey();
	}
}


bool isInside(int P, int Q, Mat src)
{
	if (P < 0 && P > src.rows && Q < 0 && Q > src.cols)
	{
		return false;
	}

	float value = src.at<float>(P, Q);
	for (int i = 0; i < src.rows; i++)
	{
		for (int j = 0; j < src.cols; j++)
		{
			if (src.at<float>(i, j) != value)
				return false;
		}
	}

	return true;
	
}

int main()
{

	int op;
	/*
	do
	{
		system("cls");
		destroyAllWindows();
		printf("Menu:\n");*/


		/*printf(" 1 - Open image\n");
		printf(" 2 - Open BMP images from folder\n");
		printf(" 3 - Image negative - diblook style\n");
		printf(" 4 - BGR->HSV\n");
		printf(" 5 - Resize image\n");
		printf(" 6 - Canny edge detection\n");
		printf(" 7 - Edges in a video sequence\n");
		printf(" 8 - Snap frame from live video\n");
		printf(" 9 - Mouse callback demo\n");
		printf(" 10 - Add\n");
		printf(" 11 - Mul\n");
		printf(" 12 - Create image\n");
		printf(" 13 - Matrix Inverse\n");*/ //lab 1

		/**

		printf(" 14 - Impartire in 3\n");
		printf(" 15- Gray Scal \n");
		printf(" 16- Show Binary \n");
		printf(" 0 - Exit\n\n");
		printf("Option: ");
		scanf("%d",&op);
		switch (op)
		{
			case 1:
				testOpenImage();
				break;
			case 2:
				testOpenImagesFld();
				break;
			case 3:
				testParcurgereSimplaDiblookStyle(); //diblook style
				break;
			case 4:
				//testColor2Gray();
				testBGR2HSV();
				break;
			case 5:
				testResize();
				break;
			case 6:
				testCanny();
				break;
			case 7:
				testVideoSequence();
				break;
			case 8:
				testSnap();
				break;
			case 9:
				testMouseClick();
				break;
			case 10:
				add_factor(30);
				break;
			case 11:
				mul_factor(2);
				break;
			case 12:
				create_image();
				break;
			case 13:
				demo_inverse();
				waitKey();
				break;
			case 14:
				afisare_RGB();
				break;
			case 15:
				convert_to_grayscale();
				break;
			case 16:
				show_binary(200);
				break;
			case 17:
				convert_to_HSV();
				break;
			case 18:
				isInside(20, 11);
				break;

				
		}
	}
	
	while (op!=0);

	
	//demo_inverse();**/

	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		
		if (isInside(20, 800, src))
			printf("Da\n");
		else
			printf("Nu\n");
	}

	return 0;

}
