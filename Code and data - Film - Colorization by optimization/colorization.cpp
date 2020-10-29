#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <math.h>
#include "mg.h"
#include "fmg.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <stdio.h>

using namespace cv;
using namespace std;
#define MAXIMAGE 100
#define MAXIMAGELAB 10


int loadfilm(VideoCapture& capture, Mat *frame) {
	int n = 0;
	Mat load;
	for (;;) {
		capture >> load;
		if (load.empty()) break;
		cvtColor(load, frame[n], COLOR_RGB2YUV);
		n++;
	}
	return n;
}

void savefilm(VideoCapture& capture, Mat* frame) {



}


int displayfilm(VideoCapture& capture) {
	int n = 0;
	Mat frame;
	for (;;) {
		capture >> frame;
		n++;
		if (frame.empty()) break;
		imshow("Robot",frame);
		char key = (char)waitKey(5); 
	}
	return n;
}

// Aim : for each labelled image, find the image in the film that corresponds best
// (due to compression, it is possible that we does not have exact mathcing)
void compute_proximity(Mat* Images_Film, Mat* Images_Film_marked, int number_images_film, int number_images_film_marked,int *best_correspondance) {

	int n = Images_Film[0].cols;									// Width
	int m = Images_Film[0].rows;									// Height
	for (int i = 0;i< number_images_film_marked;i++) {
		int best_distance = 999999999;
		int best_corresp = -1;
		for (int j = 0;j < number_images_film;j++) {
			// Compute distance between Image j in Film and Image i and Film_marked
			int distance = 0;
			for (int y = 0; y < n; y++)
				for (int x = 0; x < m; x++)
					distance += abs(Images_Film[j].at<Vec3b>(x,y)[0] - Images_Film_marked[i].at<Vec3b>(x,y)[0]);

			if (distance < best_distance) {
				best_distance = distance;
				best_corresp = j;
			}
		}
		best_correspondance[i] = best_corresp;
		cout << "Image labelled #" << i << " corresponds best with image #" << best_corresp << " with distance " << best_distance << endl;
	}
}

int main(int ac, char** av) {
	// For the image only
	//Mat Example = imread("../example.bmp");
	//Mat Example_marked = imread("../example_marked.bmp");
	//Mat Result;
	//Mat Example_YUV, Example_marked_YUV;
	//cvtColor(Example, Example_YUV, COLOR_RGB2YUV);
	//cvtColor(Example_marked, Example_marked_YUV, COLOR_RGB2YUV);
	//if (Example_marked.size().empty() || Example.size().empty()) {cout << "Cannot read image" << endl;return 1;}
	//Example_YUV.copyTo(Result); // Just to have same size and color type
	//int n = Example_marked.cols;									// Width
	//int m = Example_marked.rows;									// Height
	//m = 256;
	//int k = 1;
	//int number_images_film_marked=1;


	// For the film only
	Mat Result[MAXIMAGE];
	VideoCapture Film("../lake-gray.gif");
	if (!Film.isOpened()){printf("Ouverture du flux video impossible pour le film N&B!\n");return 1; }
	VideoCapture Film_marked("../lake-markings.gif");
	if (!Film_marked.isOpened()) {printf("Ouverture du flux video impossible pour le film labelled!\n"); return 1;}
	Mat Images_Film[MAXIMAGE];
	Mat Images_Film_marked[MAXIMAGELAB];
	int number_images_film = loadfilm(Film, Images_Film);
	int number_images_film_marked = loadfilm(Film_marked, Images_Film_marked);
	cout << "Il y a " << number_images_film << " images dans le film N&B et " << number_images_film_marked << " dans le film labellise." << endl;
	int best_matched[MAXIMAGELAB];
	compute_proximity(Images_Film, Images_Film_marked, number_images_film, number_images_film_marked, best_matched);
	/* Check we loaded correctly the image
	Mat Show;
	for (int i = 0;i < number_images_film_marked;i++) {
		cvtColor(Images_Film[best_matched[i]], Show, COLOR_YUV2RGB);
		imshow("Film image", Show);
		cvtColor(Images_Film_marked[i], Show, COLOR_YUV2RGB);
		imshow("Example_marked", Show);
	}*/
	
	for (int z=0;z< number_images_film;z++)
		Images_Film[z].copyTo(Result[z]); // Just to have same size and color type

	/*
	VideoWriter video1;
	int codec1 = VideoWriter::fourcc('M', 'J', 'P', 'G');  // select desired codec (must be available at runtime)
	double fps1 = 25.0;                          // framerate of the created video stream

	video1.open("../outcpp6.avi", codec1, fps1, Size(320, 240));
	// check if we succeeded
	if (!video1.isOpened()) {
		cerr << "Could not open the output video file for write\n";
		return -1;
	}
	for (int z = 0;z < number_images_film;z++) {
		//video << Result[z];
		video1 << Images_Film[z];
		char key = (char)waitKey(50);
		cout << "Image #" << z << endl;
	}
	*/

	//void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) {
	//int winSize = int(mxGetScalar(prhs[2]) + 0.5);		// Not used later !
	//int deg = int(mxGetScalar(prhs[3]) + 0.5);			// Not used later !

	const int* sizes;
	int n = Images_Film[0].cols;									// Width
	int m = Images_Film[0].rows;									// Height
	int k = number_images_film;		// Number of images (1 if just an image ; more if it is a film)
	int max_d, max_d1, max_d2, in_itr_num, out_itr_num, itr;
	int x, y, z;
	//n = 128;
	//m = 128;

	// To save time, we propagate on a 2**max_d smaller image (+iteration on bigger image up to normal size). Here is the computation of d:
	max_d1 = int(floor(log(n) / log(2) - 2) + 0.1);
	max_d2 = int(floor(log(m) / log(2) - 2) + 0.1);
	if (max_d1 > max_d2)
		max_d = max_d2;
	else
		max_d = max_d1;			

	Tensor3d D, G, I;				// Size (n,m,k)
	Tensor3d Dx, Dy, iDx, iDy;		// Size (n,m,k-1)
	MG smk;
	G.set(m, n, k);
	D.set(m, n, k);
	I.set(m, n, k);

	in_itr_num = 5; // It was 5
	out_itr_num = 3; // It was 2

	Vec3b Point, Point_marked,Pointfinal;

		
	////////////////////////////////////////////////////////////
	//
	// Store the images in B&W and the labelled image in I and G
	//
	///////////////////////////////////////////////////////////
	for (y = 0; y < n; y++)
		for (x = 0; x < m; x++) {

			for (z = 0; z < k; z++) {
				//G(x, y, z) = Example_YUV.at<Vec3b>(x, y)[0]; // For an image
				G(x, y, z) = Images_Film[z].at<Vec3b>(x, y)[0]; // For a film
				I(x, y, z) = 255; //Default value
			}

			for (z = 0; z < number_images_film_marked; z++) {
				//Point_marked = Example_marked_YUV.at<Vec3b>(x, y); // For an image
				//if (Point_marked[1] == 128 && Point_marked[2] == 128) I(x, y, z) = 255; //For an image

				// For a film
				Point_marked = Images_Film_marked[z].at<Vec3b>(x, y);
				if (abs(Point_marked[1]-128)+abs(Point_marked[2]-128)>=5) //Allow a tolerance to deal with compression image problems 
					I(x, y, best_matched[z]) = 0;
			}
		}
	//Problem : due to compression, we might have some black points that should be white : we have to smooth it
	Mat copy(m,n, CV_8U);
	for (z = 0; z < number_images_film_marked; z++) {
		for (y = 0; y < n; y++)
			for (x = 0; x < m; x++)
				copy.at<uchar>(x, y) = I(x, y, best_matched[z]); // Store in copy to work on it

		for (y = 1; y < n - 1; y++)
			for (x = 1; x < m - 1; x++) {
				// Hear of I
				int moy = copy.at<uchar>(x + 0, y - 1) + copy.at<uchar>(x + 0, y + 0) + copy.at<uchar>(x + 0, y + 1);
				moy = moy + copy.at<uchar>(x - 1, y - 1) + copy.at<uchar>(x - 1, y + 0) + copy.at<uchar>(x - 1, y + 1);
				moy = moy + copy.at<uchar>(x + 1, y - 1) + copy.at<uchar>(x + 1, y + 0) + copy.at<uchar>(x + 1, y + 1); //Average on 9 points
				if ((moy / 9) > 128) I(x, y, best_matched[z]) = 255; // Restore in I
				else I(x, y, best_matched[z]) = 0;
			}
		// Nothing on the contour of the image
		for (y = 0; y < n; y++) {
			I(0, y, best_matched[z]) = 255;
			I(m-1, y, best_matched[z]) = 255;
		}
		for (x = 0; x < m; x++) {
			I(x, 0, best_matched[z]) = 255;
			I(x, n - 1, best_matched[z]) = 255;
		}
	}


	////////////////////////////////////////////////////////////
	//
	// Compute optical flow and feed SetFlow
	//
	///////////////////////////////////////////////////////////
	Dx.set(m, n, k - 1);
	Dy.set(m, n, k - 1);
	iDx.set(m, n, k - 1);
	iDy.set(m, n, k - 1);

	// If k>1, have to be changed ; for k=1 this is not run at all
	// Still need to be understood for films
	for (z = 0; z < (k - 1); z++) {
		Mat flow(Images_Film[0].size(), CV_32FC2);
		Mat Im1, Im2,Im3;
		cvtColor(Images_Film[z], Im1, COLOR_YUV2RGB);
		cvtColor(Im1, Im2, COLOR_BGR2GRAY);
		cvtColor(Images_Film[z+1], Im1, COLOR_YUV2RGB);
		cvtColor(Im1, Im3, COLOR_BGR2GRAY);

		calcOpticalFlowFarneback(Im2,Im3, flow, 0.5, 3, 15, 3, 5, 1.2, 0);
		Mat flow_parts[2];
		split(flow, flow_parts);	//calcOpticalFlowFarneback returns Complex value for each point, this is a way to have x and y

		for (y = 0; y < n; y++)
			for (x = 0; x < m; x++) {
				Dy(x, y, z) = flow_parts[0].at<float>(x, y);
				Dx(x, y, z) = flow_parts[1].at<float>(x, y);
				iDy(x, y, z) = -Dy(x, y, z);	//Strange implentation but I prefer to stick to the autor's implementation
				iDx(x, y, z) = -Dx(x, y, z);
			}
		// Just to check (debugging purpose)
		cout << "Image #" << z << endl;
		cout << "Axe x - ligne 95 : ";
		int i = 95;
		for (int j = 90;j < 100;j++) cout << flow_parts[0].at<float>(i, j) << "  ";
		cout << endl;
		cout << "Axe y - ligne 95 : ";
		for (int j = 90;j < 100;j++) cout << flow_parts[1].at<float>(i, j) << "  ";
		cout << endl;
	}


	////////////////////////////////////////////////////////////
	//
	// Visualisation of the given parameters
	//
	///////////////////////////////////////////////////////////
	Mat G2(m, n, CV_8U);
	Mat I2(m, n, CV_8U);
	Mat Show;

	Mat DX2(m, n, CV_8U);
	Mat DY2(m, n, CV_8U);
	Mat IDX2(m, n, CV_8U);
	Mat IDY2(m, n, CV_8U);

	//for (z = 0;z < number_images_film_marked;z++) {
	for(z=0;z<k-1;z++) {
		for (y = 0; y < n; y++)
			for (x = 0; x < m; x++) {
				G2.at < uchar >(x, y) = G(x, y, z);
				I2.at < uchar >(x, y) = I(x, y, z);
				DX2.at < uchar >(x, y) = Dx(x, y, z);
				DY2.at < uchar >(x, y) = Dy(x, y, z);
				IDX2.at < uchar >(x, y) = iDx(x, y, z);
				IDY2.at < uchar >(x, y) = iDy(x, y, z);
			}
		cout << "Images #" << z << endl;
		//cvtColor(Images_Film_marked[z], Show, COLOR_YUV2RGB);
		//imshow("Original_marked", Show);
		imshow("Gray image", G2);
		imshow("I", I2);
		imshow("dx", DX2);
		imshow("dy", DY2);
		imshow("idx", IDX2);
		imshow("idy", IDY2);
		char key = (char)waitKey(50);
	}





	// Prepare
	smk.set(m, n, k, max_d);
	smk.setI(I);
	smk.setG(G);
	cout << "Before SetFlow" << endl;
	smk.setFlow(Dx, Dy, iDx, iDy);
	cout << "After SetFlow" << endl;

	for (int t = 1; t < 3; t++) {		// Run for t=1 (channel U) and t=2 (channel V)

		for (y = 0; y < n; y++)
			for (x = 0; x < m; x++) 
				for (z = 0; z < number_images_film_marked; z++)	{
					// For an image
					//D(x, y, z) = Example_marked_YUV.at<Vec3b>(x, y)[t];		// Load Channel U or V in D
					//smk.P()(x, y, z) = Example_marked_YUV.at<Vec3b>(x, y)[t];
					//D(x, y, z) *= (!I(x, y, z));

					// For a film
					D(x, y, best_matched[z]) = Images_Film_marked[z].at<Vec3b>(x, y)[t];		// Load Channel U or V in D
					smk.P()(x, y, best_matched[z]) = D(x, y, best_matched[z]);
					D(x, y, best_matched[z]) *= (!I(x, y, best_matched[z])); // Only the black (0) points are left -> everything not labelled at 0!
				}

		smk.Div() = D;
		Tensor3d tP2;

		if (k == 1) { // For images
			for (itr = 0; itr < out_itr_num; itr++) {
				// On propage en basse résolution puis sur des résolutions de plus en plus fine pour gagner du temps
				cout << "We are at t=" << t << " out of tmax=2 - Iteration " << itr + 1 << " out of a maxmimum of " << out_itr_num << endl;
				smk.setDepth(max_d);
				Field_MGN(&smk, in_itr_num, 2);		// 2 in the original code
				smk.setDepth(ceil(max_d / 2));		//ceil = round up
				Field_MGN(&smk, in_itr_num, 2);
				smk.setDepth(2);
				Field_MGN(&smk, in_itr_num, 2);
				smk.setDepth(1);
				Field_MGN(&smk, in_itr_num, 4); // 4 in the original code
			}
		}

		else { // Less operations for films (to slow if else
			for (itr = 0; itr < out_itr_num; itr++) {
				cout << "We are at t=" << t << " out of tmax=2 - Iteration " << itr + 1 << " out of a maxmimum of " << out_itr_num << endl;
				smk.setDepth(max_d);
				Field_MGN(&smk, in_itr_num, 20);				
				smk.setDepth(3);
				Field_MGN(&smk, in_itr_num, 30);
				smk.setDepth(2);
				Field_MGN(&smk, in_itr_num, 60);
				smk.setDepth(1);
				Field_MGN(&smk, in_itr_num, 200);
			}
		}
		

		// Call the solver
		tP2 = smk.P(); 


		///////////////////////////////////
		//
		// Give the result
		//
		///////////////////////////////////
		for (z = 0; z < k; z++)
			for (y = 0; y < n; y++)
				for (x = 0; x < m; x++) 
					Result[z].at<Vec3b>(x, y)[t]= tP2(x, y, z);
				

	}
	
	Mat Final[MAXIMAGE];
	for (z = 0; z < k; z++) {
		cvtColor(Result[z], Final[z], COLOR_YUV2RGB);
		imshow("Result in YUV", Result[z]);
		imshow("Final result", Final[z]);
		char key = (char)waitKey(50);
	}

	VideoWriter video;
	int codec = VideoWriter::fourcc('M', 'J', 'P', 'G');  // select desired codec (must be available at runtime)
	double fps = 25.0;                          // framerate of the created video stream
	string filename = "../outcppv10h41.avi";             // name of the output video file
	video.open(filename, codec, fps, Size(320, 240));

	for (z = 0;z < number_images_film;z++) {
		video << Final[z];
		char key = (char)waitKey(50);
		cout << "Image #" << z << endl;
	}
	cout << "Sortie sans erreur!!" << endl;
	return 150;
}


