#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <math.h>
#include "C:\Users\cedri\OneDrive\Bureau\Formation\INF574\libigl\external\eigen\Eigen\Dense"		   //Just put Eigen\Dense and Eigen\Sparse 
#include "C:\Users\cedri\OneDrive\Bureau\Formation\INF574\libigl\external\eigen\Eigen\Sparse"
#include <iostream>
#include <fstream>
#include <vector>
#include <stdio.h>

#define WIDTH 1

typedef Eigen::Triplet<double> T;

struct weights {
	int row;
	int col;
	double w;
};
typedef struct weights weights;

using namespace cv;
using namespace std;
using namespace Eigen;

int main(int ac, char** av) {
	// For the image only
	Mat Example = imread("../example.bmp");
	Mat Example_marked = imread("../example_marked.bmp");

	Mat Example_YUV, Example_marked_YUV;

	cvtColor(Example, Example_YUV, COLOR_RGB2YUV);
	cvtColor(Example_marked, Example_marked_YUV, COLOR_RGB2YUV);
	if (Example_marked.size().empty() || Example.size().empty()) {cout << "Cannot read image" << endl;return 1;}

	int n = Example_marked.cols;	// Width
	int m = Example_marked.rows;	// Height
	int pic_size = n * m;

	MatrixXd Y(n, m);	// To store Y channel
	MatrixXd U(n, m);	// To store U channel
	MatrixXd V(n, m);	// To store V channel
	MatrixXi Marked(n, m); // Will be used as a bool to store 255=Not marked and 0 = Marked
	Mat Result(m, n, CV_8UC3);
	Mat Final(m, n, CV_8UC3);

	Vec3b Point, Point_marked,Pointfinal;

	////////////////////////////////////////////////////////////
	//
	// Store the images in B&W and the labelled image in Y, U, V and in Marked
	//
	///////////////////////////////////////////////////////////
	// Conversion to Column / Row in the matrix format !
	for (int y = 0; y < n; y++)
		for (int x = 0; x < m; x++) {

			Y(y, x) = Example_YUV.at<Vec3b>(x, y)[0]/1.0; // For an image
			Point_marked = Example_marked_YUV.at<Vec3b>(x, y);
			U(y, x) = Point_marked[1];
			V(y, x) = Point_marked[2];
			if (Point_marked[1] != 128 || Point_marked[2] != 128) Marked(y, x) = 0; //For an image
			else Marked(y, x) = 255;
		}

	////////////////////////////////////////////////////////////
	//
	// Visualisation of the given parameters just to check
	//
	///////////////////////////////////////////////////////////
	Mat G2(m, n, CV_8U);
	Mat I2(m, n, CV_8U);

	for (int y = 0; y < n; y++)
		for (int x = 0; x < m; x++) {
			G2.at < uchar >(x, y) = Y(y, x);
			I2.at < uchar >(x, y) = Marked(y, x);
		}
	imshow("Gray image", G2);
	imshow("White=Not marked", I2);
	imshow("Labelled image", Example_marked);

	////////////////////////////////////////////////////////////
	//
	// Build sparse Matrix A together with b_u and b_v
	//
	///////////////////////////////////////////////////////////
	VectorXd b_u(pic_size), b_v(pic_size);
	SparseMatrix<double> A(pic_size, pic_size);
	SparseLU<SparseMatrix<double>, COLAMDOrdering<int> >   solver;
	std::vector<T> coefficients;            // list of non-zeros coefficients to fill the sparse Matrix
	T newcoef;
	   	  
	for (int x = 0; x < n; x++)
		for (int y = 0; y < m; y++) {
			// Initialisation fo b_u and b_v

			if (Marked(x, y) == 0) {
				b_u(y + m * x) = U(x, y);
				b_v(y + m * x) = V(x, y);
			}
			else {
				b_u(y + m * x) = 0;
				b_v(y + m * x) = 0;
			}

			// Look for the neighbors of point (y,x)
			int xmin = x - WIDTH; if (xmin < 0)	xmin = 0;
			int ymin = y - WIDTH; if (ymin < 0)	ymin = 0;
			int xmax = x + WIDTH; if (xmax >= n)	xmax = n - 1;
			int ymax = y + WIDTH; if (ymax >= m)	ymax = m - 1;

			if (Marked(x, y) == 255) {
				// We need to compute the variance of Y around the point
				double mean = 0;
				double square = 0;
				for (int row = xmin;row <= xmax; row++)
					for (int col = ymin; col <= ymax; col++) {
						mean += Y(row, col);
						square += (Y(row, col) * Y(row, col));
					}
				mean = mean / ((ymax - ymin + 1) * (xmax - xmin + 1));
				double variance = square / ((ymax - ymin + 1) * (xmax - xmin + 1)) - mean * mean;

				if (variance < 1e-6)
					variance = 1e-6;

				std::list<weights> liste;
				liste.empty();
				weights W;
				double totalw = 0;

				for (int row = xmin;row <= xmax; row++)
					for (int col = ymin; col <= ymax; col++) 
						if ((col != y) || (row != x)) {
							double value = (Y(row, col) - Y(x, y)) * (Y(row, col) - Y(x, y));		// First weight function
							value = exp(-value / (2 * variance));
							//double value = abs((Y(row, col) - mean) * (Y(x, y) - mean));		//Second weight function
							//value = 1 + (variance/ (value+1e-6));
							W.row = row;
							W.col = col;
							totalw += value;
							W.w = value;
							liste.push_back(W);
						}
				for (weights W : liste) 
					coefficients.push_back(T(y + m * x, W.col + m * W.row, W.w / -totalw));
				
			}
			coefficients.push_back(T(y + m * x, y + m * x,1.0));
		}


	////////////////////////////////////////////////////////////
	//
	//  Solve x=Ab
	//
	///////////////////////////////////////////////////////////
	A.setFromTriplets(coefficients.begin(), coefficients.end());
	solver.analyzePattern(A);
	solver.factorize(A);
	VectorXd solution_u = solver.solve(b_u);
	VectorXd solution_v = solver.solve(b_v);

	///////////////////////////////////
	//
	// Give the result
	//
	///////////////////////////////////
	for (int x = 0; x < n; x++)
		for (int y = 0; y < m; y++) 	 {
			Result.at<Vec3b>(y, x)[0] = Example_YUV.at<Vec3b>(y, x)[0];
			Result.at<Vec3b>( y ,  x )[1] = solution_u(y + x * m);
			Result.at<Vec3b>(y,x)[2] = solution_v(y + x * m);
			}

	cvtColor(Result, Final, COLOR_YUV2RGB);
	imshow("Result in YUV", Result);
	imshow("Final result", Final);
	waitKey();
}