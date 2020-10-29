// FirstPhase.cpp : Defines the entry point for the console application.
//
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/ml/ml.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include "opencv2/imgcodecs.hpp"

#include <iostream>

using namespace cv;
using namespace std;

string type2str(int type) {
	string r;

	uchar depth = type & CV_MAT_DEPTH_MASK;
	uchar chans = 1 + (type >> CV_CN_SHIFT);

	switch (depth) {
	case CV_8U:  r = "8U"; break;
	case CV_8S:  r = "8S"; break;
	case CV_16U: r = "16U"; break;
	case CV_16S: r = "16S"; break;
	case CV_32S: r = "32S"; break;
	case CV_32F: r = "32F"; break;
	case CV_64F: r = "64F"; break;
	default:     r = "User"; break;
	}

	r += "C";
	r += (chans + '0');

	return r;
}


int main(int argc, char** argv)
{
	//Step-1: Reading the images files
	Mat rinput_image = imread("../Input_BW.jpg");
	Mat temp_input[3];
	split(rinput_image, temp_input);
	Mat temp = temp_input[0];
	Mat input_image;
	Mat rref_image = imread("../Reference.jpg");
	Mat ref_image;
	cv::resize(rref_image, ref_image, cv::Size(400, 300));
	cv::resize(temp, input_image, cv::Size(400, 300));

	if (ref_image.empty()) // Check for failure
	{
		cout << "Could not open or find the image" << endl;
		system("pause"); //wait for any key press
		return -1;
	}

	//Projecting Reference Image Matrix(row,cols,3) onto Matrix(rows*cols,3), 1 row per pixel
	Mat samples(ref_image.cols * ref_image.rows, 3, CV_32F);
	for (int i = 0; i < ref_image.rows; ++i) {
		for (int j = 0; j < ref_image.cols; ++j) {
			for (int k = 0; k < 3; ++k) {
				samples.at<float>(i + j * ref_image.rows, k) = ref_image.at<Vec3b>(i, j)[k];
			}
		}
	}
	//std::cout << samples.channels() << type2str(samples.type()) << std::endl;

	//Step-2: Setting up parameters and applying K-means image segmentation to the reference image
	std::cout << "Performing the k-means clustering for generating the reference image segmentation" << std::endl;
	int clusterCount = 3;
	Mat labels;
	int attempts = 5;
	Mat centers;
	kmeans(samples, clusterCount, labels, TermCriteria(TermCriteria::MAX_ITER | TermCriteria::EPS, 10000, 0.0001), attempts, KMEANS_PP_CENTERS, centers);
	
	//Testing purposes only
	//cv::FileStorage file199("colors.ext", cv::FileStorage::WRITE);
	//file199 << "colors" << centers;
	//file199.release();

	//Displaying Image segmentation
	int r = ref_image.rows;
	int c = ref_image.cols;
	Mat rsegmented_Image(r, c, ref_image.type());
	Mat labels_train(r * c, 1, CV_32F);
	Mat checking(r, c, CV_32F);
	int label1 = 0;
	int label2 = 0;
	int label3 = 0;
	for (int i = 0; i < r; ++i) {
		for (int j = 0; j < c; ++j) {
			int cluster_indx = labels.at<int>(i + j * ref_image.rows, 0);
			labels_train.at<float>(i, 0) = cluster_indx;
			checking.at<float>(i, j) = cluster_indx;
			if (cluster_indx == 0) {
				label1++;
				rsegmented_Image.at<Vec3b>(i, j)[0] = centers.at<float>(0, 0);
				rsegmented_Image.at<Vec3b>(i, j)[1] = centers.at<float>(0, 1);
				rsegmented_Image.at<Vec3b>(i, j)[2] = centers.at<float>(0, 2);
			}
			else if (cluster_indx == 1) {
				label2++;
				rsegmented_Image.at<Vec3b>(i, j)[0] = centers.at<float>(1, 0);
				rsegmented_Image.at<Vec3b>(i, j)[1] = centers.at<float>(1, 1);
				rsegmented_Image.at<Vec3b>(i, j)[2] = centers.at<float>(1, 2);
			}
			else {
				label3++;
				rsegmented_Image.at<Vec3b>(i, j)[0] = centers.at<float>(2, 0);
				rsegmented_Image.at<Vec3b>(i, j)[1] = centers.at<float>(2, 1);
				rsegmented_Image.at<Vec3b>(i, j)[2] = centers.at<float>(2, 2);
			}
		}
	}

	//Testing purposes only
	/*cv::FileStorage file10("labels.ext", cv::FileStorage::WRITE);
	file10 << "labels" << labels;
	file10.release();*/

	//Step-3: Applying PCA and LDA on the features matrix of reference image 
	//Extracting the luminance channel of the reference image
	std::cout << "Performing the Discrete Cosine Transform of the luminance channel of reference image" << std::endl;
	Mat interm_ref;
	Mat ref_imageF;
	Mat luminace[3];
	Mat lum_f;
	cvtColor(ref_image, interm_ref, COLOR_BGR2YUV);
	std::cout << interm_ref.channels() << std::endl;
	split(interm_ref, luminace);
	luminace[0].convertTo(lum_f, CV_32F);
	imshow("Luminance Channel", lum_f);

	//Performing DCT - Discrete Cosine Transform on the reference image
	//As DCT current implmentation accepts matrixes with even number of rows & columns so the below code is just a sanity check
	int m = lum_f.rows;
	int n = lum_f.cols;
	int m2, n2 = 0;
	if (m % 2 == 0) {
		m2 = m;
	}
	else {
		m2 = m + 1;
	}
	if (n % 2 == 0) {
		n2 = n;
	}
	else {
		n2 = n + 1;
	}
	Mat padded = Mat::zeros(n, m, CV_32F);
	copyMakeBorder(lum_f, padded, 0, m2 - m, 0, n2 - n, BORDER_CONSTANT);
	Mat dct_ref = Mat::zeros(padded.rows, padded.cols, CV_32F);
	dct(padded, dct_ref);

	//Sanity check for checking dct is calculated correctly
	/*Mat dst;
	idct(dct_ref, dst);
	imshow("Reference Image(IDCT)", dst);
	Mat orig;
	dst.convertTo(orig, CV_8U);
	imshow("Reference Image(IDCT)", orig);*/

#pragma region Testing KNN-Classification with reference image provided as input image
	/*Without Dct - Reference Image used for both training and testing
	Mat woDCTLum(lum_f.rows*lum_f.cols, 1, CV_32F);
	for (int i = 0; i < lum_f.rows; i++) {
		for (int j = 0; j < lum_f.cols; j++) {
			woDCTLum.at<float>(i + j*lum_f.rows, 0) = lum_f.at<float>(i, j);
		}
	}
	std::cout << "woDCTLum:" << woDCTLum.size() << ":" << woDCTLum.channels() << ":" << type2str(woDCTLum.type()) << std::endl;
	Ptr<ml::KNearest> kclassifierPre = ml::KNearest::create();
	kclassifierPre->setIsClassifier(true);
	kclassifierPre->setAlgorithmType(ml::KNearest::Types::BRUTE_FORCE);
	kclassifierPre->setDefaultK(1000);
	kclassifierPre->train(woDCTLum, ml::ROW_SAMPLE, labels);

	Mat PrevResults;
	kclassifierPre->findNearest(woDCTLum, 1000, PrevResults);
	cv::FileStorage file10("labelsPrevPred.ext", cv::FileStorage::WRITE);
	// Write to file!
	file10 << "Preds" << PrevResults;
	file10.release();

	Mat forcoloredInput;
	Mat temp2;
	lum_f.convertTo(temp2, CV_8U);
	cvtColor(temp2, forcoloredInput, COLOR_GRAY2BGR);
	std::cout << "forcoloredInput Conv:" << forcoloredInput.size() << ":" << forcoloredInput.channels() << ":" << type2str(forcoloredInput.type()) << std::endl;
	for (int i = 0; i < forcoloredInput.rows; ++i) {
		for (int j = 0; j < forcoloredInput.cols; ++j) {
			int cluster_indx = PrevResults.at<float>(i + j*ref_image.rows, 0);
			if (cluster_indx == 0) {
				forcoloredInput.at<Vec3b>(i, j)[0] = 255;
				forcoloredInput.at<Vec3b>(i, j)[1] = 255;
				forcoloredInput.at<Vec3b>(i, j)[2] = 255;
			}
			else if (cluster_indx == 1) {
				forcoloredInput.at<Vec3b>(i, j)[0] = 0;
				forcoloredInput.at<Vec3b>(i, j)[1] = 255;
				forcoloredInput.at<Vec3b>(i, j)[2] = 255;
			}
			else {
				forcoloredInput.at<Vec3b>(i, j)[0] = 0;
				forcoloredInput.at<Vec3b>(i, j)[1] = 0;
				forcoloredInput.at<Vec3b>(i, j)[2] = 255;
			}
		}
	}

	imshow("Reference Image Coloring(For testing KNN)", forcoloredInput);

	Without DCT - Input Image Used for testing
	Mat woDCTLum(lum_f.rows*lum_f.cols, 1, CV_32F);
	for (int i = 0; i < lum_f.rows; i++) {
		for (int j = 0; j < lum_f.cols; j++) {
			woDCTLum.at<float>(i + j*lum_f.rows, 0) = lum_f.at<float>(i,j);
		}
	}
	std::cout<<"woDCTLum:"<< woDCTLum.size() << ":" << woDCTLum.channels() << ":" << type2str(woDCTLum.type()) << std::endl;
	Ptr<ml::KNearest> kclassifierPre = ml::KNearest::create();
	kclassifierPre->setIsClassifier(true);
	kclassifierPre->setAlgorithmType(ml::KNearest::Types::BRUTE_FORCE);
	kclassifierPre->setDefaultK(100);
	kclassifierPre->train(woDCTLum, ml::ROW_SAMPLE, labels);
	std::cout << "Input Image:" << input_image.size() << ":" << input_image.channels() << ":" << type2str(input_image.type()) << std::endl;
	Mat iinput;
	input_image.convertTo(iinput, CV_32F);
	std::cout << "Input Image Conv:" << iinput.size() << ":" << iinput.channels() << ":" << type2str(iinput.type()) << std::endl;
	Mat woDCTinp(iinput.rows*iinput.cols, 1, CV_32F);
	for (int i = 0; i < iinput.rows; i++) {
		for (int j = 0; j < iinput.cols; j++) {
			woDCTinp.at<float>(i + j*iinput.rows, 0) = iinput.at<float>(i, j);
		}
	}
	Mat PrevResults;
	kclassifierPre->findNearest(woDCTinp, 100, PrevResults);
	cv::FileStorage file10("labelsPrevPred.ext", cv::FileStorage::WRITE);
	// Write to file!
	file10 << "Preds" << PrevResults;
	file10.release();

	Mat forcoloredInput;
	cvtColor(input_image, forcoloredInput, COLOR_GRAY2BGR);
	std::cout << "forcoloredInput Conv:" << forcoloredInput.size() << ":" << forcoloredInput.channels() << ":" << type2str(forcoloredInput.type()) << std::endl;
	for (int i = 0; i < forcoloredInput.rows; ++i) {
		for (int j = 0; j < forcoloredInput.cols; ++j) {
			int cluster_indx = PrevResults.at<float>(i + j*ref_image.rows, 0);
			if (cluster_indx == 0) {
				forcoloredInput.at<Vec3b>(i, j)[0] = 255;
				forcoloredInput.at<Vec3b>(i, j)[1] = 255;
				forcoloredInput.at<Vec3b>(i, j)[2] = 255;
			}
			else if (cluster_indx == 1) {
				forcoloredInput.at<Vec3b>(i, j)[0] = 0;
				forcoloredInput.at<Vec3b>(i, j)[1] = 255;
				forcoloredInput.at<Vec3b>(i, j)[2] = 255;
			}
			else {
				forcoloredInput.at<Vec3b>(i, j)[0] = 0;
				forcoloredInput.at<Vec3b>(i, j)[1] = 0;
				forcoloredInput.at<Vec3b>(i, j)[2] = 255;
			}
		}
	}

	imshow("forcoloredInput", forcoloredInput);

	With DCT - Reference Image used for testing
	std::cout << dct_ref.size() << ":" << dct_ref.channels() << ":" << type2str(dct_ref.type()) << std::endl;
	int col_updated = dct_ref.cols;
	int row_updated = dct_ref.rows;
	std::cout << "Reference Image Rows:" << ref_image.rows << std::endl;
	Mat samples_train(col_updated*row_updated, 1, CV_32F);
	for (int i = 0; i < row_updated; ++i) {
		for (int j = 0; j < col_updated; ++j) {
			samples_train.at<float>(i + j*ref_image.rows, 0) = dct_ref.at<float>(i, j);
		}
	}
	std::cout << samples_train.size() << ":" << samples_train.channels() << ":" << type2str(samples_train.type()) << std::endl;
	std::cout << "Labels:" << labels.size() << ":" << labels.channels() << ":" << type2str(labels.type()) << std::endl;
	std::cout << "samples Train:" << samples_train.size() << ":" << samples_train.channels() << ":" << type2str(samples_train.type()) << std::endl;
	Ptr<ml::TrainData> trainingData;
	Ptr<ml::KNearest> kclassifier = ml::KNearest::create();
	kclassifier->setIsClassifier(true);
	kclassifier->setAlgorithmType(ml::KNearest::Types::BRUTE_FORCE);
	kclassifier->setDefaultK(5);
	kclassifier->train(samples_train, ml::ROW_SAMPLE, labels);

	Mat Results(samples_train.size(), samples_train.type());
	kclassifier->findNearest(samples_train, 5, Results);//, neigh);
													   std::cout << Results << std::endl;

	cv::FileStorage file3("labelsPred.ext", cv::FileStorage::WRITE);
	 Write to file!
	file3 << "Preds" << Results;
	file3.release();

	Mat samples_w(dct_ref.size(), dct_ref.type());
	for (int i = 0; i < dct_ref.rows; ++i) {
		for (int j = 0; j < dct_ref.cols; ++j) {
			samples_w.at<float>(i,j) = samples_train.at<float>(i + j*ref_image.rows, 0);
		}
	}

	Mat res(samples_w.size(), samples_w.type());
	idct(samples_w, res);
	Mat show;
	res.convertTo(show, CV_8U);
	imshow("IDCT Input", show);
	Mat forcoloring;
	cvtColor(show, forcoloring, COLOR_GRAY2BGR);
	std::cout << forcoloring.size() << ":" << forcoloring.channels() << ":" << type2str(forcoloring.type()) << std::endl;

	for (int i = 0; i < forcoloring.rows; ++i) {
		for (int j = 0; j < forcoloring.cols; ++j) {
			float color = Results.at<float>(i + j*forcoloring.rows, 0);
			if (color == 0) {
				forcoloring.at<Vec3b>(i, j)[0] = 255;
				forcoloring.at<Vec3b>(i, j)[1] = 255;
				forcoloring.at<Vec3b>(i, j)[2] = 255;
			}
			else if (color == 1) {
				forcoloring.at<Vec3b>(i, j)[0] = 0;
				forcoloring.at<Vec3b>(i, j)[1] = 255;
				forcoloring.at<Vec3b>(i, j)[2] = 255;
			}
			else {
				forcoloring.at<Vec3b>(i, j)[0] = 0;
				forcoloring.at<Vec3b>(i, j)[1] = 0;
				forcoloring.at<Vec3b>(i, j)[2] = 255;
			}
		}
	}
	imshow("Input(Colored)", forcoloring);*/
#pragma endregion

	//Creating nx1 training sample features matrix from the DCT of the reference image
	int col_updated = dct_ref.cols;
	int row_updated = dct_ref.rows;
	Mat samples_train(col_updated * row_updated, 1, CV_32F);
	for (int i = 0; i < row_updated; ++i) {
		for (int j = 0; j < col_updated; ++j) {
			samples_train.at<float>(i + j * ref_image.rows, 0) = dct_ref.at<float>(i, j);
		}
	}

	//Performing the DCT on the Black and White Input image
	std::cout << "Performing the Discrete Cosine Transform of the grayscale input image" << std::endl;
	Mat input(input_image.size(), CV_32F);
	input_image.convertTo(input, CV_32F);
	int m1 = input.rows;
	int n1 = input.cols;
	int m3, n3 = 0;
	if (m1 % 2 == 0) {
		m3 = m1;
	}
	else {
		m3 = m1 + 1;
	}
	if (n1 % 2 == 0) {
		n3 = n1;
	}
	else {
		n3 = n1 + 1;
	}
	Mat padded2 = Mat::zeros(n1, m1, CV_32F);
	copyMakeBorder(input, padded2, 0, m3 - m1, 0, n3 - n1, BORDER_CONSTANT);
	Mat dct_input = Mat::zeros(padded2.rows, padded2.cols, CV_32FC1);
	dct(padded2, dct_input);
	imshow("Input(DCT)", dct_input);

	//Creating nx1 testing sample features matrix from the DCT of the black and white input image
	Mat samples_test(dct_input.cols * dct_input.rows, 1, CV_32F);
	for (int i = 0; i < dct_input.rows; ++i) {
		for (int j = 0; j < dct_input.cols; ++j) {
			samples_test.at<float>(i + j * ref_image.rows, 0) = dct_input.at<float>(i, j);

		}
	}

	//Creating a neighborhood of 600x100 training sample features and associating them with their respective label values
	std::cout << "Sampling (600x100) neighborhood of reference image features" << std::endl;
	Mat comb(600, 100, samples_train.type());
	int limit = comb.rows;
	std::vector<float> clc;
	int k = 0;
	for (int i = 0; i < limit / 3; i++) {
		for (int j = 0; j < labels.rows; j++) {
			int c = labels.at<int>(j, 0);
			if (c == 0 && k < 100) {
				comb.at<float>(i, k) = samples_train.at<float>(j, 0);
				++k;
			}
		}
		if (k == 100) {
			clc.push_back(0);
			k = 0;
		}
	}
	k = 0;
	for (int i = limit / 3; i < 2 * (limit / 3); i++) {
		for (int j = 0; j < labels.rows; j++) {
			int c = labels.at<int>(j, 0);
			if (c == 1 && k < 100) {
				comb.at<float>(i, k) = samples_train.at<float>(j, 0);
				++k;
			}
		}
		if (k == 100) {
			clc.push_back(1);
			k = 0;
		}
	}
	k = 0;
	for (int i = 2 * (limit / 3); i < limit; i++) {
		for (int j = 0; j < labels.rows; j++) {
			int c = labels.at<int>(j, 0);
			if (c == 2 && k < 100) {
				comb.at<float>(i, k) = samples_train.at<float>(j, 0);
				++k;
			}
		}
		if (k == 100) {
			clc.push_back(2);
			k = 0;
		}
	}

	//Testing Purposes
	//cv::FileStorage file13("comb.ext", cv::FileStorage::WRITE);
	//// Write to file!
	//file13 << "comb" << comb;
	//file13.release();

	//Coputing the mean values of the previously sampled pixels
	Mat comb_means(1, 100, comb.type());
	float sum = 0.0;
	float ntenp = 0.0;
	int cc = 0;
	float divsr = comb.rows;
	for (int i = 0; i < comb.cols; i++) {
		for (int j = 0; j < comb.rows; j++) {
			ntenp = comb.at<float>(j, i);
			sum += ntenp;
			cc += 1;
			if (cc == divsr - 1) {
				sum /= divsr;
				comb_means.at<float>(0, i) = sum;
				sum = 0.0;
				ntenp = 0.0;
			}
		}
		cc = 0;
	}

	//Testing purposes
	/*cv::FileStorage file23("means.ext", cv::FileStorage::WRITE);
	file23 << "means" << comb_means;
	file23.release();*/

	//Performing pca on the previously sampled features and then lda on the returned pca subspace
	//Computing the product of the eigen vectors returned by the pca and lda
	//Projecting previously computed eigen vector product onto lda subspace for each feature
	std::cout << "Performing PCA and LDA on sampled reference image features" << std::endl;
	Mat leigen;
	Mat eigvec_prod;
	Mat features;
	PCA pca_analysis(comb, Mat(), PCA::DATA_AS_ROW);
	Mat scaled = pca_analysis.project(comb);
	LDA lda(scaled, clc, 1);	
	Mat eigenvectors = lda.eigenvectors();
	lda.eigenvectors().convertTo(leigen, pca_analysis.eigenvectors.type());
	gemm(pca_analysis.eigenvectors, leigen, 1.0, Mat(), 0.0, eigvec_prod, GEMM_1_T);
	for (int i = 0; i < comb.rows; i++) {
		Mat pj = lda.subspaceProject(eigvec_prod, comb_means, comb.row(i));
		features.push_back(pj);
	}

	//Step-4: Classification of the labels for the input image features 
	//Setting up Kclassifier and classifying labels for the sampled testing features
	std::cout << "Performing the classification of the grayscale input features to their respective labels" << std::endl;
	Mat Results(samples_test.size(), samples_test.type());
	Ptr<ml::TrainData> trainingData;
	Ptr<ml::KNearest> kclassifier = ml::KNearest::create();
	kclassifier->setIsClassifier(true);
	kclassifier->setAlgorithmType(ml::KNearest::Types::BRUTE_FORCE);
	kclassifier->setDefaultK(5);
	kclassifier->train(features, ml::ROW_SAMPLE, clc);
	kclassifier->findNearest(samples_test, 5, Results);

	//For testing purposes
	//cv::FileStorage file3("labelsPred.ext", cv::FileStorage::WRITE);
	//// Write to file!
	//file3 << "Preds" << Results;
	//file3.release();

	//Reshaping the sampled test features back to 400x300 for performing the inverse discrete cosine transform
	Mat forcoloring;
	Mat show;
	Mat samples_w(dct_input.size(), dct_input.type());
	for (int i = 0; i < dct_input.rows; ++i) {
		for (int j = 0; j < dct_input.cols; ++j) {
			samples_w.at<float>(i, j) = samples_test.at<float>(i + j * ref_image.rows, 0);
		}
	}
	Mat res(samples_w.size(), samples_w.type());
	idct(samples_w, res);
	res.convertTo(show, CV_8U);
	imshow("IDCT Input", show);

	//Coloring the previously computed IDCT for checking the overall distributions of the labels returned by the Kclassifier
	cvtColor(show, forcoloring, COLOR_GRAY2BGR);
	for (int i = 0; i < forcoloring.rows; ++i) {
		for (int j = 0; j < forcoloring.cols; ++j) {
			float color = Results.at<float>(i + j * forcoloring.rows, 0);
			if (color == 0) {
				forcoloring.at<Vec3b>(i, j)[0] = centers.at<float>(0, 0);
				forcoloring.at<Vec3b>(i, j)[1] = centers.at<float>(0, 1);
				forcoloring.at<Vec3b>(i, j)[2] = centers.at<float>(0, 2);
			}
			else if (color == 1) {
				forcoloring.at<Vec3b>(i, j)[0] = centers.at<float>(1, 0);
				forcoloring.at<Vec3b>(i, j)[1] = centers.at<float>(1, 1);
				forcoloring.at<Vec3b>(i, j)[2] = centers.at<float>(1, 2);
			}
			else {
				forcoloring.at<Vec3b>(i, j)[0] = centers.at<float>(2, 0);
				forcoloring.at<Vec3b>(i, j)[1] = centers.at<float>(2, 1);
				forcoloring.at<Vec3b>(i, j)[2] = centers.at<float>(2, 2);
			}
		}
	}

	//Step:5 - Gnerating the colored segmentation of the input image
	//Setting up and applying GrabCut algo for segmentation of foreground and background in the black and white input image
	std::cout << "Extracting the foreground" << std::endl;
	Mat cinput;
	Mat background;
	cv::Mat result;
	cv::Mat bgModel, fgModel;
	cvtColor(show, cinput, COLOR_GRAY2BGR);
	cv::Mat foreground(show.size(), CV_8UC3, cv::Scalar(255, 255, 255));
	cv::Rect rect(120, 60, cinput.cols - 20, cinput.rows - 110);
	cv::grabCut(cinput,    // input image
		result,   // segmentation result
		rect,// rectangle containing foreground
		bgModel, fgModel, // models
		1,        // number of iterations
		cv::GC_INIT_WITH_RECT); // use rectangle

	//Getting pixels likely to be the foreground
	cv::compare(result, cv::GC_PR_FGD, result, cv::CMP_EQ);
	cinput.copyTo(foreground, result);
	imshow("Foreground", foreground);
	cinput.copyTo(background, ~result);
	imshow("Background", background);

	//For testing Purposes
	/*cv::FileStorage file1299("foregrnd.ext", cv::FileStorage::WRITE);
	file1299 << "foregrnd" << foreground;
	file1299.release();*/


	//Setting up canny edge detection algo for edges detection in the foreground image
	std::cout << "Edges detection and contours drawing" << std::endl;
	int lowThreshold=75;
	int ratio = 3;
	int kernel_size = 3;
	Mat dst, detected_edges;
	RNG rng(12345);
	Mat processedFor;
	//Creating a matrix of the same type and size as src (for dst)
	dst.create(show.size(), show.type());
	//Reducing noise with a kernel 3x3
	cvtColor(foreground, processedFor, COLOR_BGR2GRAY);
	blur(processedFor, detected_edges, cv::Size(3, 3));
	Canny(detected_edges, detected_edges, lowThreshold, lowThreshold*ratio, kernel_size);
	/// Using Canny's output as a mask, we display our result
	dst = Scalar::all(0);
	show.copyTo(dst, detected_edges);
	imshow("edges", dst);


	//Finding and drawing contours for the edges detected previously by canny's algo and bounding them in a rectangle
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;
	//Finding contours
	findContours(detected_edges, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point(0, 0));
	//Drawing contours
	vector<Rect> boundRect(contours.size());
	vector<vector<Point> > contours_poly(contours.size());
	for (int i = 0; i < contours.size(); i++)
	{
		approxPolyDP(contours[i], contours_poly[i], 3, true);
		boundRect[i] = boundingRect(contours_poly[i]);
	}
	Mat drawing = Mat::zeros(detected_edges.size(), CV_8UC3);
	for (size_t i = 0; i < contours.size(); i++)
	{
		Scalar color = Scalar(rng.uniform(0, 256), rng.uniform(0, 256), rng.uniform(0, 256));
		drawContours(drawing, contours_poly, (int)i, color);
		rectangle(drawing, boundRect[i].tl(), boundRect[i].br(), color, 2);
	}

	//Generating a dictionary of the area and points inside the bounding rectangle for input image color segmentation
	multimap<double, Point> recPts;
	for (size_t r = 0; r < boundRect.size(); r++) {
		int left = boundRect[r].x;
		int top = boundRect[r].y;
		int width = boundRect[r].width;
		int height = boundRect[r].height;
		int x_end = left + width;
		int y_end = top + height;
		double aRec = boundRect[r].area();
		for (size_t x = left; x < x_end; x++)
		{
			for (size_t y = top; y < y_end; y++)
			{
				Point p(x, y);
				recPts.insert(make_pair(aRec, p));
			}
		}
	}
	imshow("Contours", drawing);
	
	//Coloring only the pixels associated with the foreground
	std::cout << "Generating the colored segmentation of the input image" << std::endl;
	Mat fg;
	cvtColor(foreground, fg, COLOR_BGR2GRAY);
	fg.convertTo(fg, CV_32F);
	std::cout << fg.size() << ":" << fg.channels() << ":" << type2str(fg.type()) << std::endl;
	for (auto itr = recPts.begin(); itr != recPts.end(); itr++) {
		float val = fg.at<float>(itr->second);
		if (val != 255.0) {
			foreground.at<Vec3b>(itr->second)[0] = centers.at<float>(1, 0);
			foreground.at<Vec3b>(itr->second)[1] = centers.at<float>(1, 1);
			foreground.at<Vec3b>(itr->second)[2] = centers.at<float>(1, 2);
		}
		else {
			foreground.at<Vec3b>(itr->second)[0] = centers.at<float>(2, 0);
			foreground.at<Vec3b>(itr->second)[1] = centers.at<float>(2, 1);
			foreground.at<Vec3b>(itr->second)[2] = centers.at<float>(2, 2);
		}
	}
	for (auto itr = recPts.begin(); itr != recPts.end(); itr++) {
		float val = fg.at<float>(itr->second);
		if (val != 255.0 && (itr->first == 15.0 || itr->first == 21.0 || 
			itr->first == 27.0 || itr->first == 28.0 || 
			itr->first == 32.0 || itr->first == 56.0 ||
			itr->first == 154.0 || itr->first == 192.0 || 
			itr->first == 176.0 || itr->first == 266.0 || 
			itr->first == 396.0 || itr->first == 442.0)) {
			foreground.at<Vec3b>(itr->second)[0] = centers.at<float>(0, 0);
			foreground.at<Vec3b>(itr->second)[1] = centers.at<float>(0, 1);
			foreground.at<Vec3b>(itr->second)[2] = centers.at<float>(0, 2);
		}
	}
	//Background color segmentation
	Mat bg;
	cvtColor(foreground, bg, COLOR_BGR2GRAY);
	bg.convertTo(bg, CV_32F);
	for (int i = 0; i < foreground.rows; i++) {
		for (int j = 0; j < foreground.cols; j++) {
			float val = fg.at<float>(i,j);
			if (val == 255.0) {
				foreground.at<Vec3b>(i,j)[0] = centers.at<float>(2, 0);
				foreground.at<Vec3b>(i,j)[1] = centers.at<float>(2, 1);
				foreground.at<Vec3b>(i,j)[2] = centers.at<float>(2, 2);
			}
		}
	}
	imshow("Colored Input Segmentation", foreground);



	//Displaying section
	String wn1 = "Reference Image"; //Name of the window
	namedWindow(wn1); // Create a window
	imshow(wn1, ref_image); // Show our image inside the created window.

	String wn2 = "Segmented Image";
	namedWindow(wn2);
	imshow(wn2, rsegmented_Image);

	String wn3 = "Input Image";
	namedWindow(wn3);
	imshow(wn3, input_image);

	String wn4 = "Reference Image(Luminance Chnl - DCT)";
	namedWindow(wn4);
	imshow(wn4, dct_ref);

	String wn5 = "Input(colored)";
	namedWindow(wn5);
	imshow(wn5, forcoloring);

	//Cleaning up section
	//destroyWindow(wn2); //destroy the created window
	waitKey(0); // Wait for any keystroke in the window
	//destroyWindow(wn1); //destroy the created window
	return 0;
}