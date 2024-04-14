#include <stdio.h> 
#include <filesystem>
#include <map>
#include <opencv2/opencv.hpp> 
#include <iostream>
#include <fstream>

namespace fs = std::filesystem;
using namespace fs;
using namespace cv;

std::map <std::string, cv::Mat> images;

int readFiles(const std::string& folderPath) {

    // for every entry in my input folder
    for (const auto& entry : fs::directory_iterator(folderPath)) {


        const std::string filePath = entry.path().string();
        std::ifstream inputFile(filePath);


        if (!inputFile.is_open()) {
            std::cerr << "Error opening CSV file: " << filePath << std::endl;
            return 1;
        }


        std::vector<std::vector<int>> pixelValues;
        std::string line;


        while (std::getline(inputFile, line)) {
            std::vector<int> row;
            std::stringstream ss(line);
            int value;
            while (ss >> value) {
                row.push_back(value);
                if (ss.peek() == ',') {
                    ss.ignore();
                }
            }
            pixelValues.push_back(row);
        }

        inputFile.close();

        // Convert the vector of vectors to a Mat object
        cv::Mat image(pixelValues.size(), pixelValues[0].size(), CV_8U);
        for (int i = 0; i < pixelValues.size(); ++i) { // for each row value
            for (int j = 0; j < pixelValues[0].size(); ++j) { // for each col val in that row
                image.at<uchar>(i, j) = static_cast<uchar>(pixelValues[i][j]);
            }
        }

        std::string fileName = fs::path(filePath).filename().string();
        std::cout << fileName;
        images[fileName] = image;

        }

    return 0;

}


int getStats(const std::map<std::string, cv::Mat>& images) {
    std::ofstream statsFile("statsFile.csv");

    if (!statsFile.is_open()) {
        std::cerr << "Error opening stats file!" << std::endl;
        return 1;
    }

    statsFile << "FileName,Min,Max,Sum,Mean" << std::endl;

    for (const auto& entry : images) {
        const std::string& fileName = entry.first;
        const cv::Mat& image = entry.second;

       cv::Scalar mean, stddev;
        double minVal, maxVal, sumVal;
        cv::meanStdDev(image, mean, stddev);
        cv::minMaxLoc(image, &minVal, &maxVal);
        sumVal = cv::sum(image)[0];

        // Write data to the CSV file
        statsFile << fileName << ","
                  << minVal << ","
                  << maxVal << ","
                  << sumVal << ","
                  << mean.val[0] << std::endl;
    }

    statsFile.close();

    return 0;
}


int addOverlay(const std::map<std::string, cv::Mat>& images) {


    for (const auto& entry : images) {
        const std::string& fileName = entry.first;
        const cv::Mat& image = entry.second;

        int fontFace = cv::FONT_HERSHEY_SIMPLEX;
        double fontScale = 0.5;
        int thickness = 1;
        cv::Point textOrg(10, 20);
        cv::putText(image, fileName, textOrg, fontFace, fontScale, cv::Scalar(255, 255, 255), thickness);
    }

    return 0;
}


int saveImages(const std::map<std::string, cv::Mat>& images) {


    for (const auto& entry : images) {
        const std::string& fileName = entry.first;
        const cv::Mat& image = entry.second;
        std::string outputFolderPath = "output_images";
        std::filesystem::create_directory(outputFolderPath);


        size_t lastDotPos = fileName.find_last_of('.');
        std::string baseFileName = fileName.substr(0, lastDotPos);


        std::string outputFilePath = outputFolderPath + "/" + baseFileName + "_processed" +".png";
        cv::imwrite(outputFilePath, image);


        std::ofstream outputCSV;
        std::string outputCSVPath = outputFolderPath + "/" + baseFileName +"_processed" +".csv";
        outputCSV.open(outputCSVPath);
        

        for (int i = 0; i < image.rows; ++i) {
            for (int j = 0; j < image.cols; ++j) {
            // Write the pixel value to the CSV file
                outputCSV << static_cast<int>(image.at<uchar>(i, j));

                // Add a comma after each value (except the last one in the row)
                if (j < image.cols - 1) {
                    outputCSV << ",";
                }  
            }
            outputCSV << "\n";  // Move to the next line for the next row
        }

    // Close the CSV file
    outputCSV.close();

    }

    return 0;
}


int fourierTransform(const std::map<std::string, cv::Mat>& images) {

    std::string outputFolderPath = "output_images";
    std::filesystem::create_directory(outputFolderPath);

    for (const auto& entry : images) {
        const std::string& fileName = entry.first;
        const cv::Mat& image = entry.second;

        cv::Mat initialImage;
        cv::Mat rowImage;
        cv::Mat colImage;
        cv::Mat finalImage;
        cv::Mat altImage;

        image.convertTo(initialImage, CV_64F);


        cv::dft(initialImage, altImage);

        // do a 1d on the image rows
        cv::dft(initialImage, rowImage, DFT_ROWS);
        // transpose it and do a 1d on the columns
        colImage = rowImage.t();
        cv::dft(colImage, finalImage, DFT_ROWS);
        // then transpose it back....
        finalImage = finalImage.t();

        // Display the original and Fourier-transformed images
        cv::imshow("Original Image", initialImage);

        // Compute the magnitude spectrum for visualization
        cv::Mat magnitude;
        cv::Mat altMagnitude;

        cv::magnitude(finalImage, finalImage, magnitude);
        cv::magnitude(altImage, altImage, altMagnitude);

        size_t lastDotPos = fileName.find_last_of('.');
        std::string baseFileName = fileName.substr(0, lastDotPos);

        imshow("alt", altImage);
        imshow("transposed", finalImage);

        std::string outputFilePath = outputFolderPath + "/" + baseFileName + "_fft" +".png";
        imwrite(outputFilePath, altImage);

    }
    
    return 0;
}




int main(int argc, char** argv) {

    //readFiles(argv[1]);
    //getStats(images);
    //addOverlay(images);
    //saveImages(images);
    //fourierTransform(images);

    return 0; 
}