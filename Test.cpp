#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>

#include "YOLO11.hpp" // Ensure YOLO11.hpp or other version is in your include path
#include <vector>

int main()
{
    // Configuration parameters
    //const std::string labelsPath = "models/coco.names";       // Path to class labels
    //const std::string modelPath  = "models/yolo11n.onnx";     // Path to YOLO11 model
    //const std::string imagePath  = "data/Catedra-UNESCO-UPS-Imagen-Aula.jpg";           // Path to input image
    // Configuration parameters
    const std::string labelsPath = "models/emergenci.names";       // Path to class labels
    const std::string modelPath  = "models/best.onnx";     // Path to YOLO11 model
    const std::string imagePath  = "data/001.jpg";           // Path to input image
    //bool isGPU = false;
    bool isGPU = true;                                           // Set to false for CPU processing

    // Initialize the YOLO11 detector
    YOLO11Detector detector(modelPath, labelsPath, isGPU);

    // Load an image
    cv::Mat image = cv::imread(imagePath);

    // Perform object detection to get bboxs
    std::vector<Detection> detections = detector.detect(image);

    // Draw bounding boxes on the image
    detector.drawBoundingBoxMask(image, detections);

    // Display the annotated image
    cv::imshow("YOLO11 Detections", image);
    cv::waitKey(0); // Wait indefinitely until a key is pressed

    cv::destroyAllWindows();

    cv::Mat frame;

    //cv::VideoCapture video("/dev/video0");
    cv::VideoCapture video("data/testvideo.mp4");

    if (video.isOpened()){
        cv::namedWindow("Video", cv::WINDOW_AUTOSIZE);
        //Variables para calculos FPS
        double tFrecuencia = cv::getTickFrequency();
        int contadorFrames = 0 ;
        double tiempoInicial = cv::getTickCount();

        const int smoothingFactor = 10;
        double smoothedFPS = 0;

        while(3==3){
            video >> frame;

            if(frame.empty()){
                break;
            }

            detections = detector.detect(frame);
            detector.drawBoundingBoxMask(frame, detections);

            //Calculo de FPS
            contadorFrames++;
            double tiempoFinal = cv::getTickCount();
            double tiempoTranscurrido = (tiempoFinal-tiempoInicial)/tFrecuencia;
            if(tiempoInicial >= 1.0){
                double fps = contadorFrames/tiempoTranscurrido;
                smoothedFPS = 0.9 * smoothedFPS + 0.1 * fps;
                std::cout << "FPS: " << fps << std::endl;
                contadorFrames = 0 ;
                tiempoInicial = cv::getTickCount();
            }

            //Mostrar en video
            std::ostringstream fpsText;
            fpsText << "FPS: " << int(smoothedFPS);

            // Display the FPS text in the top-left corner
            cv::putText(frame, fpsText.str(), cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2);

            cv::imshow("Video", frame);

            if(cv::waitKey(23)==27){
                break;
            }
        }

        video.release();
        cv::destroyAllWindows();

    }

    return 0;
}