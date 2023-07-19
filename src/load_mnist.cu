#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "load_mnist.cuh"
#include "stb_image_write.h"
#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iostream>

// Reads the MNIST Training data from the given files
void MNISTDataSet::ReadMNISTData(const std::filesystem::path& image_file_path, const std::filesystem::path& label_file_path) {
    // Open files
    std::ifstream image_file(image_file_path.string(), std::ios::binary);
    std::ifstream label_file(label_file_path.string(), std::ios::binary);

    std::cout << "Read MNIST data from " << image_file_path << " and " << label_file_path << std::endl;
    if (image_file.is_open() && label_file.is_open()) {
        int magic_number = 0;
        int number_of_images = 0;
        int n_rows = 0;
        int n_cols = 0;

        image_file.read((char*)&magic_number, sizeof(magic_number));
        magic_number = flipEndianness(magic_number);

        if (magic_number != 2051)
            throw std::runtime_error("Invalid MNIST image file!");

        image_file.read((char*)&number_of_images, sizeof(number_of_images));
        number_of_images = flipEndianness(number_of_images);

        image_file.read((char*)&n_rows, sizeof(n_rows));
        n_rows = flipEndianness(n_rows);

        image_file.read((char*)&n_cols, sizeof(n_cols));
        n_cols = flipEndianness(n_cols);

        // Reading labels
        int number_of_labels = 0;
        label_file.read((char*)&magic_number, sizeof(magic_number));
        magic_number = flipEndianness(magic_number);

        if (magic_number != 2049)
            throw std::runtime_error("Invalid MNIST label file!");

        label_file.read((char*)&number_of_labels, sizeof(number_of_labels));
        number_of_labels = flipEndianness(number_of_labels);

        if (number_of_images != number_of_labels)
            throw std::runtime_error("The number of labels did not match the number of images!");

        // Reading image data and label data
        _images.resize(number_of_images, std::vector<float>(n_rows * n_cols));
        _labels.resize(number_of_labels);

        for (int i = 0; i < number_of_images; ++i) {
            std::vector<uint8_t> buffer(n_rows * n_cols);
            image_file.read((char*)buffer.data(), n_rows * n_cols);
            std::transform(buffer.begin(), buffer.end(), _images[i].begin(), [](uint8_t c) { return c / 255.0f; });
            label_file.read((char*)&_labels[i], 1);
        }
        std::cout << "Loaded " << number_of_images << " images." << std::endl;
        std::cout << "Loaded " << number_of_labels << " labels." << std::endl;
    } else {
        throw std::runtime_error("Unable to open file!");
    }

    // for (int i = 0; i < 10; i++) {
    //     char filename[50];
    //     sprintf(filename, "image%d.png", i);
    //     std::vector<uint8_t> image(28 * 28);
    //     std::transform(_images[i].begin(), _images[i].end(), image.begin(), [](float f) { return static_cast<uint8_t>(f * 255); });
    //     stbi_write_png(filename, 28, 28, 1, image.data(), 28);
    // }
}

// Flips the endianness of the given uint32_t for the MNIST data
uint32_t MNISTDataSet::flipEndianness(const uint32_t u) {
    return ((u >> 24) & 0xff) | ((u << 8) & 0xff0000) | ((u >> 8) & 0xff00) | ((u << 24) & 0xff000000);
}
