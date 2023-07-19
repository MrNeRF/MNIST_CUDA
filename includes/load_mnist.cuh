#pragma once

#include <filesystem>
#include <vector>

class MNISTDataSet {
public:
    void ReadMNISTData(const std::filesystem::path& image_file_path, const std::filesystem::path& label_file_path);

    const std::vector<std::vector<float>>& GetImages() const { return _images; }

    const std::vector<uint8_t>& GetLabels() const { return _labels; }

private:
    uint32_t flipEndianness(const uint32_t u);

private:
    std::vector<std::vector<float>> _images;
    std::vector<uint8_t> _labels;
};
