//
// Created by moinshaikh on 7/2/26.
//

#include "Database/FaceDatabase.hpp"

#include <iostream>
#include <fstream>
#include <filesystem>
#include <opencv2/imgcodecs.hpp>
#include <cstring>

namespace Database {

    FaceDatabase::FaceDatabase()
    {
        // Ensure photos directory exists
        std::filesystem::create_directories(getPhotosDir());
        load();
    }

    FaceDatabase::~FaceDatabase()
    {
        save();
    }

    std::string FaceDatabase::getPhotosDir() const
    {
        return PHOTOS_DIR;
    }

    int FaceDatabase::addPerson(const std::string& name,
                                 const std::vector<float>& embedding,
                                 const cv::Mat& facePhoto)
    {
        FaceRecord record;
        record.id = nextId_++;
        record.name = name;
        record.embedding = embedding;

        // Save the face photo to disk
        std::string photoFilename = getPhotosDir() + "/person_" + std::to_string(record.id) + ".jpg";
        cv::imwrite(photoFilename, facePhoto);
        record.photoPath = photoFilename;

        records_.push_back(record);
        save();

        std::cout << "[Database] Added person '" << name << "' with ID " << record.id << std::endl;
        return record.id;
    }

    int FaceDatabase::findMatch(const std::vector<float>& embedding,
                                 float threshold) const
    {
        if (records_.empty())
        {
            return -1;
        }

        float bestSimilarity = -1.0f;
        int bestIndex = -1;

        for (int i = 0; i < static_cast<int>(records_.size()); ++i)
        {
            float sim = cosineSimilarity(embedding, records_[i].embedding);
            if (sim > bestSimilarity)
            {
                bestSimilarity = sim;
                bestIndex = i;
            }
        }

        if (bestSimilarity >= threshold)
        {
            std::cout << "[Database] Match found: '" << records_[bestIndex].name
                      << "' (ID=" << records_[bestIndex].id
                      << ", similarity=" << bestSimilarity << ")" << std::endl;
            return bestIndex;
        }

        std::cout << "[Database] No match above threshold (best=" << bestSimilarity
                  << " < threshold=" << threshold << ")" << std::endl;
        return -1;
    }

    float FaceDatabase::cosineSimilarity(const std::vector<float>& a,
                                          const std::vector<float>& b)
    {
        float dot = 0.0f, normA = 0.0f, normB = 0.0f;
        for (size_t i = 0; i < a.size(); ++i)
        {
            dot += a[i] * b[i];
            normA += a[i] * a[i];
            normB += b[i] * b[i];
        }
        float denom = std::sqrt(normA) * std::sqrt(normB);
        if (denom < 1e-10f) return 0.0f;
        return dot / denom;
    }

    const FaceRecord& FaceDatabase::getRecord(int index) const
    {
        return records_[index];
    }

    std::string FaceDatabase::getName(int index) const
    {
        return records_[index].name;
    }

    void FaceDatabase::reset()
    {
        // Clear in-memory records
        records_.clear();
        nextId_ = 1;

        // Delete all saved photo files
        if (std::filesystem::exists(getPhotosDir()))
        {
            for (const auto& entry : std::filesystem::directory_iterator(getPhotosDir()))
            {
                std::filesystem::remove(entry.path());
            }
            std::cout << "[Database] Deleted all photos from " << getPhotosDir() << std::endl;
        }

        // Overwrite the binary file with an empty database
        std::ofstream ofs(DB_FILE, std::ios::binary | std::ios::trunc);
        if (ofs)
        {
            uint32_t magic = MAGIC;
            uint32_t version = VERSION;
            uint32_t numRecords = 0;
            ofs.write(reinterpret_cast<const char*>(&magic), sizeof(magic));
            ofs.write(reinterpret_cast<const char*>(&version), sizeof(version));
            ofs.write(reinterpret_cast<const char*>(&numRecords), sizeof(numRecords));
            ofs.close();
        }

        std::cout << "[Database] Database has been reset. All records deleted." << std::endl;
    }

    void FaceDatabase::save()
    {
        std::ofstream ofs(DB_FILE, std::ios::binary);
        if (!ofs)
        {
            std::cerr << "[Database] Error: Could not write to " << DB_FILE << std::endl;
            return;
        }

        // Header: magic + version + num_records
        uint32_t numRecords = static_cast<uint32_t>(records_.size());
        ofs.write(reinterpret_cast<const char*>(&MAGIC), sizeof(MAGIC));
        ofs.write(reinterpret_cast<const char*>(&VERSION), sizeof(VERSION));
        ofs.write(reinterpret_cast<const char*>(&numRecords), sizeof(numRecords));

        for (const auto& rec : records_)
        {
            // ID
            ofs.write(reinterpret_cast<const char*>(&rec.id), sizeof(rec.id));

            // Name length + name
            uint32_t nameLen = static_cast<uint32_t>(rec.name.size());
            ofs.write(reinterpret_cast<const char*>(&nameLen), sizeof(nameLen));
            ofs.write(rec.name.data(), nameLen);

            // Embedding
            uint32_t embSize = static_cast<uint32_t>(rec.embedding.size());
            ofs.write(reinterpret_cast<const char*>(&embSize), sizeof(embSize));
            ofs.write(reinterpret_cast<const char*>(rec.embedding.data()),
                      embSize * sizeof(float));

            // Photo path length + path
            uint32_t pathLen = static_cast<uint32_t>(rec.photoPath.size());
            ofs.write(reinterpret_cast<const char*>(&pathLen), sizeof(pathLen));
            ofs.write(rec.photoPath.data(), pathLen);
        }

        std::cout << "[Database] Saved " << numRecords << " records to " << DB_FILE << std::endl;
    }

    void FaceDatabase::load()
    {
        std::ifstream ifs(DB_FILE, std::ios::binary);
        if (!ifs)
        {
            std::cout << "[Database] No existing database file found. Starting fresh." << std::endl;
            return;
        }

        uint32_t magic, version, numRecords;
        ifs.read(reinterpret_cast<char*>(&magic), sizeof(magic));
        if (magic != MAGIC)
        {
            std::cerr << "[Database] Invalid database file (bad magic)." << std::endl;
            return;
        }
        ifs.read(reinterpret_cast<char*>(&version), sizeof(version));
        if (version != VERSION)
        {
            std::cerr << "[Database] Unsupported database version." << std::endl;
            return;
        }
        ifs.read(reinterpret_cast<char*>(&numRecords), sizeof(numRecords));

        records_.clear();
        records_.reserve(numRecords);
        int maxId = 0;

        for (uint32_t i = 0; i < numRecords; ++i)
        {
            FaceRecord rec;

            // ID
            ifs.read(reinterpret_cast<char*>(&rec.id), sizeof(rec.id));
            if (rec.id > maxId) maxId = rec.id;

            // Name
            uint32_t nameLen;
            ifs.read(reinterpret_cast<char*>(&nameLen), sizeof(nameLen));
            rec.name.resize(nameLen);
            ifs.read(rec.name.data(), nameLen);

            // Embedding
            uint32_t embSize;
            ifs.read(reinterpret_cast<char*>(&embSize), sizeof(embSize));
            rec.embedding.resize(embSize);
            ifs.read(reinterpret_cast<char*>(rec.embedding.data()),
                     embSize * sizeof(float));

            // Photo path
            uint32_t pathLen;
            ifs.read(reinterpret_cast<char*>(&pathLen), sizeof(pathLen));
            rec.photoPath.resize(pathLen);
            ifs.read(rec.photoPath.data(), pathLen);

            records_.push_back(rec);
        }

        nextId_ = maxId + 1;
        std::cout << "[Database] Loaded " << numRecords << " records from " << DB_FILE << std::endl;
    }

} // namespace Database