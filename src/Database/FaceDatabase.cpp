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

        // Add first embedding + photo
        record.embeddings.push_back(embedding);
        std::string photoFilename = getPhotosDir() + "/person_" + std::to_string(record.id)
                                     + "_angle_0.jpg";
        cv::imwrite(photoFilename, facePhoto);
        record.photoPaths.push_back(photoFilename);

        records_.push_back(record);
        save();

        std::cout << "[Database] Added person '" << name << "' with ID " << record.id
                  << " (1 face angle)" << std::endl;
        return record.id;
    }

    int FaceDatabase::addPerson(int id,
                                 const std::string& name,
                                 const std::vector<float>& embedding,
                                 const cv::Mat& facePhoto)
    {
        if (hasId(id))
        {
            std::cerr << "[Database] Cannot add person: ID " << id << " already exists." << std::endl;
            return -1;
        }

        FaceRecord record;
        record.id = id;
        record.name = name;

        // Add first embedding + photo
        record.embeddings.push_back(embedding);
        std::string photoFilename = getPhotosDir() + "/person_" + std::to_string(record.id)
                                     + "_angle_0.jpg";
        cv::imwrite(photoFilename, facePhoto);
        record.photoPaths.push_back(photoFilename);

        records_.push_back(record);

        // Update nextId_ to be greater than any existing ID
        if (id >= nextId_)
            nextId_ = id + 1;

        save();

        std::cout << "[Database] Added person '" << name << "' with manual ID " << record.id
                  << " (1 face angle)" << std::endl;
        return record.id;
    }

    bool FaceDatabase::addFaceToPerson(int id,
                                        const std::vector<float>& embedding,
                                        const cv::Mat& facePhoto)
    {
        for (auto& rec : records_)
        {
            if (rec.id == id)
            {
                int angleIndex = static_cast<int>(rec.embeddings.size());
                rec.embeddings.push_back(embedding);

                std::string photoFilename = getPhotosDir() + "/person_" + std::to_string(rec.id)
                                             + "_angle_" + std::to_string(angleIndex) + ".jpg";
                cv::imwrite(photoFilename, facePhoto);
                rec.photoPaths.push_back(photoFilename);

                save();

                std::cout << "[Database] Added face angle " << angleIndex
                          << " to person '" << rec.name << "' (ID=" << rec.id << ")" << std::endl;
                return true;
            }
        }

        std::cerr << "[Database] Cannot add face: person ID " << id << " not found." << std::endl;
        return false;
    }

    bool FaceDatabase::hasId(int id) const
    {
        for (const auto& rec : records_)
        {
            if (rec.id == id)
                return true;
        }
        return false;
    }

    int FaceDatabase::findMatch(const std::vector<float>& embedding,
                                 float threshold) const
    {
        float dummyScore;
        return findMatchWithScore(embedding, dummyScore, threshold);
    }

    int FaceDatabase::findMatchWithScore(const std::vector<float>& embedding,
                                          float& bestScore,
                                          float threshold) const
    {
        bestScore = -1.0f;
        if (records_.empty())
        {
            return -1;
        }

        int bestPersonIndex = -1;
        int bestAngleIndex = -1;

        for (int personIdx = 0; personIdx < static_cast<int>(records_.size()); ++personIdx)
        {
            const auto& rec = records_[personIdx];

            // Compare against ALL embeddings for this person
            for (int angleIdx = 0; angleIdx < static_cast<int>(rec.embeddings.size()); ++angleIdx)
            {
                float sim = cosineSimilarity(embedding, rec.embeddings[angleIdx]);
                if (sim > bestScore)
                {
                    bestScore = sim;
                    bestPersonIndex = personIdx;
                    bestAngleIndex = angleIdx;
                }
            }
        }

        if (bestScore >= threshold)
        {
            std::cout << "[Database] Match found: '" << records_[bestPersonIndex].name
                      << "' (ID=" << records_[bestPersonIndex].id
                      << ", angleId=" << bestAngleIndex
                      << ", similarity=" << bestScore << ")" << std::endl;
            return bestPersonIndex;
        }

        std::cout << "[Database] No match above threshold (best=" << bestScore
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

    int FaceDatabase::totalFaces() const
    {
        int count = 0;
        for (const auto& rec : records_)
        {
            count += static_cast<int>(rec.embeddings.size());
        }
        return count;
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

    // ---- Serialization helpers ----

    template<typename Stream>
    static void writeString(Stream& s, const std::string& str)
    {
        uint32_t len = static_cast<uint32_t>(str.size());
        s.write(reinterpret_cast<const char*>(&len), sizeof(len));
        s.write(str.data(), len);
    }

    template<typename Stream>
    static std::string readString(Stream& s)
    {
        uint32_t len;
        s.read(reinterpret_cast<char*>(&len), sizeof(len));
        std::string str;
        str.resize(len);
        s.read(str.data(), len);
        return str;
    }

    template<typename Stream>
    static void writeEmbedding(Stream& s, const std::vector<float>& emb)
    {
        uint32_t embSize = static_cast<uint32_t>(emb.size());
        s.write(reinterpret_cast<const char*>(&embSize), sizeof(embSize));
        s.write(reinterpret_cast<const char*>(emb.data()), embSize * sizeof(float));
    }

    template<typename Stream>
    static std::vector<float> readEmbedding(Stream& s)
    {
        uint32_t embSize;
        s.read(reinterpret_cast<char*>(&embSize), sizeof(embSize));
        std::vector<float> emb(embSize);
        s.read(reinterpret_cast<char*>(emb.data()), embSize * sizeof(float));
        return emb;
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

            // Name
            writeString(ofs, rec.name);

            // Number of face angles
            uint32_t numAngles = static_cast<uint32_t>(rec.embeddings.size());
            ofs.write(reinterpret_cast<const char*>(&numAngles), sizeof(numAngles));

            // For each angle: embedding + photo path
            for (uint32_t a = 0; a < numAngles; ++a)
            {
                writeEmbedding(ofs, rec.embeddings[a]);

                if (a < rec.photoPaths.size())
                    writeString(ofs, rec.photoPaths[a]);
                else
                    writeString(ofs, "");  // fallback empty path
            }
        }

        std::cout << "[Database] Saved " << numRecords << " persons, "
                  << totalFaces() << " total faces to " << DB_FILE << std::endl;
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
        if (version > VERSION)
        {
            std::cerr << "[Database] Unsupported database version: " << version << std::endl;
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

            if (version == 1)
            {
                // --- v1 format: single embedding + single photo path ---
                // Name
                rec.name = readString(ifs);

                // Single embedding
                std::vector<float> emb = readEmbedding(ifs);
                rec.embeddings.push_back(emb);

                // Single photo path
                std::string path = readString(ifs);
                rec.photoPaths.push_back(path);

                records_.push_back(rec);
            }
            else
            {
                // --- v2+ format: name, numAngles, then angle data ---
                rec.name = readString(ifs);

                uint32_t numAngles;
                ifs.read(reinterpret_cast<char*>(&numAngles), sizeof(numAngles));

                for (uint32_t a = 0; a < numAngles; ++a)
                {
                    std::vector<float> emb = readEmbedding(ifs);
                    rec.embeddings.push_back(emb);

                    std::string path = readString(ifs);
                    rec.photoPaths.push_back(path);
                }

                records_.push_back(rec);
            }
        }

        nextId_ = maxId + 1;
        std::cout << "[Database] Loaded " << numRecords << " persons, "
                  << totalFaces() << " total faces from " << DB_FILE << std::endl;
    }

} // namespace Database